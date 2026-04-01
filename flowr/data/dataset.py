import gzip
import io
import itertools
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional, Union

import lmdb
import numpy as np
import torch

from flowr.util.molrepr import GeometricMol, GeometricMolBatch
from flowr.util.pocket import PocketComplex, PocketComplexBatch

# *** Util functions ***


def load_smol_data(
    data_path, smol_cls, remove_hs=False, remove_aromaticity=False, skip_non_valid=False
):
    data_path = Path(data_path)

    # TODO handle having a directory with batched data files
    if data_path.is_dir():
        raise NotImplementedError()

    # TODO maybe read in chunks if this is too big
    bytes_data = data_path.read_bytes()
    data = smol_cls.from_bytes(
        bytes_data,
        remove_hs=remove_hs,
        remove_aromaticity=remove_aromaticity,
        skip_non_valid=skip_non_valid,
    )
    return data


def load_npz_data(data_path, smol_cls):
    data_path = Path(data_path)

    # TODO handle having a directory with batched data files
    if data_path.is_dir():
        raise NotImplementedError()

    data = smol_cls.from_numpy(data_path)
    return data


# *** Abstract class for all Smol data types ***


class SmolDataset(ABC, torch.utils.data.Dataset):
    def __init__(self, smol_data, data_cls, transform=None):
        super().__init__()

        self._data = smol_data
        self.data_cls = data_cls
        self.transform = transform

    @property
    def hparams(self):
        return {}

    @property
    def lengths(self):
        return self._data.seq_length

    def __len__(self):
        return self._data.batch_size

    def __getitem__(self, item):
        molecule = self._data[item]
        if self.transform is not None:

            molecule = self.transform(molecule)

        return molecule

    @classmethod
    @abstractmethod
    def load(cls, data_path, transform=None, **kwargs):
        pass


# *** SmolDataset implementations ***


class GeometricDataset(SmolDataset):
    def sample(self, n_items, replacement=False):
        mol_samples = np.random.choice(
            self._data.to_list(), n_items, replace=replacement
        )
        data = self.data_cls.from_list(mol_samples)
        return GeometricDataset(data, self.data_cls, transform=self.transform)

    def sample_n_molecules_per_target(self, n_molecules):
        mol_samples = [
            [system for _ in range(n_molecules)] for system in self._data.to_list()
        ]
        mol_samples = list(itertools.chain(*mol_samples))
        data = self.data_cls.from_list(mol_samples)
        return GeometricDataset(data, self.data_cls, transform=self.transform)

    def split(self, idx, n_chunks):
        chunks = self._data.split(n_chunks)[idx - 1]
        data = self.data_cls.from_list(chunks)
        return GeometricDataset(data, self.data_cls, transform=self.transform)

    def ddp_split(self, num_replicas: int, rank: int) -> "GeometricDataset":
        data_list = self._data.to_list()
        dataset_size = len(data_list)
        local_data = [data_list[i] for i in range(rank, dataset_size, num_replicas)]
        new_data = self.data_cls.from_list(local_data)
        return GeometricDataset(new_data, self.data_cls, transform=self.transform)

    @classmethod
    def load(
        cls,
        data_path,
        dataset="geom-drugs",
        transform=None,
        remove_hs=False,
        remove_aromaticity=False,
        skip_non_valid=False,
        min_size=None,
    ):
        if dataset in ["geom-drugs", "qm9"]:
            data_cls = GeometricMolBatch
        else:
            data_cls = PocketComplexBatch

        if data_path.suffix == ".npz":
            data = load_npz_data(data_path, data_cls, remove_hs)
        else:
            data = load_smol_data(
                data_path, data_cls, remove_hs, remove_aromaticity, skip_non_valid
            )

        if min_size is not None:
            assert dataset in [
                "geom-drugs",
                "qm9",
            ], "min_size filtering for now only supported for geom-drugs and qm9 datasets"
            mols = [mol for mol in data if mol.seq_length >= min_size]
            data = data_cls.from_list(mols)

        return GeometricDataset(data, data_cls, transform=transform)

    def remove(self, indices):
        self._data.remove(indices)

    def append(self, new_data):
        systems = self._data.append(new_data)
        data = PocketComplexBatch(systems)
        return GeometricDataset(data, self.data_cls, transform=self.transform)

    def save(self, save_path, name="train_al", index=0):
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True, parents=True)
        save_file = save_dir / f"{name}-{index}.smol"
        bytes_data = self._data.to_bytes()
        save_file.write_bytes(bytes_data)

    def save_as_sdf(self, vocab, save_path: str):
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True, parents=True)
        self._data.to_sdf(vocab, save_path=save_path)


def get_data(env, index):
    with env.begin(write=False) as txn:
        compressed = txn.get(str(index).encode())
        buf = io.BytesIO(compressed)
        with gzip.GzipFile(fileobj=buf, mode="rb") as f:
            serialized = f.read()
        try:
            item = pickle.loads(serialized)
        except Exception:
            print("Failed to load item with index:", index)
            return None
    return item


class DatasetSubset(torch.utils.data.Subset):
    """
    Custom Subset class that preserves dataset properties like lengths and hparams.
    """

    def __init__(self, dataset, indices, lengths=None):
        super().__init__(dataset, indices)
        self.lengths = lengths
        self._cached_hparams = self._extract_hparams(dataset)

    def _extract_hparams(self, dataset):
        """Extract hparams from the original dataset."""
        if hasattr(dataset, "hparams"):
            return dataset.hparams
        return {}

    @property
    def hparams(self):
        """Get hparams from the original dataset."""
        return self._cached_hparams

    def sample_n_molecules_per_target(self, n_molecules):
        """
        Create n_molecules copies of each pocket complex in the subset.

        Args:
            n_molecules: Number of copies to create for each pocket complex

        Returns:
            DatasetSubset: New subset with duplicated indices and adjusted lengths
        """
        # Create new indices by repeating each existing index n_molecules times
        new_indices = []
        for idx in self.indices:
            new_indices.extend([idx] * n_molecules)

        # Adjust lengths if they exist
        new_lengths = []
        for length in self.lengths:
            new_lengths.extend([length] * n_molecules)
        return DatasetSubset(self.dataset, new_indices, lengths=new_lengths)

    def sample_n_molecules(self, n_molecules: int, seed: int = 42) -> "DatasetSubset":
        """
        Randomly sample n_molecules unique items from the subset.

        Args:
            n_molecules: Number of molecules to sample
            seed: Random seed for reproducibility

        Returns:
            DatasetSubset: New subset with sampled indices and corresponding lengths
        """
        if n_molecules > len(self.indices):
            raise ValueError(
                "n_molecules cannot be greater than the number of available molecules in the subset."
            )

        rng = np.random.default_rng(seed)
        sampled_indices_pos = rng.choice(
            len(self.indices), size=n_molecules, replace=False
        )
        sampled_indices = [self.indices[i] for i in sampled_indices_pos]
        sampled_lengths = [self.lengths[i] for i in sampled_indices_pos]
        return DatasetSubset(self.dataset, sampled_indices, lengths=sampled_lengths)

    def append(self, pocket_complexes: list) -> "DatasetSubset":
        """
        Append new PocketComplex objects to the underlying dataset and create a new subset.

        Args:
            pocket_complexes: List of PocketComplex objects to add

        Returns:
            New DatasetSubset with the appended items included
        """
        if not hasattr(self.dataset, "append"):
            raise RuntimeError("Underlying dataset does not support appending items")

        # Append to underlying dataset and get new indices
        new_indices = self.dataset.append(pocket_complexes, return_indices=True)

        # Create new subset with existing + new indices
        all_indices = list(self.indices) + new_indices

        # Update lengths
        new_lengths = [
            pc.remove_hs(include_ligand=self.dataset.remove_hs).seq_length
            for pc in pocket_complexes
        ]
        all_lengths = list(self.lengths) + new_lengths

        return DatasetSubset(self.dataset, all_indices, lengths=all_lengths)


class BaseLMDBDataset(ABC, torch.utils.data.Dataset):
    """
    Base class for LMDB datasets providing common functionality.
    """

    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        remove_hs: bool = False,
        remove_aromaticity: bool = False,
        skip_non_valid: bool = False,
        **kwargs,
    ):
        """
        Args:
            lmdb_path: Path to the LMDB directory or file
            transform: Optional transform to apply to each sample
            remove_hs: Whether to remove hydrogen atoms from the dataset
        """
        super().__init__()

        self.lmdb_path = Path(root)
        self.transform = transform
        self.remove_hs = remove_hs
        self.remove_aromaticity = remove_aromaticity
        self.skip_non_valid = skip_non_valid

        # Default LMDB settings
        self.lmdb_kwargs = {
            "readonly": True,
            "lock": False,
            "readahead": False,
            "meminit": False,
            "create": False,
        }
        self._env: lmdb.Environment | None = None

        # Get dataset length but don't store the environment
        self._get_length()

        # Close the environment after getting length to avoid fork issues
        # It will be reopened lazily in each worker process
        self._close_env()

    def _close_env(self):
        """Close the LMDB environment if open."""
        if self._env is not None:
            self._env.close()
            self._env = None

    def __getstate__(self):
        """Prepare state for pickling - exclude LMDB environment."""
        state = self.__dict__.copy()
        # Don't pickle the LMDB environment - it will be reopened in workers
        state["_env"] = None
        return state

    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__.update(state)
        # _env is already None from __getstate__, will be lazily reopened

    @property
    def env(self) -> lmdb.Environment:
        """Get LMDB environment"""
        if self._env is None:
            self._env = lmdb.open(str(self.lmdb_path), **self.lmdb_kwargs)
        return self._env

    def __len__(self) -> int:
        return self.length

    def _get_length(self) -> int:
        """Get the number of items in the dataset."""
        env = self.env
        with env.begin(write=False) as txn:
            # Try to get length from metadata first
            length_data = txn.get(b"__len__")
            if length_data is not None:
                try:
                    length = pickle.loads(length_data)
                    # Ensure we return a simple Python int, not numpy/tensor
                    self.length = int(length)
                except Exception as e:
                    print(f"Warning: Failed to load length metadata: {e}")
                    # Fall through to counting entries
                    self.length = int(txn.stat()["entries"])

    @abstractmethod
    def _get_lengths(self):
        """
        Get or compute item lengths.
        If lengths are not pre-computed, compute them on first access.
        """
        pass

    @abstractmethod
    def _load_item(self, key: bytes, txn) -> any:
        """Load and process a single item from LMDB."""
        pass

    @abstractmethod
    def _get_key(self, idx: int, encoding: str = "utf-8") -> bytes:
        """Convert index to LMDB key. Override for custom key formats."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        """
        Get an item by index.
        """
        pass

    @property
    def hparams(self) -> dict:
        """Return hyperparameters for this dataset."""
        pass


class PocketComplexLMDBDataset(BaseLMDBDataset):
    """
    Main LMDB dataset class for all pre-computed datasets.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        remove_hs: bool = False,
        remove_aromaticity: bool = False,
        skip_non_valid: bool = False,
        read_only: bool = True,
        **kwargs,
    ):
        self.read_only = read_only

        super().__init__(
            root, transform, remove_hs, remove_aromaticity, skip_non_valid, **kwargs
        )
        if not read_only:
            self.lmdb_kwargs.update(
                {
                    "readonly": False,
                    "map_size": 1024**4,  # 1TB max size, adjust as needed
                }
            )

        # Get the molecule sizes
        self._get_lengths()

        # Close the environment after initialization to avoid fork issues
        self._close_env()

    def _load_item(self, key: bytes, txn) -> any:
        """Load and process a single item from LMDB."""
        data_binary = txn.get(key)
        if data_binary is None:
            raise KeyError(f"Key {key!r} not found in LMDB")
        return PocketComplex.from_bytes(
            data_binary,
            remove_hs=self.remove_hs,
            remove_aromaticity=self.remove_aromaticity,
            skip_non_valid=self.skip_non_valid,
        )

    def _get_key(self, idx: int, encoding: str = "utf-8") -> bytes:
        """Convert index to LMDB key."""
        return str(int(idx)).encode(encoding)

    def __getitem__(self, idx: int):
        if idx >= len(self) or idx < 0:
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        key = self._get_key(idx)

        env = self.env
        with env.begin(write=False) as txn:
            pocket_complex = self._load_item(key, txn)

        if self.transform is not None:
            pocket_complex = self.transform(pocket_complex)

        return pocket_complex

    def _get_lengths(self):
        """Get or compute item lengths lazily."""
        if not hasattr(self, "lengths"):
            # Check if pre-computed lengths exist in LMDB metadata
            env = self.env
            with env.begin(write=False) as txn:
                lengths_data = txn.get(
                    b"lengths_no_pocket_hs"
                    if not self.remove_hs
                    else b"lengths_no_ligand_pocket_hs"
                )
                if lengths_data is not None:
                    raw_lengths = pickle.loads(lengths_data)
                    # Convert to simple Python list to ensure serializability
                    if hasattr(raw_lengths, "tolist"):
                        self.lengths = raw_lengths.tolist()
                    else:
                        self.lengths = list(raw_lengths)
                else:
                    raise RuntimeError("Lengths data not found in LMDB")

    @property
    def hparams(self) -> dict:
        """Return hyperparameters for this dataset."""
        return {}

    def append(self, pocket_complexes: list, return_indices: bool = False) -> list:
        """
        Append new PocketComplex objects to the LMDB dataset.

        Args:
            pocket_complexes: List of PocketComplex objects to add

        Returns:
            List of new indices for the appended items
        """
        if self.read_only:
            raise RuntimeError(
                "Cannot append to read-only dataset. Set read_only=False when creating the dataset."
            )

        env = self.env
        new_indices = []

        with env.begin(write=True) as txn:
            # Get current length
            current_length = self.length

            # Add each new item
            for i, pocket_complex in enumerate(pocket_complexes):
                new_idx = current_length + i
                key = self._get_key(new_idx)

                # Serialize the pocket complex
                data_binary = pocket_complex.to_bytes()
                txn.put(key, data_binary)
                new_indices.append(new_idx)

            # Update dataset length
            new_length = current_length + len(pocket_complexes)
            txn.put(b"__len__", pickle.dumps(new_length))

            # Update lengths metadata
            new_lengths_no_pocket_hs = [
                pc.remove_hs(include_ligand=False).seq_length for pc in pocket_complexes
            ]
            new_lengths_no_ligand_pocket_hs = [
                pc.remove_hs(include_ligand=True).seq_length for pc in pocket_complexes
            ]

            # Get existing lengths and append new ones
            existing_lengths = self.lengths.copy()
            existing_lengths.extend(
                new_lengths_no_pocket_hs
                if not self.remove_hs
                else new_lengths_no_ligand_pocket_hs
            )

            # Determine which lengths key to update based on remove_hs setting
            lengths_key = (
                b"lengths_no_pocket_hs"
                if not self.remove_hs
                else b"lengths_no_ligand_pocket_hs"
            )
            txn.put(lengths_key, pickle.dumps(existing_lengths))

        # Update local state
        self.length = new_length
        self.lengths.extend(
            new_lengths_no_pocket_hs
            if not self.remove_hs
            else new_lengths_no_ligand_pocket_hs
        )

        if return_indices:
            return new_indices


class GeometricMolLMDBDataset(BaseLMDBDataset):
    """
    LMDB dataset for GeometricMol objects.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        remove_hs: bool = False,
        remove_aromaticity: bool = False,
        skip_non_valid: bool = False,
        **kwargs,
    ):
        super().__init__(
            root, transform, remove_hs, remove_aromaticity, skip_non_valid, **kwargs
        )
        """
        Args:
            root: Path to the LMDB directory or file
            transform: Optional transform to apply to each GeometricMol
            remove_hs: Whether to remove hydrogen atoms from the dataset
            remove_aromaticity: Whether to remove aromaticity from the dataset
            skip_non_valid: Whether to skip non-valid molecules
        """
        # Get the molecule sizes
        self._get_lengths()

        # Close the environment after initialization to avoid fork issues
        self._close_env()

    def _load_item(self, key: bytes, txn) -> any:
        """Load and process a single item from LMDB."""
        data_binary = txn.get(key)
        if data_binary is None:
            raise KeyError(f"Key {key!r} not found in LMDB")
        return GeometricMol.from_bytes(
            data_binary,
            remove_hs=self.remove_hs,
            remove_aromaticity=self.remove_aromaticity,
            skip_non_valid=self.skip_non_valid,
            check_valid=False,
            add_rdkit_props=False,
        )

    def _get_key(self, idx: int, encoding: str = "utf-8") -> bytes:
        """Convert index to LMDB key."""
        return str(int(idx)).encode(encoding)

    def __getitem__(self, idx: int):
        if idx >= len(self) or idx < 0:
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        key = self._get_key(idx)

        env = self.env
        with env.begin(write=False) as txn:
            geometric_mol = self._load_item(key, txn)

        if self.transform is not None:
            geometric_mol = self.transform(geometric_mol)

        return geometric_mol

    def _get_lengths(self):
        """Get or compute item lengths lazily."""
        if not hasattr(self, "lengths"):
            # Check if pre-computed lengths exist in LMDB metadata
            env = self.env
            with env.begin(write=False) as txn:
                lengths_data = txn.get(
                    b"lengths_with_hs" if not self.remove_hs else b"lengths_no_hs"
                )
                if lengths_data is not None:
                    raw_lengths = pickle.loads(lengths_data)
                    # Convert to simple Python list to ensure serializability
                    if hasattr(raw_lengths, "tolist"):
                        self.lengths = raw_lengths.tolist()
                    else:
                        self.lengths = list(raw_lengths)
                else:
                    raise RuntimeError("Lengths data not found in LMDB")

    @property
    def hparams(self) -> dict:
        """Return hyperparameters for this dataset."""
        return {}


class SDFDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pdb_file: str,
        sdf_path: str,
        remove_hs: bool = False,
        protonate_pocket: bool = False,
        cut_pocket: bool = True,
        pocket_cutoff: float = 6.0,
        compute_interactions: bool = False,
        transform=None,
        save_name: str = "sdf_data",
        **kwargs,
    ):
        """
        Constructor for SDFDataset.
        """
        super().__init__()

        # Check if there's already a .smol file in the parent directory
        smol_file = Path(sdf_path).parent / f"{save_name}.smol"
        if smol_file.exists():
            # Load existing data
            print(f"Loading existing data from {smol_file}")
            bytes_data = smol_file.read_bytes()
            self._data = PocketComplexBatch.from_bytes(bytes_data, remove_hs=remove_hs)
        else:
            # Create new data from SDF
            print(f"Creating data from SDF: {sdf_path}")
            self._data = PocketComplexBatch.from_sdf(
                pdb_file=pdb_file,
                sdf_path=sdf_path,
                remove_hs=remove_hs,
                protonate_pocket=protonate_pocket,
                cut_pocket=cut_pocket,
                pocket_cutoff=pocket_cutoff,
                compute_interactions=compute_interactions,
                pocket_type="holo",
            )
            self.save(smol_file.parent, name=save_name)

        self._num_graphs = len(self._data)
        self.transform = transform

    def save(self, save_dir, name="data"):
        save_dir = Path(save_dir)
        save_file = save_dir / f"{name}.smol"
        bytes_data = self._data.to_bytes()
        save_file.write_bytes(bytes_data)

    def __getitem__(self, idx):
        molecule = self._data[idx]
        if self.transform is not None:
            molecule = self.transform(molecule)
        return molecule

    def __len__(self) -> int:
        return self._num_graphs

    @property
    def lengths(self):
        return []

    @property
    def hparams(self):
        return {}


class GeometricMolSDFDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sdf_path: str,
        ligand_idx: int | None = None,
        remove_hs: bool = False,
        remove_aromaticity: bool = False,
        transform=None,
        save_name: str = "sdf_mol_data",
        **kwargs,
    ):
        """
        Constructor for SDFDataset.
        """
        super().__init__()

        # Check if there's already a .smol file in the parent directory
        smol_file = Path(sdf_path).parent / f"{save_name}.smol"
        if smol_file.exists():
            # Load existing data
            print(f"Loading existing data from {smol_file}")
            bytes_data = smol_file.read_bytes()
            self._data = GeometricMolBatch.from_bytes(
                bytes_data,
                remove_hs=remove_hs,
                remove_aromaticity=remove_aromaticity,
                keep_orig_data=True,
            )
        else:
            # Create new data from SDF
            print(f"Creating data from SDF: {sdf_path}")
            self._data = GeometricMolBatch.from_sdf(
                sdf_path=sdf_path,
                ligand_idx=ligand_idx,
                remove_hs=remove_hs,
                remove_aromaticity=remove_aromaticity,
                keep_orig_data=True,
            )
            self.save(smol_file.parent, name=save_name)

        self._num_graphs = len(self._data)
        self.transform = transform

    def save(self, save_dir, name="data"):
        save_dir = Path(save_dir)
        save_file = save_dir / f"{name}.smol"
        bytes_data = self._data.to_bytes()
        save_file.write_bytes(bytes_data)

    def sample_n_molecules_per_mol(self, n_items):
        mol_samples = [[mol for _ in range(n_items)] for mol in self._data.to_list()]
        mol_samples = list(itertools.chain(*mol_samples))
        data = GeometricMolBatch.from_list(mol_samples)
        return GeometricDataset(data, GeometricMolBatch, transform=self.transform)

    def __getitem__(self, idx):
        molecule = self._data[idx]
        if self.transform is not None:
            molecule = self.transform(molecule)
        return molecule

    def __len__(self) -> int:
        return self._num_graphs

    @property
    def lengths(self):
        return []

    @property
    def hparams(self):
        return {}

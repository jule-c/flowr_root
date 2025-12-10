from __future__ import annotations

import copy
import pickle
from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Generic, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
from rdkit import Chem
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from typing_extensions import Self

import flowr.util.functional as smolF
import flowr.util.rdkit as smolRD
from flowr.data.util import mol_to_torch
from flowr.util.tokeniser import Vocabulary

# Type aliases
_T = torch.Tensor
TDevice = Union[torch.device, str]
TCoord = Tuple[float, float, float]

# Generics
TSmolMol = TypeVar("TSmolMol", bound="SmolMol")

# Constants
PICKLE_PROTOCOL = 4


# **********************
# *** Util functions ***
# **********************


def _check_mol_valid(mol: dict):
    coords = mol["coords"].numpy()
    atomics = mol["atomics"].numpy()
    bond_types = mol["bond_types"]
    bond_indices = mol["bond_indices"]
    bonds = torch.cat((bond_indices, bond_types.unsqueeze(1)), dim=-1).numpy()
    charges = mol["charges"].numpy()
    hybridization = mol.get("hybridization", None)
    if hybridization is not None:
        hybridization = [
            smolRD.IDX_ADD_FEAT_MAP["hybridization"][int(a)]
            for a in mol["hybridization"]
        ]
    atomics = atomics.tolist()
    tokens = [smolRD.PT.symbol_from_atomic(a) for a in atomics]
    mol = smolRD.mol_from_atoms(
        coords,
        tokens,
        bonds,
        charges,
        hybridization=hybridization,
        sanitise=False,
        kekulize=False,
    )
    return smolRD.mol_is_valid(mol, connected=True)


def _remove_hs_from_dict(mol: dict, remove_aromaticity: bool = True):
    device = mol["device"]

    coords = mol["coords"].numpy()
    atomics = mol["atomics"].numpy()
    try:
        bond_types = mol["bond_types"]
        bond_indices = mol["bond_indices"]
        bonds = torch.cat((bond_indices, bond_types.unsqueeze(1)), dim=-1).numpy()
    except KeyError:
        # Support for an older representation of the bonds
        bonds = mol["bonds"].numpy()
    charges = mol["charges"].numpy()
    atomics = atomics.tolist()
    tokens = [smolRD.PT.symbol_from_atomic(a) for a in atomics]
    if mol.get("hybridization") is not None:
        hybridization = [
            smolRD.IDX_ADD_FEAT_MAP["hybridization"][int(a)]
            for a in mol["hybridization"]
        ]
    else:
        hybridization = None
    mol = smolRD.mol_from_atoms(
        coords,
        tokens,
        bonds,
        charges,
        hybridization=hybridization,
        sanitise=False,
        kekulize=False,
    )
    try:
        mol_data = mol_to_torch(
            mol, remove_hs=True, add_feats=True, remove_aromaticity=remove_aromaticity
        )
    except ValueError:
        return
    return {
        "coords": mol_data["coords"],
        "atomics": mol_data["atomics"],
        "bond_indices": mol_data["bond_indices"],
        "bond_types": mol_data["bond_types"],
        "charges": mol_data["charges"],
        "hybridization": mol_data["hybridization"],
        "id": mol_data["smiles"],
        "device": device,
    }


def _remove_aromaticity_from_bonds(mol: dict):
    device = mol["device"]

    coords = mol["coords"].numpy()
    atomics = mol["atomics"].numpy()
    try:
        bond_types = mol["bond_types"]
        bond_indices = mol["bond_indices"]
        bonds = torch.cat((bond_indices, bond_types.unsqueeze(1)), dim=-1).numpy()
    except KeyError:
        # Support for an older representation of the bonds
        bonds = mol["bonds"].numpy()
    charges = mol["charges"].numpy()
    hybridization = mol.get("hybridization", None)
    atomics = atomics.tolist()
    tokens = [smolRD.PT.symbol_from_atomic(a) for a in atomics]
    mol = smolRD.mol_from_atoms(
        coords,
        tokens,
        bonds,
        charges,
        hybridization=hybridization,
        sanitise=False,
        kekulize=False,
    )
    try:
        mol_data = mol_to_torch(mol, remove_hs=False, remove_aromaticity=True)
    except ValueError:
        return
    return {
        "coords": mol_data["coords"],
        "atomics": mol_data["atomics"],
        "bond_indices": mol_data["bond_indices"],
        "bond_types": mol_data["bond_types"],
        "charges": mol_data["charges"],
        "id": mol_data["smiles"],
        "device": device,
    }


def _check_type(obj, obj_type, name="object"):
    if not isinstance(obj, obj_type):
        raise TypeError(
            f"{name} must be an instance of {obj_type} or one of its subclasses, got {type(obj)}"
        )


def _check_shape_len(tensor, allowed, name="object"):
    num_dims = len(tensor.size())
    allowed = [allowed] if isinstance(allowed, int) else allowed
    if num_dims not in allowed:
        raise RuntimeError(
            f"Number of dimensions of {name} must be in {str(allowed)}, got {num_dims}"
        )


def _check_shapes_equal(t1, t2, dims=None):
    if dims is None:
        if t1.size() != t2.size():
            raise RuntimeError(
                f"objects must have the same shape, got {t1.shape} and {t2.shape}"
            )
        else:
            return

    if isinstance(dims, int):
        dims = [dims]

    t1_dims = [t1.size(dim) for dim in dims]
    t2_dims = [t2.size(dim) for dim in dims]
    if t1_dims != t2_dims:
        raise RuntimeError(
            f"Expected dimensions {str(dims)} to match, got {t1.size()} and {t2.size()}"
        )


def _check_dim_shape(tensor, dim, allowed, name="object"):
    shape = tensor.size(dim)
    allowed = [allowed] if isinstance(allowed, int) else allowed
    if shape not in allowed:
        raise RuntimeError(f"Shape of {name} for dim {dim} must be in {allowed}")


def _check_dict_key(map, key, dict_name="dictionary"):
    if key not in map:
        raise RuntimeError(f"{dict_name} must contain key {key}")


# *************************
# *** MolRepr Interface ***
# *************************


class SmolMol(ABC):
    """Interface for molecule representations for the Smol library"""

    def __init__(self, str_id: str):
        self._str_id = str_id

    # *** Properties for molecule objects ***

    @property
    def str_id(self):
        return self.__str__()

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass

    @property
    @abstractmethod
    def seq_length(self) -> int:
        pass

    # *** Static constructor methods ***

    @staticmethod
    @abstractmethod
    def from_bytes(data: bytes) -> SmolMol:
        pass

    @staticmethod
    @abstractmethod
    def from_rdkit(rdkit, mol: Chem.rdchem.Mol, *args) -> SmolMol:
        pass

    # *** Conversion functions for molecule objects ***

    @abstractmethod
    def to_bytes(self) -> bytes:
        pass

    @abstractmethod
    def to_rdkit(self, *args) -> Chem.rdchem.Mol:
        pass

    # *** Other functionality for molecule objects ***

    @abstractmethod
    def _copy_with(self, *args) -> Self:
        pass

    # *** Interface util functions for all molecule representations ***

    def __len__(self):
        return self.seq_length

    def __str__(self):
        if self._str_id is not None:
            return self._str_id

        return super().__str__()

    # Note: only performs a shallow copy
    def copy(self) -> Self:
        return copy.copy(self)

    def to_device(self, device: TDevice) -> Self:
        obj_copy = self.copy()
        for attr_name in vars(self):
            value = getattr(self, attr_name, None)
            if value is not None and isinstance(value, _T):
                setattr(obj_copy, attr_name, value.to(device))

        return obj_copy


class SmolBatch(Sequence, Generic[TSmolMol]):
    """Abstract class for molecule batch representations for the Smol library"""

    # All subclasses must call super init
    def __init__(self, mols: list[TSmolMol], device: Optional[TDevice] = None):
        if len(mols) == 0:
            raise RuntimeError(f"Batch must be non-empty")

        if device is None:
            device = mols[0].device

        mols = [mol.to_device(device) for mol in mols]

        self._mols = mols
        self._device = torch.device(device)

    # *** Properties for molecular batches ***

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def seq_length(self) -> list[int]:
        return [mol.seq_length for mol in self._mols]

    @property
    def batch_size(self) -> int:
        return len(self._mols)

    @property
    @abstractmethod
    def mask(self) -> _T:
        pass

    # *** Sequence methods ***

    def __len__(self) -> int:
        return self.batch_size

    def __getitem__(self, item: int) -> TSmolMol:
        return self._mols[item]

    # *** Default methods which may need overriden ***

    def to_bytes(self) -> bytes:
        mol_bytes = [mol.to_bytes() for mol in self._mols]
        return pickle.dumps(mol_bytes)

    def to_list(self) -> list[TSmolMol]:
        return self._mols

    def apply(self, fn: Callable[[TSmolMol, int], TSmolMol]) -> Self:
        applied = [fn(mol, idx) for idx, mol in enumerate(self._mols)]
        [_check_type(mol, SmolMol, "apply result") for mol in applied]
        return self.from_list(applied)

    def copy(self) -> Self:
        # Only performs shallow copy on individual mols
        mol_copies = [mol.copy() for mol in self._mols]
        return self.from_list(mol_copies)

    def to_device(self, device: TDevice) -> Self:
        applied = [mol.to_device(device) for mol in self._mols]
        return self.from_list(applied)

    @classmethod
    def collate(cls, batches: list[SmolBatch]) -> Self:
        all_mols = [mol for batch in batches for mol in batch]
        return cls.from_list(all_mols)

    # *** Abstract methods for batches ***

    @staticmethod
    @abstractmethod
    def from_bytes(data: bytes) -> SmolBatch:
        pass

    @staticmethod
    @abstractmethod
    def from_list(mols: list[TSmolMol]) -> SmolBatch:
        pass

    @staticmethod
    @abstractmethod
    def from_tensors(*tensors: _T) -> SmolBatch:
        pass

    @staticmethod
    @abstractmethod
    def load(save_dir: str, lazy: bool = False) -> SmolBatch:
        pass

    @abstractmethod
    def save(self, save_dir: str, shards: int = 0, threads: int = 0) -> None:
        pass


# *******************************
# *** MolRepr Implementations ***
# *******************************


# TODO remove distributions for atomics
# TODO documentation
class GeometricMol(SmolMol):
    def __init__(
        self,
        coords: _T,
        atomics: _T,
        bond_indices: Optional[_T] = None,
        bond_types: Optional[_T] = None,
        charges: Optional[_T] = None,
        hybridization: Optional[_T] = None,
        is_aromatic: Optional[_T] = None,
        device: Optional[TDevice] = None,
        is_mmap: bool = False,
        str_id: Optional[str] = None,
        forces: Optional[_T] = None,
        potential_energy: Optional[float] = None,
        fragment_mask: Optional[_T] = None,
        fragment_mode: Optional[str] = None,
        orig_mol: Optional[GeometricMol] = None,
        rdkit_feats_cont: Optional[_T] = None,
        rdkit_feats_disc: Optional[_T] = None,
    ):
        # Check that each tensor has correct number of dimensions
        _check_shape_len(coords, 2, "coords")

        _check_shape_len(atomics, [1, 2], "atomics")
        _check_shapes_equal(coords, atomics, 0)

        if forces is not None:
            _check_shape_len(forces, 2, "coords")
            _check_shapes_equal(coords, forces, 0)

        if bond_indices is None and bond_types is not None:
            raise ValueError(
                "bond_indices must be provided if bond_types are provided."
            )

        # Create an empty edge list if bonds are not provided
        # Or assume single bonds if bond_indices is provided but bond_types is not
        bond_indices = (
            torch.tensor([[]] * 2).T if bond_indices is None else bond_indices
        )
        bond_types = (
            torch.tensor([1] * bond_indices.size(0))
            if bond_types is None
            else bond_types
        )

        _check_shape_len(bond_indices, 2, "bond indices")
        _check_dim_shape(bond_indices, 1, 2, "bond indices")

        _check_shape_len(bond_types, [1, 2], "bond types")
        _check_shapes_equal(bond_indices, bond_types, 0)

        charges = torch.zeros(coords.size(0)) if charges is None else charges

        _check_shape_len(charges, [1, 2], "charges")
        _check_shapes_equal(coords, charges, 0)

        device = coords.device if device is None else torch.device(device)

        self._coords = coords
        self._atomics = atomics
        self._bond_indices = bond_indices
        self._bond_types = bond_types
        self._charges = charges
        self._hybridization = hybridization
        self._is_aromatic = is_aromatic
        self._device = device
        self._forces = forces
        self._potential_energy = potential_energy
        self._rdkit_feats_cont = rdkit_feats_cont
        self._rdkit_feats_disc = rdkit_feats_disc
        self.fragment_mask = fragment_mask
        self.fragment_mode = fragment_mode
        self.orig_mol = orig_mol

        # If the data are not stored in mmap tensors, then convert to expected type and move to device
        if not is_mmap:
            # Use float if atomics is a distribution over atomic numbers
            atomics = atomics.float() if len(atomics.size()) == 2 else atomics.long()
            bond_types = (
                bond_types.float() if len(bond_types.size()) == 2 else bond_types.long()
            )

            self._atomics = atomics.to(device)
            self._coords = coords.float().to(device)
            self._bond_indices = bond_indices.long().to(device)
            self._charges = charges.long().to(device)

        super().__init__(str_id)

    # *** General Properties ***

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def seq_length(self) -> int:
        return self._coords.shape[0]

    # *** Geometric Specific Properties ***

    @property
    def coords(self) -> _T:
        return self._coords.float().to(self._device)

    @property
    def forces(self) -> _T:
        if self._forces is not None:
            return self._forces.float().to(self._device)
        return None

    @property
    def potential_energy(self) -> Optional[float]:
        return (
            self._potential_energy.float().to(self._device)
            if self._potential_energy is not None
            else None
        )

    @property
    def atomics(self) -> _T:
        if len(self._atomics.size()) == 2:
            return self._atomics.float().to(self._device)

        return self._atomics.long().to(self._device)

    @property
    def bond_indices(self) -> _T:
        return self._bond_indices.long().to(self._device)

    @property
    def bond_types(self) -> _T:
        if len(self._bond_types.size()) == 2:
            return self._bond_types.float().to(self._device)

        return self._bond_types.long().to(self._device)

    @property
    def bonds(self) -> _T:
        bond_types = self.bond_types
        if len(bond_types.size()) == 2:
            bond_types = torch.argmax(bond_types, dim=-1)

        return torch.cat((self.bond_indices, bond_types.unsqueeze(1)), dim=-1)

    @property
    def charges(self) -> _T:
        if len(self._charges.size()) == 2:
            return self._charges.float().to(self._device)
        return self._charges.long().to(self._device)

    @property
    def hybridization(self) -> Optional[_T]:
        if self._hybridization is not None:
            if len(self._hybridization.size()) == 2:
                return self._hybridization.float().to(self._device)
            return self._hybridization.long().to(self._device)

        return None

    @property
    def is_aromatic(self) -> Optional[_T]:
        if self._is_aromatic is not None:
            if len(self._is_aromatic.size()) == 2:
                return self._is_aromatic.float().to(self._device)
            return self._is_aromatic.long().to(self._device)
        return None

    @property
    def rdkit_feats_cont(self) -> Optional[_T]:
        if self._rdkit_feats_cont is not None:
            return self._rdkit_feats_cont.float().to(self._device)
        return None

    @property
    def rdkit_feats_disc(self) -> Optional[_T]:
        if self._rdkit_feats_disc is not None:
            return self._rdkit_feats_disc.long().to(self._device)
        return None

    # Note: this will always return a symmetric NxN matrix
    @property
    def adjacency(self) -> _T:
        bond_indices = self.bond_indices
        bond_types = self.bond_types
        return smolF.adj_from_edges(
            bond_indices, bond_types, self.seq_length, symmetric=True
        )

    @property
    def com(self):
        return self.coords.sum(dim=0) / self.seq_length

    # *** Interface Methods ***

    def remove_hs(self, remove_aromaticity: bool = True) -> GeometricMol:

        if self.orig_mol is None:
            orig_mol = GeometricMol(
                self._coords,
                self._atomics,
                bond_indices=self._bond_indices,
                bond_types=self._bond_types,
                charges=self._charges,
                hybridization=self._hybridization,
                forces=self._forces,
                potential_energy=self._potential_energy,
                rdkit_feats_cont=self._rdkit_feats_cont,
                rdkit_feats_disc=self._rdkit_feats_disc,
                device=self.device,
                is_mmap=False,
                str_id=self._str_id,
            )
        else:
            orig_mol = self.orig_mol

        coords = self._coords.numpy()
        atomics = self._atomics.tolist()
        tokens = [smolRD.PT.symbol_from_atomic(a) for a in atomics]
        bonds = torch.cat(
            (self._bond_indices, self._bond_types.unsqueeze(1)), dim=-1
        ).numpy()
        charges = self._charges.numpy()
        if self._hybridization is not None:
            hybridization = [
                smolRD.IDX_ADD_FEAT_MAP["hybridization"][int(a)]
                for a in self._hybridization
            ]
        else:
            hybridization = None
        forces = self._forces if self._forces is not None else None
        if forces is not None:
            hs_mask = torch.tensor([bool(a != 1) for a in atomics])
            forces = forces[hs_mask]
        potential_energy = (
            self._potential_energy if self._potential_energy is not None else None
        )

        mol = smolRD.mol_from_atoms(
            coords,
            tokens,
            bonds,
            charges,
            hybridization=hybridization,
            sanitise=False,
            kekulize=False,
        )
        mol_data = mol_to_torch(
            mol, remove_hs=True, add_feats=True, remove_aromaticity=remove_aromaticity
        )

        mol = GeometricMol(
            coords=mol_data["coords"],
            atomics=mol_data["atomics"],
            bond_indices=mol_data["bond_indices"],
            bond_types=mol_data["bond_types"],
            charges=mol_data["charges"],
            hybridization=mol_data["hybridization"],
            forces=forces,
            potential_energy=potential_energy,
            rdkit_feats_cont=mol_data["rdkit_feats_cont"],
            rdkit_feats_disc=mol_data["rdkit_feats_disc"],
            device=self.device,
            is_mmap=False,
            str_id=mol_data["smiles"],
            orig_mol=orig_mol,
        )
        return mol

    @staticmethod
    def from_biotite(
        obj,
        remove_hs: bool = False,
        add_feats: bool = False,
        remove_aromaticity: bool = False,
    ) -> GeometricMol:
        atoms = [el.capitalize() for el in obj.element.tolist()]
        atomics = torch.tensor([smolRD.PT.atomic_from_symbol(atom) for atom in atoms])
        coords = torch.tensor(obj.coord)
        charges = torch.tensor(obj.charge)
        hybridization = (
            torch.tensor(obj.hybridization) if hasattr(obj, "hybridization") else None
        )

        bonds = obj.bonds.as_array().astype(np.int32)
        if bonds.shape[0] == 0:
            bond_types = None
        else:
            bond_types = torch.tensor(bonds[:, 2]).long()

        mol = smolRD.mol_from_atoms(
            coords,
            atomics,
            bonds=bond_types,
            charges=charges,
            hybridization=hybridization,
            sanitise=True,
        )
        mol_data = mol_to_torch(
            mol,
            remove_hs=remove_hs,
            add_feats=add_feats,
            remove_aromaticity=remove_aromaticity,
        )

        mol = GeometricMol(
            coords=mol_data["coords"],
            atomics=mol_data["atomics"],
            bond_indices=mol_data["bond_indices"],
            bond_types=mol_data["bond_types"],
            charges=mol_data["charges"],
            hybridization=mol_data["hybridization"],
            rdkit_feats_cont=mol_data["rdkit_feats_cont"],
            rdkit_feats_disc=mol_data["rdkit_feats_disc"],
            is_mmap=False,
            str_id=mol_data["smiles"],
        )
        return mol

    @staticmethod
    def from_bytes(
        data: bytes,
        remove_hs: bool = False,
        remove_aromaticity: bool = False,
        keep_orig_data: bool = False,
        skip_non_valid: bool = False,
        check_valid: bool = True,
        add_rdkit_props: bool = False,
    ) -> GeometricMol:
        """
        Load a GeometricMol from bytes data.
        Args:
            data (bytes): The bytes data to load the GeometricMol from.
            remove_hs (bool): Whether to remove hydrogens from the molecule.
            keep_orig_data (bool): Whether to keep the original data from bytes.
            check_valid (bool): Whether to check if the unpickled object is a valid RDKit mol.
            skip_non_valid (bool): Whether to skip non-valid molecules instead of raising an error.
        Returns:
            GeometricMol: The loaded GeometricMol object.
        """

        obj = pickle.loads(data)
        if check_valid:
            if not _check_mol_valid(obj):
                if skip_non_valid:
                    print(
                        "The unpickled object is not a valid RDKit mol. "
                        "Skipping this molecule."
                    )
                    return None
                raise RuntimeError(
                    "The unpickled object is not a valid RDKit mol. "
                    "Ensure the bytes data is valid."
                )

        if keep_orig_data:
            orig_mol = GeometricMol.from_bytes(data)
        else:
            orig_mol = None

        if remove_hs:
            obj = _remove_hs_from_dict(obj, remove_aromaticity=remove_aromaticity)
            if obj is None:
                return
        # else:
        #     obj = _remove_aromaticity_from_bonds(obj)
        #     if obj is None:
        #         return

        _check_type(obj, dict, "unpickled object")
        _check_dict_key(obj, "coords")
        _check_dict_key(obj, "atomics")
        _check_dict_key(obj, "charges")
        _check_dict_key(obj, "device")
        _check_dict_key(obj, "id")

        if obj.get("bond_types") is not None:
            bond_indices = obj["bond_indices"]
            bond_types = obj["bond_types"]
        else:
            # Support for an older representation of the bonds
            _check_dict_key(obj, "bonds")
            bonds = obj["bonds"]
            bond_indices = bonds[:, :2]
            bond_types = bonds[:, 2]

        if obj.get("forces") is not None:
            forces = obj["forces"]
        else:
            forces = None
        if obj.get("potential_energy") is not None:
            potential_energy = obj["potential_energy"]
        else:
            potential_energy = None

        mol = GeometricMol(
            obj["coords"],
            obj["atomics"],
            bond_indices=bond_indices,
            bond_types=bond_types,
            charges=obj["charges"],
            forces=forces,
            potential_energy=potential_energy,
            device=obj["device"],
            is_mmap=False,
            str_id=obj["id"],
            orig_mol=orig_mol,
        )

        if add_rdkit_props:
            # Get additional features if they are provided, otherwise calculate them
            if (
                "hybridization" not in obj
                or "rdkit_feats_cont" not in obj
                or "rdkit_feats_disc" not in obj
            ):
                rdkit_mol = mol.to_rdkit(sanitise=True)
                if rdkit_mol is None:
                    raise RuntimeError(
                        "Failed to convert GeometricMol to RDKit Mol while loading bytes data and extracting additional features. "
                        "Ensure the molecules are valid."
                    )

            if obj.get("hybridization") is not None:
                hybridization = obj["hybridization"]
            else:
                hybridization = torch.tensor(
                    smolRD.retrieve_hybridization_from_mol(rdkit_mol)
                )
            mol._hybridization = hybridization

            if obj.get("rdkit_feats_cont") is not None:
                rdkit_feats_cont = obj["rdkit_feats_cont"]
            else:
                rdkit_feats_cont = torch.tensor(
                    smolRD.retrieve_rdkit_cont_feats_from_mol(rdkit_mol)
                ).float()

            if obj.get("rdkit_feats_disc") is not None:
                rdkit_feats_disc = obj["rdkit_feats_disc"]
            else:
                rdkit_feats_disc = torch.tensor(
                    smolRD.retrieve_rdkit_disc_feats_from_mol(rdkit_mol)
                ).float()
            mol._rdkit_feats_cont = rdkit_feats_cont
            mol._rdkit_feats_disc = rdkit_feats_disc

        return mol

    # Note: currently only uses the default conformer for mol
    @staticmethod
    def from_rdkit(
        mol: Chem.rdchem.Mol, infer_bonds: bool = False, kekulize: bool = False
    ) -> GeometricMol:
        # TODO handle this better - maybe create 3D info if not provided, with a warning
        if mol.GetNumConformers() == 0 or not mol.GetConformer().Is3D():
            raise RuntimeError("The default conformer must have 3D coordinates")

        if kekulize:
            Chem.Kekulize(mol)

        has_hydrogens = smolRD.has_explicit_hydrogens(mol)
        if not kekulize and not has_hydrogens:
            mol = Chem.AddHs(mol, addCoords=True)

        conf = mol.GetConformer()
        smiles = smolRD.smiles_from_mol(mol)

        coords = np.array(conf.GetPositions())
        atomics = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]

        bonds = smolRD.retrieve_bonds_from_mol(mol, infer_bonds=infer_bonds)
        bonds = torch.tensor(bonds)
        bond_indices = bonds[:, :2]
        bond_types = bonds[:, 2]

        coords = torch.tensor(coords)
        atomics = torch.tensor(atomics)
        charges = torch.tensor(charges)

        hybridization = torch.tensor(
            [
                smolRD.ADD_FEAT_IDX_MAP["hybridization"][a.GetHybridization()]
                for a in mol.GetAtoms()
            ],
            dtype=torch.long,
        )

        rdkit_feats_cont = torch.tensor(
            smolRD.retrieve_rdkit_cont_feats_from_mol(mol), dtype=torch.float
        )
        rdkit_feats_disc = torch.tensor(
            smolRD.retrieve_rdkit_disc_feats_from_mol(mol), dtype=torch.float
        )

        mol = GeometricMol(
            coords,
            atomics,
            bond_indices,
            bond_types,
            charges=charges,
            hybridization=hybridization,
            rdkit_feats_cont=rdkit_feats_cont,
            rdkit_feats_disc=rdkit_feats_disc,
            str_id=smiles,
        )
        if not has_hydrogens:
            mol = mol.remove_hs()
        return mol

    def to_bytes(self) -> bytes:
        dict_repr = {
            "coords": self.coords,
            "atomics": self.atomics,
            "bond_indices": self.bond_indices,
            "bond_types": self.bond_types,
            "charges": self.charges,
            "device": str(self.device),
            "id": self._str_id,
        }
        if self.rdkit_feats_cont is not None:
            dict_repr["rdkit_feats_cont"] = self.rdkit_feats_cont
        if self.rdkit_feats_disc is not None:
            dict_repr["rdkit_feats_disc"] = self.rdkit_feats_disc
        if self.hybridization is not None:
            dict_repr["hybridization"] = self.hybridization
        if self.forces is not None:
            dict_repr["forces"] = self.forces
        if self.potential_energy is not None:
            dict_repr["potential_energy"] = self.potential_energy
        if self.fragment_mask is not None:
            dict_repr["fragment_mask"] = self.fragment_mask
        if self.fragment_mode is not None:
            dict_repr["fragment_mode"] = self.fragment_mode

        byte_obj = pickle.dumps(dict_repr, protocol=PICKLE_PROTOCOL)
        return byte_obj

    def write_sdf(self, path: str) -> None:
        mol = self.to_rdkit()
        w = Chem.SDWriter(path)
        w.write(mol)
        w.close()

    def to_rdkit(
        self,
        vocab: Optional[Vocabulary] = None,
        vocab_charges: Optional[Vocabulary] = None,
        vocab_hybridization: Optional[Vocabulary] = None,
        sanitise: Optional[bool] = False,
        kekulize: Optional[bool] = False,
        remove_hs: Optional[bool] = False,
        fix_aromaticity: Optional[bool] = False,
    ) -> Chem.rdchem.Mol:
        if len(self.atomics.size()) == 2:
            assert (
                vocab is not None
            ), "Vocabulary must be provided if atomics is a distribution over atom types."
            vocab_indices = torch.argmax(self.atomics, dim=1).tolist()
            tokens = vocab.tokens_from_indices(vocab_indices)

        else:
            atomics = self.atomics.tolist()
            tokens = [smolRD.PT.symbol_from_atomic(a) for a in atomics]

        if len(self.charges.size()) == 2:
            assert (
                vocab_charges is not None
            ), "Vocabulary for charges must be provided if charges is a distribution."
            charges = torch.argmax(self.charges, dim=1).tolist()
            charges = np.array(vocab_charges.tokens_from_indices(charges))
        else:
            charges = self.charges.numpy()

        if self.hybridization is not None and len(self.hybridization.size()) == 2:
            assert (
                vocab_hybridization is not None
            ), "Vocabulary for hybridization must be provided if hybridization is a distribution."
            hybridization_indices = torch.argmax(self.hybridization, dim=1).tolist()
            hybridization = vocab_hybridization.tokens_from_indices(
                hybridization_indices
            )
        elif self.hybridization is not None:
            hybridization = [
                smolRD.IDX_ADD_FEAT_MAP["hybridization"][int(a)]
                for a in self.hybridization
            ]
        else:
            hybridization = None

        coords = self.coords.numpy()
        bonds = self.bonds.numpy()

        mol = smolRD.mol_from_atoms(
            coords,
            tokens,
            bonds,
            charges,
            hybridization=hybridization,
            remove_hs=remove_hs,
            sanitise=sanitise,
            kekulize=kekulize,
        )

        if mol is None:
            return None

        if fix_aromaticity:
            try:
                mol_copy = Chem.Mol(mol)
                # Clear any existing aromatic flags
                for atom in mol_copy.GetAtoms():
                    atom.SetIsAromatic(False)
                for bond in mol_copy.GetBonds():
                    bond.SetIsAromatic(False)

                Chem.Kekulize(mol_copy)
                Chem.SanitizeMol(mol_copy)
                return mol_copy
            except Exception as e:
                print(f"Failed to sanitize molecule during to_rdkit conversion: {e}")
                return None
        return mol

    def _copy_with(
        self,
        coords: Optional[_T] = None,
        atomics: Optional[_T] = None,
        bond_indices: Optional[_T] = None,
        bond_types: Optional[_T] = None,
        charges: Optional[_T] = None,
        hybridization: Optional[_T] = None,
        is_aromatic: Optional[_T] = None,
        forces: Optional[_T] = None,
        potential_energy: Optional[float] = None,
        rdkit_feats_cont: Optional[_T] = None,
        rdkit_feats_disc: Optional[_T] = None,
        orig_mol: Optional[GeometricMol] = None,
        fragment_mask: Optional[_T] = None,
        fragment_mode: Optional[str] = None,
        com: Optional[_T] = None,
    ) -> GeometricMol:

        coords = self.coords if coords is None else coords
        atomics = self.atomics if atomics is None else atomics
        bond_indices = self.bond_indices if bond_indices is None else bond_indices
        bond_types = self.bond_types if bond_types is None else bond_types
        charges = self.charges if charges is None else charges
        hybridization = self.hybridization if hybridization is None else hybridization
        is_aromatic = self.is_aromatic if is_aromatic is None else is_aromatic
        forces = self.forces if forces is None else forces
        potential_energy = (
            self.potential_energy if potential_energy is None else potential_energy
        )
        rdkit_feats_cont = (
            self.rdkit_feats_cont if rdkit_feats_cont is None else rdkit_feats_cont
        )
        rdkit_feats_disc = (
            self.rdkit_feats_disc if rdkit_feats_disc is None else rdkit_feats_disc
        )
        orig_mol = self.orig_mol if orig_mol is None else orig_mol
        fragment_mask = self.fragment_mask if fragment_mask is None else fragment_mask
        fragment_mode = self.fragment_mode if fragment_mode is None else fragment_mode

        obj = GeometricMol(
            coords,
            atomics,
            bond_indices=bond_indices,
            bond_types=bond_types,
            charges=charges,
            hybridization=hybridization,
            is_aromatic=is_aromatic,
            forces=forces,
            potential_energy=potential_energy,
            rdkit_feats_cont=rdkit_feats_cont,
            rdkit_feats_disc=rdkit_feats_disc,
            device=self.device,
            is_mmap=False,
            str_id=self._str_id,
            orig_mol=orig_mol,
            fragment_mask=fragment_mask,
            fragment_mode=fragment_mode,
        )
        return obj

    # TODO add tests
    def permute(self, indices: list[int]) -> GeometricMol:
        """Used for permuting atom order. Can also be used for taking a subset of atoms but not duplicating."""

        if len(set(indices)) != len(indices):
            raise ValueError("Indices list cannot contain duplicates.")

        if max(indices) >= self.seq_length:
            raise ValueError(
                f"Index {max(indices)} is out of bounds for molecule with {self.seq_length} atoms."
            )

        indices = torch.tensor(indices)

        coords = self.coords[indices]
        atomics = self.atomics[indices]
        charges = self.charges[indices]
        hybridization = (
            self.hybridization[indices] if self.hybridization is not None else None
        )

        # Relabel bond from and to indices with new indices
        from_idxs = self.bond_indices[:, 0].clone()
        to_idxs = self.bond_indices[:, 1].clone()
        curr_indices = torch.arange(indices.size(0))

        old_from, new_from = torch.nonzero(
            from_idxs.unsqueeze(1) == curr_indices, as_tuple=True
        )
        old_to, new_to = torch.nonzero(
            to_idxs.unsqueeze(1) == curr_indices, as_tuple=True
        )

        from_idxs[old_from] = indices[new_from]
        to_idxs[old_to] = indices[new_to]

        # Remove bonds whose indices do not appear in new indices list
        bond_idxs = torch.cat((from_idxs.unsqueeze(-1), to_idxs.unsqueeze(-1)), dim=-1)
        mask = bond_idxs.unsqueeze(-1) == indices.view(1, 1, -1)
        mask = ~(~mask.any(dim=-1)).any(dim=-1)
        bond_indices = bond_idxs[mask]
        bond_types = self.bond_types[mask]

        mol_copy = self._copy_with(
            coords=coords,
            atomics=atomics,
            bond_indices=bond_indices,
            bond_types=bond_types,
            charges=charges,
            hybridization=hybridization,
            rdkit_feats_cont=self.rdkit_feats_cont,
            rdkit_feats_disc=self.rdkit_feats_disc,
        )
        return mol_copy

    # *** Geometric Specific Methods ***

    def zero_com(self) -> GeometricMol:
        shifted = self.coords - self.com.unsqueeze(0)
        return self._copy_with(coords=shifted)

    def rotate(self, rotation: Union[Rotation, TCoord] = None) -> GeometricMol:
        coords = torch.tensor(rotation.apply(self.coords), dtype=torch.float32)
        return self._copy_with(coords=coords)

    def shift(self, shift: Union[float, TCoord]) -> GeometricMol:
        shift_tensor = torch.tensor(shift).view(1, -1)
        shifted = self.coords + shift_tensor
        return self._copy_with(coords=shifted)

    def scale(self, scale: float) -> GeometricMol:
        scaled = self.coords * scale
        return self._copy_with(coords=scaled)


class GeometricMolBatch(SmolBatch[GeometricMol]):
    def __init__(self, mols: list[GeometricMol], device: Optional[TDevice] = None):
        for mol in mols:
            _check_type(mol, GeometricMol, "molecule object")

        super().__init__(mols, device)

        # Cache for batched tensors
        self._coords = None
        self._mask = None
        self._atomics = None
        self._bond_indices = None
        self._bond_types = None
        self._bonds = None
        self._charges = None
        self._hybridization = None
        self._forces = None
        self._potential_energy = None
        self._rdkit_feats_cont = None
        self._rdkit_feats_disc = None
        self._fragment_mask = None
        self._fragment_mode = None
        self._orig_mols = None

    # *** General Properties ***

    @property
    def mask(self) -> _T:
        if self._mask is None:
            masks = [torch.ones(mol.seq_length) for mol in self._mols]
            self._mask = smolF.pad_tensors(masks)

        return self._mask

    @property
    def orig_mols(self) -> list[GeometricMol]:
        if self._orig_mols is None:
            orig_mols = [mol.orig_mol for mol in self._mols]
            self._orig_mols = orig_mols

        return self._orig_mols

    # *** Geometric Specific Properties ***

    @property
    def coords(self) -> _T:
        if self._coords is None:
            coords = [mol.coords for mol in self._mols]
            self._coords = smolF.pad_tensors(coords)

        return self._coords

    @property
    def forces(self) -> _T:
        if self._forces is None:
            forces = [mol.forces for mol in self._mols]
            self._forces = smolF.pad_tensors(forces)

        return self._forces

    @property
    def potential_energy(self) -> _T:
        if self._potential_energy is None:
            potential_energy = [mol.potential_energy for mol in self._mols]
            self._potential_energy = smolF.pad_tensors(potential_energy)

        return self._potential_energy

    @property
    def atomics(self) -> _T:
        if self._atomics is None:
            atomics = [mol.atomics for mol in self._mols]
            self._atomics = smolF.pad_tensors(atomics)

        return self._atomics

    @property
    def bond_indices(self) -> _T:
        if self._bond_indices is None:
            bond_indices = [mol.bond_indices for mol in self._mols]
            self._bond_indices = smolF.pad_tensors(bond_indices)

        return self._bond_indices

    @property
    def bond_types(self) -> _T:
        if self._bond_types is None:
            bond_types = [mol.bond_types for mol in self._mols]
            self._bond_types = smolF.pad_tensors(bond_types)

        return self._bond_types

    @property
    def bonds(self) -> _T:
        if self._bonds is None:
            bonds = [mol.bonds for mol in self._mols]
            self._bonds = smolF.pad_tensors(bonds)

        return self._bonds

    @property
    def charges(self) -> _T:
        if self._charges is None:
            charges = [mol.charges for mol in self._mols]
            self._charges = smolF.pad_tensors(charges)

        return self._charges

    @property
    def hybridization(self) -> _T:
        if self._hybridization is None:
            hybridizations = [mol.hybridization for mol in self._mols]
            if any(h is None for h in hybridizations):
                return None
            self._hybridization = smolF.pad_tensors(hybridizations)

        return self._hybridization

    @property
    def rdkit_feats_cont(self) -> _T:
        if self._rdkit_feats_cont is None:
            self._rdkit_feats_cont = [mol.rdkit_feats_cont for mol in self._mols]
        return self._rdkit_feats_cont

    @property
    def rdkit_feats_disc(self) -> _T:
        if self._rdkit_feats_disc is None:
            self._rdkit_feats_disc = [mol.rdkit_feats_disc for mol in self._mols]
        return self._rdkit_feats_disc

    @property
    def fragment_mask(self) -> _T:
        if self._fragment_mask is None and self._mols[0].fragment_mask is not None:
            self._fragment_mask = smolF.pad_tensors(
                [mol.fragment_mask for mol in self._mols]
            ).long()
            return self._fragment_mask
        elif self._mols[0].fragment_mask is None:
            return []
        else:
            return self._fragment_mask

    @property
    def fragment_mode(self) -> Optional[str]:
        if self._fragment_mode is None and self._mols[0].fragment_mode is not None:
            self._fragment_mode = [mol.fragment_mode for mol in self._mols]
            return self._fragment_mode
        elif self._mols[0].fragment_mode is None:
            return []
        else:
            return self._fragment_mode

    @property
    def adjacency(self) -> _T:
        n_atoms = max(self.seq_length)
        adjs = [
            smolF.adj_from_edges(
                mol.bond_indices, mol.bond_types, n_atoms, symmetric=True
            )
            for mol in self._mols
        ]
        return torch.stack(adjs)

    @property
    def com(self) -> _T:
        return smolF.calc_com(self.coords, node_mask=self.mask)

    # *** Interface Methods ***

    def _copy(self):
        return GeometricMolBatch(self._mols)

    def to_dict(self) -> dict:
        return {
            "coords": self.coords,
            "atomics": self.atomics,
            "bonds": self.bonds,
            "mask": self.mask,
        }

    @staticmethod
    def from_bytes(
        data: bytes,
        remove_hs: bool = False,
        remove_aromaticity: bool = False,
        keep_orig_data: bool = False,
    ) -> GeometricMolBatch:
        mols = [
            GeometricMol.from_bytes(
                mol_bytes,
                remove_hs=remove_hs,
                remove_aromaticity=remove_aromaticity,
                keep_orig_data=keep_orig_data,
            )
            for mol_bytes in tqdm(pickle.loads(data))
        ]
        failed_mols = mols.count(None)
        mols = [m for m in mols if m is not None]
        print(f"Failed to load {failed_mols} molecules.")
        return GeometricMolBatch.from_list(mols)

    @staticmethod
    def from_list(mols: list[GeometricMol]) -> GeometricMolBatch:
        return GeometricMolBatch(mols)

    @staticmethod
    def from_sdf(
        sdf_path: str,
        ligand_idx: int | None = None,
        add_feats: bool = False,
        remove_hs: bool = False,
        remove_aromaticity: bool = False,
        keep_orig_data: bool = False,
    ) -> GeometricMolBatch:
        suppl = Chem.SDMolSupplier(sdf_path, sanitize=True, removeHs=False)
        if ligand_idx is not None and ligand_idx != -1:
            suppl = [suppl[ligand_idx]]
        mols = []
        for mol in tqdm(suppl):
            if mol is None:
                continue
            geom_mol = GeometricMol.from_rdkit(mol, infer_bonds=False, kekulize=False)
            if keep_orig_data:
                geom_mol.orig_mol = geom_mol._copy_with()
            if remove_hs:
                geom_mol = geom_mol.remove_hs(remove_aromaticity=remove_aromaticity)
            if add_feats:
                rdkit_mol = geom_mol.to_rdkit(sanitise=True)
                hybridization = torch.tensor(
                    smolRD.retrieve_hybridization_from_mol(rdkit_mol)
                )
                rdkit_feats_cont = torch.tensor(
                    smolRD.retrieve_rdkit_cont_feats_from_mol(rdkit_mol)
                ).float()
                rdkit_feats_disc = torch.tensor(
                    smolRD.retrieve_rdkit_disc_feats_from_mol(rdkit_mol)
                ).float()
                geom_mol._hybridization = hybridization
                geom_mol._rdkit_feats_cont = rdkit_feats_cont
                geom_mol._rdkit_feats_disc = rdkit_feats_disc

            mols.append(geom_mol)

        return GeometricMolBatch.from_list(mols)

    # TODO add bonds and charges
    @staticmethod
    def from_tensors(
        coords: _T,
        atomics: Optional[_T] = None,
        num_atoms: Optional[_T] = None,
        is_mmap: bool = False,
    ) -> GeometricMolBatch:

        _check_shape_len(coords, 3, "coords")

        if atomics is not None:
            _check_shape_len(atomics, [2, 3], "atomics")
            _check_shapes_equal(coords, atomics, [0, 1])

        if num_atoms is not None:
            _check_shape_len(num_atoms, 1, "num_atoms")
            _check_shapes_equal(coords, num_atoms, 0)

        device = coords.device
        batch_size, max_atoms = coords.size()[:2]

        num_atoms = (
            torch.tensor([max_atoms] * batch_size) if num_atoms is None else num_atoms
        )
        seq_lens = num_atoms.int().tolist()

        mols = []
        for idx in range(coords.size(0)):
            mol_coords = coords[idx, : seq_lens[idx]]
            mol_types = atomics[idx, : seq_lens[idx]] if atomics is not None else None
            mol = GeometricMol(mol_coords, mol_types, device=device, is_mmap=is_mmap)
            mols.append(mol)

        batch = GeometricMolBatch(mols, device)

        # Put all tensors on same device and set batched tensor cache if they are not mem mapped
        if not is_mmap:

            # Use float if types is a distribution over atom types
            if atomics is not None:
                if len(atomics.size()) == 3:
                    atomics = atomics.float().to(device)
                else:
                    atomics = atomics.long().to(device)

            batch._atomics = atomics
            batch._coords = coords.float().to(device)

        return batch

    @staticmethod
    def load(save_dir: str, lazy: bool = False) -> GeometricMolBatch:
        save_path = Path(save_dir)

        if not save_path.exists() or not save_path.is_dir():
            raise RuntimeError(f"Folder {save_dir} does not exist.")

        batches = []
        curr_folders = [save_path]

        while len(curr_folders) != 0:
            curr_path = curr_folders[0]
            if (curr_path / "atoms.npy").exists():
                batch = GeometricMolBatch._load_batch(curr_path, lazy=lazy)
                batches.append(batch)

            children = [path for path in curr_path.iterdir() if path.is_dir()]
            curr_folders = curr_folders[1:]
            curr_folders.extend(children)

        collated = GeometricMolBatch.collate(batches)
        return collated

    def save(
        self, save_dir: Union[str, Path], shards: int = 0, threads: int = 0
    ) -> None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        if shards is None or shards <= 0:
            return self._save_batch(self, save_path)

        items_per_shard = (len(self) // shards) + 1
        start_idxs = [idx * items_per_shard for idx in range(shards)]
        end_idxs = [(idx + 1) * items_per_shard for idx in range(shards)]
        end_idxs[-1] = len(self)

        batches = [
            self._mols[s_idx:e_idx] for s_idx, e_idx in zip(start_idxs, end_idxs)
        ]
        batches = [GeometricMolBatch.from_list(batch_list) for batch_list in batches]

        f_len = len(str(shards - 1))
        dir_names = [
            f"{str(b_idx):0{f_len}}_n{str(b.batch_size)}"
            for b_idx, b in enumerate(batches)
        ]
        save_paths = [save_path / name for name in dir_names]

        if threads is not None and threads > 0:
            executor = ThreadPoolExecutor(threads)
            futures = [
                executor.submit(self._save_batch, b, path)
                for b, path in zip(batches, save_paths)
            ]
            [future.result() for future in futures]

        else:
            [self._save_batch(batch, path) for batch, path in zip(batches, save_paths)]

    # *** Geometric Specific Methods ***

    def zero_com(self) -> GeometricMolBatch:
        shifted = self.coords - self.com
        shifted = shifted * self.mask.unsqueeze(2)
        return self._from_coords(shifted)

    def rotate(self, rotation: Union[Rotation, TCoord]) -> GeometricMolBatch:
        return self.apply(lambda mol, idx: mol.rotate(rotation))

    def shift(self, shift: TCoord) -> GeometricMolBatch:
        shift_tensor = torch.tensor(shift).view(1, 1, -1)
        shifted = (self.coords + shift_tensor) * self.mask.unsqueeze(2)
        return self._from_coords(shifted)

    def scale(self, scale: float) -> GeometricMolBatch:
        scaled = (self.coords * scale) * self.mask.unsqueeze(2)
        return self._from_coords(scaled)

    # *** Util Methods ***

    def _from_coords(self, coords: _T) -> GeometricMolBatch:
        _check_shape_len(coords, 3, "coords")
        _check_shapes_equal(coords, self.coords, [0, 1, 2])

        if coords.size(0) != self.batch_size:
            raise RuntimeError(f"coords batch size must be the same as self batch size")

        if coords.size(1) != max(self.seq_length):
            raise RuntimeError(f"coords num atoms must be the same as largest molecule")

        coords = coords.float().to(self.device)

        mol_coords = [
            cs[:num_atoms, :] for cs, num_atoms in zip(list(coords), self.seq_length)
        ]
        mols = [mol._copy_with(coords=cs) for mol, cs in zip(self._mols, mol_coords)]
        batch = GeometricMolBatch(mols)

        # Set the cache for the tensors that have already been created
        batch._coords = coords
        batch._mask = self.mask if self._mask is not None else None
        batch._atomics = self.atomics if self._atomics is not None else None
        batch._bonds = self.bonds if self._bonds is not None else None

        return batch

    # TODO add bonds and charges
    @staticmethod
    def _load_batch(batch_dir: Path, lazy: bool) -> GeometricMolBatch:
        mmap_mode = "r+" if lazy else None

        num_atoms_arr = np.load(batch_dir / "atoms.npy")
        num_atoms = torch.tensor(num_atoms_arr)

        # torch now supports loading mmap tensors but np mmap seems a lot more mature and creating a tensor from a
        # mmap array using from_numpy() preserves the mmap array without reading in the data until required
        coords_arr = np.load(batch_dir / "coords.npy", mmap_mode=mmap_mode)
        coords = torch.from_numpy(coords_arr)

        atomics_arr = np.load(batch_dir / "atomics.npy", mmap_mode=mmap_mode)
        atomics = torch.from_numpy(atomics_arr)

        # bonds = None
        # bonds_path = batch_dir / "bonds.npy"
        # if edges_path.exists():
        #     bonds_arr = np.load(bonds_path, mmap_mode=mmap_mode)
        #     bonds = torch.from_numpy(bonds_arr)

        batch = GeometricMolBatch.from_tensors(coords, atomics, num_atoms, is_mmap=lazy)
        return batch

    @staticmethod
    def _save_batch(batch, save_path: Path) -> None:
        save_path.mkdir(exist_ok=True, parents=True)

        coords = batch.coords.cpu().numpy()
        np.save(save_path / "coords.npy", coords)

        num_atoms = np.array(batch.seq_length).astype(np.int16)
        np.save(save_path / "atoms.npy", num_atoms)

        atomics = batch.atomics.cpu().numpy()
        np.save(save_path / "atomics.npy", atomics)

        bonds = batch.bonds.cpu().numpy()
        if bonds.shape[1] != 0:
            np.save(save_path / "bonds.npy", bonds)

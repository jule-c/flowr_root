import os
from functools import partial

import lightning as L
import torch
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    SubsetRandomSampler,
)

from flowr.data.util import (
    BucketBatchSampler,
    DistributedBatchSamplerSimilarLength,
    get_ddp_info,
)
from flowr.util.molrepr import GeometricMol, GeometricMolBatch
from flowr.util.pocket import PocketComplex, PocketComplexBatch


def worker_init_fn(worker_id):
    """Initialize each DataLoader worker with thread limits."""

    # Set threading limits for this worker
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    # Set random seed for reproducibility
    import random

    import numpy as np

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class SmolDM(L.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        batch_cost,
        val_batch_cost,
        bucket_limits=None,
        bucket_cost_scale="constant",
        pad_to_bucket=False,
        use_bucket_sampler=False,
        use_adaptive_sampler=False,
        use_weighted_sampler=False,
        target_memory_gb=40.0,
        num_workers=4,
        train_mols=None,
        val_mols=None,
        test_mols=None,
    ):
        super().__init__()

        if bucket_cost_scale not in [None, "constant", "linear", "quadratic"]:
            raise ValueError(
                f"Bucket cost scale '{bucket_cost_scale}' is not supported."
            )

        if bucket_limits is not None and use_bucket_sampler:
            bucket_limits = sorted(bucket_limits)
            largest_padding = bucket_limits[-1]

            if (
                train_dataset is not None
                and max(train_dataset.lengths) > largest_padding
            ):
                raise ValueError(
                    "At least one item in train dataset is larger than largest padded size."
                )
        else:
            bucket_limits = None

            # if val_dataset is not None and max(val_dataset.lengths) > largest_padding:
            #     raise ValueError(
            #         "At least one item in val dataset is larger than largest padded size."
            #     )

            # if test_dataset is not None and max(test_dataset.lengths) > largest_padding:
            #     raise ValueError(
            #         "At least one item in test dataset is larger than largest padded size."
            #     )

        self._num_workers = num_workers  # len(os.sched_getaffinity(0))

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_mols = train_mols
        self.val_mols = val_mols
        self.test_mols = test_mols

        self.batch_cost = batch_cost
        self.val_batch_cost = val_batch_cost
        self.bucket_limits = bucket_limits
        self.bucket_cost_scale = bucket_cost_scale
        self.pad_to_bucket = pad_to_bucket
        self.use_bucket_sampler = use_bucket_sampler
        self.use_adaptive_sampler = use_adaptive_sampler
        self.use_weighted_sampler = use_weighted_sampler
        self.target_memory_gb = target_memory_gb

    @property
    def hparams(self):

        train_hps = (
            {f"train-{k}": v for k, v in self.train_dataset.hparams.items()}
            if self.train_dataset is not None and hasattr(self.train_dataset, "hparams")
            else {}
        )
        val_hps = (
            {f"val-{k}": v for k, v in self.val_dataset.hparams.items()}
            if self.val_dataset is not None and hasattr(self.val_dataset, "hparams")
            else {}
        )
        test_hps = (
            {f"test-{k}": v for k, v in self.test_dataset.hparams.items()}
            if self.test_dataset is not None and hasattr(self.test_dataset, "hparams")
            else {}
        )

        hparams = {
            "batch-cost": self.batch_cost,
            "buckets": len(self.bucket_limits) if self.bucket_limits is not None else 0,
            "bucket-cost-scale": self.bucket_cost_scale,
            **train_hps,
            **val_hps,
            **test_hps,
        }
        return hparams

    def train_dataloader(self):
        if self.use_weighted_sampler:
            assert (
                not self.use_bucket_sampler
            ), "Cannot use both weighted and bucket samplers"

            from torch.utils.data import WeightedRandomSampler

            weighted_sampler = WeightedRandomSampler(
                weights=self.train_dataset.sample_weights,
                num_samples=len(self.train_dataset),
                replacement=True,
            )
            if weighted_sampler is not None:
                dataloader = DataLoader(
                    self.train_dataset,
                    sampler=weighted_sampler,
                    batch_size=self.batch_cost,
                    num_workers=self._num_workers,
                    pin_memory=True,
                    persistent_workers=self._num_workers > 0,
                    # worker_init_fn=worker_init_fn,
                    collate_fn=partial(self._collate, dataset="train"),
                )
                return dataloader

        if self.use_bucket_sampler:
            batch_sampler = self._batch_sampler(
                self.train_dataset, drop_last=True, shuffle=True
            )

            dataloader = DataLoader(
                self.train_dataset,
                # sampler=sampler,
                batch_sampler=batch_sampler,
                num_workers=self._num_workers,
                pin_memory=False,
                persistent_workers=False,  # self._num_workers > 0,
                # worker_init_fn=worker_init_fn,
                collate_fn=partial(self._collate, dataset="train"),
            )

        elif self.use_adaptive_sampler:
            from flowr.data.util import AdaptiveBatchSampler

            lengths = self.train_dataset.lengths
            sampler = RandomSampler(self.train_dataset)
            batch_sampler = AdaptiveBatchSampler(
                sampler=sampler,
                lengths=lengths,
                target_memory_gb=self.target_memory_gb,
                min_batch_size=1,
                max_batch_size=64,
                similarity_threshold=0.2,
                shuffle=True,
                drop_last=True,
            )

            dataloader = DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                num_workers=self._num_workers,
                pin_memory=True,
                # worker_init_fn=worker_init_fn,
                collate_fn=partial(self._collate, dataset="train"),
            )
        else:
            dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_cost,
                shuffle=True,
                num_workers=self._num_workers,
                pin_memory=True,
                persistent_workers=self._num_workers > 0,
                # worker_init_fn=worker_init_fn,
                collate_fn=partial(self._collate, dataset="train"),
            )
        return dataloader

    def val_dataloader(self, subset=None, shuffle=False):
        dataset = self.val_dataset
        sampler = None

        # Create a subset sampler if subset is specified
        if subset is not None:
            if subset > len(dataset):
                raise ValueError(
                    f"Subset size {subset} is larger than dataset size {len(dataset)}"
                )

            indices = torch.randperm(len(dataset))[:subset].tolist()
            sampler = SubsetRandomSampler(indices)

        dataloader = DataLoader(
            dataset,
            batch_size=self.val_batch_cost,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self._num_workers,
            pin_memory=True,
            persistent_workers=self._num_workers > 0,
            # worker_init_fn=worker_init_fn,
            collate_fn=partial(self._collate, dataset="val"),
        )

        return dataloader

    def test_dataloader(self, subset=None, shuffle=False):
        dataset = self.test_dataset
        sampler = None

        # Create a subset sampler if subset is specified
        if subset is not None:
            if subset > len(dataset):
                raise ValueError(
                    f"Subset size {subset} is larger than dataset size {len(dataset)}"
                )

            indices = torch.randperm(len(dataset))[:subset].tolist()
            sampler = SubsetRandomSampler(indices)

        dataloader = DataLoader(
            dataset,
            batch_size=self.val_batch_cost,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self._num_workers,
            pin_memory=True,
            persistent_workers=self._num_workers > 0,
            # worker_init_fn=worker_init_fn,
            collate_fn=partial(self._collate, dataset="test"),
        )

        return dataloader

    def _sampler(self, dataset, drop_last=False):
        sampler = None
        _, world_size = get_ddp_info()
        if self.bucket_limits is not None:
            costs = self._get_bucket_costs()
            if world_size == 1:
                sampler = BucketBatchSampler(
                    self.bucket_limits,
                    dataset.lengths,
                    self.batch_cost,
                    bucket_costs=costs,
                    drop_last=drop_last,
                    round_batch_to_8=True,
                )
            else:
                raise ("Only single GPU training with bucketing is supported for now")
            # else:
            #     sampler = DistributedBucketBatchSampler(
            #         self.bucket_limits,
            #         dataset.lengths,
            #         self.batch_cost,
            #         bucket_costs=costs,
            #         drop_last=drop_last,
            #         round_batch_to_8=True,
            #     )

        return sampler

    def _batch_sampler(self, dataset, drop_last=False, shuffle=True):
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

        if self.bucket_limits is not None:
            costs = self._get_bucket_costs()
            batch_sampler = BucketBatchSampler(
                sampler=sampler,
                bucket_limits=self.bucket_limits,
                lengths=dataset.lengths,
                batch_cost=self.batch_cost,
                bucket_costs=costs,
                drop_last=drop_last,
                round_batch_to_8=True,
            )
        return batch_sampler

    def lengths_sampler(self, dataset, drop_last, stage):
        rank, world_size = get_ddp_info()
        sampler = DistributedBatchSamplerSimilarLength(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=stage == "train",
            seed=42,
            drop_last=drop_last,
            batch_size=self.batch_cost,
        )

        return sampler

    def _get_bucket_costs(self):
        if self.bucket_cost_scale is None:
            return None
        elif self.bucket_cost_scale == "constant":
            return [1] * len(self.bucket_limits)
        elif self.bucket_cost_scale == "linear":
            return self.bucket_limits
        elif self.bucket_cost_scale == "quadratic":
            # Divide by 256 and add one to approximate the linear and constant overheads
            # A molecule with 16 atoms will therefore have a cost of 1 + 1
            return [((limit**2) / 256) + 1 for limit in self.bucket_limits]
        else:
            raise ValueError(
                f"Unknown value for bucket_cost_scale '{self.bucket_cost_scale}'"
            )

    # TODO implement this using GeometricDM stuff and add extra collations for other types of SmolMol
    def _collate(self, batch):
        raise NotImplementedError()


# TODO could make this more general for all types of SmolMol
# Just have to allow for different types of SmolMol in collate and take different tensors in batch_to_dict
class GeometricDM(SmolDM):
    def _collate(self, batch):
        if isinstance(batch, GeometricMolBatch):
            return self._batch_to_dict(batch)

        elif isinstance(batch[0], GeometricMol):
            smol_batch = GeometricMolBatch.from_list(list(batch))
            return self._batch_to_dict(smol_batch)

        # THE DEFAULT OUTPUT OF THE DATAMODULE: A list of tuples of either GeometricMols or PocketComplexes and interpolation times as torch tensors or list:
        # If this is a list of PocketComplexes pass the state (apo/holo) to the collate function (tuple: apo, holo, interpolated, times)
        collated = [
            self._collate_objs(list(objs), i=i)
            for i, objs in enumerate(tuple(zip(*batch)))
        ]
        return collated

    def _collate_objs(self, objs, i=None):
        if isinstance(objs, GeometricMolBatch):
            return self._batch_to_dict(objs)

        elif isinstance(objs, dict):
            return {key: self._collate_objs(val) for key, val in objs.items()}

        elif isinstance(objs[0], GeometricMol):
            smol_batch = GeometricMolBatch.from_list(list(objs))
            return self._batch_to_dict(smol_batch)

        elif isinstance(objs[0], PocketComplex):
            smol_batch = PocketComplexBatch.from_list(list(objs))
            return self._complex_batch_to_dict(smol_batch, i=i)

        elif isinstance(objs, list) and isinstance(objs[0], torch.Tensor):
            return torch.stack(objs)

        elif isinstance(objs, list) and isinstance(objs[0], list):
            return torch.stack([torch.tensor(obj) for obj in objs], dim=1)

        elif isinstance(objs, torch.Tensor) and isinstance(objs[0], torch.Tensor):
            return objs

        elif isinstance(objs[0], dict):
            collated = {k: [obj[k] for obj in objs] for k in list(objs[0].keys)}
            return self._collate_objs(collated)

        return objs

    def _batch_to_dict(self, smol_batch):
        # Pad batch to n_atoms using a fake mol
        # If we are not padding to bucket size get_padded_size will just return largest mol size
        n_atoms = self._get_padded_size(smol_batch)
        batch = [self._fake_mol_like(smol_batch[0], n_atoms)] + smol_batch.to_list()
        batch = GeometricMolBatch.from_list(batch)
        coords = batch.coords.float()[1:]
        atomics = batch.atomics.float()[1:]
        bonds = batch.adjacency.float()[1:]
        charges = batch.charges.float()[1:]
        hybridization = (
            batch.hybridization.float()[1:] if batch.hybridization is not None else None
        )
        mask = batch.mask.long()[1:]
        if len(batch.fragment_mask) > 0:
            fragment_mask = batch.fragment_mask.long()[1:]
        else:
            fragment_mask = []
        if len(batch.fragment_mode) > 0:
            fragment_mode = batch.fragment_mode[1:]
        else:
            fragment_mode = []

        data = {
            "coords": coords,
            "atomics": atomics,
            "bonds": bonds,
            "charges": charges,
            "mask": mask,
            "fragment_mask": fragment_mask,
            "fragment_mode": fragment_mode,
        }
        if hybridization is not None:
            data["hybridization"] = hybridization
        return data

    def _complex_batch_to_dict(self, smol_batch, i=None):

        # PocketComplexBatch either contains apo or holo data. Apo covers either the prior or the interpolated data
        state_dict = {0: "apo", 1: "holo", 2: "apo"}
        state = state_dict[i]

        coords = smol_batch.coords(state=state).float()
        atomics = smol_batch.atomics(state=state).float()
        bonds = smol_batch.adjacency(state=state).float()
        interactions = smol_batch.interactions(state=state)
        charges = smol_batch.charges(state=state).float()
        hybridization = smol_batch.hybridization(state=state)
        affinity = smol_batch.affinity().float()
        docking_score = smol_batch.docking_score().float()
        atom_names = smol_batch.atom_names(state=state).long()
        res_names = smol_batch.res_names(state=state).long()
        _complex = smol_batch._systems
        lig_mask = smol_batch.lig_mask(state=state).long()
        pocket_mask = smol_batch.pocket_mask(state=state).long()
        fragment_mask = smol_batch.fragment_mask()
        fragment_mode = smol_batch.fragment_mode()
        mask = smol_batch.mask.long()

        data = {
            "coords": coords,
            "atomics": atomics,
            "bonds": bonds,
            "interactions": interactions,
            "charges": charges,
            "affinity": affinity,
            "docking_score": docking_score,
            "atom_names": atom_names,
            "res_names": res_names,
            "lig_mask": lig_mask,
            "pocket_mask": pocket_mask,
            "fragment_mask": fragment_mask,
            "fragment_mode": fragment_mode,
            "mask": mask,
            "complex": _complex,
        }
        if hybridization is not None:
            data["hybridization"] = hybridization
        return data

    def _get_padded_size(self, smol_batch):
        largest_mol_size = max(smol_batch.seq_length)

        if self.bucket_limits is None or not self.pad_to_bucket:
            return largest_mol_size

        # Find smallest bucket which all mols will fit in
        for size in self.bucket_limits:
            if size >= largest_mol_size:
                return size

        raise ValueError(
            f"Mol size of {largest_mol_size} is larger than largest padded size."
        )

    def _fake_mol_like(self, mol, n_atoms):
        coords = torch.zeros((n_atoms, 3))
        if len(mol.atomics.shape) == 1:
            atomics = torch.zeros((n_atoms,))
        else:
            atomics = torch.zeros((n_atoms, mol.atomics.size(1)))
        if len(mol.charges.shape) == 1:
            charges = torch.zeros((n_atoms,))
        else:
            charges = torch.zeros((n_atoms, mol.charges.size(1)))
        if mol.hybridization is not None:
            if len(mol.hybridization.shape) == 1:
                hybridization = torch.zeros((n_atoms,))
            else:
                hybridization = torch.zeros((n_atoms, mol.hybridization.size(1)))
        else:
            hybridization = None
        bond_indices = torch.tensor([[0, 0]])
        if len(mol.bond_types.shape) == 1:
            bond_types = torch.tensor([0])
        else:
            bond_types = torch.zeros((1, mol.bond_types.size(1)))

        if mol.fragment_mask is not None:
            fragment_mask = torch.zeros((n_atoms,))
        else:
            fragment_mask = None
        if mol.fragment_mode is not None:
            fragment_mode = ""
        else:
            fragment_mode = None

        return GeometricMol(
            coords,
            atomics,
            charges=charges,
            hybridization=hybridization,
            bond_indices=bond_indices,
            bond_types=bond_types,
            fragment_mask=fragment_mask,
            fragment_mode=fragment_mode,
        )


class GeometricInterpolantDM(GeometricDM):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size,
        val_batch_size,
        train_interpolant=None,
        val_interpolant=None,
        test_interpolant=None,
        bucket_limits=None,
        bucket_cost_scale=None,
        pad_to_bucket=False,
        use_bucket_sampler=False,
        use_adaptive_sampler=False,
        use_weighted_sampler=False,
        target_memory_gb=40.0,
        num_workers=4,
        train_mols=None,
        val_mols=None,
        test_mols=None,
    ):

        self.train_interpolant = train_interpolant
        self.val_interpolant = val_interpolant
        self.test_interpolant = test_interpolant

        super().__init__(
            train_dataset,
            val_dataset,
            test_dataset,
            batch_size,
            val_batch_size,
            bucket_limits=bucket_limits,
            bucket_cost_scale=bucket_cost_scale,
            pad_to_bucket=pad_to_bucket,
            use_bucket_sampler=use_bucket_sampler,
            use_adaptive_sampler=use_adaptive_sampler,
            use_weighted_sampler=use_weighted_sampler,
            target_memory_gb=target_memory_gb,
            num_workers=num_workers,
            train_mols=train_mols,
            val_mols=val_mols,
            test_mols=test_mols,
        )

    @property
    def hparams(self):
        interps = [self.train_interpolant, self.val_interpolant, self.test_interpolant]
        datasets = ["train", "val", "test"]

        hparams = []
        for dataset, interp in zip(datasets, interps):
            if interp is not None:
                interp_hparams = {
                    f"{dataset}-{k}": v for k, v in interp.hparams.items()
                }
                hparams.append(interp_hparams)

        hparams = {
            k: v for interp_hparams in hparams for k, v in interp_hparams.items()
        }
        return {**hparams, **super().hparams}

    def _collate(self, batch, dataset):

        if dataset == "train" and self.train_interpolant is not None:
            objs = self.train_interpolant.interpolate(batch)
            batch = list(zip(*objs))

        elif dataset == "val" and self.val_interpolant is not None:
            objs = self.val_interpolant.interpolate(batch)
            batch = list(zip(*objs))

        elif dataset == "test" and self.test_interpolant is not None:
            objs = self.test_interpolant.interpolate(batch)
            batch = list(zip(*objs))

        return super()._collate(batch)

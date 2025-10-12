import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import flowr.util.rdkit as smolRD
from flowr.util.pocket import PocketComplex, PocketComplexBatch
from flowr.util.tokeniser import Vocabulary

DEFAULT_FILTER_MIN_RESIDUES = 4

ALLOWED_ATOMS = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Se"] # , "Br", "B", "Si", "As", "I", "Bi"]
MAX_ATOMIC_NUM = 118
BAD_LIGAND_SMILES = "CC(=O)NC1COC(CO)C(O)C1O"

PT = smolRD.PeriodicTable()


class FilterFunc(ABC):
    def __init__(self, func_repr: str, func_desc: str):
        self.func_repr = func_repr
        self.func_desc = func_desc

    @property
    def filter_repr(self):
        return f"{self.func_repr}  ({self.func_desc})"

    @abstractmethod
    def filter(self, system: PocketComplex) -> bool:
        """Returns True if the system should be removed, False otherwise"""
        pass

    def filter_batch(self, systems: PocketComplexBatch) -> PocketComplexBatch:
        remaining = [system for system in systems if not self.filter(system)]
        return PocketComplexBatch(remaining)


class FilterSet(FilterFunc):
    def __init__(self, filters: list[FilterFunc]):
        func_repr = "FilterSet"
        func_desc = "Applies a set of filter functions"
        super().__init__(func_repr, func_desc)

        self.filters = filters
        self._filter_idx_removed_map = {idx: 0 for idx in range(len(filters))}

    @property
    def filter_repr(self) -> str:
        str_repr = f"{self.func_repr} (["
        for filter_fn in self.filters:
            str_repr += f"\n  {filter_fn.filter_repr}"
        str_repr += "\n])"

        return str_repr

    @property
    def n_filtered_str(self) -> str:
        filtered_str = f"{'Filter name':<20}Num systems failing filter"
        for filter_idx, filter_fn in enumerate(self.filters):
            n_failing = self._filter_idx_removed_map[filter_idx]
            filtered_str += f"\n{filter_fn.func_repr:<20}{n_failing}"

        return filtered_str

    def filter(self, system: PocketComplex) -> bool:
        is_filtered = [filter_fn.filter(system) for filter_fn in self.filters]
        for filter_idx, failed_filter in enumerate(is_filtered):
            if failed_filter:
                self._filter_idx_removed_map[filter_idx] += 1

        return any(is_filtered)


class FilterCovalent(FilterFunc):
    def __init__(self):
        func_repr = "FilterCovalent"
        func_desc = "Removes covalently bonded systems"
        super().__init__(func_repr, func_desc)

    def filter(self, system: PocketComplex) -> bool:
        if system.is_covalent is None:
            print(f"WARNING -- is_covalent not set for system {system.system_id}")
            return False

        return system.is_covalent


class FilterSize(FilterFunc):
    def __init__(
        self,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        use_residues: Optional[bool] = False
    ):
        if min_size is None and max_size is None:
            raise ValueError(f"At least one of min_size and max_size must be provided.")

        func_repr = f"FilterSize < {min_size}"
        func_desc = f"Removes systems with fewer than {min_size} atoms"
        super().__init__(func_repr, func_desc)

        self.min_size = min_size
        self.max_size = max_size
        self.use_residues = use_residues

    def filter(self, system: PocketComplex) -> bool:
        system_size = system.holo.n_residues if self.use_residues else len(system)
        if self.min_size is not None and system_size < self.min_size:
            return True
        
        if self.max_size is not None and system_size > self.max_size:
            return True

        return False


class FilterAtomTypes(FilterFunc):
    """Removes any systems which contain any of the given atom symbols"""

    def __init__(self, allowed_atoms: list[str]):
        func_repr = "FilterAtomTypes"
        func_desc = "Removes systems which have atom types which are not in the allowed set"
        super().__init__(func_repr, func_desc)

        self.allowed_atom_set = set(allowed_atoms)

    def filter(self, system: PocketComplex) -> bool:
        ligand_symbols = [PT.symbol_from_atomic(atom) for atom in system.ligand.atomics.tolist()]
        is_allowed_ligand = [symbol in self.allowed_atom_set for symbol in ligand_symbols]
        is_allowed_holo = [symbol in self.allowed_atom_set for symbol in system.holo.atom_symbols]
        is_allowed = is_allowed_ligand + is_allowed_holo

        if system.apo is not None:
            is_allowed_apo = [symbol in self.allowed_atom_set for symbol in system.holo.atom_symbols]
            is_allowed += is_allowed_apo

        return not all(is_allowed)


class FilterLigands(FilterFunc):
    """Removes any systems with the given ligands"""

    def __init__(self, smiles: list[str]):
        func_repr = "FilterLigands"
        func_desc = "Removes systems with a ligand which matches the provided SMILES"
        super().__init__(func_repr, func_desc)

        all_atom_symbols = [PT.symbol_from_atomic(atomic) for atomic in range(MAX_ATOMIC_NUM + 1)]
        self.smiles_set = set([self._to_canonical(smi) for smi in smiles])
        self.vocab = Vocabulary(all_atom_symbols)

    def filter(self, system: PocketComplex) -> bool:
        smiles = smolRD.smiles_from_mol(system.ligand.to_rdkit(self.vocab), canonical=True)
        return smiles in self.smiles_set

    def _to_canonical(self, smiles: str) -> str:
        return smolRD.smiles_from_mol(smolRD.mol_from_smiles(smiles), canonical=True, explicit_hs=False)


def load_dataset(args):
    data_path = Path(args.save_path) / "intermediate"

    batches = []
    for group_path in data_path.iterdir():
        if group_path.is_file() and group_path.suffix == ".smol":
            batch_bytes = group_path.read_bytes()
            group_batch = PocketComplexBatch.from_bytes(batch_bytes)
            batches.append(group_batch)

    full_batch = PocketComplexBatch.from_batches(batches)
    return full_batch


def build_filter_set(args):
    filter_funcs = []

    if args.filter_covalent:
        filter_funcs.append(FilterCovalent())

    if args.filter_atom_types:
        filter_funcs.append(FilterAtomTypes(ALLOWED_ATOMS))

    if args.filter_min_residues >= 1:
        filter_funcs.append(FilterSize(args.filter_min_residues, use_residues=True))

    if args.filter_bad_ligand:
        filter_funcs.append(FilterLigands([BAD_LIGAND_SMILES]))

    filter_set = FilterSet(filter_funcs)
    return filter_set


def save_systems_(args, systems: list[PocketComplex], split: str):
    batch = PocketComplexBatch(systems)
    save_dir = Path(args.save_path) / "processed"
    save_dir.mkdir(exist_ok=True, parents=True)
    save_file = save_dir / f"{split}.smol"
    bytes_data = batch.to_bytes()
    save_file.write_bytes(bytes_data)
    print(f"Saved {len(systems)} systems to {save_file.resolve()}")


def save_splits_(args, systems: PocketComplexBatch):
    train_systems = []
    val_systems = []
    test_systems = []

    for system in systems:
        if system.split == "train":
            train_systems.append(system)
        elif system.split == "val":
            val_systems.append(system)
        elif system.split == "test":
            test_systems.append(system)
        elif system.split is None:
            raise ValueError(f"System {system.system_id} has no assigned data split.")
        else:
            raise ValueError(f"Unknown split for system {system.system_id}, got split {system.split}")

    save_systems_(args, train_systems, "train")
    save_systems_(args, val_systems, "val")
    save_systems_(args, test_systems, "test")


def main(args):
    print("Running dataset accumulation script...")

    filter_set = build_filter_set(args)

    print(f"Loading data intermediate from {args.save_path}")
    data_batch = load_dataset(args)
    print(f"Successfully loaded {len(data_batch)} systems.")

    print("\nFiltering systems with the following filters:")
    print(filter_set.filter_repr)
    print()

    remaining_data = filter_set.filter_batch(data_batch)

    print(f"Filtering complete. {len(remaining_data)} systems remaining.")
    print("\nThe following number of systems were filtered:")
    print(filter_set.n_filtered_str)
    print()

    print("Saving split files...")
    save_splits_(args, remaining_data)

    print("\nAccumulation and saving script complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_path", type=str)

    parser.add_argument("--no_filter_covalent", action="store_false", dest="filter_covalent")
    parser.add_argument("--no_filter_atom_types", action="store_false", dest="filter_atom_types")
    parser.add_argument("--no_filter_bad_ligand", action="store_false", dest="filter_bad_ligand")
    parser.add_argument("--filter_min_residues", type=int, default=DEFAULT_FILTER_MIN_RESIDUES)

    parser.set_defaults(
        filter_covalent=True,
        filter_atom_types=True,
        fiter_bad_ligand=True
    )

    args = parser.parse_args()
    main(args)

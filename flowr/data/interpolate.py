import random
from abc import ABC, abstractmethod
from itertools import zip_longest
from typing import Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdMMPA import FragmentMol
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

import flowr.util.functional as smolF
from flowr.util.ifg import identify_functional_groups
from flowr.util.molrepr import GeometricMol, GeometricMolBatch, SmolBatch, SmolMol
from flowr.util.pocket import PocketComplex, ProteinPocket
from flowr.util.rdkit import ConformerGenerator

PLINDER_MOLECULE_SIZE_MEAN = 48.3841740914044
PLINDER_MOLECULE_SIZE_STD_DEV = 20.328270251327584
PLINDER_MOLECULE_SIZE_MAX = 182
PLINDER_MOLECULE_SIZE_MIN = 8
CROSSDOCKED_MOLECULE_SIZE_MEAN = 40.0
CROSSDOCKED_MOLECULE_SIZE_STD_DEV = 10.0
CROSSDOCKED_MOLECULE_SIZE_MAX = 82
CROSSDOCKED_MOLECULE_SIZE_MIN = 5
KINODATA_MOLECULE_SIZE_MEAN = 31.24166706404082
KINODATA_MOLECULE_SIZE_STD_DEV = 6.369577265037612
KINODATA_MOLECULE_SIZE_MAX = 84
KINODATA_MOLECULE_SIZE_MIN = 4

_InterpT = tuple[list[SmolMol], list[SmolMol], list[SmolMol], list[torch.Tensor]]
_GeometricInterpT = tuple[
    list[GeometricMol], list[GeometricMol], list[GeometricMol], list[torch.Tensor]
]
_ComplexInterpT = tuple[
    list[PocketComplex], list[PocketComplex], list[PocketComplex], list[torch.Tensor]
]


def extract_fragments(
    to_mols: list[Chem.Mol],
    maxCuts: int = 3,
    fragment_mode: str = "fragment",
    prefer_rings: bool = False,
):
    """Extract molecular fragments using MMPA fragmentation.

    Args:
        to_mols: List of RDKit Mol objects to extract fragments from.
        maxCuts: Maximum number of cuts for MMPA fragmentation.
        fragment_mode: Mode for fragment selection:
            - "single": Select one connected fragment
            - "multi-2": Select two disconnected fragments
            - "multi-3": Select three disconnected fragments
            - other: Original behavior (can select disconnected fragments)
        prefer_rings: Whether to prefer ring-containing fragments

    Returns:
        List of boolean masks indicating selected fragments for each molecule.
        For multi-fragment modes, returns list of lists of masks.
    """

    def single_fragment_per_mol(
        mol: Chem.Mol, maxCuts: int, max_fragment_size=15, prefer_rings=False
    ):
        mask = torch.zeros(mol.GetNumAtoms(), dtype=bool)
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            print(
                "Fragments could not be extracted as molecule could not be sanitized. Skipping!"
            )
            return mask

        def clean_fragment(frag):
            for a in frag.GetAtoms():
                if a.GetAtomicNum() == 0:
                    a.SetAtomicNum(1)
            frag = Chem.RemoveHs(frag)
            frags = Chem.GetMolFrags(frag, asMols=True)
            return frags

        def is_connected_component(atom_indices: tuple) -> bool:
            """Check if atom indices form a single connected component in the molecule"""
            if len(atom_indices) <= 1:
                return True

            atom_set = set(atom_indices)
            # Build a graph of the fragment
            visited = set()
            stack = [atom_indices[0]]

            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)

                atom = mol.GetAtomWithIdx(current)
                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    if neighbor_idx in atom_set and neighbor_idx not in visited:
                        stack.append(neighbor_idx)

            # If all atoms were visited, it's connected
            return len(visited) == len(atom_indices)

        def contains_ring(atom_indices: tuple) -> bool:
            """Check if fragment contains at least one ring atom"""
            ring_info = mol.GetRingInfo()
            ring_atoms = set()
            for ring in ring_info.AtomRings():
                ring_atoms.update(ring)
            return any(idx in ring_atoms for idx in atom_indices)

        # Generate fragments using MMPA fragmentation
        frags = FragmentMol(mol=mol, maxCuts=maxCuts)

        # FragmentMol returns (cores, side_chains), need to flatten properly
        all_frags = []
        for frag_tuple in frags:  # cores or side_chains
            if frag_tuple:  # if not None/empty
                for frag in frag_tuple:  # individual fragments
                    if frag:
                        cleaned = clean_fragment(frag)
                        all_frags.extend([f for f in cleaned if f.GetNumAtoms() > 1])

        if not all_frags:
            return mask

        # Match fragments back to original molecule and filter for local fragments
        local_fragment_indices = []

        for frag in all_frags:
            matches = mol.GetSubstructMatches(frag)
            if matches:
                for match in matches:
                    # Filter criteria: must be connected, reasonably sized, and local
                    if is_connected_component(match):
                        local_fragment_indices.append(match)

        # Further filter by preferring ring-containing fragments if requested
        if prefer_rings and local_fragment_indices:
            ring_fragments = [f for f in local_fragment_indices if contains_ring(f)]
            if ring_fragments:
                local_fragment_indices = ring_fragments

        # Randomly select one local fragment from the filtered list
        if local_fragment_indices:
            selected_fragment = random.choice(local_fragment_indices)
            mask[torch.tensor(selected_fragment)] = 1
        else:
            all_matches = []
            for frag in all_frags:
                matches = mol.GetSubstructMatches(frag)
                if matches:
                    for match in matches:
                        all_matches.append(match)

            if all_matches:
                # Pick the smallest fragment
                smallest = min(all_matches, key=len)
                mask[torch.tensor(smallest)] = 1

        return mask

    def multi_fragment_per_mol(
        mol: Chem.Mol, maxCuts: int, n_fragments: int = 2, prefer_rings: bool = False
    ):
        """Select multiple disconnected fragments for replacement.

        Returns:
            List of masks, one per selected fragment. Empty list if not enough fragments found.
        """
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            print(
                "Fragments could not be extracted as molecule could not be sanitized. Skipping!"
            )
            return []

        def clean_fragment(frag):
            for a in frag.GetAtoms():
                if a.GetAtomicNum() == 0:
                    a.SetAtomicNum(1)
            frag = Chem.RemoveHs(frag)
            frags = Chem.GetMolFrags(frag, asMols=True)
            return frags

        def is_connected_component(atom_indices: tuple) -> bool:
            """Check if atom indices form a single connected component in the molecule"""
            if len(atom_indices) <= 1:
                return True

            atom_set = set(atom_indices)
            visited = set()
            stack = [atom_indices[0]]

            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)

                atom = mol.GetAtomWithIdx(current)
                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    if neighbor_idx in atom_set and neighbor_idx not in visited:
                        stack.append(neighbor_idx)

            return len(visited) == len(atom_indices)

        def contains_ring(atom_indices: tuple) -> bool:
            """Check if fragment contains at least one ring atom"""
            ring_info = mol.GetRingInfo()
            ring_atoms = set()
            for ring in ring_info.AtomRings():
                ring_atoms.update(ring)
            return any(idx in ring_atoms for idx in atom_indices)

        def fragments_are_disconnected(
            frag1_indices: tuple, frag2_indices: tuple
        ) -> bool:
            """Check if two fragments are disconnected (no shared atoms or direct bonds)"""
            set1 = set(frag1_indices)
            set2 = set(frag2_indices)

            # Check for overlapping atoms
            if set1 & set2:
                return False

            # Check for direct bonds between fragments
            for idx1 in frag1_indices:
                atom = mol.GetAtomWithIdx(idx1)
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetIdx() in set2:
                        return False

            return True

        # Generate fragments using MMPA fragmentation
        frags = FragmentMol(mol=mol, maxCuts=maxCuts)

        # Collect all valid connected fragments
        all_frags = []
        for frag_tuple in frags:
            if frag_tuple:
                for frag in frag_tuple:
                    if frag:
                        cleaned = clean_fragment(frag)
                        all_frags.extend([f for f in cleaned if f.GetNumAtoms() > 1])

        if not all_frags:
            return []

        # Match fragments to original molecule
        local_fragment_indices = []
        for frag in all_frags:
            matches = mol.GetSubstructMatches(frag)
            if matches:
                for match in matches:
                    if is_connected_component(match):
                        local_fragment_indices.append(match)

        # Filter by ring preference if requested
        if prefer_rings and local_fragment_indices:
            ring_fragments = [f for f in local_fragment_indices if contains_ring(f)]
            if ring_fragments:
                local_fragment_indices = ring_fragments

        # If not enough fragments, return empty
        if len(local_fragment_indices) < n_fragments:
            return []

        # Randomly select n_fragments disconnected fragments
        selected_fragments = []
        available_fragments = local_fragment_indices.copy()

        # Select first fragment randomly
        first_frag = random.choice(available_fragments)
        selected_fragments.append(first_frag)
        available_fragments.remove(first_frag)

        # Select remaining fragments ensuring they're disconnected from all previously selected
        for _ in range(n_fragments - 1):
            # Filter for fragments disconnected from all selected ones
            disconnected_candidates = []
            for candidate in available_fragments:
                is_disconnected_from_all = all(
                    fragments_are_disconnected(candidate, selected)
                    for selected in selected_fragments
                )
                if is_disconnected_from_all:
                    disconnected_candidates.append(candidate)

            if not disconnected_candidates:
                # Not enough disconnected fragments, return empty
                return []

            # Select one randomly
            next_frag = random.choice(disconnected_candidates)
            selected_fragments.append(next_frag)
            available_fragments.remove(next_frag)

        # Convert to masks
        masks = []
        for frag_indices in selected_fragments:
            mask = torch.zeros(mol.GetNumAtoms(), dtype=bool)
            mask[torch.tensor(frag_indices)] = 1
            masks.append(mask)

        return masks

    def fragment_per_mol(mol: Chem.Mol, maxCuts: int):
        mask = torch.zeros(mol.GetNumAtoms(), dtype=bool)
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            print(
                "Fragments could not be extracted as molecule could not be sanitized. Skipping!"
            )
            return mask

        def clean_fragment(frag):
            for a in frag.GetAtoms():
                if a.GetAtomicNum() == 0:
                    a.SetAtomicNum(1)
            frag = Chem.RemoveHs(frag)
            frags = Chem.GetMolFrags(frag, asMols=True)
            return frags

        # Generate fragments
        frags = FragmentMol(mol=mol, maxCuts=maxCuts)
        # FragmentMol returns (cores, side_chains), need to flatten properly
        all_frags = []
        for frag_tuple in frags:  # cores or side_chains
            if frag_tuple:  # if not None/empty
                for frag in frag_tuple:  # individual fragments
                    if frag:
                        cleaned = clean_fragment(frag)
                        all_frags.extend([f for f in cleaned if f.GetNumAtoms() > 1])

        if not all_frags:
            return mask

        substructure_ids = []
        for frag in all_frags:
            matches = mol.GetSubstructMatches(frag)
            if matches:
                substructure_ids.append(matches[0])

        # Randomly select a fragment
        if substructure_ids:
            # Keep sampling until we find a fragment with at least one atom
            valid_frags = [f for f in substructure_ids if len(f) > 0]
            if valid_frags:
                frag = random.choice(valid_frags)
            else:
                # Fallback: if no valid fragments, return empty mask
                return mask
            mask[torch.tensor(frag)] = 1

        return mask

    # Route to appropriate function based on fragment_mode
    if fragment_mode == "single":
        return [
            single_fragment_per_mol(mol, maxCuts=maxCuts, prefer_rings=prefer_rings)
            for mol in to_mols
        ]
    elif fragment_mode == "multi-2":
        return [
            multi_fragment_per_mol(
                mol, maxCuts=maxCuts, n_fragments=2, prefer_rings=prefer_rings
            )
            for mol in to_mols
        ]
    elif fragment_mode == "multi-3":
        return [
            multi_fragment_per_mol(
                mol, maxCuts=maxCuts, n_fragments=3, prefer_rings=prefer_rings
            )
            for mol in to_mols
        ]
    else:
        # Original behavior
        return [fragment_per_mol(mol, maxCuts=maxCuts) for mol in to_mols]


def extract_substructure(
    to_mols: list[Chem.Mol],
    substructure_query: str,
    use_smarts: bool = False,
    invert_mask: bool = False,
):
    def substructure_per_mol(mol, substructure_query):
        mask = torch.zeros(mol.GetNumAtoms(), dtype=bool)
        _mol = Chem.Mol(mol)
        try:
            Chem.SanitizeMol(_mol)
        except Exception:
            print(
                "Substructure could not be extracted as reference molecule could not be sanitized. Skipping."
            )
            return mask
        if use_smarts:
            substructure = Chem.MolFromSmarts(substructure_query)
        else:
            substructure = Chem.MolFromSmiles(substructure_query)
        if substructure is None or substructure.GetNumAtoms() == 0:
            print(
                "Substructure could not be extracted from the reference molecule. Skipping."
            )
            return mask
        if mol.HasSubstructMatch(
            substructure
        ):  # TODO: handle the case where multiple substructures are present
            try:
                substructure_atoms = mol.GetSubstructMatches(substructure)
            except Exception as e:
                print(e)
        if len(substructure_atoms) > 0:
            mask[torch.tensor(substructure_atoms)] = 1
        return mask

    def substructure_per_mol_list(mol, substructure_atoms):
        mask = torch.zeros(mol.GetNumAtoms(), dtype=bool)
        _mol = Chem.Mol(mol)
        try:
            Chem.SanitizeMol(_mol)
        except Exception:
            print(
                "Substructure could not be extracted as reference molecule could not be sanitized. Skipping."
            )
            return mask
        mask[torch.tensor(substructure_atoms)] = 1
        return mask

    assert isinstance(substructure_query, list) or isinstance(
        substructure_query, str
    ), "substructure_query must be a list or string either containing atom indices or a SMILES/SMARTS pattern"
    if isinstance(substructure_query, str):
        mask = [substructure_per_mol(mol, substructure_query) for mol in to_mols]
    else:
        mask = [substructure_per_mol_list(mol, substructure_query) for mol in to_mols]

    # Invert the mask if requested (to exclude the specified atoms instead of including them)
    if invert_mask:
        mask = [~m for m in mask]

    return mask


def extract_func_groups(to_mols: list[Chem.Mol], includeHs=False):
    def func_groups_per_mol(mol, includeHs=True):
        mask = torch.zeros(mol.GetNumAtoms(), dtype=bool)
        _mol = Chem.Mol(mol)
        try:
            Chem.SanitizeMol(_mol)
        except Exception:
            print(
                "Functional groups could not be extracted as reference molecule could not be sanitized. Skipping."
            )
            return mask
        fgroups = identify_functional_groups(_mol)
        findices = []
        for f in fgroups:
            findices.extend(f.atomIds)
        if includeHs:  # include neighboring H atoms in functional groups
            findices_incl_h = []
            for fi in findices:
                hidx = [
                    n.GetIdx()
                    for n in _mol.GetAtomWithIdx(fi).GetNeighbors()
                    if n.GetSymbol() == "H"
                ]
                findices_incl_h.extend([fi] + hidx)
            findices = findices_incl_h
        if len(findices) > 0:
            try:
                mask[torch.tensor(findices)] = 1
            except Exception as e:
                print(e)
        return mask

    return [func_groups_per_mol(mol, includeHs) for mol in to_mols]


def extract_cores(to_mols: list[Chem.Mol]):
    def cores_per_mol(mol):
        _mol = Chem.Mol(mol)
        try:
            Chem.SanitizeMol(_mol)
        except Exception:
            print(
                "Cores could not be extracted as molecule could not be sanitized. Skipping."
            )
            return torch.zeros(mol.GetNumAtoms(), dtype=bool)
        # Retain original atom indices
        for a in _mol.GetAtoms():
            a.SetIntProp("org_idx", a.GetIdx())

        scaffold = GetScaffoldForMol(_mol)
        scaffold_atoms = [a.GetIntProp("org_idx") for a in scaffold.GetAtoms()]

        if scaffold is None:
            print("Scaffold could not be extracted. Returning zero mask.")
            return torch.zeros(mol.GetNumAtoms(), dtype=bool)
        # Collect indices for all atoms in rings
        ring_atoms = set()
        for ring in _mol.GetRingInfo().AtomRings():
            ring_atoms.update(ring)
        # Define core atoms as atoms in the scaffold that are in any ring
        core_atoms = [a for a in scaffold_atoms if a in ring_atoms]
        if not core_atoms:
            # No core atoms found, return a zero mask (i.e. mask with all zeros)
            # print("No core atoms found. Returning zero mask.")
            return torch.zeros(_mol.GetNumAtoms(), dtype=bool)
        # Otherwise, start with a mask that is 0 for every atom, then unmask core atoms (set them to 1)
        mask = torch.zeros(_mol.GetNumAtoms(), dtype=bool)
        try:
            mask[torch.tensor(core_atoms)] = 1
        except Exception as e:
            print(e)
        return mask

    return [cores_per_mol(mol) for mol in to_mols]


def extract_linkers(to_mols: list[Chem.Mol]):
    def linker_per_mol(mol):
        _mol = Chem.Mol(mol)
        try:
            Chem.SanitizeMol(_mol)
        except Exception:
            print(
                "Linker could not be extracted as molecule could not be sanitized. Skipping."
            )
            return torch.zeros(mol.GetNumAtoms(), dtype=bool)
        # Retain original atom indices
        for a in _mol.GetAtoms():
            a.SetIntProp("org_idx", a.GetIdx())
        scaffold = GetScaffoldForMol(_mol)
        if scaffold is None:
            print("Scaffold could not be extracted. Returning zero mask.")
            return torch.zeros(mol.GetNumAtoms(), dtype=bool)
        # Collect indices for all atoms in rings
        ring_atoms = set()
        for ring in _mol.GetRingInfo().AtomRings():
            ring_atoms.update(ring)
        # Define linker atoms as atoms in the scaffold that are not in any ring
        linker_atoms = [
            a.GetIntProp("org_idx")
            for a in scaffold.GetAtoms()
            if a.GetIdx() not in ring_atoms
        ]
        if not linker_atoms:
            # No linker atoms found, return a zero mask (i.e. mask with all zeros)
            # print("No linker atoms found. Returning zero mask.")
            return torch.zeros(_mol.GetNumAtoms(), dtype=bool)
        # Otherwise, start with a mask that is 1 for every atom, then unmask linker atoms (set them to 0)
        mask = torch.zeros(_mol.GetNumAtoms(), dtype=bool)
        try:
            mask[torch.tensor(linker_atoms)] = 1
        except Exception as e:
            print(e)
        return mask

    return [linker_per_mol(mol) for mol in to_mols]


def extract_scaffolds(to_mols: list[Chem.Mol]):
    def scaffold_per_mol(mol):
        mask = torch.zeros(mol.GetNumAtoms(), dtype=bool)
        _mol = Chem.Mol(mol)
        try:
            Chem.SanitizeMol(_mol)
        except Exception:
            print(
                "Scaffold could not be extracted as reference molecule could not be sanitized. Skipping."
            )
            return mask
        for a in _mol.GetAtoms():
            a.SetIntProp("org_idx", a.GetIdx())

        scaffold = GetScaffoldForMol(_mol)
        scaffold_atoms = [a.GetIntProp("org_idx") for a in scaffold.GetAtoms()]
        if len(scaffold_atoms) > 0:
            try:
                mask[torch.tensor(scaffold_atoms)] = 1
            except Exception as e:
                print(e)
        return mask

    return [scaffold_per_mol(mol) for mol in to_mols]


def sample_mol_sizes(
    molecule_size,
    dataset: str = None,
    by_mean_and_std=False,
    upper_bound=0.1,
    lower_bound=0.1,
    n_molecules=1,
    seed=None,
    default_max=80,
    default_min=5,
):
    """
    Sample molecule sizes either by mean and std dev or by a fixed range
    :param molecule_size: The given molecule size
    :param dataset: The dataset used for sampling
    :param by_mean_and_std: If True (default: False), sample from a distribution with mean and std dev based on the dataset
    :param upper_bound: If by_mean_and_std is False, sample around the molecule size with this upper bound (percentage of molecule size)
    :param lower_bound: If by_mean_and_std is False, sample around the molecule size with this lower bound (percentage of molecule size)
    :param n_molecules: The number of molecule sizes to sample
    :param seed: The seed for reproducibility
    :return: list of sampled molecule sizes
    """

    if seed is not None:
        # set the seed for reproducibility
        np.random.seed(seed)

    if dataset is not None:
        max_size = (
            PLINDER_MOLECULE_SIZE_MAX
            if dataset == "plinder"
            else (
                KINODATA_MOLECULE_SIZE_MAX
                if dataset == "kinodata"
                else (
                    CROSSDOCKED_MOLECULE_SIZE_MAX
                    if dataset == "crossdocked"
                    else default_max
                )
            )
        )
        min_size = (
            PLINDER_MOLECULE_SIZE_MIN
            if dataset == "plinder"
            else (
                KINODATA_MOLECULE_SIZE_MIN
                if dataset == "kinodata"
                else (
                    CROSSDOCKED_MOLECULE_SIZE_MIN
                    if dataset == "crossdocked"
                    else default_min
                )
            )
        )
    else:
        max_size = default_max
        min_size = default_min

    if by_mean_and_std:
        assert (
            dataset is not None
        ), "Dataset must be specified when sampling by mean and std dev"
        mean = (
            PLINDER_MOLECULE_SIZE_MEAN
            if dataset == "plinder"
            else (
                KINODATA_MOLECULE_SIZE_MEAN
                if dataset == "kinodata"
                else (
                    CROSSDOCKED_MOLECULE_SIZE_MEAN if dataset == "crossdocked" else None
                )
            )
        )
        std_dev = (
            PLINDER_MOLECULE_SIZE_STD_DEV
            if dataset == "plinder"
            else (
                KINODATA_MOLECULE_SIZE_STD_DEV
                if dataset == "kinodata"
                else (
                    CROSSDOCKED_MOLECULE_SIZE_STD_DEV
                    if dataset == "crossdocked"
                    else None
                )
            )
        )
        if mean is None:
            raise ValueError(f"Invalid dataset {dataset}")

        if molecule_size < std_dev:
            lower_bound = molecule_size
            upper_bound = molecule_size + std_dev / 2
        elif molecule_size < mean - std_dev:
            lower_bound = molecule_size - std_dev / 2
            upper_bound = molecule_size + std_dev
        elif molecule_size > mean + std_dev:
            lower_bound = molecule_size - std_dev
            upper_bound = molecule_size + std_dev / 2
        else:
            lower_bound = molecule_size - std_dev
            upper_bound = molecule_size + std_dev

        sampled_sizes = np.random.uniform(lower_bound, upper_bound, n_molecules)
        sampled_sizes = np.round(sampled_sizes).astype(int)
        sampled_sizes = np.clip(sampled_sizes, min_size, max_size)

    else:
        sampled_sizes = np.random.uniform(
            molecule_size - lower_bound * molecule_size,
            molecule_size + upper_bound * molecule_size,
            n_molecules,
        )
        sampled_sizes = np.round(sampled_sizes).astype(int)
        sampled_sizes = np.clip(sampled_sizes, min_size, max_size)

    if n_molecules == 1:
        return sampled_sizes[0]
    else:
        return sampled_sizes


def get_cosine_scheduler_coefficients(t: torch.Tensor, nu: int = 1):
    """
    Compute the cosine scheduler coefficients.
    Args:
        t: Time tensor (between 0 and 1)
        nu: Exponent for the cosine function
    Returns:
        alpha_t: Coefficient for the signal
        sigma_t: Coefficient for the noise
    """
    y = 0.5 * torch.pi * ((1 - t) ** nu)
    y = torch.cos(y)
    alpha_t = y**2
    sigma_t = 1.0 - alpha_t
    return alpha_t, sigma_t


class NoiseSchedule(ABC):
    """Abstract base class for noise scheduling strategies."""

    @abstractmethod
    def get_noise_scale(self, t: torch.Tensor) -> torch.Tensor:
        """Get the noise scale for given time t."""
        pass

    def sample_noise(self, coords_mean: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample noise and return coords_mean + noise."""
        noise_scale = self.get_noise_scale(t)
        noise = torch.randn_like(coords_mean)
        return noise_scale * noise


class ConstantThenDecayNoiseSchedule(NoiseSchedule):
    """Constant noise until decay_start, then smooth decay."""

    def __init__(
        self, base_scale: float = 0.2, decay_start: float = 0.9, decay_rate: float = 1.0
    ):
        self.base_scale = base_scale
        self.decay_start = decay_start
        self.decay_rate = decay_rate

    def get_noise_scale(self, t: torch.Tensor) -> torch.Tensor:
        """
        Constant noise until decay_start, then smooth decay.
        Args:
            t: Time value between 0 and 1
        """
        # Handle both scalar and tensor inputs
        if isinstance(t, (int, float)):
            t = torch.tensor(t, dtype=torch.float32)

        noise_scale = torch.full_like(t, self.base_scale, dtype=torch.float32)

        # Apply decay where t > decay_start
        decay_mask = t > self.decay_start
        if decay_mask.any():
            decay_progress = (t[decay_mask] - self.decay_start) / (
                1.0 - self.decay_start
            )
            decay_factor = (1 - decay_progress) ** self.decay_rate
            noise_scale[decay_mask] = self.base_scale * decay_factor

        return noise_scale


class CosineSquaredNoiseSchedule(NoiseSchedule):
    """Cosine squared noise scheduling: cos(0.5 * π * t²)."""

    def __init__(self):
        pass

    def get_noise_scale(self, t: torch.Tensor) -> torch.Tensor:
        """
        Cosine squared noise scale: cos(0.5 * π * t²).
        Args:
            t: Time tensor (between 0 and 1)
        """
        if isinstance(t, (int, float)):
            t = torch.tensor(t, dtype=torch.float32)

        return torch.cos(0.5 * torch.pi * (t**2))


class StandardNoiseSchedule(NoiseSchedule):
    """Standard noise scheduling with t*(1-t) scaling."""

    def __init__(self):
        pass

    def get_noise_scale(self, t: torch.Tensor) -> torch.Tensor:
        """
        Standard noise scale: t*(1-t).
        Args:
            t: Time tensor (between 0 and 1)
        """
        if isinstance(t, (int, float)):
            t = torch.tensor(t, dtype=torch.float32)

        return t * (1 - t)


class GeometryDistortionNoiseSchedule(NoiseSchedule):
    """Geometry distortion noise scheduling that adds perturbations to subset of atoms."""

    def __init__(
        self, p_distort: float = 0.2, t_distort: float = 0.5, sigma_distort: float = 0.5
    ):
        """
        Initialize geometry distortion noise schedule.

        Args:
            p_distort: Probability of distorting each atom (Bernoulli parameter)
            t_distort: Time threshold after which distortion is applied
            sigma_distort: Standard deviation of per-atom displacement
        """
        self.p_distort = p_distort
        self.t_distort = t_distort
        self.sigma_distort = sigma_distort

    def get_noise_scale(self, t: torch.Tensor) -> torch.Tensor:
        """This method is not used for geometry distortion."""
        return torch.zeros_like(t)

    def sample_noise(self, coords_mean: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample geometry distortion noise.

        Args:
            coords_mean: Mean coordinates of shape (N, 3)
            t: Time value between 0 and 1

        Returns:
            Distortion noise to be added to coords_mean
        """
        if isinstance(t, (int, float)):
            t = torch.tensor(t, dtype=torch.float32)

        N = coords_mean.shape[0]
        device = coords_mean.device
        dtype = coords_mean.dtype

        # Initialize noise as zeros
        noise = torch.zeros_like(coords_mean)

        # Apply distortion only if t >= t_distort
        if t.item() >= self.t_distort:
            # Sample binary mask M ~ Bernoulli(p_distort) for each atom
            M = torch.bernoulli(
                torch.full((N, 1), self.p_distort, device=device, dtype=dtype)
            )

            # Sample per-atom displacement ε ~ N(0, σ_distort * I_3)
            epsilon = torch.randn_like(coords_mean) * self.sigma_distort

            # Apply mask via Hadamard product (element-wise multiplication)
            # M is (N, 1) and epsilon is (N, 3), so broadcast M across 3 dimensions
            noise = M * epsilon

        return noise


class Interpolant(ABC):
    @property
    @abstractmethod
    def hparams(self):
        pass

    @abstractmethod
    def interpolate(self, to_batch: list[SmolMol]) -> _InterpT:
        pass


class NoiseSampler(ABC):
    @property
    def hparams(self):
        pass

    @abstractmethod
    def sample_molecule(self, num_atoms: int) -> SmolMol:
        pass

    @abstractmethod
    def sample_batch(self, num_atoms: list[int]) -> SmolBatch:
        pass


class GeometricNoiseSampler(NoiseSampler):
    def __init__(
        self,
        vocab_size: int,
        n_bond_types: int,
        n_charge_types: int,
        n_hybridization_types: Optional[int] = None,
        n_aromatic_types: Optional[int] = None,
        coord_noise: str = "gaussian",
        type_noise: str = "uniform-sample",
        bond_noise: str = "uniform-sample",
        zero_com: bool = True,
        type_mask_index: Optional[int] = None,
        bond_mask_index: Optional[int] = None,
        atom_types_distribution: Optional[torch.Tensor] = None,
        bond_types_distribution: Optional[torch.Tensor] = None,
        conformer_generator: Optional[ConformerGenerator] = None,
    ):
        if coord_noise != "gaussian":
            raise NotImplementedError(f"Coord noise {coord_noise} is not supported.")

        self.conformer_generator = conformer_generator

        self._check_cat_noise_type(type_noise, type_mask_index, "type")
        self._check_cat_noise_type(bond_noise, bond_mask_index, "bond")

        self.vocab_size = vocab_size
        self.n_bond_types = n_bond_types
        self.n_charge_types = n_charge_types
        self.n_hybridization_types = n_hybridization_types
        self.n_aromatic_types = n_aromatic_types
        self.coord_noise = coord_noise
        self.type_noise = type_noise
        self.bond_noise = bond_noise
        self.zero_com = zero_com
        self.type_mask_index = type_mask_index
        self.bond_mask_index = bond_mask_index
        self.atom_types_distribution = atom_types_distribution
        self.bond_types_distribution = bond_types_distribution

        self.coord_dist = torch.distributions.Normal(
            torch.tensor(0.0), torch.tensor(1.0)
        )

    @property
    def hparams(self):
        return {
            "coord-noise": self.coord_noise,
            "type-noise": self.type_noise,
            "bond-noise": self.bond_noise,
            "zero-com": self.zero_com,
        }

    def fuse_fragments(
        self,
        fixed_fragment: GeometricMol,
        variable_fragment: GeometricMol,
        fragment_mode: str,
    ) -> GeometricMol:
        """
        Fuse fixed and variable fragments into a single molecule.
        Fixed fragment occupies indices [0:N_fixed], variable fragment [N_fixed:N_total].

        Args:
            fixed_fragment: The fixed part (e.g., scaffold)
            variable_fragment: The variable part to generate
            fragment_mode: The inpainting mode

        Returns:
            Fused GeometricMol with proper indexing and fragment metadata
        """
        N_fixed = fixed_fragment.seq_length
        N_var = variable_fragment.seq_length
        N_total = N_fixed + N_var

        if N_fixed == 0:
            raise ValueError(
                f"Fixed fragment is empty for mode {fragment_mode}. "
                "This indicates incorrect mask extraction."
            )

        if N_var == 0:
            raise ValueError(
                f"Variable fragment is empty for mode {fragment_mode}. "
                "This indicates incorrect size calculation or sampling."
            )

        # Concatenate atom features
        coords = torch.cat([fixed_fragment.coords, variable_fragment.coords], dim=0)
        atomics = torch.cat([fixed_fragment.atomics, variable_fragment.atomics], dim=0)
        charges = torch.cat([fixed_fragment.charges, variable_fragment.charges], dim=0)

        # Concatenate hybridization if present
        if (
            fixed_fragment.hybridization is not None
            and variable_fragment.hybridization is not None
        ):
            hybridization = torch.cat(
                [fixed_fragment.hybridization, variable_fragment.hybridization], dim=0
            )
        else:
            hybridization = None

        # Handle bonds - shift variable fragment indices by N_fixed
        fixed_bond_indices = fixed_fragment.bond_indices
        fixed_bond_types = fixed_fragment.bond_types

        var_bond_indices = variable_fragment.bond_indices + N_fixed
        var_bond_types = variable_fragment.bond_types

        # Combine bonds
        bond_indices = torch.cat([fixed_bond_indices, var_bond_indices], dim=0)
        bond_types = torch.cat([fixed_bond_types, var_bond_types], dim=0)

        # Create fragment mask: True for fixed fragment atoms
        fragment_mask = torch.zeros(N_total, dtype=torch.bool, device=coords.device)
        fragment_mask[:N_fixed] = True

        # Create fused molecule
        fused_mol = GeometricMol(
            coords=coords,
            atomics=atomics,
            charges=charges,
            hybridization=hybridization,
            bond_indices=bond_indices,
            bond_types=bond_types,
            fragment_mask=fragment_mask,
            fragment_mode=fragment_mode,
        )

        return fused_mol

    def inpaint_molecule(
        self,
        to_mol: GeometricMol,
        from_mol: GeometricMol,
        fragment_mask: torch.Tensor,
        fragment_mode: Optional[str] = "",
        harmonic_prior: Optional[bool] = False,
        symmetrize: Optional[bool] = False,
    ) -> GeometricMol:

        # Convert multi-fragment masks to single combined mask
        if (
            fragment_mask is not None
            and isinstance(fragment_mask, list)
            and len(fragment_mask) > 1
        ):
            # Combine all fragment masks
            combined_mask = torch.zeros_like(fragment_mask[0])
            for m in fragment_mask:
                combined_mask = combined_mask | m
            fragment_mask = ~combined_mask

        if fragment_mask is None or not fragment_mask.any():
            # if fragment_mask is already specified by graph_inpainting, return from_mol
            from_mol.fragment_mode = fragment_mode
            if from_mol.fragment_mask is not None:
                return from_mol
            from_mol.fragment_mask = torch.zeros(
                from_mol.seq_length, dtype=torch.bool, device=from_mol.coords.device
            )
            return from_mol

        if harmonic_prior:
            harmonic_coords = self._generate_harmonic_coordinates(to_mol)
            inp_coords = harmonic_coords - (
                harmonic_coords.sum(dim=0) / harmonic_coords.size(0)
            )
        else:
            inp_coords = from_mol.coords.clone()

        inp_atomics, inp_charges = (
            from_mol.atomics.clone(),
            from_mol.charges.clone(),
        )
        inp_coords[fragment_mask, :] = to_mol.coords[fragment_mask, :]
        inp_atomics[fragment_mask, :] = to_mol.atomics[fragment_mask, :]
        inp_charges[fragment_mask, :] = to_mol.charges[fragment_mask, :]
        if self.n_hybridization_types is not None:
            inp_hybridization = from_mol.hybridization.clone()
            inp_hybridization[fragment_mask, :] = to_mol.hybridization[fragment_mask, :]
        else:
            inp_hybridization = None

        # Overwrite bond types
        N = from_mol.seq_length
        bond_indices = torch.ones((N, N), device=from_mol.coords.device).nonzero(
            as_tuple=False
        )
        num_bond_types = to_mol.adjacency.size(-1)
        from_adj = torch.argmax(from_mol.adjacency, dim=-1)
        to_adj = torch.argmax(to_mol.adjacency, dim=-1)

        ## only update bonds if both atoms are inpainted
        fixed_mask = fragment_mask.unsqueeze(0) & fragment_mask.unsqueeze(
            1
        )  # & (to_adj > 0)
        new_adj = from_adj.clone()
        new_adj[fixed_mask] = to_adj[fixed_mask]
        if symmetrize:
            new_adj = smolF.symmetrize_bonds(new_adj, is_one_hot=False)

        new_bond_types = smolF.one_hot_encode_tensor(
            new_adj, num_bond_types
        )  # shape: (N, N, num_bond_types)
        new_bond_types = new_bond_types[bond_indices[:, 0], bond_indices[:, 1]]

        return GeometricMol(
            inp_coords,
            inp_atomics,
            charges=inp_charges,
            hybridization=inp_hybridization,
            bond_indices=bond_indices,
            bond_types=new_bond_types,
            fragment_mask=fragment_mask,
            fragment_mode=fragment_mode,
        )

    def inpaint_graph(
        self,
        to_mol: GeometricMol,
        from_mol: GeometricMol,
        mask: torch.Tensor,
        mode: str = "random",
    ) -> GeometricMol:

        # if all mask is False, return from_mol with fragment_mask
        if mask is None or not mask.any():
            mask = torch.zeros(
                from_mol.seq_length, dtype=torch.bool, device=from_mol.coords.device
            )
            from_mol.fragment_mask = mask
            from_mol.fragment_mode = "graph"
            return from_mol
        else:
            # create atom-wise mask
            mask = torch.ones(
                from_mol.seq_length, dtype=torch.bool, device=from_mol.coords.device
            )

        if mode == "conformer":
            return self.inpaint_graph_with_conformer(to_mol, from_mol, mask)
        elif mode == "harmonic":
            return self.inpaint_graph_with_harmonic(to_mol, from_mol, mask)
        elif mode == "random":
            coords = self.coord_dist.sample((to_mol.seq_length, 3))
            inp_atomics, inp_charges, inp_bond_types, inp_bond_indices = (
                to_mol.atomics.clone(),
                to_mol.charges.clone(),
                to_mol.bond_types.clone(),
                to_mol.bond_indices.clone(),
            )
            if self.n_hybridization_types is not None:
                inp_hybridization = from_mol.hybridization.clone()

            mol = GeometricMol(
                coords,
                inp_atomics,
                charges=inp_charges,
                hybridization=inp_hybridization,
                bond_indices=inp_bond_indices,
                bond_types=inp_bond_types,
                fragment_mask=mask,
                fragment_mode="graph",
            )
            if self.zero_com:
                mol = mol.zero_com()
            return mol

    def inpaint_graph_with_conformer(
        self,
        to_mol: GeometricMol,
        from_mol: GeometricMol,
        mask: torch.Tensor,
    ) -> GeometricMol:
        """
        Graph inpainting using RDKit conformer as starting coordinates
        instead of noisy coordinates
        """

        # Generate conformer coordinates
        conformer_coords = None
        assert (
            self.conformer_generator is not None
        ), "Conformer generator must be provided for conformer inpainting."
        conformer_coords = self.conformer_generator.generate_conformer_from_graph(
            to_mol
        )

        mol = GeometricMol(
            conformer_coords,
            to_mol.atomics.clone(),
            charges=to_mol.charges.clone(),
            hybridization=(
                to_mol.hybridization.clone()
                if self.n_hybridization_types is not None
                else None
            ),
            bond_indices=to_mol.bond_indices.clone(),
            bond_types=to_mol.bond_types.clone(),
            fragment_mask=mask,
        )
        if self.zero_com:
            mol = mol.zero_com()
        return mol

    def inpaint_graph_with_harmonic(
        self,
        to_mol: GeometricMol,
        from_mol: GeometricMol,
        mask: torch.Tensor,
    ) -> GeometricMol:
        """
        Graph inpainting using harmonic prior based on molecular bonds
        """

        try:
            # Generate harmonic coordinates
            harmonic_coords = self._generate_harmonic_coordinates(to_mol)

            if harmonic_coords is None:
                # Fallback to random coordinates
                harmonic_coords = self.coord_dist.sample((to_mol.seq_length, 3))

            mol = GeometricMol(
                harmonic_coords,
                to_mol.atomics.clone(),
                charges=to_mol.charges.clone(),
                hybridization=(
                    to_mol.hybridization.clone()
                    if self.n_hybridization_types is not None
                    else None
                ),
                bond_indices=to_mol.bond_indices.clone(),
                bond_types=to_mol.bond_types.clone(),
                fragment_mask=mask,
            )

            if self.zero_com:
                mol = mol.zero_com()

            return mol

        except Exception as e:
            print(f"Harmonic coordinate generation failed: {e}")
            # Fallback to standard random sampling
            return self.inpaint_graph(to_mol, from_mol, mask)

    def _generate_harmonic_coordinates(
        self, to_mol: GeometricMol
    ) -> Optional[torch.Tensor]:
        """
        Generate coordinates using harmonic potential based on molecular graph
        """
        try:
            n_atoms = to_mol.seq_length
            device = to_mol.device

            # Extract bond information from GeometricMol
            bond_indices = to_mol.bond_indices  # [n_bonds, 2]
            bond_types = torch.argmax(to_mol.bond_types, dim=-1)  # [n_bonds]

            # Filter out non-bonded pairs (bond_type = 0 typically means no bond)
            real_bonds_mask = bond_types > 0
            if not real_bonds_mask.any():
                print("No real bonds found in molecule")
                return None

            edges = bond_indices[real_bonds_mask]  # [n_real_bonds, 2]

            # Remove duplicate edges (keep only i < j)
            edge_mask = edges[:, 0] < edges[:, 1]
            edges = edges[edge_mask]

            if edges.shape[0] == 0:
                print("No valid edges found after filtering")
                return None

            # Build harmonic matrix
            D, P = self._diagonalize_harmonic_matrix(n_atoms, edges, device=device)

            if D is None or P is None:
                return None

            # Sample from harmonic distribution
            noise = torch.randn(n_atoms, 3, device=device)

            # Ensure D is positive for numerical stability
            D_safe = torch.clamp(D, min=1e-8)

            # Generate harmonic coordinates: P @ (noise / sqrt(D))
            harmonic_coords = P @ (noise / torch.sqrt(D_safe).unsqueeze(-1))

            return harmonic_coords

        except Exception as e:
            print(f"Error in harmonic coordinate generation: {e}")
            return None

    def _diagonalize_harmonic_matrix(
        self,
        n_atoms: int,
        edges: torch.Tensor,
        device: torch.device,
        spring_constant: float = 1.0,
        regularization: float = 0.01,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Build and diagonalize harmonic matrix for molecular bonds

        Args:
            n_atoms: Number of atoms
            edges: Bond indices [n_bonds, 2]
            device: Target device
            spring_constant: Harmonic spring constant for bonds
            regularization: Regularization to ensure positive definiteness

        Returns:
            Eigenvalues D and eigenvectors P, or (None, None) if failed
        """
        try:
            # Build harmonic matrix (graph Laplacian-like)
            J = torch.zeros(n_atoms, n_atoms, device=device, dtype=torch.float32)

            # Add harmonic springs for each bond
            for edge in edges:
                i, j = edge[0].item(), edge[1].item()
                if i >= n_atoms or j >= n_atoms:
                    continue

                # Harmonic potential: 0.5 * k * (r_i - r_j)^2
                # This gives force matrix elements:
                J[i, i] += spring_constant
                J[j, j] += spring_constant
                J[i, j] -= spring_constant
                J[j, i] -= spring_constant

            # Add regularization for numerical stability
            J += torch.eye(n_atoms, device=device) * regularization

            # Check if matrix is reasonable
            if torch.isnan(J).any() or torch.isinf(J).any():
                print("Harmonic matrix contains NaN or Inf")
                return None, None

            # Eigendecomposition
            try:
                D, P = torch.linalg.eigh(J)
            except Exception as e:
                print(f"Eigendecomposition failed: {e}")
                return None, None

            # Check for numerical issues
            if torch.isnan(D).any() or torch.isnan(P).any():
                print("Eigendecomposition produced NaN values")
                return None, None

            # Ensure positive eigenvalues (due to regularization, should be positive)
            if (D <= 0).any():
                print(f"Found non-positive eigenvalues: min={D.min().item()}")
                D = torch.clamp(D, min=1e-6)

            return D, P

        except Exception as e:
            print(f"Error in harmonic matrix diagonalization: {e}")
            return None, None

    def sample_molecule(self, n_atoms: int, symmetrize: bool = False) -> GeometricMol:

        # Sample coords and scale, if required
        coords = self.coord_dist.sample((n_atoms, 3))

        if self.type_noise == "uniform-sample":
            atomics = torch.randint(1, self.vocab_size, (n_atoms,))
            atomics = smolF.one_hot_encode_tensor(atomics, self.vocab_size)

        elif self.type_noise == "prior-sample":
            atom_types_distribution = torch.zeros(
                (self.vocab_size,), dtype=torch.float32
            )
            atom_types_distribution[1:] = (
                self.atom_types_distribution + 1.0e-6
            )  # skip pad tokens and add a bit of signal to all states
            atomics = torch.multinomial(
                atom_types_distribution, n_atoms, replacement=True
            )
            atomics = smolF.one_hot_encode_tensor(atomics, self.vocab_size)
        else:
            raise ValueError(f"Unknown atom noise type: {self.type_noise}")

        if self.type_noise == "uniform-sample":
            charges = torch.randint(1, self.n_charge_types, (n_atoms,))
            charges = smolF.one_hot_encode_tensor(charges, self.n_charge_types)
        elif self.charge_noise == "prior-sample":
            charge_types_distribution = torch.zeros(
                (self.n_charge_types,), dtype=torch.float32
            )
            charge_types_distribution[1:] = (
                self.charge_types_distribution + 1.0e-6
            )  # skip pad tokens and add a bit of signal to all states
            charges = torch.multinomial(
                charge_types_distribution, n_atoms, replacement=True
            )
            charges = smolF.one_hot_encode_tensor(charges, self.n_charge_types)

        if self.n_hybridization_types is not None:
            if self.type_noise == "uniform-sample":
                hybridization = torch.randint(1, self.n_hybridization_types, (n_atoms,))
                hybridization = smolF.one_hot_encode_tensor(
                    hybridization, self.n_hybridization_types
                )
            elif self.type_noise == "prior-sample":
                hybridization_types_distribution = torch.zeros(
                    (self.n_hybridization_types,), dtype=torch.float32
                )
                hybridization_types_distribution[1:] = (
                    self.hybridization_types_distribution + 1.0e-6
                )  # skip pad tokens and add a bit of signal to all states
                hybridization = torch.multinomial(
                    hybridization_types_distribution, n_atoms, replacement=True
                )
                hybridization = smolF.one_hot_encode_tensor(
                    hybridization, self.n_hybridization_types
                )

        # Create bond indices and sample bond types
        bond_indices = torch.ones((n_atoms, n_atoms)).nonzero()
        n_bonds = bond_indices.size(0)
        if self.bond_noise == "uniform-sample":
            bond_types = torch.randint(0, self.n_bond_types, size=(n_bonds,))

        elif self.bond_noise == "prior-sample":
            bond_types = torch.multinomial(
                self.bond_types_distribution, n_bonds, replacement=True
            )

        # Convert to adjacency
        bond_adj = torch.zeros((n_atoms, n_atoms), dtype=torch.long)
        bond_adj[bond_indices[:, 0], bond_indices[:, 1]] = bond_types
        if symmetrize:
            bond_adj = smolF.symmetrize_bonds(bond_adj, is_one_hot=False)
        # Convert back to edge list format
        bond_types = smolF.one_hot_encode_tensor(bond_adj, self.n_bond_types)
        bond_types = bond_types[bond_indices[:, 0], bond_indices[:, 1]]

        # Create smol mol object
        mol = GeometricMol(
            coords,
            atomics,
            charges=charges,
            hybridization=(
                hybridization if self.n_hybridization_types is not None else None
            ),
            bond_indices=bond_indices,
            bond_types=bond_types,
        )
        if self.zero_com:
            mol = mol.zero_com()

        return mol

    def sample_batch(self, num_atoms: list[int]) -> GeometricMolBatch:
        mols = [self.sample_molecule(n) for n in num_atoms]
        batch = GeometricMolBatch.from_list(mols)
        return batch

    def _check_cat_noise_type(self, noise_type, mask_index, name):
        if noise_type not in ["mask", "uniform-sample", "prior-sample"]:
            raise ValueError(f"{name} noise {noise_type} is not supported.")

        if noise_type == "mask" and mask_index is None:
            raise ValueError(
                f"{name}_mask_index must be provided if {name}_noise is 'mask'."
            )


class MixedTimeSampler:
    def __init__(self, alpha, beta, mix_prob=0.5):
        self.beta_dist = torch.distributions.Beta(alpha, beta)
        self.mix_prob = mix_prob

    def sample(self, sample_shape):
        beta_sample = self.beta_dist.sample(sample_shape)
        uniform_sample = torch.rand(sample_shape)
        mix_mask = torch.rand(sample_shape) < self.mix_prob
        return torch.where(mix_mask, uniform_sample, beta_sample)


class GeometricInterpolant(Interpolant):
    def __init__(
        self,
        prior_sampler: GeometricNoiseSampler,
        coord_interpolation: str = "linear",
        type_interpolation: str = "unmask",
        bond_interpolation: str = "unmask",
        coord_noise_std: float = 0.0,
        coord_noise_schedule: str = "standard",
        type_dist_temp: float = 1.0,
        time_alpha: float = 1.0,
        time_beta: float = 1.0,
        dataset: str = "geom-drugs",
        fixed_time: Optional[float] = None,
        split_continuous_discrete_time: bool = False,
        mixed_uniform_beta_time: bool = False,
        scaffold_inpainting: bool = False,
        func_group_inpainting: bool = False,
        linker_inpainting: bool = False,
        core_inpainting: bool = False,
        max_fragment_cuts: int = 3,
        fragment_inpainting: bool = False,
        substructure_inpainting: bool = False,
        substructure: Optional[str] = None,
        graph_inpainting: str | None = None,
        mixed_uncond_inpaint: bool = False,
        inference: bool = False,
        sample_mol_sizes: Optional[bool] = False,
        vocab: Optional[dict] = None,
        vocab_charges: Optional[dict] = None,
        vocab_hybridization: Optional[dict] = None,
        vocab_aromatic: Optional[dict] = None,
        batch_ot: bool = False,
        rotation_alignment: bool = False,
        permutation_alignment: bool = False,
        fragment_size_variation: float = 0.15,
    ):

        if fixed_time is not None and (fixed_time < 0 or fixed_time > 1):
            raise ValueError("fixed_time must be between 0 and 1 if provided.")

        if coord_interpolation not in ["linear", "cosine"]:
            raise ValueError(
                f"coord interpolation '{coord_interpolation}' not supported."
            )

        if type_interpolation not in ["unmask", "sample"]:
            raise ValueError(
                f"type interpolation '{type_interpolation}' not supported."
            )

        if bond_interpolation not in ["unmask", "sample"]:
            raise ValueError(
                f"bond interpolation '{bond_interpolation}' not supported."
            )

        self.prior_sampler = prior_sampler
        self.coord_interpolation = coord_interpolation
        self.type_interpolation = type_interpolation
        self.bond_interpolation = bond_interpolation
        self.coord_noise_std = coord_noise_std
        self.type_dist_temp = type_dist_temp
        self.time_alpha = time_alpha if fixed_time is None else None
        self.time_beta = time_beta if fixed_time is None else None
        self.fixed_time = fixed_time
        self.split_continuous_discrete_time = split_continuous_discrete_time
        self.graph_inpainting = graph_inpainting
        self.sample_mol_sizes = sample_mol_sizes
        self.scaffold_inpainting = scaffold_inpainting
        self.func_group_inpainting = func_group_inpainting
        self.linker_inpainting = linker_inpainting
        self.core_inpainting = core_inpainting
        self.fragment_inpainting = fragment_inpainting
        self.fragment_modes = None  # ["single", "multi-2", "multi-3"]
        self.max_fragment_cuts = max_fragment_cuts
        self.substructure_inpainting = substructure_inpainting
        if substructure_inpainting:
            if len(substructure) == 1 and isinstance(substructure[0], str):
                # Single SMILES/SMARTS string
                self.substructure = substructure[0]
            else:
                # List of atom indices
                self.substructure = substructure
        self.mixed_uncond_inpaint = mixed_uncond_inpaint
        self.batch_ot = batch_ot
        self.rotation_alignment = rotation_alignment
        self.permutation_alignment = permutation_alignment
        self.dataset = dataset
        self.inference = inference
        self.fragment_size_variation = fragment_size_variation

        self.vocab = vocab
        self.vocab_charges = vocab_charges
        self.vocab_hybridization = vocab_hybridization
        self.inpainting_mode = (
            scaffold_inpainting
            or func_group_inpainting
            or linker_inpainting
            or core_inpainting
            or substructure_inpainting
            or fragment_inpainting
        )
        if self.inpainting_mode and self.graph_inpainting:
            print(
                "Inpainting mode and graph inpainting are both enabled - if inference mode, graph_inpainting is set to the default!"
            )

        if mixed_uniform_beta_time:
            self.time_dist = MixedTimeSampler(alpha=4.0, beta=1.0, mix_prob=0.15)
            if self.split_continuous_discrete_time:
                self.time_dist_disc = MixedTimeSampler(
                    alpha=4.0, beta=1.0, mix_prob=0.15
                )
        else:
            self.time_dist = torch.distributions.Beta(time_alpha, time_beta)
            if self.split_continuous_discrete_time:
                self.time_dist_disc = torch.distributions.Beta(time_alpha, time_beta)

        # Define noise scheduler for coordinate interpolation
        self.coord_noise_schedule = coord_noise_schedule
        if coord_noise_schedule == "constant_decay":
            self.noise_scheduler = ConstantThenDecayNoiseSchedule(
                base_scale=coord_noise_std, decay_start=1.0, decay_rate=1.5
            )
        elif coord_noise_schedule == "cosine_squared":
            self.noise_scheduler = CosineSquaredNoiseSchedule()
        elif coord_noise_schedule == "standard":
            self.noise_scheduler = StandardNoiseSchedule()
        elif coord_noise_schedule == "geometry_distortion":
            self.noise_scheduler = GeometryDistortionNoiseSchedule(
                p_distort=0.25, t_distort=0.5, sigma_distort=coord_noise_std
            )
        else:
            raise ValueError(f"Unknown noise schedule: {coord_noise_schedule}")

        self.symmetrize = (
            True  # NOTE: Whether or not to symmetrize bonds: Should be true
        )

        # Define which modes are local vs global
        # Local modes: fragment and substructure inpainting (invert mask to keep everything except fragment)
        # Global modes: everything else (keep the specified structure, generate the rest)
        self.local_modes = {"fragment_inpainting", "substructure_inpainting"}
        self.global_modes = {
            "scaffold_inpainting",
            "func_group_inpainting",
            "linker_inpainting",
            "core_inpainting",
        }

    @property
    def hparams(self):
        prior_hparams = {f"prior-{k}": v for k, v in self.prior_sampler.hparams.items()}
        hparams = {
            "coord-interpolation": self.coord_interpolation,
            "type-interpolation": self.type_interpolation,
            "bond-interpolation": self.bond_interpolation,
            "coord-noise-std": self.coord_noise_std,
            "coord-noise-schedule": self.coord_noise_schedule,
            "type-dist-temp": self.type_dist_temp,
            "rotation-alignment": self.rotation_alignment,
            "permutation-alignment": self.permutation_alignment,
            "batch-ot": self.batch_ot,
            "time-alpha": self.time_alpha,
            "time-beta": self.time_beta,
            "dataset": self.dataset,
            **prior_hparams,
        }

        if self.fixed_time is not None:
            hparams["fixed-interpolation-time"] = self.fixed_time

        return hparams

    def _calculate_fragment_center_of_mass(
        self, mol: GeometricMol, mask: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the center of mass for atoms specified by the mask."""
        if not mask.any():
            return torch.zeros(3, device=mol.coords.device)

        fragment_coords = mol.coords[mask]
        return fragment_coords.mean(dim=0)

    def _align_prior_to_fragment(
        self,
        from_mol: GeometricMol,
        to_mol: GeometricMol,
        to_mol_mask: torch.Tensor,
        from_mol_mask: torch.Tensor,
    ) -> GeometricMol:
        """
        Align the variable fragment in from_mol to match the position in to_mol.

        Args:
            from_mol: Fused molecule with fixed + variable fragments
            to_mol: Reference molecule (original size)
            to_mol_mask: Mask for to_mol (True = fixed/keep, False = variable/generate)
            from_mol_mask: Mask for from_mol (True = fixed/keep, False = variable/generate)
        """
        # Handle multi-fragment masks
        if isinstance(to_mol_mask, list) and len(to_mol_mask) > 1:
            return self._align_prior_to_multi_fragments(from_mol, to_mol, to_mol_mask)

        # Get variable atoms in to_mol (the reference positions)
        variable_mask_to = ~to_mol_mask
        if not variable_mask_to.any():
            return from_mol

        # Get variable atoms in from_mol (using from_mol's fragment_mask)
        variable_mask_from = ~from_mol_mask
        if not variable_mask_from.any():
            return from_mol

        # Calculate center of mass of the variable fragment in the target molecule
        target_com = self._calculate_fragment_center_of_mass(to_mol, variable_mask_to)

        # Calculate center of mass of the variable fragment in the prior
        prior_com = self._calculate_fragment_center_of_mass(
            from_mol, variable_mask_from
        )

        # Calculate translation vector
        translation = target_com - prior_com

        # Apply translation only to variable fragment atoms in from_mol
        new_coords = from_mol.coords.clone()
        new_coords[variable_mask_from] = (
            from_mol.coords[variable_mask_from] + translation
        )

        return from_mol._copy_with(coords=new_coords)

    def _align_prior_to_multi_fragments(
        self,
        from_mol: GeometricMol,
        to_mol: GeometricMol,
        fragment_masks: list[torch.Tensor],
    ) -> GeometricMol:
        """Align multiple disconnected prior fragments to their corresponding target fragments.

        Each fragment's atoms in the prior are aligned to the corresponding fragment's
        center of mass in the target molecule.

        Args:
            from_mol: Prior molecule with noise
            to_mol: Target molecule
            fragment_masks: List of masks, one per fragment to be replaced

        Returns:
            Aligned prior molecule
        """

        if not fragment_masks or not any(mask.any() for mask in fragment_masks):
            return from_mol

        new_coords = from_mol.coords.clone()

        # Align each fragment independently
        for mask in fragment_masks:
            if not mask.any():
                continue

            # Calculate COM of this fragment in the target molecule
            target_com = self._calculate_fragment_center_of_mass(to_mol, mask)

            # Calculate COM of the same atoms in the prior molecule
            prior_com = self._calculate_fragment_center_of_mass(from_mol, mask)

            # Calculate translation vector
            translation = target_com - prior_com

            # Apply translation to these specific atoms in the prior
            new_coords[mask] = from_mol.coords[mask] + translation

        return from_mol._copy_with(coords=new_coords)

    def interpolate(
        self, to_mols: list[GeometricMol], interaction_mask: torch.Tensor = None
    ) -> _GeometricInterpT:
        """
        Main interpolation entry point.
        Routes to appropriate path based on inference/training and inpainting mode.
        """

        # Determine if we're doing inpainting
        is_inpainting = (
            self.inpainting_mode
            or self.graph_inpainting
            or interaction_mask is not None
        )

        if not is_inpainting:
            # Standard full molecule generation
            return self._interpolate_standard(to_mols)

        # Inpainting modes
        if self.inference:
            # Inference with inpainting: variable sizes, no interpolation
            return self._inference_inpainting(to_mols, interaction_mask)
        else:
            # Training with inpainting: fixed sizes, standard interpolation
            return self._training_inpainting(to_mols, interaction_mask)

    def _interpolate_standard(self, to_mols: list[GeometricMol]) -> _GeometricInterpT:
        """Standard full molecule generation (no inpainting)."""
        batch_size = len(to_mols)

        if self.sample_mol_sizes and self.inference:
            mol_sizes = [
                sample_mol_sizes(
                    mol.seq_length,
                    by_mean_and_std=False,
                    upper_bound=0.1,
                    lower_bound=0.1,
                    n_molecules=1,
                )
                for mol in to_mols
            ]
        else:
            mol_sizes = [mol.seq_length for mol in to_mols]
        num_atoms = max(mol_sizes)

        # Sample prior
        from_mols = [
            self.prior_sampler.sample_molecule(num_atoms, symmetrize=self.symmetrize)
            for _ in to_mols
        ]

        # Align prior to reference
        from_mols = [
            self._match_mols(from_mol, to_mol, mol_size=mol_size)
            for from_mol, to_mol, mol_size in zip(from_mols, to_mols, mol_sizes)
        ]

        # Sample the batch times
        if self.fixed_time is not None:
            times_cont = torch.tensor([self.fixed_time] * batch_size)
            times_disc = torch.tensor([self.fixed_time] * batch_size)
        else:
            times_cont = self.time_dist.sample((batch_size,))
            if self.split_continuous_discrete_time:
                times_disc = self.time_dist_disc.sample((batch_size,))
            else:
                times_disc = times_cont.clone()

        # Create interpolated states
        tuples = zip_longest(
            from_mols,
            to_mols,
            times_cont.tolist(),
            times_disc.tolist(),
            fillvalue=None,
        )
        interp_mols = (
            [
                self._interpolate_mol(
                    from_mol,
                    to_mol,
                    t_cont=t_cont,
                    t_disc=t_disc,
                )
                for from_mol, to_mol, t_cont, t_disc in tuples
            ]
            if not (self.sample_mol_sizes and self.inference)
            else from_mols
        )

        times = torch.stack([times_cont, times_disc], dim=1)
        return from_mols, to_mols, interp_mols, times

    def _training_inpainting(
        self,
        to_mols: list[GeometricMol],
        interaction_mask: Optional[torch.Tensor] = None,
    ) -> _GeometricInterpT:
        """
        Training with inpainting: use exact reference sizes, apply masks, standard interpolation.
        """
        batch_size = len(to_mols)
        mol_sizes = [mol.seq_length for mol in to_mols]
        num_atoms = max(mol_sizes)

        # Pre-compute RDKit molecules for mask extraction
        rdkit_mols = [
            mol.to_rdkit(
                vocab=self.vocab,
                vocab_charges=self.vocab_charges,
                vocab_hybridization=self.vocab_hybridization,
            )
            for mol in to_mols
        ]

        # Determine modes and extract masks
        modes_and_masks = self._determine_modes_and_extract_masks(
            to_mols, rdkit_mols, interaction_mask
        )

        # Sample prior molecules (standard size)
        from_mols = [
            self.prior_sampler.sample_molecule(num_atoms, symmetrize=self.symmetrize)
            for _ in to_mols
        ]

        # Align prior to reference (permutation + rotation)
        from_mols = [
            self._match_mols(from_mol, to_mol, mol_size=mol_size)
            for from_mol, to_mol, mol_size in zip(from_mols, to_mols, mol_sizes)
        ]

        # Apply inpainting logic based on mode
        from_mols = [
            self._apply_training_inpainting(
                from_mol, to_mol, rdkit_mol, mode, mask, is_local
            )
            for from_mol, to_mol, rdkit_mol, (mode, mask, is_local) in zip(
                from_mols, to_mols, rdkit_mols, modes_and_masks
            )
        ]

        # Sample the batch times
        if self.fixed_time is not None:
            times_cont = torch.tensor([self.fixed_time] * batch_size)
            times_disc = torch.tensor([self.fixed_time] * batch_size)
        else:
            times_cont = self.time_dist.sample((batch_size,))
            if self.split_continuous_discrete_time:
                times_disc = self.time_dist_disc.sample((batch_size,))
            else:
                times_disc = times_cont.clone()

        # Create interpolated states
        tuples = zip_longest(
            from_mols,
            to_mols,
            times_cont.tolist(),
            times_disc.tolist(),
            fillvalue=None,
        )
        interp_mols = [
            self._interpolate_mol(
                from_mol,
                to_mol,
                t_cont=t_cont,
                t_disc=t_disc,
            )
            for from_mol, to_mol, t_cont, t_disc in tuples
        ]

        times = torch.stack([times_cont, times_disc], dim=1)
        return from_mols, to_mols, interp_mols, times

    def _apply_training_inpainting(
        self,
        from_mol: GeometricMol,
        to_mol: GeometricMol,
        rdkit_mol: Chem.Mol,
        mode: str,
        mask: torch.Tensor,
        is_local: bool,
    ) -> GeometricMol:
        """
        Apply inpainting at training time.
        Handles graph inpainting and fragment/substructure CoM alignment.
        """
        if mode == "de_novo":
            # For de novo, set metadata to indicate no inpainting
            from_mol = from_mol._copy_with(
                fragment_mask=torch.zeros(
                    from_mol.seq_length, dtype=torch.bool, device=from_mol.coords.device
                ),
                fragment_mode="de_novo",
            )
            return from_mol

        if mode == "graph" or (self.graph_inpainting and not self.inpainting_mode):
            return self.prior_sampler.inpaint_graph(
                to_mol,
                from_mol,
                mask=torch.ones(to_mol.seq_length, dtype=torch.bool),
                mode=self.graph_inpainting,
            )

        # Align variable fragment to reference for local modes
        if is_local:
            from_mol = self._align_prior_to_fragment(from_mol, to_mol, mask, mask)

        # Fragment-based inpainting
        # Apply mask-based inpainting
        from_mol = self.prior_sampler.inpaint_molecule(
            to_mol,
            from_mol,
            fragment_mask=mask,
            fragment_mode=mode,
            harmonic_prior=False,
            symmetrize=self.symmetrize,
        )
        return from_mol

    def _inference_inpainting(
        self,
        to_mols: list[GeometricMol],
        interaction_mask: Optional[torch.Tensor] = None,
    ) -> _GeometricInterpT:
        """
        Inference with inpainting: build fragment-based priors with variable sizes.
        No interpolation, return from_mols as interp_mols.
        """
        batch_size = len(to_mols)

        # Pre-compute RDKit molecules
        rdkit_mols = [
            mol.to_rdkit(
                vocab=self.vocab,
                vocab_charges=self.vocab_charges,
                vocab_hybridization=self.vocab_hybridization,
            )
            for mol in to_mols
        ]

        # Determine modes and extract masks
        modes_and_masks = self._determine_modes_and_extract_masks(
            to_mols, rdkit_mols, interaction_mask
        )

        # Build from_mols using fragment-based construction OR graph inpainting
        from_mols = [
            self._build_inference_prior(to_mol, rdkit_mol, mode, mask, is_local)
            for to_mol, rdkit_mol, (mode, mask, is_local) in zip(
                to_mols, rdkit_mols, modes_and_masks
            )
        ]

        # No interpolation at inference
        times_cont = torch.zeros(batch_size)
        times_disc = torch.zeros(batch_size)
        times = torch.stack([times_cont, times_disc], dim=1)

        return from_mols, to_mols, from_mols, times

    def _build_inference_prior(
        self,
        to_mol: GeometricMol,
        rdkit_mol: Chem.Mol,
        mode: str,
        mask: torch.Tensor,
        is_local: bool,
    ) -> GeometricMol:
        """
        Build prior molecule at inference time.
        Handles both graph inpainting and fragment-based construction.
        """
        if mode == "de_novo":
            from_mol = self.prior_sampler.sample_molecule(
                to_mol.seq_length, symmetrize=self.symmetrize
            )
            # Set metadata for de novo generation
            from_mol = from_mol._copy_with(
                fragment_mask=torch.zeros(
                    to_mol.seq_length, dtype=torch.bool, device=to_mol.coords.device
                ),
                fragment_mode="de_novo",
            )
            return from_mol

        if mode == "graph" or (self.graph_inpainting and not self.inpainting_mode):
            from_mol = self.prior_sampler.sample_molecule(
                to_mol.seq_length, symmetrize=self.symmetrize
            )
            # inpaint_graph sets the fragment_mask and fragment_mode
            return self.prior_sampler.inpaint_graph(
                to_mol,
                from_mol,
                mask=torch.ones(to_mol.seq_length, dtype=torch.bool),
                mode=self.graph_inpainting,
            )

        # Fragment-based construction with variable sizes
        fixed_indices = torch.where(mask)[0]
        variable_indices = torch.where(~mask)[0]

        N_fixed = len(fixed_indices)
        N_variable = len(variable_indices)

        # Handle empty fragments
        if N_fixed == 0:
            if is_local:
                raise ValueError(
                    f"Local mode '{mode}' has no atoms to generate. "
                    f"Mask has {mask.sum().item()} True values out of {len(mask)}."
                )
            else:
                # Global mode with no fixed atoms - generate entire molecule (de novo)
                print(
                    f"Warning: Falling back to de novo, couldn't extract conditional fragment for mode: {mode}!"
                )
                from_mol = self.prior_sampler.sample_molecule(
                    to_mol.seq_length, symmetrize=self.symmetrize
                )
                from_mol = from_mol._copy_with(
                    fragment_mask=torch.zeros(
                        to_mol.seq_length, dtype=torch.bool, device=to_mol.coords.device
                    ),
                    fragment_mode=mode,
                )
                return from_mol

        if N_variable == 0:
            # Everything is fixed, nothing to generate
            print(
                f"For mode '{mode}': All {len(mask)} atoms are marked as fixed, nothing to generate. Falling back to de novo."
            )
            from_mol = self.prior_sampler.sample_molecule(
                to_mol.seq_length, symmetrize=self.symmetrize
            )
            from_mol = from_mol._copy_with(
                fragment_mask=torch.zeros(
                    to_mol.seq_length, dtype=torch.bool, device=to_mol.coords.device
                ),
                fragment_mode=mode,
            )
            return from_mol

        # Extract fixed fragment
        fixed_fragment = self._extract_submolecule(to_mol, fixed_indices)

        # For interaction inpainting, keep original size (no size variation, not yet supported)
        if mode == "interaction" or not self.sample_mol_sizes:
            N_variable_sampled = N_variable
        else:
            # Calculate variable fragment size
            assert (
                self.fragment_size_variation is not None
                and self.fragment_size_variation > 0
            ), "Fragment size variation must be positive when sampling variable fragment sizes."
            N_variable_sampled = self._apply_size_variation(N_variable)

        # Sample variable fragment
        variable_fragment = self.prior_sampler.sample_molecule(
            N_variable_sampled, symmetrize=self.symmetrize
        )

        # Fuse fragments
        from_mol = self.prior_sampler.fuse_fragments(
            fixed_fragment, variable_fragment, mode
        )

        # For local modes, align variable fragment to reference
        if is_local:
            from_mol_mask = from_mol.fragment_mask
            from_mol = self._align_prior_to_fragment(
                from_mol, to_mol, mask, from_mol_mask
            )

        return from_mol

    def _determine_modes_and_extract_masks(
        self,
        to_mols: list[GeometricMol],
        rdkit_mols: list[Chem.Mol],
        interaction_mask: Optional[torch.Tensor],
    ) -> list[tuple[str, torch.Tensor, bool]]:
        """
        Determine inpainting mode and extract mask for each molecule.
        Returns list of (mode, mask, is_local) tuples.
        Mask: True = keep fixed, False = generate.
        """
        batch_size = len(to_mols)

        # Collect active modes
        active_global_modes = []
        active_local_modes = []

        if self.scaffold_inpainting:
            active_global_modes.append("scaffold")
        if self.func_group_inpainting:
            active_global_modes.append("func_group")
        if self.linker_inpainting:
            active_global_modes.append("linker")
        if self.core_inpainting:
            active_global_modes.append("core")
        if interaction_mask is not None:
            active_global_modes.append("interaction")
        if self.fragment_inpainting:
            active_local_modes.append("fragment")
        if self.substructure_inpainting and self.inference:
            active_local_modes.append("substructure")

        # Handle graph inpainting
        if self.graph_inpainting and not (active_global_modes or active_local_modes):
            # Only graph inpainting active
            if self.inference or not self.mixed_uncond_inpaint:
                return [
                    ("graph", torch.ones(mol.seq_length, dtype=torch.bool), False)
                    for mol in to_mols
                ]
            else:
                # Training with mixed_uncond: 50% graph, 50% de novo
                return [
                    (
                        "graph" if torch.rand(1).item() < 0.5 else "de_novo",
                        (
                            torch.ones(mol.seq_length, dtype=torch.bool)
                            if torch.rand(1).item() < 0.5
                            else torch.zeros(mol.seq_length, dtype=torch.bool)
                        ),
                        False,
                    )
                    for mol in to_mols
                ]

        all_active = active_local_modes + active_global_modes

        if not all_active:
            raise ValueError("No inpainting modes active")

        # Determine mode for each molecule
        results = []
        rand_vals = torch.rand(batch_size)
        for i in range(batch_size):
            # Select mode
            if self.inference and len(all_active) == 1:
                mode = all_active[0]
            elif self.inference:
                mode = random.choice(all_active)
            else:
                # Training: 25% de novo, 50% local, 25% global
                if rand_vals[i] < 0.25:
                    mode = "de_novo"
                elif (
                    rand_vals[i] < 0.75 and active_local_modes
                ) or not active_global_modes:
                    mode = random.choice(active_local_modes)
                else:
                    mode = random.choice(active_global_modes)

            # Extract mask
            is_local = mode in active_local_modes

            if mode == "de_novo":
                mask = torch.zeros(to_mols[i].seq_length, dtype=torch.bool)
            elif mode == "scaffold":
                mask = extract_scaffolds([rdkit_mols[i]])[0]
            elif mode == "func_group":
                mask = extract_func_groups([rdkit_mols[i]], includeHs=True)[0]
            elif mode == "linker":
                mask = extract_linkers([rdkit_mols[i]])[0]
            elif mode == "core":
                mask = extract_cores([rdkit_mols[i]])[0]
            elif mode == "fragment":
                if self.fragment_modes is not None:
                    fragment_mode = random.choice(self.fragment_modes)
                    masks = extract_fragments(
                        [rdkit_mols[i]],
                        maxCuts=self.max_fragment_cuts,
                        fragment_mode=fragment_mode,
                    )[0]
                    # masks is now a list of masks for multi-fragment mode
                    if isinstance(masks, list) and len(masks) > 1:
                        # Store as list for later processing
                        mask = masks
                    else:
                        # Fallback to single fragment if multi-fragment extraction failed
                        mask = extract_fragments(
                            [rdkit_mols[i]],
                            maxCuts=self.max_fragment_cuts,
                            fragment_mode="single",
                        )[0]
                else:
                    mask = extract_fragments(
                        [rdkit_mols[i]],
                        maxCuts=self.max_fragment_cuts,
                        fragment_mode="fragment",
                    )[0]
            elif mode == "substructure":
                mask = extract_substructure(
                    [rdkit_mols[i]],
                    substructure_query=self.substructure,
                    invert_mask=False,
                )[0]
            elif mode == "interaction":
                mask = interaction_mask[i]
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # For local modes, INVERT mask (generate selected part, keep rest)
            if is_local:
                if not isinstance(mask, list):
                    mask = ~mask

            results.append((mode, mask.bool(), is_local))

        return results

    def _extract_submolecule(
        self,
        mol: GeometricMol,
        atom_indices: torch.Tensor,
    ) -> GeometricMol:
        """
        Extract submolecule containing only specified atoms.
        Maintains full NxN bond connectivity structure.
        """
        if len(atom_indices) == 0:
            raise ValueError("Cannot extract submolecule from empty atom indices")

        N_sub = len(atom_indices)

        # Extract atom features
        coords = mol.coords[atom_indices]
        atomics = mol.atomics[atom_indices]
        charges = mol.charges[atom_indices]
        hybridization = (
            mol.hybridization[atom_indices] if mol.hybridization is not None else None
        )

        # Extract bonds by building full adjacency matrix for submolecule
        # Get original adjacency matrix (N_orig × N_orig)
        adjacency = mol.adjacency  # Shape: (N_orig, N_orig, n_bond_types)

        # Extract submolecule adjacency (N_sub × N_sub)
        # Index both dimensions with atom_indices
        sub_adjacency = adjacency[atom_indices][
            :, atom_indices
        ]  # (N_sub, N_sub, n_bond_types)

        # Create full bond_indices for submolecule (all pairs)
        new_bond_indices = torch.ones((N_sub, N_sub), device=mol.device).nonzero(
            as_tuple=False
        )

        # Extract corresponding bond types
        new_bond_types = sub_adjacency[new_bond_indices[:, 0], new_bond_indices[:, 1]]

        return GeometricMol(
            coords=coords,
            atomics=atomics,
            charges=charges,
            hybridization=hybridization,
            bond_indices=new_bond_indices,
            bond_types=new_bond_types,
            device=mol.device,
        )

    def _apply_size_variation(self, base_size: int) -> int:
        """Apply size variation for inference fragment generation."""
        if base_size == 0:
            raise ValueError("Cannot apply size variation to zero-sized fragment")

        variation = max(1, int(base_size * self.fragment_size_variation))
        min_size = max(1, base_size - variation)
        max_size = base_size + variation

        return random.randint(min_size, max_size)

    def _ot_map(
        self, from_mols: list[GeometricMol], to_mols: list[GeometricMol]
    ) -> list[GeometricMol]:
        """Permute the from_mols batch so that it forms an approximate mini-batch OT map with to_mols"""

        mol_matrix = []
        cost_matrix = []

        # Create matrix with to mols on outer axis and from mols on inner axis
        for to_mol in to_mols:
            best_from_mols = [
                self._match_mols(from_mol, to_mol) for from_mol in from_mols
            ]
            best_costs = [self._match_cost(mol, to_mol) for mol in best_from_mols]
            mol_matrix.append(list(best_from_mols))
            cost_matrix.append(list(best_costs))

        row_indices, col_indices = linear_sum_assignment(np.array(cost_matrix))
        best_from_mols = [mol_matrix[r][c] for r, c in zip(row_indices, col_indices)]
        return best_from_mols

    def _match_mols(
        self,
        from_mol: GeometricMol,
        to_mol: GeometricMol,
        mol_size: int,
    ) -> GeometricMol:
        """Permute the from_mol to best match the to_mol and return the permuted from_mol"""

        if mol_size > from_mol.seq_length:
            raise RuntimeError("from_mol must have at least as many atoms as to_mol.")

        # Find best permutation first, then best rotation
        # As done in Equivariant Flow Matching (https://arxiv.org/abs/2306.15030)

        # Keep the same number of atoms as the data mol in the noise mol
        mol_size = list(range(mol_size))
        from_mol = from_mol.permute(mol_size)

        if not self.permutation_alignment and not self.rotation_alignment:
            return from_mol

        assert (
            not self.sample_mol_sizes
        ), "Cannot use equivariant OT with sampled molecule sizes"

        if self.permutation_alignment:
            # Use linear sum assignment to find best permutation
            cost_matrix = smolF.inter_distances(
                to_mol.coords.cpu(), from_mol.coords.cpu(), sqrd=True
            )
            _, from_mol_indices = linear_sum_assignment(cost_matrix.numpy())
            from_mol = from_mol.permute(from_mol_indices.tolist())

        if self.rotation_alignment:
            padded_coords = smolF.pad_tensors(
                [from_mol.coords.cpu(), to_mol.coords.cpu()]
            )
            from_mol_coords = padded_coords[0].numpy()
            to_mol_coords = padded_coords[1].numpy()
            rotation, _ = Rotation.align_vectors(to_mol_coords, from_mol_coords)
            from_mol = from_mol.rotate(rotation)

        return from_mol

    def _match_cost(self, from_mol: GeometricMol, to_mol: GeometricMol) -> float:
        """Calculate MSE between mol coords as a match cost"""

        sqrd_dists = smolF.inter_distances(
            from_mol.coords.cpu(), to_mol.coords.cpu(), sqrd=True
        )
        mse = sqrd_dists.mean().item()
        return mse

    def _interpolate_mol(
        self,
        from_mol: GeometricMol,
        to_mol: GeometricMol,
        t_cont: float,
        t_disc: float,
    ) -> GeometricMol:
        """Interpolates mols which have already been sampled according to OT map, if required"""

        if from_mol.seq_length != to_mol.seq_length:
            raise RuntimeError(
                "Both molecules to be interpolated must have the same number of atoms."
            )

        # Interpolate coords and add gaussian noise
        if self.coord_interpolation == "cosine":
            alpha_t, sigma_t = get_cosine_scheduler_coefficients(
                torch.tensor(t_cont), nu=1
            )
            coords_mean = (from_mol.coords * sigma_t) + (to_mol.coords * alpha_t)
        else:
            coords_mean = (from_mol.coords * (1 - t_cont)) + (to_mol.coords * t_cont)
        coords = coords_mean + self.noise_scheduler.sample_noise(
            coords_mean, torch.tensor(t_cont)
        )

        # Interpolate atom types using unmasking or sampling
        if self.type_interpolation == "unmask":
            to_atomics = torch.argmax(to_mol.atomics, dim=-1)
            from_atomics = torch.argmax(from_mol.atomics, dim=-1)
            atom_mask = torch.rand(from_mol.seq_length) > t_disc
            to_atomics[atom_mask] = from_atomics[atom_mask]
            atomics = smolF.one_hot_encode_tensor(to_atomics, to_mol.atomics.size(-1))

        elif self.type_interpolation == "sample":
            atomics_mean = (from_mol.atomics * (1 - t_disc)) + (to_mol.atomics * t_disc)
            atomics_sample = torch.distributions.Categorical(atomics_mean).sample()
            atomics = smolF.one_hot_encode_tensor(
                atomics_sample, to_mol.atomics.size(-1)
            )
        else:
            raise ValueError(f"Unknown type interpolation: {self.type_interpolation}")

        # Interpolate charges using unmasking or sampling
        if self.type_interpolation == "unmask":
            to_charges = torch.argmax(to_mol.charges, dim=-1)
            from_charges = torch.argmax(from_mol.charges, dim=-1)
            charge_mask = torch.rand(from_mol.seq_length) > t_disc
            to_charges[charge_mask] = from_charges[charge_mask]
            charges = smolF.one_hot_encode_tensor(to_charges, to_mol.charges.size(-1))
        elif self.type_interpolation == "sample":
            charges_mean = (from_mol.charges * (1 - t_disc)) + (to_mol.charges * t_disc)
            charges_sample = torch.distributions.Categorical(charges_mean).sample()
            charges = smolF.one_hot_encode_tensor(
                charges_sample, to_mol.charges.size(-1)
            )

        if self.vocab_hybridization is not None and to_mol.hybridization is not None:
            # Interpolate hybridization types using unmasking or sampling
            if self.type_interpolation == "unmask":
                to_hybridization = torch.argmax(to_mol.hybridization, dim=-1)
                from_hybridization = torch.argmax(from_mol.hybridization, dim=-1)
                hybrid_mask = torch.rand(from_mol.seq_length) > t_disc
                to_hybridization[hybrid_mask] = from_hybridization[hybrid_mask]
                hybridization = smolF.one_hot_encode_tensor(
                    to_hybridization, to_mol.hybridization.size(-1)
                )
            elif self.type_interpolation == "sample":
                hybridization_mean = (
                    from_mol.hybridization * (1 - t_disc)
                    + to_mol.hybridization * t_disc
                )
                hybridization_sample = torch.distributions.Categorical(
                    hybridization_mean
                ).sample()
                hybridization = smolF.one_hot_encode_tensor(
                    hybridization_sample, to_mol.hybridization.size(-1)
                )
        else:
            hybridization = None

        # Interpolate bonds using unmasking or sampling
        if self.bond_interpolation == "unmask":
            to_adj = torch.argmax(to_mol.adjacency, dim=-1)
            from_adj = torch.argmax(from_mol.adjacency, dim=-1)
            bond_mask = torch.rand_like(from_adj.float()) > t_disc
            new_adj = to_adj.clone()
            new_adj[bond_mask] = from_adj[bond_mask]

        elif self.bond_interpolation == "sample":
            adj_mean = (from_mol.adjacency * (1 - t_disc)) + (to_mol.adjacency * t_disc)
            new_adj = torch.distributions.Categorical(adj_mean).sample()

        if self.symmetrize:
            new_adj = smolF.symmetrize_bonds(new_adj, is_one_hot=False)
        new_adj = smolF.one_hot_encode_tensor(new_adj, to_mol.adjacency.size(-1))
        bond_indices = torch.ones((from_mol.seq_length, from_mol.seq_length)).nonzero()
        bond_types = new_adj[bond_indices[:, 0], bond_indices[:, 1]]

        interp_mol = GeometricMol(
            coords,
            atomics,
            charges=charges,
            hybridization=hybridization,
            bond_indices=bond_indices,
            bond_types=bond_types,
        )
        if from_mol.fragment_mask is not None:
            interp_mol.fragment_mask = from_mol.fragment_mask
        if from_mol.fragment_mode is not None:
            interp_mol.fragment_mode = from_mol.fragment_mode

        return interp_mol


class ComplexInterpolant(GeometricInterpolant):
    """Provides apo-holo and noise to ligand interpolation by wrapping a ligand interpolant"""

    def __init__(
        self,
        prior_sampler: GeometricNoiseSampler,
        ligand_coord_interpolation="linear",
        ligand_type_interpolation="unmask",
        ligand_bond_interpolation="unmask",
        ligand_coord_noise_std: float = 0.0,
        ligand_coord_noise_schedule: str = "standard",
        ligand_time_alpha: float = 1.0,
        ligand_time_beta: float = 1.0,
        ligand_fixed_time: Optional[float] = None,
        split_continuous_discrete_time: bool = False,
        pocket_time_alpha: float = 1.0,
        pocket_time_beta: float = 1.0,
        pocket_fixed_time: Optional[float] = None,
        interaction_time_alpha: float = 1.0,
        interaction_time_beta: float = 1.0,
        interaction_fixed_time: Optional[float] = None,
        pocket_coord_noise_std: float = 0.0,
        pocket_noise: str = "fix",
        separate_pocket_interpolation: bool = False,
        separate_interaction_interpolation: bool = False,
        n_interaction_types: Optional[int] = None,
        flow_interactions: bool = False,
        dataset: str = "plinder",
        sample_mol_sizes: Optional[bool] = False,
        interaction_inpainting: bool = False,
        scaffold_inpainting: bool = False,
        func_group_inpainting: bool = False,
        linker_inpainting: bool = False,
        core_inpainting: bool = False,
        max_fragment_cuts: int = 3,
        fragment_inpainting: bool = False,
        substructure_inpainting: bool = False,
        substructure: Optional[str] = None,
        graph_inpainting: str = None,
        mixed_uncond_inpaint: bool = False,
        mixed_uniform_beta_time: bool = False,
        inference: bool = False,
        vocab: Optional[dict] = None,
        vocab_charges: Optional[dict] = None,
        vocab_hybridization: Optional[dict] = None,
        batch_ot: bool = False,
        rotation_alignment: bool = False,
        permutation_alignment: bool = False,
    ):

        super().__init__(
            prior_sampler,
            coord_interpolation=ligand_coord_interpolation,
            type_interpolation=ligand_type_interpolation,
            bond_interpolation=ligand_bond_interpolation,
            coord_noise_std=ligand_coord_noise_std,
            coord_noise_schedule=ligand_coord_noise_schedule,
            time_alpha=ligand_time_alpha,
            time_beta=ligand_time_beta,
            fixed_time=ligand_fixed_time,
            split_continuous_discrete_time=split_continuous_discrete_time,
            mixed_uniform_beta_time=mixed_uniform_beta_time,
            scaffold_inpainting=scaffold_inpainting,
            func_group_inpainting=func_group_inpainting,
            linker_inpainting=linker_inpainting,
            core_inpainting=core_inpainting,
            fragment_inpainting=fragment_inpainting,
            max_fragment_cuts=max_fragment_cuts,
            substructure_inpainting=substructure_inpainting,
            substructure=substructure,
            graph_inpainting=graph_inpainting,
            mixed_uncond_inpaint=mixed_uncond_inpaint,
            inference=inference,
            sample_mol_sizes=sample_mol_sizes,
            dataset=dataset,
            vocab=vocab,
            vocab_charges=vocab_charges,
            vocab_hybridization=vocab_hybridization,
            batch_ot=batch_ot,
            rotation_alignment=rotation_alignment,
            permutation_alignment=permutation_alignment,
        )
        if sample_mol_sizes:
            print("Running inference with sampled molecule sizes!")

        self.pocket_noise = pocket_noise
        self.separate_pocket_interpolation = separate_pocket_interpolation
        self.separate_interaction_interpolation = separate_interaction_interpolation
        self.pocket_coord_noise_std = pocket_coord_noise_std
        self.pocket_time_alpha = pocket_time_alpha
        self.pocket_time_beta = pocket_time_beta
        self.pocket_fixed_time = pocket_fixed_time
        self.interaction_time_alpha = interaction_time_alpha
        self.interaction_time_beta = interaction_time_beta
        self.interaction_fixed_time = interaction_fixed_time
        self.flow_interactions = flow_interactions
        self.n_interaction_types = n_interaction_types
        self.interaction_inpainting = interaction_inpainting
        self.sample_mol_sizes = sample_mol_sizes

        self.inference = inference

        self.pocket_time_dist = torch.distributions.Beta(
            pocket_time_alpha, pocket_time_beta
        )
        self.interaction_time_dist = torch.distributions.Beta(
            interaction_time_alpha, interaction_time_beta
        )

    @property
    def hparams(self):
        ligand_hparams = {f"ligand-{k}": v for k, v in super().hparams.items()}
        hparams = {
            "pocket-noise": self.pocket_noise,
            "separate-pocket-interpolation": self.separate_pocket_interpolation,
            "pocket-coord-noise-std": self.pocket_coord_noise_std,
            **ligand_hparams,
        }
        hparams["separate-interaction-interpolation"] = (
            self.separate_interaction_interpolation
        )
        hparams["n-interaction-types"] = self.n_interaction_types
        hparams["flow-interactions"] = self.flow_interactions
        hparams["interaction-inpainting"] = self.interaction_inpainting

        if self.separate_pocket_interpolation:
            hparams["pocket-time-alpha"] = self.pocket_time_alpha
            hparams["pocket-time-beta"] = self.pocket_time_beta
            if self.pocket_fixed_time is not None:
                hparams["pocket-fixed-interpolation-time"] = self.fixed_time

        if self.separate_interaction_interpolation:
            hparams["interaction-time-alpha"] = self.interaction_time_alpha
            hparams["interaction-time-beta"] = self.interaction_time_beta
            if self.interaction_fixed_time is not None:
                hparams["interaction-fixed-interpolation-time"] = self.fixed_time

        return hparams

    # NOTE the apo and holo pairs must come with 1-1 match on atoms and bonds, except coordinate values
    # NOTE this also assumes that each system has already been shifted so that the apo pocket has a zero com
    def interpolate(self, to_mols: list[PocketComplex]) -> _ComplexInterpT:
        batch_size = len(to_mols)

        # Interpolate ligands
        interaction_mask = None
        if self.interaction_inpainting:
            assert (
                not self.sample_mol_sizes
            ), "Inpainting currently not supported with sampled mol sizes"
            if self.interaction_inpainting:
                interaction_mask = [
                    (
                        mol.interactions[:, :, 1:].sum(dim=(0, 2)) > 0
                        if mol.interactions is not None
                        else None
                    )
                    for mol in to_mols
                ]

        ligands = [system.ligand for system in to_mols]
        (
            from_ligands,
            to_ligands,
            interp_ligands,
            ligand_times,
        ) = super().interpolate(ligands, interaction_mask=interaction_mask)

        # Retrieve inpaint masks
        inpaint_mask = [
            (
                mol.fragment_mask
                if mol.fragment_mask is not None
                else torch.tensor([0] * len(mol)).bool()
            )
            for mol in from_ligands
        ]
        inpaint_mode = [
            mol.fragment_mode if mol.fragment_mode is not None else ""
            for mol in from_ligands
        ]

        # Save metadata and center-of-mass for each system
        metadata = [system.metadata for system in to_mols]
        com = [system.com for system in to_mols]

        # Interpolate interactions
        to_interactions = (
            [system.interactions for system in to_mols]
            if self.n_interaction_types is not None
            else []
        )
        if self.flow_interactions:
            assert (
                self.n_interaction_types is not None
            ), "Flowing interactions requires n_interaction_types to be specified"
            if self.separate_interaction_interpolation:
                if self.interaction_fixed_time is not None:
                    interaction_times = [
                        torch.tensor(self.interaction_fixed_time)
                    ] * batch_size
                else:
                    interaction_times = self.interaction_time_dist.sample((batch_size,))
                    interaction_times = interaction_times.tolist()
            else:
                interaction_times = ligand_times[:, 0]
            from_interactions = [
                self._noise_interactions(
                    n_pocket_atoms=to_mols[i].holo.seq_length,
                    n_ligand_atoms=from_ligands[i].seq_length,
                    n_interactions=self.n_interaction_types,
                )
                for i in range(len(to_mols))
            ]
            if self.inference:
                interp_interactions = from_interactions
            else:
                interp_interactions = [
                    self._interpolate_interactions(from_interaction, to_interaction, t)
                    for from_interaction, to_interaction, t in zip(
                        from_interactions, to_interactions, interaction_times
                    )
                ]
        else:
            interaction_times = torch.tensor([0.0] * batch_size)
            from_interactions = interp_interactions = to_interactions

        # Interpolate pockets
        if self.separate_pocket_interpolation:
            if self.pocket_fixed_time is not None:
                pocket_times = torch.tensor([self.pocket_fixed_time] * batch_size)
            else:
                pocket_times = self.pocket_time_dist.sample((batch_size,))
        else:
            pocket_times = ligand_times[:, 0]

        holo_pockets = [system.holo for system in to_mols]
        apo_pockets = (
            holo_pockets
            if self.pocket_noise == "fix"
            else [system.apo for system in to_mols]
        )
        interp_pockets = [
            self._interpolate_pocket(apo_pocket, holo_pocket, t)
            for apo_pocket, holo_pocket, t in zip(
                apo_pockets, holo_pockets, pocket_times
            )
        ]

        # Combine everything back into PocketComplex objects
        from_systems = [
            PocketComplex(
                apo=apo_pocket,
                ligand=ligand,
                interactions=interaction,
                metadata=meta,
                fragment_mask=_mask,
                fragment_mode=_mode,
                com=_com,
            )
            for apo_pocket, ligand, interaction, meta, _mask, _mode, _com in zip_longest(
                apo_pockets,
                from_ligands,
                from_interactions,
                metadata,
                inpaint_mask,
                inpaint_mode,
                com,
                fillvalue=None,
            )
        ]
        to_systems = [
            PocketComplex(
                holo=holo_pocket,
                apo=apo_pocket,
                ligand=ligand,
                interactions=interaction,
                metadata=meta,
                fragment_mask=_mask,
                fragment_mode=_mode,
                com=_com,
            )
            for holo_pocket, apo_pocket, ligand, interaction, meta, _mask, _mode, _com in zip_longest(
                holo_pockets,
                apo_pockets,
                to_ligands,
                to_interactions,
                metadata,
                inpaint_mask,
                inpaint_mode,
                com,
                fillvalue=None,
            )
        ]
        interp_systems = [
            PocketComplex(
                apo=interp_pocket,
                ligand=ligand,
                interactions=interaction,
                metadata=meta,
                fragment_mask=_mask,
                fragment_mode=_mode,
            )
            for interp_pocket, ligand, meta, interaction, _mask, _mode in zip_longest(
                interp_pockets,
                interp_ligands,
                metadata,
                interp_interactions,
                inpaint_mask,
                inpaint_mode,
                fillvalue=None,
            )
        ]

        # Save times for ligand, pocket
        times = torch.stack(
            [
                ligand_times[:, 0],
                ligand_times[:, 1],
                pocket_times,
            ],
            dim=1,
        )

        return from_systems, to_systems, interp_systems, times

    def _interpolate_pocket(
        self, apo_pocket: ProteinPocket, holo_pocket: ProteinPocket, t: torch.Tensor
    ) -> ProteinPocket:
        assert len(apo_pocket) == len(
            holo_pocket
        ), "apo and holo pockets must have the same number of atoms"

        if self.pocket_noise == "fix" or self.pocket_noise == "apo":
            # Interpolate coords and add gaussian noise
            # Apo and holo should come pre-aligned so no need for any alignment here
            coords_mean = (apo_pocket.mol.coords * (1 - t)) + (
                holo_pocket.mol.coords * t
            )
            coords_noise = torch.randn_like(coords_mean) * self.pocket_coord_noise_std
            coords = coords_mean + coords_noise
        elif self.pocket_noise == "random":
            # In the random case, we use holo coordinates as apo and add gaussian noise to it with less noise when t=1
            coords_noise = torch.randn_like(holo_pocket.mol.coords) * (t * (1 - t))
            coords = holo_pocket.mol.coords + coords_noise
        else:
            raise ValueError(f"Unknown pocket noise type: {self.pocket_noise}")

        # NOTE Assumes apo and holo have a 1-1 match on everything except coordinates
        interp_pocket_mol = holo_pocket.mol._copy_with(coords=coords)
        interp_pocket = holo_pocket._copy_with(mol=interp_pocket_mol)

        return interp_pocket

    def _interpolate_interactions(
        self,
        from_interactions: torch.Tensor,
        to_interactions: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        # Interpolate interactions

        num_interactions = to_interactions.size(-1)
        to_inter = torch.argmax(to_interactions, dim=-1)
        from_inter = torch.argmax(from_interactions, dim=-1)
        interaction_mask = torch.rand_like(from_inter.float()) > t
        to_inter[interaction_mask] = from_inter[interaction_mask]
        interp_interactions = smolF.one_hot_encode_tensor(to_inter, num_interactions)
        return interp_interactions

    def _noise_interactions(
        self, n_pocket_atoms: int, n_ligand_atoms: int, n_interactions: int
    ):
        num_pairs = n_pocket_atoms * n_ligand_atoms
        prior_flat = torch.zeros((num_pairs, n_interactions))
        prior_interactions = torch.randint(0, n_interactions, size=(num_pairs,))
        prior_flat[torch.arange(num_pairs), prior_interactions] = 1.0
        from_interaction = prior_flat.reshape(
            n_pocket_atoms, n_ligand_atoms, n_interactions
        )

        return from_interaction

    def _interaction_ot_map(self, from_systems, from_ligands, from_interactions):
        """
        Permute the from_ligands batch so that it forms an approximate mini-batch OT map with from_interactions:
        Meaning, align the from_ligands batch with the from_systems batch, such that the ligand atoms that are
        involved in the interactions between pocket and ligand are close to the pocket atoms that are involved
        in the interactions between pocket and ligand given by the from_interactions batch.
        Hence, permute ligand atoms based on the distance to the pocket atoms that show interactions with the ligand atoms.
        """
        raise NotImplementedError("Interaction OT map not implemented yet.")

        def kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
            """
            Compute the optimal rotation matrix that aligns Q onto P
            using the Kabsch algorithm (no reflection allowed).
            """
            P_mean = P.mean(axis=0)
            Q_mean = Q.mean(axis=0)
            P_centered = P - P_mean
            Q_centered = Q - Q_mean

            # Covariance matrix
            C = np.dot(Q_centered.T, P_centered)  # (3xK) dot (Kx3) -> (3x3)

            # SVD
            V, S, Wt = np.linalg.svd(C)
            d = np.linalg.det(np.dot(V, Wt))
            if d < 0.0:
                V[:, -1] = -V[:, -1]
            R = np.dot(V, Wt)
            return R

        def align_points_kabsch(
            P: np.ndarray, Q: np.ndarray, all_Q: np.ndarray
        ) -> np.ndarray:
            """
            Given matched points P, Q of shape (K, 3), compute the rigid transform
            (rotation + translation) that best aligns Q to P. Then apply that transform
            to all_Q (shape (N, 3)), returning the aligned points of same shape (N, 3).
            """
            P_mean = P.mean(axis=0)
            Q_mean = Q.mean(axis=0)
            R = kabsch_rotation(P, Q)

            aligned_all_Q = (all_Q - Q_mean) @ R + P_mean
            return aligned_all_Q

        def optimal_ligand_alignment(
            pocket_coords: np.ndarray,
            ligand_coords: np.ndarray,
            interaction_matrix: np.ndarray,
            large_penalty: float = 1e6,
        ) -> np.ndarray:
            """
            Find a partial matching between pocket atoms (N_p x 3) and ligand atoms (N_l x 3)
            by minimizing distance while respecting interactions, then compute
            the rigid transform that best aligns the entire ligand to the pocket.
            """
            # pairwise distance matrix (N_p x N_l)
            diffs = pocket_coords[:, None, :] - ligand_coords[None, :, :]
            distance_matrix = np.linalg.norm(diffs, axis=2)

            # interaction mask
            feasible_mask = np.any(interaction_matrix, axis=2)  # (N_p, N_l)
            cost_matrix = np.where(feasible_mask, distance_matrix, large_penalty)

            # linear assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            pocket_matched = pocket_coords[row_ind]  # shape (K, 3)
            ligand_matched = ligand_coords[col_ind]  # shape (K, 3)

            # transform
            aligned_ligand_coords = align_points_kabsch(
                P=pocket_matched, Q=ligand_matched, all_Q=ligand_coords
            )
            return aligned_ligand_coords

        # aligned_ligand = optimal_ligand_alignment(
        #     pocket_coords=pocket_coords,
        #     ligand_coords=ligand_coords,
        #     interaction_matrix=to_interaction,
        #     large_penalty=1e6
        # )

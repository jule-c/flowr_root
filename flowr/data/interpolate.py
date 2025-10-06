import random
from abc import ABC, abstractmethod
from itertools import chain, zip_longest
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


def extract_fragments(to_mols: list[Chem.Mol], maxCuts: int = 3):
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
        frags = [clean_fragment(frag) for frag in chain(*frags) if frag]
        frags = [
            frag
            for frag_tuple in frags
            for frag in frag_tuple
            if frag.GetNumAtoms() > 1
        ]
        substructure_ids = [mol.GetSubstructMatches(frag)[0] for frag in frags]
        # Randomly select a fragment
        findices = []
        if substructure_ids:
            frag = random.choice(substructure_ids)
            findices.extend(frag)

            if findices:
                mask[torch.tensor(findices)] = 1
        return mask

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

    if isinstance(substructure_query, str):
        mask = [substructure_per_mol(mol, substructure_query) for mol in to_mols]
    elif isinstance(substructure_query, list):
        mask = [substructure_per_mol_list(mol, substructure_query) for mol in to_mols]
    else:
        raise ValueError("substructure_query must be a string or a list of atom indices.")

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
    dataset,
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

    if by_mean_and_std:
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

    def inpaint_molecule(
        self,
        to_mol: GeometricMol,
        from_mol: GeometricMol,
        mask: torch.Tensor,
        harmonic_prior: Optional[bool] = False,
    ) -> GeometricMol:

        if mask is None or not mask.any():
            # if fragment_mask is already specified by graph_inpainting, return from_mol
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
        inp_coords[mask, :] = to_mol.coords[mask, :]
        inp_atomics[mask, :] = to_mol.atomics[mask, :]
        inp_charges[mask, :] = to_mol.charges[mask, :]
        if self.n_hybridization_types is not None:
            inp_hybridization = from_mol.hybridization.clone()
            inp_hybridization[mask, :] = to_mol.hybridization[mask, :]
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

        ## only update bonds if both atoms are inpainted and if they are bonded
        fixed_mask_matrix = (mask.unsqueeze(0) & mask.unsqueeze(1)) & (to_adj != 0)
        new_adj = from_adj.clone()
        new_adj[fixed_mask_matrix] = to_adj[fixed_mask_matrix]
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
            fragment_mask=mask,
        )

    def inpaint_graph(
        self,
        to_mol: GeometricMol,
        from_mol: GeometricMol,
        mask: torch.Tensor,
    ) -> GeometricMol:

        # if all mask is False, return from_mol with fragment_mask
        if mask is None or not mask.any():
            mask = torch.zeros(
                from_mol.seq_length, dtype=torch.bool, device=from_mol.coords.device
            )
            from_mol.fragment_mask = mask
            return from_mol
        else:
            # create atom-wise mask
            mask = torch.ones(
                from_mol.seq_length, dtype=torch.bool, device=from_mol.coords.device
            )

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

        # if all mask is False, return from_mol with fragment_mask
        if mask is None or not mask.any():
            mask = torch.zeros(
                from_mol.seq_length, dtype=torch.bool, device=from_mol.coords.device
            )
            from_mol.fragment_mask = mask
            return from_mol
        else:
            # create atom-wise mask
            mask = torch.ones(
                from_mol.seq_length, dtype=torch.bool, device=from_mol.coords.device
            )

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

        # if all mask is False, return from_mol with fragment_mask
        if mask is None or not mask.any():
            mask = torch.zeros(
                from_mol.seq_length, dtype=torch.bool, device=from_mol.coords.device
            )
            from_mol.fragment_mask = mask
            return from_mol
        else:
            # create atom-wise mask
            mask = torch.ones(
                from_mol.seq_length, dtype=torch.bool, device=from_mol.coords.device
            )

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

    def sample_molecule(self, n_atoms: int) -> GeometricMol:

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
            bond_types = smolF.one_hot_encode_tensor(bond_types, self.n_bond_types)

        elif self.bond_noise == "prior-sample":
            bond_types = torch.multinomial(
                self.bond_types_distribution, n_bonds, replacement=True
            )
            bond_types = smolF.one_hot_encode_tensor(bond_types, self.n_bond_types)

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

        # Add cache for extracted inpainted masks
        self._mask_cache = {}

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

    def _get_cache_key(self, mol: GeometricMol) -> str:
        """Generate a unique cache key for a molecule based on its 3D structure"""
        try:
            import hashlib

            # Get coordinates and round to reasonable precision (e.g., 3 decimal places)
            # This handles small numerical differences while preserving conformational differences
            coords = mol.coords.numpy()
            coords_rounded = np.round(coords, decimals=3)

            # Get atom types for context (same conformation of different molecules should be different)
            atomics = torch.argmax(mol.atomics, dim=-1).numpy()

            # Create a deterministic string representation
            # Include both atom types and coordinates
            cache_components = [
                f"atoms:{atomics.tolist()}",
                f"coords:{coords_rounded.flatten().tolist()}",
                f"seq_len:{mol.seq_length}",
            ]

            # Use SHA256 hash for uniqueness
            cache_str = "|".join(cache_components)
            return hashlib.sha256(cache_str.encode()).hexdigest()[:16]

        except Exception as e:
            # Fallback to object id if anything fails
            print(f"Warning: Cache key generation failed: {e}")
            return f"fallback_{id(mol)}"

    def _extract_masks_cached(
        self, to_mols: list[GeometricMol], interaction_mask: torch.Tensor = None
    ):
        """Extract masks with caching to avoid recomputation"""
        all_masks = []

        for mol in to_mols:
            cache_key = self._get_cache_key(mol)

            # Check if we have cached results for this molecule
            if cache_key in self._mask_cache:
                cached_data = self._mask_cache[cache_key]
                all_masks.append(cached_data["masks"])
                continue

            # Convert to RDKit molecule (only if not cached)
            rdkit_mol = mol.to_rdkit(
                vocab=self.vocab,
                vocab_charges=self.vocab_charges,
                vocab_hybridization=self.vocab_hybridization,
            )

            # Extract all possible masks for this molecule
            mol_masks = {}

            if self.scaffold_inpainting:
                mol_masks["scaffold"] = extract_scaffolds([rdkit_mol])[0]
            if self.func_group_inpainting:
                mol_masks["func_group"] = extract_func_groups(
                    [rdkit_mol], includeHs=True
                )[0]
            if self.substructure_inpainting:
                mol_masks["substructure"] = extract_substructure(
                    [rdkit_mol], substructure_query=self.substructure
                )[0]
            if self.linker_inpainting:
                mol_masks["linker"] = extract_linkers([rdkit_mol])[0]
            if self.core_inpainting:
                mol_masks["core"] = extract_cores([rdkit_mol])[0]
            if self.fragment_inpainting:
                mol_masks["fragment"] = extract_fragments(
                    [rdkit_mol], maxCuts=self.max_fragment_cuts
                )[0]

            # Cache the results
            self._mask_cache[cache_key] = {
                "masks": mol_masks,
            }
            all_masks.append(mol_masks)

        # Now select which mask to use for each molecule
        inpaint_mask = []
        for i, mol_masks in enumerate(all_masks):
            available_masks = []

            # Add interaction mask if provided
            if interaction_mask is not None:
                available_masks.append(interaction_mask[i])

            # Add cached structural masks
            for mask_type, mask in mol_masks.items():
                available_masks.append(mask)

            # Randomly select one mask (or create empty mask if none available)
            if available_masks:
                selected_mask = random.choice(available_masks)
            else:
                selected_mask = torch.zeros(to_mols[i].seq_length, dtype=torch.bool)

            inpaint_mask.append(selected_mask)

        return inpaint_mask

    def interpolate(
        self, to_mols: list[GeometricMol], interaction_mask: torch.Tensor = None
    ) -> _GeometricInterpT:
        batch_size = len(to_mols)
        if not self.sample_mol_sizes:
            mol_sizes = [mol.seq_length for mol in to_mols]
        else:
            mol_sizes = [
                sample_mol_sizes(mol.seq_length, self.dataset) for mol in to_mols
            ]
        num_atoms = max(mol_sizes)
        # Within match_mols either just truncate noise to match size of data molecule
        # Or also permute and rotate the noise to best match data molecule
        from_mols = [self.prior_sampler.sample_molecule(num_atoms) for mol in to_mols]
        from_mols = [
            self._match_mols(
                from_mol,
                to_mol,
                mol_size=mol_size,
            )
            for from_mol, to_mol, mol_size in zip(from_mols, to_mols, mol_sizes)
        ]

        # Interpolate ligands
        inpaint_mask = []
        if (
            interaction_mask is not None
            or self.inpainting_mode
            or self.graph_inpainting
        ):
            assert (
                not self.sample_mol_sizes
            ), "Inpainting currently not supported with sampled mol sizes"

            mols = [
                mol.to_rdkit(
                    vocab=self.vocab,
                    vocab_charges=self.vocab_charges,
                    vocab_hybridization=self.vocab_hybridization,
                )
                for mol in to_mols
            ]

            if self.inpainting_mode:
                if self.scaffold_inpainting:
                    scaffold_mask = extract_scaffolds(mols)
                if self.func_group_inpainting:
                    func_group_mask = extract_func_groups(mols, includeHs=True)
                if self.substructure_inpainting:
                    assert (
                        self.substructure is not None
                    ), "Substructure query must be provided"
                    assert isinstance(self.substructure, str) or isinstance(
                        self.substructure, list
                    ), "Substructure must be a string or a list of indices"
                    custom_mask = extract_substructure(
                        mols,
                        substructure_query=self.substructure, invert_mask=True
                    )
                if self.linker_inpainting:
                    linker_mask = extract_linkers(mols)
                if self.core_inpainting:
                    core_mask = extract_cores(mols)
                if self.fragment_inpainting:
                    fragment_mask = extract_fragments(
                        mols, maxCuts=self.max_fragment_cuts
                    )

                # Randomly select one of the mask for both training and inference if more than one mode is active
                # NOTE: normally, at inference time, only one mode should be active, but this allows for more flexibility
                masks = []
                if interaction_mask is not None:
                    masks.append(interaction_mask)
                if self.scaffold_inpainting:
                    masks.append(scaffold_mask)
                if self.func_group_inpainting:
                    masks.append(func_group_mask)
                if self.substructure_inpainting:
                    masks.append(custom_mask)
                if self.linker_inpainting:
                    masks.append(linker_mask)
                if self.core_inpainting:
                    masks.append(core_mask)
                if self.fragment_inpainting:
                    masks.append(fragment_mask)
                inpaint_mask = random.choice(masks)

            if self.graph_inpainting is not None:
                if not self.inference and self.mixed_uncond_inpaint:
                    if self.inpainting_mode or interaction_mask is not None:
                        # Both modes are active: 0.25 uncond, 0.25 graph, 0.5 inpaint
                        rand_vals = torch.rand(batch_size)
                        graph_mask = (rand_vals >= 0.25) & (rand_vals < 0.5)
                        zero_out_mask = (
                            rand_vals < 0.5
                        )  # Zero out for uncond (25%) + graph-only (25%)
                        inpaint_mask = [
                            torch.zeros_like(mask) if zero_out_mask[i] else mask
                            for i, mask in enumerate(inpaint_mask)
                        ]
                    else:
                        # Only graph mode is active: 0.5 uncond, 0.5 graph
                        graph_mask = torch.rand(batch_size) < 0.5
                else:
                    graph_mask = torch.ones(batch_size, dtype=torch.bool)
                    # NOTE: if at inference several conditional modes are specified and graph_inpainting is true as well,
                    # which should only be the case at training time + validation, use graph_inpainting only as default!
                    if self.inpainting_mode or interaction_mask is not None:
                        inpaint_mask = []

                # Apply graph inpainting...
                if self.graph_inpainting == "harmonic":
                    from_mols = [
                        (
                            self.prior_sampler.inpaint_graph_with_harmonic(
                                to_mol, from_mol, mask
                            )
                        )
                        for to_mol, from_mol, mask in zip(
                            to_mols, from_mols, graph_mask
                        )
                    ]
                elif self.graph_inpainting == "conformer":
                    from_mols = [
                        (
                            self.prior_sampler.inpaint_graph_with_conformer(
                                to_mol, from_mol, mask
                            )
                        )
                        for to_mol, from_mol, mask in zip(
                            to_mols, from_mols, graph_mask
                        )
                    ]
                elif self.graph_inpainting == "random":
                    from_mols = [
                        (self.prior_sampler.inpaint_graph(to_mol, from_mol, mask))
                        for to_mol, from_mol, mask in zip(
                            to_mols, from_mols, graph_mask
                        )
                    ]
                else:
                    raise ValueError(
                        "Graph inpainting is not supported with the current prior. Set prior to be either 'harmonic', 'conformer' or 'random'."
                    )
            else:
                # No graph inpainting, handle only inpaint modes if they exist
                if not self.inference and self.mixed_uncond_inpaint:
                    # Only inpainting mode is active: 0.5 uncond, 0.5 inpaint
                    rand_vals = torch.rand(batch_size)
                    uncond_batch_mask = rand_vals < 0.5
                    inpaint_mask = [
                        torch.zeros_like(mask) if uncond_batch_mask[i] else mask
                        for i, mask in enumerate(inpaint_mask)
                    ]

        # Apply the conditional masks if inpainting
        if inpaint_mask and len(inpaint_mask) > 0:
            from_mols = [
                (
                    self.prior_sampler.inpaint_molecule(
                        to_mol, from_mol, mask, harmonic_prior=False
                    )
                )
                for to_mol, from_mol, mask in zip(to_mols, from_mols, inpaint_mask)
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
            if not self.sample_mol_sizes
            else from_mols
        )

        times = torch.stack([times_cont, times_disc], dim=1)
        return (
            from_mols,
            to_mols,
            interp_mols,
            times,
        )

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
        to_atomics = torch.argmax(to_mol.atomics, dim=-1)
        from_atomics = torch.argmax(from_mol.atomics, dim=-1)
        if self.type_interpolation == "unmask":
            atom_mask = torch.rand(from_mol.seq_length) > t_disc
            to_atomics[atom_mask] = from_atomics[atom_mask]
            atomics = smolF.one_hot_encode_tensor(to_atomics, to_mol.atomics.size(-1))

        elif self.type_interpolation == "sample":
            atomics_mean = (from_atomics * (1 - t_disc)) + (to_atomics * t_disc)
            atomics_sample = torch.distributions.Categorical(atomics_mean).sample()
            atomics = smolF.one_hot_encode_tensor(
                atomics_sample, to_mol.atomics.size(-1)
            )
        else:
            raise ValueError(f"Unknown type interpolation: {self.type_interpolation}")

        # Interpolate charges using unmasking or sampling
        to_charges = torch.argmax(to_mol.charges, dim=-1)
        from_charges = torch.argmax(from_mol.charges, dim=-1)
        if self.type_interpolation == "unmask":
            charge_mask = torch.rand(from_mol.seq_length) > t_disc
            to_charges[charge_mask] = from_charges[charge_mask]
            charges = smolF.one_hot_encode_tensor(to_charges, to_mol.charges.size(-1))
        elif self.type_interpolation == "sample":
            charges_mean = (from_charges * (1 - t_disc)) + (to_charges * t_disc)
            charges_sample = torch.distributions.Categorical(charges_mean).sample()
            charges = smolF.one_hot_encode_tensor(
                charges_sample, to_mol.charges.size(-1)
            )

        if self.vocab_hybridization is not None and to_mol.hybridization is not None:
            # Interpolate hybridization types using unmasking or sampling
            to_hybridization = torch.argmax(to_mol.hybridization, dim=-1)
            from_hybridization = torch.argmax(from_mol.hybridization, dim=-1)
            if self.type_interpolation == "unmask":
                hybrid_mask = torch.rand(from_mol.seq_length) > t_disc
                to_hybridization[hybrid_mask] = from_hybridization[hybrid_mask]
                hybridization = smolF.one_hot_encode_tensor(
                    to_hybridization, to_mol.hybridization.size(-1)
                )
            elif self.type_interpolation == "sample":
                hybridization_mean = (from_hybridization * (1 - t_disc)) + (
                    to_hybridization * t_disc
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
        to_adj = torch.argmax(to_mol.adjacency, dim=-1)
        from_adj = torch.argmax(from_mol.adjacency, dim=-1)
        if self.bond_interpolation == "unmask":
            bond_mask = torch.rand_like(from_adj.float()) > t_disc
            to_adj[bond_mask] = from_adj[bond_mask]
            interp_adj = smolF.one_hot_encode_tensor(to_adj, to_mol.adjacency.size(-1))

        elif self.bond_interpolation == "sample":
            adj_mean = (from_adj * (1 - t_disc)) + (to_adj * t_disc)
            adj_sample = torch.distributions.Categorical(adj_mean).sample()
            interp_adj = smolF.one_hot_encode_tensor(
                adj_sample, to_mol.adjacency.size(-1)
            )

        bond_indices = torch.ones((from_mol.seq_length, from_mol.seq_length)).nonzero()
        bond_types = interp_adj[bond_indices[:, 0], bond_indices[:, 1]]

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
                    mol.interactions[:, :, 1:].sum(dim=(0, 2)) > 0 for mol in to_mols
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
                com=_com,
            )
            for apo_pocket, ligand, interaction, meta, _mask, _com in zip_longest(
                apo_pockets,
                from_ligands,
                from_interactions,
                metadata,
                inpaint_mask,
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
                com=_com,
            )
            for holo_pocket, apo_pocket, ligand, interaction, meta, _mask, _com in zip_longest(
                holo_pockets,
                apo_pockets,
                to_ligands,
                to_interactions,
                metadata,
                inpaint_mask,
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
            )
            for interp_pocket, ligand, meta, interaction, _mask in zip_longest(
                interp_pockets,
                interp_ligands,
                metadata,
                interp_interactions,
                inpaint_mask,
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

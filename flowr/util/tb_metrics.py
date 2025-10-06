from collections import Counter
from typing import List, Optional

import torch
import yaml
from datamol import pdist
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski
from torchmetrics import Metric

from flowr.scriptutil import ATOM_NAMES
from flowr.util.pylogger import RankedLogger
from flowr.util.rdkit import calc_energy, largest_component, mol_is_valid, optimise_mol
from posebusters import PoseBusters
from posecheck.utils.strain import calculate_strain_energy

log = RankedLogger(__name__, rank_zero_only=True)


class MolecularValidity(Metric):
    """Fraction of *valid* RDKit molecules among generated samples.

    This and the other metrics are mostly based on the code from the diffusion-hopping repo:
    https://github.com/jostorge/diffusion-hopping/blob/main/diffusion_hopping/analysis/metrics.py
    """

    higher_is_better = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state("num_valid", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, molecules: List[Chem.Mol]):
        """Update the validity metric with a list of molecules."""
        self.num_valid += sum(1 for mol in molecules if mol is not None)
        self.num_total += len(molecules)

    def compute(self):
        """Compute the validity metric."""
        return self.num_valid / self.num_total


class MolecularConnectivity(Metric):
    """Share of molecules whose graph is a single connected component."""

    higher_is_better = True

    def __init__(self, *args, **kwargs):
        """Initialize the connectivity metric.

        The connectivity metric measures whether generated molecules are fully connected graphs.
        A molecule is considered connected if all atoms belong to a single component,
        with no isolated fragments.

        This metric is important for molecular generation since valid molecules should
        not have disconnected components floating in space.
        """
        super().__init__(*args, **kwargs)
        self.add_state("num_connected", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, molecules: List[Chem.Mol]):
        """Update the connectivity metric with a list of molecules.

        Counts the number of fully-connected molecules.
        """
        molecules = [mol for mol in molecules if mol is not None]
        largest_components = largest_component(molecules)

        self.num_total += len(molecules)
        self.num_connected += sum(
            1
            for mol, ref in zip(largest_components, molecules)
            if mol.GetNumAtoms() == ref.GetNumAtoms()
        )

    def compute(self):
        """Compute the connectivity metric."""
        return self.num_connected / self.num_total


class FCValidity(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("valid", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        is_valid = [mol_is_valid(mol, connected=True) for mol in mols]
        self.valid += sum(is_valid)
        self.total += len(mols)

    def compute(self) -> torch.Tensor:
        return self.valid.float() / self.total


class PoseBustersValidityMolecule(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("valid", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_mols", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, gen_ligs: list[Chem.rdchem.Mol]) -> None:
        total = len(gen_ligs)
        self.n_mols += total

        gen_ligs = [lig for lig in gen_ligs if lig is not None]
        if len(gen_ligs) == 0:
            return

        buster = PoseBusters(config="mol")
        buster_df = buster.bust(gen_ligs, None, None)
        self.valid += sum(list(buster_df.all(axis=1)))

    def compute(self) -> torch.Tensor:
        return self.valid.float() / self.n_mols


class AverageStrainEnergy(Metric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state(
            "total_energy_diff", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        opt_mols = [
            (idx, optimise_mol(mol))
            for idx, mol in list(enumerate(mols))
            if mol is not None
        ]
        energies = [
            (idx, calc_energy(mol, per_atom=False))
            for idx, mol in opt_mols
            if mol is not None
        ]
        valids = [(idx, energy) for idx, energy in energies if energy is not None]

        if len(valids) == 0:
            return

        valid_indices, valid_energies = tuple(zip(*valids))
        original_energies = [
            calc_energy(mols[idx], per_atom=False) for idx in valid_indices
        ]
        energy_diffs = [
            orig - opt for orig, opt in zip(original_energies, valid_energies)
        ]

        self.total_energy_diff += sum(energy_diffs)
        self.n_valid += len(energy_diffs)

    def compute(self) -> torch.Tensor:
        return self.total_energy_diff / self.n_valid


class MolecularUniqueness(Metric):
    """Ratio of unique canonical SMILES among all *valid* samples.
    Uses hashing to be fancy and compatible with distributed training.
    """

    higher_is_better = True

    def __init__(self, *args, sync_on_compute=True, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_state(
            "hash_tensor",
            default=torch.empty(0, dtype=torch.int64),
            dist_reduce_fx="cat",
        )
        self.add_state("num_total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.sync_on_compute = sync_on_compute

        self._local_hashes = set()

    def update(self, molecules: List[Chem.Mol]):
        """Update the uniqueness metric with a list of molecules."""
        molecules = [mol for mol in molecules if mol is not None]
        molecules = largest_component(molecules)

        if not molecules:
            return

        self.num_total += len(molecules)

        smiles = [Chem.MolToSmiles(mol) for mol in molecules]

        new_hashes = []
        for smile in smiles:
            h = hash(smile) % (2**63 - 1)

            if h not in self._local_hashes:
                self._local_hashes.add(h)
                new_hashes.append(h)

        if new_hashes:
            new_hash_tensor = torch.tensor(
                new_hashes, dtype=torch.int64, device=self.hash_tensor.device
            )
            self.hash_tensor = torch.cat([self.hash_tensor, new_hash_tensor])

    def compute(self):
        """Calculate uniqueness ratio."""
        if self.sync_on_compute:
            synced_hashes = self.hash_tensor.clone()
            unique_count = torch.unique(synced_hashes).numel()
            return torch.tensor(unique_count / max(self.num_total.item(), 1))
        else:
            return torch.tensor(len(self._local_hashes) / max(self.num_total.item(), 1))

    def reset(self):
        """Reset the metric state."""
        super().reset()
        self._local_hashes = set()


class MolecularNovelty(Metric):
    """Fraction of generated SMILES absent from the training set."""

    higher_is_better = True

    def __init__(self, original_smiles: List[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self.original_smiles = set(original_smiles)
        self.add_state("num_novel", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, molecules: List[Chem.Mol]):
        """Compute smiles of generated molecules and compare to original smiles."""
        valid_mols = [mol for mol in molecules if mol is not None]
        valid_mols = largest_component(valid_mols)

        self.num_total += len(molecules)
        smiles = [Chem.MolToSmiles(mol) for mol in valid_mols]
        self.num_novel += sum(
            1 for smile in smiles if smile not in self.original_smiles
        )

    def compute(self):
        return self.num_novel / self.num_total


class MolecularDiversity(Metric):
    """Mean pair-wise fingerprint distance (higher ⇒ more diverse).

    Uses `datamol.pdist` with fingerprints such as ECFP; see
    `datamol.list_supported_fingerprints()` for available types.
    """

    higher_is_better = True

    def __init__(self, fp_size: int = 2048, fp_type: str = "ecfp", **kwargs):
        super().__init__(**kwargs)
        self.add_state("mols", default=[], dist_reduce_fx=None)
        self.fp_size = fp_size
        self.fp_type = fp_type

    def update(self, molecules: List[Chem.Mol]):
        """Update the diversity metric with a list of molecules."""
        valid_mols = [mol for mol in molecules if mol is not None]
        self.mols.extend(valid_mols)

    def compute(self):
        n = len(self.mols)
        if n < 2:
            # not enough molecules to compute pairwise diversity.
            return torch.tensor(0.0)

        dist_vec = pdist(
            self.mols,
            n_jobs=1,
            squareform=False,
            fpSize=self.fp_size,
            fp_type=self.fp_type,
        )

        return torch.tensor(dist_vec.mean())


class MolecularQEDValue(Metric):
    """Average QED score (0-1) over valid molecules."""

    higher_is_better = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state("qed_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, molecules: List[Chem.Mol]):
        molecules = [mol for mol in molecules if mol is not None]
        molecules = largest_component(molecules)

        self.num_total += len(molecules)
        self.qed_sum += sum(Descriptors.qed(mol) for mol in molecules)

    def compute(self):
        return self.qed_sum / self.num_total


class MolecularLogP(Metric):
    """Average logP (hydrophobicity) via `Crippen.MolLogP`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state("logp_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, molecules: List[Chem.Mol]):
        molecules = [mol for mol in molecules if mol is not None]
        molecules = largest_component(molecules)

        self.num_total += len(molecules)
        self.logp_sum += sum(Crippen.MolLogP(mol) for mol in molecules)

    def compute(self):
        return self.logp_sum / self.num_total


class MolecularLipinski(Metric):
    """Use Lipinski's rules to compute the lipinski score of a list of molecules."""

    higher_is_better = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("lipinski_sum", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, molecules: List[Chem.Mol]):
        molecules = [mol for mol in molecules if mol is not None]
        molecules = largest_component(molecules)

        self.num_total += len(molecules)
        self.lipinski_sum += sum(self._lipinski_score(mol) for mol in molecules)

    def _lipinski_score(self, mol: Chem.Mol) -> int:
        """Computes the lipinski score of a molecule."""
        logp = Crippen.MolLogP(mol)
        value = 0
        if Descriptors.ExactMolWt(mol) < 500:
            value += 1
        if Lipinski.NumHDonors(mol) <= 5:
            value += 1
        if Lipinski.NumHAcceptors(mol) <= 10:
            value += 1
        if -2 <= logp <= 5:
            value += 1
        if Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10:
            value += 1
        return value

    def compute(self):
        return self.lipinski_sum / self.num_total


class AtomTypeDistribution(Metric):
    """Similarity of atom-type histograms between generated and training sets."""

    def __init__(
        self,
        original_smiles: List[str],
        atom_names: Optional[List[str]] = ATOM_NAMES,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.atom_names = atom_names
        self.add_state(
            "seen_atoms",
            default=torch.tensor([], dtype=torch.int64),
            dist_reduce_fx="cat",
        )

        atom_type_counts = Counter()

        for smile in original_smiles:
            mol = Chem.MolFromSmiles(smile)
            for atom in mol.GetAtoms():
                atom_type = atom.GetSymbol()
                atom_type_counts[self.name_to_idx(atom_type)] += 1

        total = sum(atom_type_counts.values())
        self.atom_type_dict = {
            k: (v * 1.0) / total for k, v in atom_type_counts.items()
        }

    def name_to_idx(self, atom_type: str) -> int:
        """Convert an atom type to an index."""
        if atom_type not in self.atom_names:
            return len(self.atom_names) - 1
        return self.atom_names.index(atom_type)

    def update(self, molecules: List[Chem.Mol]):
        """Update the atom type distribution with a list of molecules."""
        for mol in molecules:
            if mol is None:
                continue

            for atom in mol.GetAtoms():
                atom_idx = self.name_to_idx(atom.GetSymbol())
                self.seen_atoms = torch.cat(
                    [self.seen_atoms, torch.tensor([atom_idx], dtype=torch.int64)]
                )

    def distribution_similarity(self, histo1, histo2):
        """Compute the similarity between two histograms."""

        assert torch.allclose(torch.tensor(sum(histo1.values())), torch.tensor(1.0))
        assert torch.allclose(torch.tensor(sum(histo2.values())), torch.tensor(1.0))

        similarity = 0.0
        all_keys = set(histo1.keys()) | set(histo2.keys())

        for key in all_keys:
            similarity += min(histo1.get(key, 0), histo2.get(key, 0))

        return torch.tensor(similarity)

    def compute(self):
        """Compute the atom type distribution."""

        if self.seen_atoms.shape[0] == 0:
            return torch.tensor(0.0)

        counts = Counter(self.seen_atoms.tolist())
        total = sum(counts.values())
        seen_dict = {k: (v * 1.0) / total for k, v in counts.items()}

        return self.distribution_similarity(self.atom_type_dict, seen_dict)


class AtomFractionMetric(Metric):
    """Proportion of atoms matching a given element symbol (e.g. 'C' or 'c')."""

    def __init__(self, atom_symbol, **kwargs):
        super().__init__(**kwargs)
        self.add_state(
            "num_atoms_of_symbol", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.add_state("num_total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.atom_symbol = atom_symbol.lower()

    def update(self, molecules: List[Chem.Mol]):
        """Update the fraction of carbons with a list of molecules."""
        for mol in molecules:
            if mol is None:
                continue

            self.num_total += mol.GetNumAtoms()

            num_atoms_of_symbol = 0
            for atom in mol.GetAtoms():
                if atom.GetSymbol().lower() == self.atom_symbol:
                    num_atoms_of_symbol += 1

            self.num_atoms_of_symbol += num_atoms_of_symbol

    def compute(self):
        return self.num_atoms_of_symbol / self.num_total


class PoseBustersValidity(Metric):
    """Fraction of molecules passing PoseBusters checks."""

    higher_is_better = True

    def __init__(self, **kwargs):
        """Args:
            **kwargs: Forwarded to `Metric`; may include `cfg_file` to
                override the default PoseBusters YAML.

        Note:
            Strain-energy evaluation is very slow—omit it during training
            unless strictly required (see the YAML in `utils/posebusters_no_strain.yaml`).
        """
        super().__init__(**kwargs)
        self.add_state(
            "posebusters_sum", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("num_total", default=torch.tensor(0), dist_reduce_fx="sum")

        if "cfg_file" in kwargs:
            self.cfg_file = kwargs["cfg_file"]
            self.cfg = yaml.safe_load(open(self.cfg_file, encoding="utf-8"))
        else:
            self.cfg = "mol"

    def update(self, molecules: List[Chem.Mol]):
        """Update the PoseBusters validity metric with a list of molecules, bad molecules cause the
        list to fail."""
        self.num_total += len(molecules)
        molecules = [mol for mol in molecules if mol is not None]

        if len(molecules) == 0:
            return

        pb = PoseBusters(config=self.cfg)
        try:
            results = pb.bust(mol_pred=molecules)
        except RuntimeError as e:
            log.warning(f"Error computing PoseBusters validity: {e}")
            self.posebusters_sum += 0
            return

        for _, row in results.iterrows():
            self.posebusters_sum += 0 if row.isin([False]).any() else 1

    def compute(self):
        return self.posebusters_sum / self.num_total


class PoseCheckStrainEnergy(Metric):
    """Average or median strain energy computed by PoseCheck."""

    higher_is_better = False

    def __init__(self, mode="median", num_confs=50, **kwargs) -> None:
        """
        Initialize the PoseCheck strain energy metric.

        Args:
            mode: Either "mean" or "median" to determine how the strain energy is aggregated
            num_confs: Number of conformations to use for strain energy calculation
            **kwargs: Additional arguments to pass to the Metric constructor
        """
        super().__init__(**kwargs)
        if mode not in ["mean", "median"]:
            raise ValueError(f"Mode must be 'mean' or 'median', got {mode}")
        self.mode = mode
        self.num_confs = num_confs
        if self.mode == "mean":
            self.add_state(
                "strain_energies", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )
            self.add_state("num_valid", default=torch.tensor(0), dist_reduce_fx="sum")
        else:  # median
            self.add_state(
                "strain_energies",
                default=torch.tensor([], dtype=torch.float32),
                dist_reduce_fx="cat",
            )

    def update(self, molecules: List[Chem.Mol]):
        """Update the PoseCheck strain energy metric with a list of valid molecules."""
        valid_molecules = [mol for mol in molecules if mol is not None]

        if not valid_molecules:
            return

        # For median mode, collect energies in a list first
        median_energies = []

        for mol in valid_molecules:
            try:
                energy = calculate_strain_energy(mol, num_confs=self.num_confs)
                if energy is None:
                    log.warning("Strain energy calculation returned None for molecule.")
                    continue
                if self.mode == "mean":
                    self.strain_energies += torch.tensor(energy)
                    self.num_valid += 1
                else:  # median
                    median_energies.append(energy)
            except RuntimeError as e:
                log.warning(f"Error computing strain energy for molecule: {e}")

        # For median mode, convert collected energies to tensor and concatenate
        if self.mode == "median" and median_energies:
            energies_tensor = torch.tensor(
                median_energies, dtype=torch.float32, device=self.strain_energies.device
            )
            self.strain_energies = torch.cat((self.strain_energies, energies_tensor))

    def compute(self):
        """Compute the strain energy according to the specified mode."""
        if self.mode == "mean":
            return (
                self.strain_energies / self.num_valid
                if self.num_valid > 0
                else torch.tensor(0.0)
            )
        else:  # median
            return (
                torch.median(self.strain_energies)
                if self.strain_energies.numel() > 0
                else torch.tensor(0.0)
            )

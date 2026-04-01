import itertools
import os
import shutil
import sys
import tempfile
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from pathlib import Path
from typing import Optional

import dill as pkl
import numpy as np
import prolif as plf
import torch
from biopandas.pdb import PandasPdb
from rdkit import Chem
from rdkit.Chem import (
    QED,
    AllChem,
    Crippen,
    Descriptors,
    Lipinski,
    Mol,
    RDConfig,
    rdchem,
    rdMolTransforms,
)
from scipy.stats import wasserstein_distance
from torchmetrics import Metric
from tqdm import tqdm

import flowr.util.functional as smolF
import flowr.util.rdkit as smolRD
from flowr.eval.evaluate_pose import shape_tanimoto_similarity
from flowr.util.calc_rmsd import align_rmsd
from flowr.util.functional import (
    LigandPocketOptimization,
    ligand_from_mol,
    prepare_complex_data,
    prepare_prolif_mols,
)
from flowr.util.interaction_util import InteractionFingerprints
from flowr.util.molecule import Molecule
from flowr.util.sampling.utils import (
    angle_distance,
    atom_types_distance,
    bond_length_distance,
    bond_types_distance,
    number_nodes_distance,
    valency_distance,
)
from posebusters import PoseBusters
from posecheck import PoseCheck

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer
from rdkit.Chem import rdMolDescriptors

warnings.filterwarnings(
    "ignore",
    category=Warning,
    message="WARNING: Search space volume is greater than 27000 Angstrom^3 (See FAQ)",
)
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def blockPrint():
    sys.stdout = open(os.devnull, "w")


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


ALLOWED_VALENCIES = {
    "H": {0: 1, 1: 0, -1: 0},
    "C": {0: [3, 4], 1: 3, -1: 3},
    "N": {
        0: [2, 3],
        1: [2, 3, 4],  # In QM9, N+ seems to be present in the form NH+ and NH2+
        -1: 2,
    },
    "O": {0: 2, 1: 3, -1: 1},
    "F": {0: 1, -1: 0},
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": {0: [3, 5], 1: 4},
    "S": {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
    "Cl": 1,
    "As": 3,
    "Br": {0: 1, 1: 2},
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
    "Se": [2, 4, 6],
}


def calc_atom_stabilities(mol):
    stabilities = []

    for atom in mol.GetAtoms():
        atom_type = atom.GetSymbol()
        valence = atom.GetExplicitValence()
        charge = atom.GetFormalCharge()

        if atom_type not in ALLOWED_VALENCIES:
            stabilities.append(False)
            continue

        allowed = ALLOWED_VALENCIES[atom_type]
        atom_stable = _is_valid_valence(valence, allowed, charge)
        stabilities.append(atom_stable)

    return stabilities


def _is_valid_valence(valence, allowed, charge):
    if isinstance(allowed, int):
        valid = allowed == valence

    elif isinstance(allowed, list):
        valid = valence in allowed

    elif isinstance(allowed, dict):
        allowed = allowed.get(charge)
        if allowed is None:
            return False

        valid = _is_valid_valence(valence, allowed, charge)

    return valid


def _is_valid_float(num):
    return num not in [None, float("inf"), float("-inf"), float("nan")]


class GenerativeMetric(Metric):
    # TODO add metric attributes - see torchmetrics doc

    def __init__(self, **kwargs):
        # Pass extra kwargs (defined in Metric class) to parent
        super().__init__(**kwargs)

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        raise NotImplementedError()

    def compute(self) -> torch.Tensor:
        raise NotImplementedError()


class PairMetric(Metric):
    def __init__(self, **kwargs):
        # Pass extra kwargs (defined in Metric class) to parent
        super().__init__(**kwargs)

    def update(
        self, predicted: list[Chem.rdchem.Mol], actual: list[Chem.rdchem.Mol]
    ) -> None:
        raise NotImplementedError()

    def compute(self) -> torch.Tensor:
        raise NotImplementedError()


class AtomStability(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("atom_stable", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, stabilities: list[list[bool]]) -> None:
        all_atom_stables = [
            atom_stable for atom_stbs in stabilities for atom_stable in atom_stbs
        ]
        self.atom_stable += sum(all_atom_stables)
        self.total += len(all_atom_stables)

    def compute(self) -> torch.Tensor:
        return self.atom_stable.float() / self.total


class MoleculeStability(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("mol_stable", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, stabilities: list[list[bool]]) -> None:
        mol_stables = [sum(atom_stbs) == len(atom_stbs) for atom_stbs in stabilities]
        self.mol_stable += sum(mol_stables)
        self.total += len(mol_stables)

    def compute(self) -> torch.Tensor:
        return self.mol_stable.float() / self.total


class Validity(GenerativeMetric):
    def __init__(self, connected=False, **kwargs):
        super().__init__(**kwargs)
        self.connected = connected

        self.add_state("valid", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        is_valid = [smolRD.mol_is_valid(mol, connected=self.connected) for mol in mols]
        self.valid += sum(is_valid)
        self.total += len(mols)

    def compute(self) -> torch.Tensor:
        return self.valid.float() / self.total


class DistributionDistance(GenerativeMetric):
    def __init__(self, dataset_info, train_mols=None, **kwargs):
        super().__init__(**kwargs)
        self.dataset_info = dataset_info
        self.train_mols = [mol for mol in train_mols if mol is not None]
        self.atom_encoder = dataset_info.atom_encoder
        self.atom_decoder = dataset_info.atom_decoder

        self.add_state("num_nodes_w1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("atom_types_tv", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("edge_types_tv", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("charge_w1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("valency_w1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "bond_lengths_w1", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("angles_w1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("dihedrals_cw1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, gen_ligs: list[Chem.rdchem.Mol], state="val") -> None:
        gen_ligs = [mol for mol in gen_ligs if mol is not None]
        molecules = [Molecule(mol, device=self.device) for mol in gen_ligs]
        if len(molecules) == 0:
            return

        # Compute statistics
        stat = self.dataset_info.statistics[state]

        self.num_nodes_w1 += number_nodes_distance(molecules, stat.num_nodes)
        atom_types_tv, atom_tv_per_class = atom_types_distance(
            molecules, stat.atom_types, save_histogram=False
        )
        self.atom_types_tv += atom_types_tv
        edge_types_tv, bond_tv_per_class, sparsity_level = bond_types_distance(
            molecules, stat.bond_types, save_histogram=False
        )
        self.edge_types_tv += edge_types_tv
        valency_w1, valency_w1_per_class = valency_distance(
            molecules, stat.valencies, stat.atom_types, self.atom_encoder
        )
        self.valency_w1 += valency_w1
        bond_lengths_w1, bond_lengths_w1_per_type = bond_length_distance(
            molecules, stat.bond_lengths, stat.bond_types
        )
        self.bond_lengths_w1 += bond_lengths_w1
        angles_w1, angles_w1_per_type = angle_distance(
            molecules,
            stat.bond_angles,
            stat.atom_types,
            stat.valencies,
            atom_decoder=self.atom_decoder,
            save_histogram=False,
        )
        self.angles_w1 += angles_w1
        self.dihedrals_cw1 += calc_circular_wasserstein_distance(
            gen_ligs, self.train_mols
        )
        self.total += 1

    def compute(self) -> torch.Tensor:
        return {
            "num_nodes_w1": self.num_nodes_w1 / self.total,
            "atom_types_tv": self.atom_types_tv / self.total,
            "edge_types_tv": self.edge_types_tv / self.total,
            "valency_w1": self.valency_w1 / self.total,
            "bond_lengths_w1": self.bond_lengths_w1 / self.total,
            "angles_w1": self.angles_w1 / self.total,
            "dihedrals_cw1": self.dihedrals_cw1 / self.total,
        }


class PocketValidity(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("pdb_valid", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("pocket_rmsds", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, ref_pdbs: list[str], pred_pdbs: list[str]) -> None:
        if len(pred_pdbs) == 0:
            return
        num_total = len(pred_pdbs)
        valid = len([pdb for pdb in pred_pdbs if pdb is not None])

        rmsds = []
        for ref_pdb, pred_pdb in zip(ref_pdbs, pred_pdbs):
            if pred_pdb is None:
                continue
            rmsds.append(align_rmsd(ref_pdb, pred_pdb))

        self.pocket_rmsds += sum(rmsds)
        self.pdb_valid += valid
        self.total += num_total

    def compute(self) -> torch.Tensor:
        out_dict = {
            "pdb_valid": self.pdb_valid / self.total,
            "pocket_rmsds": self.pocket_rmsds / self.pdb_valid,
        }
        return out_dict


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


class PoseBustersValidity(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("valid", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_mols", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        gen_ligs: list[Chem.rdchem.Mol],
        pdb_files: list,
        filter_pdbs: bool = False,
    ) -> None:

        total = len(gen_ligs)
        out = smolRD.sanitize_list(
            gen_ligs,
            pdbs=pdb_files,
            filter_pdb=filter_pdbs,
        )
        if isinstance(out, tuple):
            gen_ligs, pdb_files = out
        elif len(out) == 0:
            return

        validities = []
        for lig, pdb in zip(gen_ligs, pdb_files):
            assert isinstance(lig, Chem.Mol), "Expected a single ligand"
            lig = [lig]
            buster_dock = PoseBusters(config="dock")
            buster_dock_df = buster_dock.bust(lig, None, pdb)
            validities.extend(list(buster_dock_df.all(axis=1)))

        self.n_mols += total
        self.valid += sum(validities)

    def compute(self) -> torch.Tensor:
        return self.valid.float() / self.n_mols


class InteractionAccuracy(GenerativeMetric):
    def __init__(
        self,
        optimization_method: str = "prolif_mmff",
        pocket_cutoff: float = 6.0,
        strip_invalid: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.optim_method = optimization_method
        self.optimizer = LigandPocketOptimization(
            pocket_cutoff=pocket_cutoff, strip_invalid=strip_invalid
        )
        self.get_interactions = InteractionFingerprints()

        interaction_types = [
            "Hydrophobic",
            "VdWContact",
            "Cationic",
            "Anionic",
            "CationPi",
            "PiCation",
            "PiStacking",
            "HBAcceptor",
            "HBDonor",
        ]
        metrics = ["recall", "precision", "f1", "support"]
        for interaction in interaction_types:
            for metric in metrics:
                self.add_state(
                    f"{interaction}_{metric}",
                    default=torch.tensor(0.0),
                    dist_reduce_fx="sum",
                )

        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        gen_ligs: list[Chem.rdchem.Mol],
        pdb_files: list,
        true_interaction_arr: torch.Tensor,
        pred_interaction_arr: Optional[list[torch.Tensor]] = None,
    ) -> None:
        """
        Calculate the interaction accuracy for a given target
        gen_ligs: list of RDKit molecules of sampled ligands for each target (batch_size,)
        pdb_files: list of PDB files for each target (batch_size,) - NOTE: Must contain hydrogens!
        true_interaction_arr: true interaction tensor for each target (batch_size, N_l, N_p, num_interactions)
        pred_interaction_arr: list of N predicted interaction tensors for each target (batch_size,)
        NOTE: Expects the ligands to be valid
        """
        assert (
            len(gen_ligs)
            == len(true_interaction_arr)
            == len(pred_interaction_arr)
            == len(pdb_files)
        )

        true_interaction_arr = true_interaction_arr.permute(0, 2, 1, 3)
        pred_interaction_arr = (
            pred_interaction_arr.permute(0, 2, 1, 3)
            if pred_interaction_arr is not None
            else None
        )

        actual_interaction_arr = []
        for i, (lig, pdb_file) in enumerate(zip(gen_ligs, pdb_files)):
            lig_mol = self.add_and_optimize_hs(lig, pdb_file)
            pocket_mol = prepare_prolif_mols(pdb_file=pdb_file)
            interactions = self._get_interaction_array(
                lig_mol, pocket_mol, shape=true_interaction_arr[i].shape
            )
            actual_interaction_arr.append(interactions)

        # stack all interactions to get (batch_size, N_p, N_l, num_interactions+1)
        actual_interaction_arr = torch.stack(actual_interaction_arr)

        # get evaluation metrics
        complex_metrics = self._evaluate_per_class_per_complex(
            true_interaction_arr, actual_interaction_arr
        )
        avg_metrics = self._average_metrics_across_complexes(complex_metrics)

        for interaction in avg_metrics:
            for metric in avg_metrics[interaction]:
                self.__dict__[f"{interaction}_{metric}"] += avg_metrics[interaction][
                    metric
                ]

    def compute(self) -> dict:
        out_dict = {}
        for interaction in self.get_interaction_arr.PROLIF_INTERACTIONS:
            for metric in ["precision", "recall", "f1"]:
                out_dict[f"{interaction}_{metric}"] = (
                    self.__dict__[f"{interaction}_{metric}"].float() / self.total
                )
        return out_dict

    def _add_and_optimize_hs(
        self,
        lig: Chem.Mol,
        pdb_file: str,
        process_and_return_pocket: bool = False,
    ) -> plf.Molecule:
        return self.optimizer(
            lig,
            pdb_file,
            method=self.optim_method,
            process_and_return_pocket=process_and_return_pocket,
        )

    def _get_interaction_array(
        self, lig_mol: plf.Molecule, pocket_mol: plf.Molecule, shape: tuple
    ) -> torch.Tensor:
        interaction_arr = self.get_interactions(lig_mol, pocket_mol, return_array=True)
        return self._reshape_interaction_array(interaction_arr, shape=shape)

    def _reshape_interaction_array(self, interaction_arr, shape) -> torch.Tensor:
        n_pocket, n_lig, n_classes = shape
        interactions_arr = np.zeros((n_pocket, n_lig, n_classes))
        interactions_arr[:, :, 1:] = interaction_arr
        interactions_flat = interactions_arr.reshape(n_pocket * n_lig, n_classes)
        interactions_flat = np.argmax(interactions_flat, axis=-1)
        interactions_arr = smolF.one_hot_encode_tensor(
            torch.from_numpy(interactions_flat), n_classes
        )
        interactions_arr = interactions_arr.reshape(n_pocket, n_lig, -1)
        return interactions_arr

    def _evaluate_per_class_per_complex(
        self,
        true_interactions: torch.Tensor,
        pred_interactions: torch.Tensor,
        ignore_class: str = None,
    ):
        """
        Compute per-complex, per-class metrics (precision, recall, F1)
        for multi-class one-hot interaction arrays of shape:
            (B, N_pocket, N_ligand, K).
        Args:
            ground_truth (torch.Tensor): One-hot ground truth, shape (B, N_p, N_l, num_interactions+1).
            predictions (torch.Tensor): One-hot or multi-logit predictions, same shape.
            class_names (List[str], optional): Class labels for K classes.
                E.g. ["no_interaction", "Hbond", "PiStacking", ...].
            ignore_class (str, optional): Name of a class to skip in the final metrics
                (e.g., "no_interaction").
        Returns:
            A list (length B) of dictionaries:
                per_complex_metrics[b][class_name] = {
                    "precision": float,
                    "recall": float,
                    "f1": float,
                    "support": int
                }
            The "support" = number of ground-truth samples for that class in complex b.
        """
        bs, Np, Nl, num_interactions = true_interactions.shape
        class_names = ["NoInteraction"] + self.get_interaction_arr.PROLIF_INTERACTIONS
        assert num_interactions == len(class_names)

        pred_labels_all = pred_interactions.argmax(dim=-1)
        gt_labels_all = true_interactions.argmax(dim=-1)

        per_complex_metrics = []
        for b_idx in range(bs):
            gt_labels = gt_labels_all[b_idx]  # (Np, Nl)
            pred_labels = pred_labels_all[b_idx]  # (Np, Nl)

            class_metrics = {}
            for class_idx, class_name in enumerate(class_names):
                if ignore_class is not None and class_name == ignore_class:
                    continue

                # Ground-truth positives
                gt_mask = gt_labels == class_idx
                gt_count = gt_mask.sum().item()

                # Predicted positives
                pred_mask = pred_labels == class_idx

                # True positives
                tp_mask = gt_mask & pred_mask
                TP = tp_mask.sum().item()

                # False negatives
                FN = (gt_mask & (pred_labels != class_idx)).sum().item()

                # False positives: Prediction is class_idx, but GT differs
                FP = ((gt_labels != class_idx) & pred_mask).sum().item()

                precision = TP / (TP + FP) if TP + FP > 0 else 0.0
                recall = TP / (TP + FN) if TP + FN > 0 else 0.0
                # F1
                if (precision + recall) > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0.0

                class_metrics[class_name] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "support": gt_count,
                }

            per_complex_metrics.append(class_metrics)

        return per_complex_metrics

    def _average_metrics_across_complexes(self, per_complex_metrics):
        """
        Given the output of evaluate_per_class_per_complex(),
        compute the average (mean) metrics across B complexes
        for each class.
        Args:
            per_complex_metrics (list of dicts): length B.
            e.g. [
                { "HBAcceptor": { "precision": ..., "recall": ..., "f1": ..., "support": ...},
                "PiStacking": {...}, ...
                },
                {...}, # for complex 2
                ...
            ]
        Returns:
            avg_metrics (dict):
            { class_name: {"precision": float, "recall": float, "f1": float, "support": int} }
        """

        class_sums = defaultdict(
            lambda: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
        )
        class_counts = defaultdict(int)  # how many complexes had this class?
        for complex_dict in per_complex_metrics:
            for class_name, metrics in complex_dict.items():
                class_sums[class_name]["precision"] += metrics["precision"]
                class_sums[class_name]["recall"] += metrics["recall"]
                class_sums[class_name]["f1"] += metrics["f1"]
                class_sums[class_name]["support"] += metrics["support"]
                class_counts[class_name] += 1
        avg_metrics = {}
        for class_name, sums in class_sums.items():
            count = class_counts[class_name]
            avg_metrics[class_name] = {
                "precision": sums["precision"] / count,
                "recall": sums["recall"] / count,
                "f1": sums["f1"] / count,
                "support": sums["support"],  # total support across all complexes
            }
        return avg_metrics


class InteractionRecovery(GenerativeMetric):
    def __init__(
        self,
        optimization_method: str = "prolif_mmff",
        pocket_cutoff: float = 6.0,
        strip_invalid: bool = True,
        save_dir: str = None,
        **kwargs,
    ):
        super().__init__()

        self.optim_method = optimization_method
        self.optimizer = LigandPocketOptimization(
            pocket_cutoff=pocket_cutoff, strip_invalid=strip_invalid
        )
        self.save_dir = save_dir

        self.get_interactions = InteractionFingerprints()

        self.add_state("plif_recovery", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("failed", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        gen_ligs: list[list[Chem.Mol]],
        native_ligs: list[Chem.Mol],
        pdb_files: list[str],
        add_optimize_gen_lig_hs: bool = True,
        add_optimize_ref_lig_hs: bool = False,
        optimize_pocket_hs: bool = False,
        process_pocket: bool = False,
    ) -> None:
        """
        Calculate the interaction recovery rate for a given target
        gen_ligs: list of list RDKit molecules of N (sampled) valid ligands for each target
        native_ligs: list of sdf files of native ligands for each target - NOTE: the ligands need to have hydrogens!
        pdb_files: list of PDB files for each target - NOTE: the pocket needs to have hydrogens!
        """

        assert (
            not optimize_pocket_hs
        ), "Pocket optimization is not supported yet. The computation for every generated ligand would be too slow."
        # Pocket Hs optimization would require a separate pocket for every generated ligand
        # and hence prolif calculation per ligand-pocket pair, which currently is not supported.
        # The pocket should come with optimized hydrogens anyway. However, the reference ligands can be Hs optimized.

        for gen_lig, native_lig, pdb_file in tqdm(
            zip(gen_ligs, native_ligs, pdb_files),
            total=len(pdb_files),
            desc="PLIF recovery calculation...",
        ):
            complex_id = Path(pdb_file).stem
            if add_optimize_gen_lig_hs:
                if isinstance(gen_lig, Chem.Mol):
                    gen_lig_mol, pocket_mol = self._add_and_optimize_hs(
                        gen_lig,
                        pdb_file,
                        return_pocket=True,
                        optimize_pocket_hs=optimize_pocket_hs,
                        process_pocket=process_pocket,
                    )
                elif isinstance(gen_lig, list):
                    gen_lig_pocket_mol = [
                        self._add_and_optimize_hs(
                            lig,
                            pdb_file,
                            return_pocket=True,
                            optimize_pocket_hs=optimize_pocket_hs,
                            process_pocket=process_pocket,
                        )
                        for lig in gen_lig
                    ]
                    gen_lig_mol = [cmplx[0] for cmplx in gen_lig_pocket_mol]
                    pocket_mol = gen_lig_pocket_mol[0][
                        1
                    ]  # as pocket Hs optimization per ligand is currently not supported
                else:
                    raise ValueError("Invalid ligand format")
            else:
                if isinstance(gen_lig, Chem.Mol):
                    gen_lig_mol = ligand_from_mol(gen_lig, add_hydrogens=False)
                elif isinstance(gen_lig, list):
                    gen_lig_mol = [
                        ligand_from_mol(lig, add_hydrogens=False) for lig in gen_lig
                    ]
                else:
                    raise ValueError("Invalid ligand format")

            if add_optimize_ref_lig_hs:
                native_lig = Chem.RemoveHs(native_lig)
                native_lig_mol, pocket_ref_mol = self._add_and_optimize_hs(
                    native_lig,
                    pdb_file,
                    return_pocket=True,
                    process_pocket=process_pocket,
                    optimize_pocket_hs=optimize_pocket_hs,
                )
            else:
                native_lig_mol = ligand_from_mol(native_lig, add_hydrogens=False)
                pocket_ref_mol = self.optimizer.pocket_from_pdb(
                    pdb_file, native_lig_mol, process_pocket=process_pocket
                )

            # Calculate the interaction recovery rate
            plif_recovery = self._get_plif_recovery(
                gen_lig_mol,
                native_lig_mol,
                pocket_mol,
                pocket_ref_mol,
                complex_id=complex_id,
                optimized_ground_truth=add_optimize_ref_lig_hs,
            )

            if plif_recovery is not None:
                self.plif_recovery += plif_recovery
                self.total += 1
            else:
                self.failed += 1

    def compute(self) -> float:
        out_dict = {
            "plif_recovery": self.plif_recovery.float() / self.total,
            "failed": self.failed,
        }
        return out_dict

    def _add_and_optimize_hs(
        self,
        lig: Chem.Mol,
        pdb_file: str,
        process_pocket: bool = False,
        return_pocket: bool = False,
        optimize_pocket_hs: bool = False,
    ) -> plf.Molecule:
        return self.optimizer(
            lig,
            pdb_file,
            method=self.optim_method,
            process_pocket=process_pocket,
            return_pocket=return_pocket,
            optimize_pocket_hs=optimize_pocket_hs,
        )

    def _get_plif_recovery(
        self,
        pred_ligand_mol: list[plf.Molecule],
        native_ligand_mol: plf.Molecule,
        pocket_mol: list[plf.Molecule],
        pocket_ref_mol: plf.Molecule,
        complex_id: str = None,
        optimized_ground_truth: bool = False,
    ) -> float:
        """
        Calculate the interaction recovery rate for a given target
        pred_ligand_mol: (list of) prolif molecule(s) of the predicted ligand(s)
        native_ligand_mol: prolif molecule of the native ligand
        pocket_mol: prolif molecule of the protein pocket

        NOTE: recovery rate can become None if the native plif is empty
        """
        # Load or calculate the native plifs
        if optimized_ground_truth:
            plifs_pkl_path = Path(self.save_dir) / "plifs_optimized"
        else:
            plifs_pkl_path = Path(self.save_dir) / "plifs"
        if not os.path.exists(plifs_pkl_path):
            os.makedirs(plifs_pkl_path)
        plifs_pkl_file = plifs_pkl_path / f"{complex_id}_native_plifs.pkl"
        if plifs_pkl_file.is_file():
            with open(plifs_pkl_file, "rb") as f:
                native_plif = pkl.load(f)
        else:
            native_plif = self.get_interactions(native_ligand_mol, pocket_ref_mol)
            with open(plifs_pkl_file, "wb") as f:
                pkl.dump(native_plif, f)

        if isinstance(pocket_mol, list):
            assert len(pred_ligand_mol) == len(pocket_mol)
            gen_plif = [
                self.get_interactions(lig, pocket)
                for lig, pocket in zip(pred_ligand_mol, pocket_mol)
            ]
            plif_result = [
                self.get_interactions.get_plif_recovery_rates(
                    true_fp=native_plif, pred_fp=gen_plif[i]
                )
                for i in range(len(pred_ligand_mol))
            ]
            recovery_rate = [r.count_recovery for r in plif_result if r is not None]
            recovery_rate = np.mean(recovery_rate) if len(recovery_rate) > 0 else None
        elif isinstance(pred_ligand_mol, list):
            gen_plif = self.get_interactions(pred_ligand_mol, pocket_mol)
            if isinstance(pred_ligand_mol, list):
                plif_result = [
                    self.get_interactions.get_plif_recovery_rates(
                        true_fp=native_plif, pred_fp=gen_plif, ifp_idx=i
                    )
                    for i in range(len(pred_ligand_mol))
                ]
            recovery_rate = [r.count_recovery for r in plif_result if r is not None]
            recovery_rate = np.mean(recovery_rate) if len(recovery_rate) > 0 else None
        else:
            gen_plif = self.get_interactions(pred_ligand_mol, pocket_mol)
            plif_result = self.get_interactions.get_plif_recovery_rates(
                true_fp=native_plif, pred_fp=gen_plif
            )
            recovery_rate = (
                plif_result.count_recovery if plif_result is not None else None
            )

        return recovery_rate


# TODO I don't think this will work with DDP
class Uniqueness(Metric):
    """Note: only tracks uniqueness of molecules which can be converted into SMILES"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.valid_smiles = []

    def reset(self):
        self.valid_smiles = []

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        smiles = [
            smolRD.smiles_from_mol(mol, canonical=True)
            for mol in mols
            if mol is not None
        ]
        valid_smiles = [smi for smi in smiles if smi is not None]
        self.valid_smiles.extend(valid_smiles)

    def compute(self) -> torch.Tensor:
        num_unique = len(set(self.valid_smiles))
        uniqueness = torch.tensor(num_unique) / len(self.valid_smiles)
        return uniqueness


class Novelty(GenerativeMetric):
    def __init__(self, existing_mols: list[Chem.rdchem.Mol], **kwargs):
        super().__init__(**kwargs)

        n_workers = min(4, len(os.sched_getaffinity(0)))
        # executor = ProcessPoolExecutor(max_workers=n_workers)
        executor = ThreadPoolExecutor(max_workers=n_workers)

        futures = [
            executor.submit(smolRD.smiles_from_mol, mol, canonical=True)
            for mol in existing_mols
        ]
        smiles = [future.result() for future in futures]
        smiles = [smi for smi in smiles if smi is not None]

        executor.shutdown()

        self.smiles = set(smiles)

        self.add_state("novel", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        smiles = [
            smolRD.smiles_from_mol(mol, canonical=True)
            for mol in mols
            if mol is not None
        ]
        valid_smiles = [smi for smi in smiles if smi is not None]
        novel = [smi not in self.smiles for smi in valid_smiles]

        self.novel += sum(novel)
        self.total += len(novel)

    def compute(self) -> torch.Tensor:
        return self.novel.float() / self.total


class EnergyValidity(GenerativeMetric):
    def __init__(self, optimise=False, **kwargs):
        super().__init__(**kwargs)

        self.optimise = optimise

        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        num_mols = len(mols)

        if self.optimise:
            mols = [smolRD.optimise_mol(mol) for mol in mols if mol is not None]

        energies = [smolRD.calc_energy(mol) for mol in mols if mol is not None]
        valid_energies = [energy for energy in energies if _is_valid_float(energy)]

        self.n_valid += len(valid_energies)
        self.total += num_mols

    def compute(self) -> torch.Tensor:
        return self.n_valid.float() / self.total


class AverageEnergy(GenerativeMetric):
    """Average energy for molecules for which energy can be calculated

    Note that the energy cannot be calculated for some molecules (specifically invalid ones) and the pose optimisation
    is not guaranteed to succeed. Molecules for which the energy cannot be calculated do not count towards the metric.

    This metric doesn't require that input molecules have been sanitised by RDKit, however, it is usually a good idea
    to do this anyway to ensure that all of the required molecular and atom properties are calculated and stored.
    """

    def __init__(self, optimise=False, per_atom=False, **kwargs):
        super().__init__(**kwargs)

        self.optimise = optimise
        self.per_atom = per_atom

        self.add_state("energy", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "n_valid_energies", default=torch.tensor(0), dist_reduce_fx="sum"
        )

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        if self.optimise:
            mols = [smolRD.optimise_mol(mol) for mol in mols if mol is not None]

        energies = [
            smolRD.calc_energy(mol, per_atom=self.per_atom)
            for mol in mols
            if mol is not None
        ]
        valid_energies = [energy for energy in energies if _is_valid_float(energy)]

        self.energy += sum(valid_energies)
        self.n_valid_energies += len(valid_energies)

    def compute(self) -> torch.Tensor:
        return self.energy / self.n_valid_energies


class AverageStrainEnergy(GenerativeMetric):
    """
    The strain energy is the energy difference between a molecule's pose and its optimised pose. Estimated using RDKit.
    Only calculated when all of the following are true:
    1. The molecule is valid and an energy can be calculated
    2. The pose optimisation succeeds
    3. The energy can be calculated for the optimised pose

    Note that molecules which do not meet these criteria will not count towards the metric and can therefore give
    unexpected results. Use the EnergyValidity metric with the optimise flag set to True to track the proportion of
    molecules for which this metric can be calculated.

    This metric doesn't require that input molecules have been sanitised by RDKit, however, it is usually a good idea
    to do this anyway to ensure that all of the required molecular and atom properties are calculated and stored.
    """

    def __init__(self, per_atom=False, **kwargs):
        super().__init__(**kwargs)

        self.per_atom = per_atom

        self.add_state(
            "total_energy_diff", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        opt_mols = [
            (idx, smolRD.optimise_mol(mol))
            for idx, mol in list(enumerate(mols))
            if mol is not None
        ]
        energies = [
            (idx, smolRD.calc_energy(mol, per_atom=self.per_atom))
            for idx, mol in opt_mols
            if mol is not None
        ]
        valids = [(idx, energy) for idx, energy in energies if energy is not None]

        if len(valids) == 0:
            return

        valid_indices, valid_energies = tuple(zip(*valids))
        original_energies = [
            smolRD.calc_energy(mols[idx], per_atom=self.per_atom)
            for idx in valid_indices
        ]
        energy_diffs = [
            orig - opt for orig, opt in zip(original_energies, valid_energies)
        ]

        self.total_energy_diff += sum(energy_diffs)
        self.n_valid += len(energy_diffs)

    def compute(self) -> torch.Tensor:
        return self.total_energy_diff / self.n_valid


class AverageOptRmsd(GenerativeMetric):
    """
    Average RMSD between a molecule and its optimised pose. Only calculated when all of the following are true:
    1. The molecule is valid
    2. The pose optimisation succeeds

    Note that molecules which do not meet these criteria will not count towards the metric and can therefore give
    unexpected results.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("total_rmsd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        valids = [
            (idx, smolRD.optimise_mol(mol))
            for idx, mol in list(enumerate(mols))
            if mol is not None
        ]
        valids = [(idx, mol) for idx, mol in valids if mol is not None]

        if len(valids) == 0:
            return

        valid_indices, opt_mols = tuple(zip(*valids))
        original_mols = [mols[idx] for idx in valid_indices]
        rmsds = [
            smolRD.conf_distance(mol1, mol2)
            for mol1, mol2 in zip(original_mols, opt_mols)
        ]

        self.total_rmsd += sum(rmsds)
        self.n_valid += len(rmsds)

    def compute(self) -> torch.Tensor:
        return self.total_rmsd / self.n_valid


class MolecularAccuracy(PairMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("n_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, predicted: list[Chem.rdchem.Mol], actual: list[Chem.rdchem.Mol]
    ) -> None:
        predicted_smiles = [
            smolRD.smiles_from_mol(pred, canonical=True) for pred in predicted
        ]
        actual_smiles = [smolRD.smiles_from_mol(act, canonical=True) for act in actual]
        matches = [
            pred == act
            for pred, act in zip(predicted_smiles, actual_smiles)
            if act is not None
        ]

        self.n_correct += sum(matches)
        self.total += len(matches)

    def compute(self) -> torch.Tensor:
        return self.n_correct.float() / self.total


class MolecularPairRMSD(PairMetric):
    def __init__(self, fix_order: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.fix_order = fix_order
        self.add_state("total_rmsd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, predicted: list[Chem.rdchem.Mol], actual: list[Chem.rdchem.Mol]
    ) -> None:
        valid_pairs = [
            (pred, act)
            for pred, act in zip(predicted, actual)
            if pred is not None and act is not None
        ]
        rmsds = [
            smolRD.conf_distance(pred, act, fix_order=self.fix_order)
            for pred, act in valid_pairs
        ]
        rmsds = [rmsd for rmsd in rmsds if rmsd is not None]

        self.total_rmsd += sum(rmsds)
        self.n_valid += len(rmsds)

    def compute(self) -> torch.tensor:
        return self.total_rmsd / self.n_valid


class MolecularPairShapeTanimotoSim(PairMetric):
    def __init__(self, align: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.align = align
        self.add_state(
            "total_shape_tanimoto_sim", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, predicted: list[Chem.rdchem.Mol], actual: list[Chem.rdchem.Mol]
    ) -> None:
        valid_pairs = [
            (pred, act)
            for pred, act in zip(predicted, actual)
            if pred is not None and act is not None
        ]
        values = [
            shape_tanimoto_similarity(pred, act, align=self.align)
            for pred, act in valid_pairs
        ]
        values = [value for value in values if value is not None]

        self.total_shape_tanimoto_sim += sum(values)
        self.n_valid += len(values)

    def compute(self) -> torch.tensor:
        return self.total_shape_tanimoto_sim / self.n_valid


def evaluate_uniqueness(gen_ligs: list[Chem.Mol]):
    smiles = [
        smolRD.smiles_from_mol(mol, canonical=True)
        for mol in gen_ligs
        if mol is not None
    ]
    valid_smiles = [smi for smi in smiles if smi is not None]
    num_unique = len(set(valid_smiles))
    uniqueness = num_unique / len(valid_smiles)
    return uniqueness


def calculate_qed(rdmol):
    return QED.qed(rdmol)


def calculate_sa(rdmol):
    sa = sascorer.calculateScore(rdmol)
    sa = (sa - 1.0) / (10.0 - 1.0)
    sa = 1.0 - sa
    return round(sa, 2)


def calculate_logp(rdmol):
    return Crippen.MolLogP(rdmol)


def calculate_hdonors(rdmol):
    num_hdonors = Lipinski.NumHDonors(rdmol)
    return num_hdonors


def calculate_hacceptors(rdmol):
    num_hacceptors = Lipinski.NumHAcceptors(rdmol)
    return num_hacceptors


def calculate_molwt(rdmol):
    mol_weight = Descriptors.MolWt(rdmol)
    return mol_weight


def calculate_lipinski(rdmol):
    rule_1 = Descriptors.ExactMolWt(rdmol) < 500
    rule_2 = Lipinski.NumHDonors(rdmol) <= 5
    rule_3 = Lipinski.NumHAcceptors(rdmol) <= 10
    rule_4 = (logp := Crippen.MolLogP(rdmol) >= -2) & (logp <= 5)
    rule_5 = rdMolDescriptors.CalcNumRotatableBonds(rdmol) <= 10
    return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])


def num_rings(mol):
    return rdMolDescriptors.CalcNumRings(mol)


def num_aromatic_rings(mol):
    return rdMolDescriptors.CalcNumAromaticRings(mol)


def calculate_tpsa(mol):
    return rdMolDescriptors.CalcTPSA(mol)


def calculate_rotatable_bonds(mol):
    return rdMolDescriptors.CalcNumRotatableBonds(mol)


def evaluate_mol_metrics(gen_ligs: list[Chem.Mol], train_smiles: list[str]):
    # novelty
    smiles = [
        smolRD.smiles_from_mol(mol, canonical=True)
        for mol in gen_ligs
        if mol is not None
    ]
    valid_smiles = [smi for smi in smiles if smi is not None]
    novel = [smi not in train_smiles for smi in valid_smiles]
    mean_novel = np.mean(novel)
    std_novel = np.std(novel)

    # Mol properties
    qed = [calculate_qed(mol) for mol in gen_ligs if mol is not None]
    mean_qed = np.mean(qed)
    std_qed = np.std(qed)

    sa = [calculate_sa(mol) for mol in gen_ligs if mol is not None]
    mean_sa = np.mean(sa)
    std_sa = np.std(sa)

    logp = [calculate_logp(mol) for mol in gen_ligs if mol is not None]
    mean_logp = np.mean(logp)
    std_logp = np.std(logp)

    hdonors = [calculate_hdonors(mol) for mol in gen_ligs if mol is not None]
    mean_hdonors = np.mean(hdonors)
    std_hdonors = np.std(hdonors)

    hacceptors = [calculate_hacceptors(mol) for mol in gen_ligs if mol is not None]
    mean_hacceptors = np.mean(hacceptors)
    std_hacceptors = np.std(hacceptors)

    molwt = [calculate_molwt(mol) for mol in gen_ligs if mol is not None]
    mean_molwt = np.mean(molwt)
    std_molwt = np.std(molwt)

    lipinski = [calculate_lipinski(mol) for mol in gen_ligs if mol is not None]
    mean_lipinski = np.mean(lipinski)
    std_lipinski = np.std(lipinski)

    # Rings
    rings = [num_rings(mol) for mol in gen_ligs if mol is not None]
    mean_rings = np.mean(rings)
    std_rings = np.std(rings)

    aromatic_rings = [num_aromatic_rings(mol) for mol in gen_ligs if mol is not None]
    mean_aromatic_rings = np.mean(aromatic_rings)
    std_aromatic_rings = np.std(aromatic_rings)

    # average strain energy
    opt_mols = [
        (idx, smolRD.optimise_mol(mol))
        for idx, mol in enumerate(gen_ligs)
        if mol is not None
    ]
    energies = [
        (idx, smolRD.calc_energy(mol, per_atom=False))
        for idx, mol in opt_mols
        if mol is not None
    ]
    valids = [(idx, energy) for idx, energy in energies if energy is not None]
    valid_indices, valid_energies = tuple(zip(*valids))
    original_energies = [
        smolRD.calc_energy(gen_ligs[idx], per_atom=False) for idx in valid_indices
    ]
    energy_diffs = [orig - opt for orig, opt in zip(original_energies, valid_energies)]
    mean_strain_energy = np.mean(energy_diffs)
    std_strain_energy = np.std(energy_diffs)

    # energy validity
    energies = [smolRD.calc_energy(mol) for mol in gen_ligs if mol is not None]
    valid_energies = [energy for energy in energies if _is_valid_float(energy)]
    energy_validity = [_is_valid_float(energy) for energy in energies]
    mean_energy_validity = np.mean(energy_validity)
    std_energy_validity = np.std(energy_validity)

    # energy
    mean_energy = np.mean(valid_energies)
    std_energy = np.std(valid_energies)

    # opt-rmsd
    valids = [(idx, opt_mol) for idx, opt_mol in opt_mols if opt_mol is not None]
    valid_indices, opt_mols = tuple(zip(*valids))
    original_mols = [gen_ligs[idx] for idx in valid_indices]
    rmsds = [
        smolRD.conf_distance(mol1, mol2) for mol1, mol2 in zip(original_mols, opt_mols)
    ]
    mean_opt_rmsd = np.mean(rmsds)
    std_opt_rmsd = np.std(rmsds)

    # number of rotatable bonds
    rotatable_bonds = [
        calculate_rotatable_bonds(mol) for mol in gen_ligs if mol is not None
    ]
    mean_rotatable_bonds = np.mean(rotatable_bonds)
    std_rotatable_bonds = np.std(rotatable_bonds)

    # tpsa
    tpsa = [calculate_tpsa(mol) for mol in gen_ligs if mol is not None]
    mean_tpsa = np.mean(tpsa)
    std_tpsa = np.std(tpsa)

    return {
        "Energy validity (mean)": mean_energy_validity,
        "Energy validity (std)": std_energy_validity,
        "Energy (mean)": mean_energy,
        "Energy (std)": std_energy,
        "Strain energy (mean)": mean_strain_energy,
        "Strain energy (std)": std_strain_energy,
        "Opt-RMSD (mean)": mean_opt_rmsd,
        "Opt-RMSD (std)": std_opt_rmsd,
        "Novelty (mean)": mean_novel,
        "Novelty (std)": std_novel,
        "QED (mean)": mean_qed,
        "QED (std)": std_qed,
        "SA (mean)": mean_sa,
        "SA (std)": std_sa,
        "LogP (mean)": mean_logp,
        "LogP (std)": std_logp,
        "HDonors (mean)": mean_hdonors,
        "HDonors (std)": std_hdonors,
        "HAcceptors (mean)": mean_hacceptors,
        "HAcceptors (std)": std_hacceptors,
        "MolWt (mean)": mean_molwt,
        "MolWt (std)": std_molwt,
        "Lipinski (mean)": mean_lipinski,
        "Lipinski (std)": std_lipinski,
        "Rings (mean)": mean_rings,
        "Rings (std)": std_rings,
        "Aromatic Rings (mean)": mean_aromatic_rings,
        "Aromatic Rings (std)": std_aromatic_rings,
        "Rotatable Bonds (mean)": mean_rotatable_bonds,
        "Rotatable Bonds (std)": std_rotatable_bonds,
        "TPSA (mean)": mean_tpsa,
        "TPSA (std)": std_tpsa,
    }


def evaluate_validity(gen_ligs: list[Chem.Mol]):
    # validity
    is_valid = [smolRD.mol_is_valid(lig, connected=False) for lig in gen_ligs]
    mean_valid = np.mean(is_valid)
    std_valid = np.std(is_valid)
    is_valid_fc = [smolRD.mol_is_valid(lig, connected=True) for lig in gen_ligs]
    mean_valid_fc = np.mean(is_valid_fc)
    std_valid_fc = np.std(is_valid_fc)
    return {
        "Validity (mean)": mean_valid,
        "Validity (std)": std_valid,
        "Fc-validity (mean)": mean_valid_fc,
        "Fc-validity (std)": std_valid_fc,
    }


def evaluate_pb_validity_mol(
    gen_ligs: list[Chem.Mol],
    return_list: bool = False,
    return_dict: bool = False,
):
    if isinstance(gen_ligs, Chem.Mol):
        gen_ligs = [gen_ligs]
    buster_dock = PoseBusters(config="mol")
    buster_dock_df = buster_dock.bust(gen_ligs, None, None)
    validities = list(buster_dock_df.all(axis=1))
    if return_dict:
        # Return all PoseBusters metrics as a dictionary (column-wise, as lists)
        return buster_dock_df.to_dict(orient="list")
    if return_list:
        return validities
    return {
        "PB-validity (mean)": np.mean(validities),
        "PB-validity (std)": np.std(validities),
    }


def evaluate_pb_validity(
    gen_ligs: list[Chem.Mol],
    pdb_file: str,
    ref_lig: Chem.Mol = None,
    minimize: bool = False,
    config: dict = None,
    return_list: bool = False,
):
    if isinstance(gen_ligs, Chem.Mol):
        gen_ligs = [gen_ligs]
    buster_dock = PoseBusters(config="dock")
    buster_dock_df = buster_dock.bust(gen_ligs, None, pdb_file)
    validities = list(buster_dock_df.all(axis=1))
    if return_list:
        return validities
    return {
        "PB-validity (mean)": np.mean(validities),
        "PB-validity (std)": np.std(validities),
    }


def evaluate_pb_validity_list(
    gen_ligs: list[list[Chem.Mol]],
    pdb_files: list[str],
    ref_ligs: list[Chem.Mol] = [],
    minimize: bool = False,
    config: dict = None,
    return_list: bool = False,
):
    validities_strict_sbdd = [] if return_list else defaultdict(list)
    for ligs, ref_lig, pdb_file in tqdm(
        zip_longest(gen_ligs, ref_ligs, pdb_files, fillvalue=None),
        total=len(gen_ligs),
        desc="PoseBusters validity calculation...",
    ):
        validities = evaluate_pb_validity(
            ligs,
            pdb_file,
            ref_lig=ref_lig,
            minimize=minimize,
            config=config,
            return_list=return_list,
        )
        if return_list:
            validities_strict_sbdd.append(validities)
        else:
            for k, v in validities.items():
                validities_strict_sbdd[k].append(v)

    if return_list:
        return validities_strict_sbdd
    return {k: np.mean(v) for k, v in validities_strict_sbdd.items()}


def evaluate_posecheck(
    gen_ligs: list[Chem.Mol],
    pdb_file: str,
    return_list: bool = False,
):
    if isinstance(gen_ligs, Chem.Mol):
        gen_ligs = [gen_ligs]

    pc = PoseCheck()
    pc.load_protein_from_pdb(pdb_file)
    pc.load_ligands_from_mols(gen_ligs)

    clashes = pc.calculate_clashes()
    strain = pc.calculate_strain_energy()

    if return_list:
        return {
            "PC-Clashes": clashes,
            "PC-Strain": strain,
        }
    return {
        "PC-Clashes (mean)": np.nanmean([c for c in clashes if c is not None]),
        "PC-Clashes (std)": np.nanstd([c for c in clashes if c is not None]),
        "PC-Strain (mean)": np.nanmean([s for s in strain if s is not None]),
        "PC-Strain (std)": np.nanstd([s for s in strain if s is not None]),
    }


def evaluate_posecheck_list(
    gen_ligs: list[list[Chem.Mol]],
    pdb_files: list[str],
    return_list: bool = False,
):
    pc_dict = defaultdict(list)
    for gen_lig, pdb_file in tqdm(zip(gen_ligs, pdb_files), total=len(gen_ligs)):
        pc_results = evaluate_posecheck(gen_lig, pdb_file, return_list=return_list)
        for metric_name, value in pc_results.items():
            pc_dict[metric_name].append(value)

    if return_list:
        return pc_dict
    pc_dict = {k: np.nanmean(v) for k, v in pc_dict.items()}
    return pc_dict


def calc_strain(
    mol: Chem.Mol,
    add_hs: bool = True,
    n_steps: int = 1000,
    force_field_name: str = "MMFF94s",
):
    if add_hs:
        mol = Chem.AddHs(mol, addCoords=True)

    try:
        if force_field_name == "MMFF94s":
            mol_properties = AllChem.MMFFGetMoleculeProperties(
                mol, mmffVariant="MMFF94s"
            )
            force_field = AllChem.MMFFGetMoleculeForceField(
                mol, mol_properties, confId=0
            )
        elif force_field_name == "MMFF94":
            mol_properties = AllChem.MMFFGetMoleculeProperties(mol)
            force_field = AllChem.MMFFGetMoleculeForceField(
                mol, mol_properties, confId=0
            )
        elif force_field_name == "UFF":
            force_field = AllChem.UFFGetMoleculeForceField(mol, confId=0)

        start_energy = force_field.CalcEnergy()
        not_converged = force_field.Minimize(maxIts=n_steps)
        if not_converged:
            print(
                "Energy minimization did not converge - using intermediate state for strain calculation"
            )
        final_energy = force_field.CalcEnergy()
        strain_energy = start_energy - final_energy
        assert strain_energy > 0, "Strain energy should be positive"
        return strain_energy
    except Exception:
        print("Force field error - skipping strain calculation")
        return np.nan


def calculate_minimization_metrics(
    mol_before: Optional[Mol],
    mol_after: Optional[Mol],
) -> tuple[float, float, float]:
    """
    Calculate strain and RMSD metrics for a molecule before and after minimization.

    Args:
        mol_before: Molecule before minimization (with hydrogens).
        mol_after: Molecule after minimization (with hydrogens).

    Returns:
        Tuple of (strain_before, strain_after, rmsd_before_after).
        Returns (np.nan, np.nan, np.nan) if calculation fails.
    """
    if mol_before is None or mol_after is None:
        return np.nan, np.nan, np.nan

    try:
        # Calculate strain before minimization
        strain_before = evaluate_strain([mol_before], add_hs=False, return_list=True)[0]

        # Calculate strain after minimization
        strain_after = evaluate_strain([mol_after], add_hs=False, return_list=True)[0]

        # Calculate all-atom RMSD without alignment
        rmsd_value = AllChem.CalcRMS(mol_before, mol_after)

        return strain_before, strain_after, rmsd_value

    except Exception as e:
        print(f"Failed to calculate metrics: {e}")
        return np.nan, np.nan, np.nan


def evaluate_strain(
    gen_ligs: list[Chem.Mol],
    n_steps: int = 500,
    add_hs: bool = True,
    force_field_name: str = "MMFF94s",
    return_list: bool = False,
):
    assert force_field_name in [
        "MMFF94s",
        "MMFF94",
        "UFF",
    ], "Please select a valid force field name among [MMFF94s, MMFF94, UFF]"

    strain_energies = []
    for mol in gen_ligs:
        new_mol = Mol(mol)
        strain_energy = calc_strain(
            new_mol, add_hs=add_hs, n_steps=n_steps, force_field_name=force_field_name
        )
        strain_energies.append(strain_energy)

    if return_list:
        return strain_energies
    else:
        return {
            "Strain energy (mean)": np.nanmean(strain_energies),
            "Strain energy (std)": np.nanstd(strain_energies),
        }


def evaluate_clashes(
    gen_ligs: list[Chem.Mol],
    pdb_file: str,
    tolerance: float = 0.5,
    add_pocket_hs: bool = False,
    add_ligand_hs: bool = False,
    return_list: bool = False,
):
    """
    Counts the number of clashes between atoms in a protein and a ligand.

    Args:
        prot: RDKit Mol object representing the protein.
        lig: RDKit Mol object representing the ligand.
        tolerance: Distance tolerance for clash detection (default: 0.5).

    Returns:
        clashes: Number of clashes between the protein and the ligand.
    """

    if add_ligand_hs:
        gen_ligs = [Chem.AddHs(mol, addCoords=True) for mol in gen_ligs]
    # prot = load_protein_from_pdb(pdb_file, add_hs=add_pocket_hs)
    prot = Chem.MolFromPDBFile(pdb_file, removeHs=False, proximityBonding=False)
    if prot is None:
        tmpdir = tempfile.mkdtemp()
        receptor = PandasPdb().read_pdb(pdb_file)
        pdb_file = os.path.join(tmpdir, "receptor.pdb")
        receptor.to_pdb(pdb_file)
        prot = Chem.MolFromPDBFile(pdb_file, removeHs=False)
        shutil.rmtree(tmpdir)
        if prot is None:
            print("Could not read protein file. Skipping...")
            return [] if return_list else {}
    clashes = []
    for lig in gen_ligs:
        count = 0
        # Check if the molecule has radicals
        assert not smolRD.has_radicals(
            lig
        ), "Molecule has radicals, consider removing them first. (`posecheck.utils.chem.remove_radicals()`)"
        # try:
        # Get the positions of atoms in the protein and ligand
        prot_pos = prot.GetConformer().GetPositions()
        lig_pos = lig.GetConformer().GetPositions()

        pt = Chem.GetPeriodicTable()

        # Get the number of atoms in the protein and ligand
        num_prot_atoms = prot.GetNumAtoms()
        num_lig_atoms = lig.GetNumAtoms()

        # Calculate the Euclidean distances between all atom pairs in the protein and ligand
        dists = np.linalg.norm(
            prot_pos[:, np.newaxis, :] - lig_pos[np.newaxis, :, :], axis=-1
        )

        # Iterate over the ligand atoms
        for i in range(num_lig_atoms):
            lig_vdw = pt.GetRvdw(lig.GetAtomWithIdx(i).GetAtomicNum())

            # Iterate over the protein atoms
            for j in range(num_prot_atoms):
                prot_vdw = pt.GetRvdw(prot.GetAtomWithIdx(j).GetAtomicNum())

                # Check for clash by comparing the distances with tolerance
                if dists[j, i] + tolerance < lig_vdw + prot_vdw:
                    count += 1
        clashes.append(count)
        # except AttributeError:
        #     print(
        #         "Invalid input molecules. Please provide valid RDKit Mol objects."
        #     )
        #     clashes.append(np.nan)

    if return_list:
        return clashes
    else:
        return {
            "Clashes (mean)": np.nanmean(clashes),
            "Clashes (std)": np.nanstd(clashes),
        }


def evaluate_statistics(gen_ligs, dataset_info, ref_mols, state="test", verbose=False):
    if isinstance(gen_ligs[0], list):
        molecules = list(
            itertools.chain(
                *[
                    [Molecule(mol, device="cpu") for mol in mols if mol is not None]
                    for mols in gen_ligs
                ]
            )
        )
    else:
        molecules = [Molecule(mol, device="cpu") for mol in gen_ligs if mol is not None]
    stat = dataset_info.statistics[state]
    atom_encoder = dataset_info.atom_encoder
    atom_decoder = dataset_info.atom_decoder

    # Calculate statistics
    num_nodes_w1 = number_nodes_distance(molecules, stat.num_nodes)
    atom_types_tv, atom_tv_per_class = atom_types_distance(
        molecules, stat.atom_types, save_histogram=False
    )
    edge_types_tv, bond_tv_per_class, sparsity_level = bond_types_distance(
        molecules, stat.bond_types, save_histogram=False
    )
    valency_w1, valency_w1_per_class = valency_distance(
        molecules, stat.valencies, stat.atom_types, atom_encoder
    )
    bond_lengths_w1, bond_lengths_w1_per_type = bond_length_distance(
        molecules, stat.bond_lengths, stat.bond_types
    )
    angles_w1, angles_w1_per_type = angle_distance(
        molecules,
        stat.bond_angles,
        stat.atom_types,
        stat.valencies,
        atom_decoder=atom_decoder,
        save_histogram=False,
    )
    dihedrals_cw1 = calc_circular_wasserstein_distance(gen_ligs, ref_mols)

    if verbose:
        print(f"Number of nodes W1: {num_nodes_w1}")
        print(f"Edge type TV: {edge_types_tv}")
        print(f"Valency W1: {valency_w1}")
        print(f"Bond lengths W1: {bond_lengths_w1}")
        print(f"Angles W1: {angles_w1}")
        print(f"Dihedrals cW1: {dihedrals_cw1}")

    return {
        "NumNodesW1": num_nodes_w1,
        "AtomTypesTV": atom_types_tv,
        "EdgeTypesTV": edge_types_tv,
        "ValencyW1": valency_w1,
        "BondLengthsW1": bond_lengths_w1,
        "AnglesW1": angles_w1,
        "DihedralscW1": dihedrals_cw1,
    }


def interaction_recovery_per_complex(
    gen_ligs: list[Chem.Mol],
    native_lig: Chem.Mol,
    pdb_file: str,
    add_optimize_gen_lig_hs: bool = True,
    add_optimize_ref_lig_hs: bool = False,
    optimize_pocket_hs: bool = False,
    process_pocket: bool = False,
    optimization_method: str = "prolif_mmff",
    pocket_cutoff: float = 6.0,
    strip_invalid: bool = True,
    save_dir: str = None,
    return_list: bool = False,
):

    interaction_fingerprint = InteractionFingerprints()

    complex_id = Path(pdb_file).stem
    gen_lig_mol, native_lig_mol, pocket_mol = prepare_complex_data(
        gen_ligs,
        native_lig,
        pdb_file,
        add_optimize_gen_lig_hs=add_optimize_gen_lig_hs,
        add_optimize_ref_lig_hs=add_optimize_ref_lig_hs,
        optimize_pocket_hs=optimize_pocket_hs,
        process_pocket=process_pocket,
        optimization_method=optimization_method,
        pocket_cutoff=pocket_cutoff,
        strip_invalid=strip_invalid,
    )

    # Calculate the interaction recovery
    plif_recovery, tanimoto_sim = _get_plif_recovery(
        gen_lig_mol,
        native_lig_mol,
        pocket_mol,
        save_dir=save_dir,
        complex_id=complex_id,
        interaction_fingerprint=interaction_fingerprint,
        optimized_ground_truth=add_optimize_ref_lig_hs,
        return_list=return_list,
    )
    return plif_recovery, tanimoto_sim


def evaluate_interaction_recovery(
    gen_ligs: list[list[Chem.Mol]],
    native_ligs: list[Chem.Mol],
    pdb_files: list[str],
    add_optimize_gen_lig_hs: bool = True,
    add_optimize_ref_lig_hs: bool = False,
    optimize_pocket_hs: bool = False,
    process_pocket: bool = False,
    optimization_method: str = "prolif_mmff",
    pocket_cutoff: float = 6.0,
    strip_invalid: bool = True,
    save_dir: str = None,
    return_list: bool = False,
):
    """
    Calculate the interaction recovery rate for a given target
    gen_ligs: list of list RDKit molecules of N (sampled) valid ligands for each target
    native_ligs: list of sdf files of native ligands for each target - NOTE: the ligands need to have hydrogens!
    pdb_files: list of PDB files for each target - NOTE: the pocket needs to have hydrogens!
    """

    assert len(gen_ligs) == len(native_ligs) == len(pdb_files)
    assert not optimize_pocket_hs, "Pocket optimization is not supported yet."

    total = 0
    failed = 0
    recovery_rates = defaultdict(list) if return_list else []
    tanimoto_sims = defaultdict(list) if return_list else []
    for gen_lig, native_lig, pdb_file in tqdm(
        zip(gen_ligs, native_ligs, pdb_files),
        total=len(gen_ligs),
        desc="PLIF recovery calculation...",
    ):
        plif_recovery, tanimoto_sim = interaction_recovery_per_complex(
            gen_lig,
            native_lig,
            pdb_file,
            add_optimize_gen_lig_hs=add_optimize_gen_lig_hs,
            add_optimize_ref_lig_hs=add_optimize_ref_lig_hs,
            optimize_pocket_hs=optimize_pocket_hs,
            process_pocket=process_pocket,
            optimization_method=optimization_method,
            pocket_cutoff=pocket_cutoff,
            strip_invalid=strip_invalid,
            save_dir=save_dir,
            return_list=return_list,
        )
        if plif_recovery is not None and tanimoto_sim is not None:
            if return_list:
                assert isinstance(plif_recovery, list)
                recovery_rates[Path(pdb_file).stem].extend(plif_recovery)
                tanimoto_sims[Path(pdb_file).stem].extend(tanimoto_sim)
            else:
                recovery_rates.append(plif_recovery)
                tanimoto_sims.append(tanimoto_sim)
            total += len(gen_lig)
        else:
            failed += 1

    if return_list:
        return {"Recovery rate": recovery_rates, "Tanimoto similarity": tanimoto_sims}

    return {
        "PLIF recovery (mean)": np.nanmean(recovery_rates),
        "PLIF recovery (std)": np.nanstd(recovery_rates),
        "PLIF Tanimoto similarity (mean)": np.nanmean(tanimoto_sims),
        "PLIF Tanimoto similarity (std)": np.nanstd(tanimoto_sims),
        "Number of tested molecules": int(total),
        "Number of failed molecules": int(failed),
    }


def _get_plif_recovery(
    pred_ligand_mol: list[plf.Molecule],
    native_ligand_mol: plf.Molecule,
    pocket_mol: plf.Molecule,
    interaction_fingerprint: InteractionFingerprints,
    complex_id: str,
    save_dir: str,
    optimized_ground_truth: bool = False,
    return_list: bool = False,
) -> float:
    """
    Calculate the interaction recovery rate for a given target
    pred_ligand_mol: (list of) prolif molecule(s) of the predicted ligand(s)
    native_ligand_mol: prolif molecule of the native ligand
    pocket_mol: prolif molecule of the protein pocket

    NOTE: recovery rate can become None if the native plif is empty
    """
    # Load or calculate the native plifs
    if optimized_ground_truth:
        plifs_pkl_path = Path(save_dir) / "plifs_optimized"
    else:
        plifs_pkl_path = Path(save_dir) / "plifs"
    if not os.path.exists(plifs_pkl_path):
        os.makedirs(plifs_pkl_path, exist_ok=True)
    plifs_pkl_file = plifs_pkl_path / f"{complex_id}_native_plifs.pkl"
    if plifs_pkl_file.is_file():
        with open(plifs_pkl_file, "rb") as f:
            native_plif = pkl.load(f)
    else:
        native_plif = interaction_fingerprint(native_ligand_mol, pocket_mol)
        with open(plifs_pkl_file, "wb") as f:
            pkl.dump(native_plif, f)

    gen_plif = interaction_fingerprint(pred_ligand_mol, pocket_mol)

    # Get recovery rate
    if isinstance(pred_ligand_mol, list):
        plif_result = [
            interaction_fingerprint.get_plif_recovery_rates(
                true_fp=native_plif,
                pred_fp=gen_plif,
                ifp_idx=i,
                recovery_type="recovery_rate",
            )
            for i in range(len(pred_ligand_mol))
        ]
        recovery_rate = [r.count_recovery for r in plif_result if r is not None]
        if not return_list:
            recovery_rate = np.mean(recovery_rate) if len(recovery_rate) > 0 else None
    else:
        plif_result = interaction_fingerprint.get_plif_recovery_rates(
            true_fp=native_plif,
            pred_fp=gen_plif,
            recovery_type="recovery_rate",
        )
        recovery_rate = plif_result.count_recovery if plif_result is not None else None
        if return_list:
            recovery_rate = [recovery_rate]

    # Get Tanimoto similarity
    tanimoto_sim = None  # interaction_fingerprint.get_plif_recovery_rates(
    #     true_fp=native_plif,
    #     pred_fp=gen_plif,
    #     recovery_type="similarity",
    # )
    if tanimoto_sim is not None:
        tanimoto_sim = [sim for sim in tanimoto_sim if sim is not None]
    if not return_list:
        tanimoto_sim = (
            np.mean(tanimoto_sim)
            if tanimoto_sim is not None and len(tanimoto_sim) > 0
            else None
        )

    return recovery_rate, tanimoto_sim


def calc_strain(
    mol: Chem.Mol,
    add_hs: bool = True,
    n_steps: int = 1000,
    force_field_name: str = "MMFF94s",
):
    if add_hs:
        mol = Chem.AddHs(mol, addCoords=True)

    try:
        if force_field_name == "MMFF94s":
            mol_properties = AllChem.MMFFGetMoleculeProperties(
                mol, mmffVariant="MMFF94s"
            )
            force_field = AllChem.MMFFGetMoleculeForceField(
                mol, mol_properties, confId=0
            )
        elif force_field_name == "MMFF94":
            mol_properties = AllChem.MMFFGetMoleculeProperties(mol)
            force_field = AllChem.MMFFGetMoleculeForceField(
                mol, mol_properties, confId=0
            )
        elif force_field_name == "UFF":
            force_field = AllChem.UFFGetMoleculeForceField(mol, confId=0)

        start_energy = force_field.CalcEnergy()
        not_converged = force_field.Minimize(maxIts=n_steps)
        if not_converged:
            print(
                "Energy minimization did not converge - using intermediate state for strain calculation"
            )
        final_energy = force_field.CalcEnergy()
        strain_energy = start_energy - final_energy
        assert strain_energy > 0, "Strain energy should be positive"
        return strain_energy
    except Exception:
        print("Force field error - skipping strain calculation")
        return np.nan


def calc_dihedrals(molecules):
    """
    Calculates dihedral angles (in degrees) for all rotatable bonds in a list of RDKit molecules.
    Assumes each molecule has at least one conformer with valid 3D coordinates.
    """
    dihedrals = []
    for mol in molecules:
        try:
            conf = mol.GetConformer()
        except ValueError:
            continue
        for bond in mol.GetBonds():
            if bond.IsInRing():
                continue
            if bond.GetBondType() != rdchem.BondType.SINGLE:
                continue

            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()

            if begin_atom.GetDegree() < 2 or end_atom.GetDegree() < 2:
                continue

            begin_neighbors = [
                nbr.GetIdx()
                for nbr in begin_atom.GetNeighbors()
                if nbr.GetIdx() != end_atom.GetIdx()
            ]
            end_neighbors = [
                nbr.GetIdx()
                for nbr in end_atom.GetNeighbors()
                if nbr.GetIdx() != begin_atom.GetIdx()
            ]

            if not begin_neighbors or not end_neighbors:
                continue

            idx1 = begin_neighbors[0]
            idx2 = begin_atom.GetIdx()
            idx3 = end_atom.GetIdx()
            idx4 = end_neighbors[0]

            try:
                angle = rdMolTransforms.GetDihedralDeg(conf, idx1, idx2, idx3, idx4)
                dihedrals.append(angle)
            except Exception:
                continue
    return dihedrals


def calc_wasserstein_distance(molecules1, molecules2):
    """
    Calculates the Wasserstein distance between two sets of dihedral angles (raw values)
    computed from two lists of RDKit molecules.

    Args:
        molecules1 (list): List of RDKit molecule objects for distribution 1.
        molecules2 (list): List of RDKit molecule objects for distribution 2.

    Returns:
        float: The Wasserstein distance between the two distributions of dihedral angles.
    """
    angles1 = calc_dihedrals(molecules1)
    angles2 = calc_dihedrals(molecules2)

    # Compute the 1D Wasserstein distance using SciPy
    return wasserstein_distance(angles1, angles2)


def calc_circular_wasserstein_distance(molecules1, molecules2, nbins=36):
    """
    Calculates the circular Wasserstein distance between two dihedral angle distributions.

    The method bins the dihedral angles over the range [-180, 180] and computes the
    cumulative distribution functions (CDFs) for both. The optimal alignment over the circle
    is achieved by subtracting the median difference between the CDFs. The distance is then
    computed as the sum of absolute differences (multiplied by the bin width).

    Args:
        molecules1 (list): List of RDKit molecule objects for distribution 1.
        molecules2 (list): List of RDKit molecule objects for distribution 2.
        nbins (int): Number of bins to use for the histograms (default: 36, i.e., 10° per bin).

    Returns:
        float: The circular Wasserstein distance between the two distributions.
    """
    # Compute dihedral angles for each set of molecules
    angles1 = np.array(calc_dihedrals(molecules1))
    angles2 = np.array(calc_dihedrals(molecules2))

    # Define histogram bins for the periodic angles ([-180, 180])
    bins = np.linspace(-180, 180, nbins + 1)
    bin_width = bins[1] - bins[0]

    # Compute histograms (counts)
    hist1, _ = np.histogram(angles1, bins=bins)
    hist2, _ = np.histogram(angles2, bins=bins)

    # Normalize histograms to obtain probability mass functions.
    if hist1.sum() > 0:
        p1 = hist1.astype(float) / hist1.sum()
    else:
        p1 = np.zeros_like(hist1)

    if hist2.sum() > 0:
        p2 = hist2.astype(float) / hist2.sum()
    else:
        p2 = np.zeros_like(hist2)

    # Compute cumulative distributions (CDFs)
    cdf1 = np.cumsum(p1)
    cdf2 = np.cumsum(p2)

    # Calculate the difference between the CDFs
    diff = cdf1 - cdf2
    # The optimal circular alignment is found by subtracting the median difference
    shift = np.median(diff)
    # Circular Wasserstein distance is computed as the total absolute deviation, scaled by bin width
    distance = np.sum(np.abs(diff - shift)) * bin_width
    return distance

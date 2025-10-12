import tempfile
from pathlib import Path
from typing import NamedTuple, TypeAlias, Union

import numpy as np
import pandas as pd
import prolif as plf
import torch
from Bio import pairwise2
from MDAnalysis.lib.util import inverse_aa_codes
from rdkit import Chem, DataStructs

from flowr.util.functional import (
    LigandPocketOptimization,
    add_and_optimize_hs,
    ligand_from_mol,
    prepare_prolif_mols,
)

_T = torch.Tensor
TupleRot = tuple[float, float, float]
IFPType: TypeAlias = dict[int, dict[tuple[plf.ResidueId, plf.ResidueId], dict]]
THREE_TO_ONE = inverse_aa_codes.copy()
THREE_TO_ONE.update({"HIP": "H"})


class PlifResults(NamedTuple):
    count_recovery: float | None = None
    true_plif: str = ""
    plif: str = ""


class InteractionFingerprints:
    INTERACTION_PARAMETERS = {
        "HBAcceptor": {
            "distance": 3.7,
            "donor": "[$([O,S,#7;+0]),$([N+1])]-[H]",
        },
        "HBDonor": {"distance": 3.7},
        "CationPi": {"distance": 5.5},
        "PiCation": {"distance": 5.5},
        "Anionic": {"distance": 5},
        "Cationic": {"distance": 5},
    }
    PROLIF_INTERACTIONS = [
        # "Hydrophobic",
        # "VdWContact",
        # "MetalAcceptor",
        # "MetalDonor",
        "Cationic",
        "Anionic",
        "XBAcceptor",
        "XBDonor",
        "CationPi",
        "PiCation",
        "PiStacking",
        "HBAcceptor",
        "HBDonor",
    ]

    def __init__(self, interaction_types: list[str] = None):
        self.interaction_types = (
            self.PROLIF_INTERACTIONS if interaction_types is None else interaction_types
        )

    def __call__(
        self,
        ligand_mol: list[plf.Molecule],
        pocket_mol: plf.Molecule,
        return_array: bool = False,
    ):

        if isinstance(pocket_mol, list):
            raise NotImplementedError("Multiple pockets not supported yet")
        plf_fp = self.interaction_fp(ligand_mol, pocket_mol=pocket_mol)
        if return_array:
            if isinstance(ligand_mol, list):
                raise NotImplementedError("Multiple ligands not supported yet")
                interaction_arr = [
                    self.interaction_array(
                        plf_fp,
                        n_pocket=pocket_mol.GetNumAtoms(),
                        n_ligand=lig.GetNumAtoms(),
                        ifp_idx=i,
                    )
                    for i, lig in enumerate(ligand_mol)
                ]
                return plf_fp, interaction_arr
            else:
                interaction_arr = self.interaction_array(
                    plf_fp,
                    n_pocket=pocket_mol.GetNumAtoms(),
                    n_ligand=ligand_mol.GetNumAtoms(),
                )
                return plf_fp, interaction_arr
        else:
            return plf_fp

    def interaction_array(
        self,
        plf_fp: plf.Fingerprint,
        n_pocket: int,
        n_lig: int,
        ifp_idx: int = 0,
    ) -> Union[np.ndarray, tuple[np.ndarray, plf.Fingerprint]]:
        """Build an atom-level interaction array, array shape [n_atoms_ligand, n_atoms_pocket, n_int_types]"""

        plf_interaction_map = {
            int_type: idx for idx, int_type in enumerate(self.interaction_types)
        }
        atom_interactions = self._interactions_from_ifp(plf_fp.ifp[ifp_idx])

        # Create a bit vector for each protein-ligand atom pair and set the bits for the given interactions
        arr_shape = (n_pocket, n_lig, len(self.interaction_types))
        interaction_arr = np.zeros(arr_shape, dtype=np.int8)

        for int_type, p_idx, l_idx in atom_interactions:
            int_idx = plf_interaction_map[int_type]
            interaction_arr[p_idx, l_idx, int_idx] = 1

        return interaction_arr

    def interaction_fp(
        self,
        ligand_mol: list[plf.Molecule],
        pocket_mol: plf.Molecule,
    ) -> plf.Fingerprint:
        """Run the prolif interaction algorithm and retrun the prolif IFP object"""

        plf_fp = plf.Fingerprint(
            interactions=self.interaction_types,
            parameters=self.INTERACTION_PARAMETERS,
            count=True,
        )
        if not isinstance(ligand_mol, list):
            ligand_mol = [ligand_mol]

        plf_fp.run_from_iterable(ligand_mol, pocket_mol, residues="all", progress=False)
        return plf_fp

    def _interactions_from_ifp(self, ifp):

        interaction_list = []
        for _, res_interactions in ifp.items():
            for int_type, interactions in res_interactions.items():
                for interaction in interactions:
                    l_atom_idxs = interaction["parent_indices"]["ligand"]
                    p_atom_idxs = interaction["parent_indices"]["protein"]
                    int_tuples = [
                        (int_type, p_idx, l_idx)
                        for p_idx in p_atom_idxs
                        for l_idx in l_atom_idxs
                    ]
                    interaction_list.extend(int_tuples)

        return interaction_list

    def get_plif_recovery_rates(
        self,
        true_fp: plf.Fingerprint,
        pred_fp: plf.Fingerprint,
        recovery_type: str = "recovery_rate",
        ifp_idx: int = 0,
    ):
        # no interaction
        if not true_fp.ifp[0]:
            print("No interactions found in native complex. Skipping...")
            return

        if recovery_type == "similarity":
            df_pred = pred_fp.to_dataframe(index_col="Pose")
            df_ref = true_fp.to_dataframe(index_col="Pose")
            df_ref.rename(index={0: -1}, inplace=True)
            df_ref.rename(columns={"UNL1": df_pred.columns.levels[0][0]}, inplace=True)
            df_ref_poses = (
                pd.concat([df_ref, df_pred])
                .fillna(False)
                .sort_index(
                    axis=1,
                    level=1,
                    key=lambda index: [plf.ResidueId.from_string(x) for x in index],
                )
            )
            bitvectors = plf.to_bitvectors(df_ref_poses)
            tanimoto_sims = DataStructs.BulkTanimotoSimilarity(
                bitvectors[0], bitvectors[1:]
            )
            return tanimoto_sims

        elif recovery_type == "recovery_rate":
            if not pred_fp.ifp[ifp_idx]:
                return PlifResults(0, fp_to_str(true_fp.to_dataframe()))

            true_df = plf.to_dataframe(true_fp.ifp, true_fp.interactions, count=True)
            true_counts = true_df.droplevel("ligand", axis=1).to_dict("records")[0]
            total_true_count = sum(true_counts.values())
            true_plifs = fp_to_str(true_df)

            pred_df = plf.to_dataframe(pred_fp.ifp, pred_fp.interactions, count=True)
            pred_counts = pred_df.droplevel("ligand", axis=1).to_dict("records")[
                ifp_idx
            ]
            count_recovery = (
                sum([min(true_counts[k], pred_counts.get(k, 0)) for k in true_counts])
                / total_true_count
            )
            pred_plifs = fp_to_str(pred_df)
            return PlifResults(count_recovery, true_plifs, pred_plifs)
        else:
            raise ValueError("Invalid recovery type")


def fp_to_str(ifp_df: pd.DataFrame) -> str:
    return "/".join(
        [
            f"{'_'.join(key)}={count}"
            for key, count in ifp_df.droplevel("ligand", axis=1)
            .to_dict("records")[0]
            .items()
        ]
    )


def map_fingerprints(
    fp_a: plf.Fingerprint,
    fp_b: plf.Fingerprint,
    pocket_a: plf.Molecule,
    pocket_b: plf.Molecule,
) -> tuple[IFPType, IFPType]:
    """
    Adjusts the fingerprints protein residues to use the same numbering based on a
    sequence alignment (in case the number of residues is not the same between the
    different methods).
    """
    # get sequence and residue ids of the chain(s) corresponding to the pocket
    seq_a, res_ids_a = get_sequence(pocket_a)
    seq_b, res_ids_b = get_sequence(pocket_b)

    # map
    map_a, map_b = get_common_map(seq_a, seq_b, res_ids_a, res_ids_b)

    ifp_a = adjust_fingerprint_residues(fp_a, map_a)
    ifp_b = adjust_fingerprint_residues(fp_b, map_b)
    return ifp_a, ifp_b


def get_sequence(pocket: plf.Molecule) -> tuple[str, list[str]]:
    """
    Extracts sequence from chains that are found in the residues from the pocket.
    This avoids mapping to the wrong chain in homo-oligomers.
    """
    pocket_chain = {resid.chain for resid in pocket.residues}

    if pocket_chain:

        def chain_predicate(resid: plf.ResidueId) -> bool:
            return resid.chain in pocket_chain

    else:

        def chain_predicate(resid: plf.ResidueId) -> bool:
            return True

    with tempfile.NamedTemporaryFile(suffix=".pdb") as tmp:
        pdb_file = tmp.name
        pdb_writer = Chem.PDBWriter(pdb_file)
        pdb_writer.write(pocket)
        pdb_writer.close()
        pocket = Chem.MolFromPDBFile(pdb_file, sanitize=False, proximityBonding=False)
    residues = list(
        dict.fromkeys(
            resid
            for atom in pocket.GetAtoms()
            if chain_predicate(resid := plf.ResidueId.from_atom(atom))
        )
    )
    sequence = "".join(THREE_TO_ONE.get(resid.name, "X") for resid in residues)
    res_ids = [f"{resid.number}.{resid.chain}" for resid in residues]
    return sequence, res_ids


def get_common_map(
    seq_a: str, seq_b: str, res_ids_a: list[str], res_ids_b: list[str]
) -> tuple[dict[str, int], dict[str, int]]:
    """Maps 2 sequences to the numbering used in sequence A."""
    map_a = {}
    map_b = {}

    # pick alignment where dashes are as far as possible from the start
    # (avoids alignments with gaps in the middle)
    alignment = max(
        pairwise2.align.globalxs(seq_a, seq_b, -0.5, -0.1),
        key=lambda a: sum((a.seqA.find("-"), a.seqB.find("-"))),
    )

    running_idx_a = 0
    running_idx_b = 0
    for i, (align_a, align_b) in enumerate(zip(alignment.seqA, alignment.seqB)):
        if align_a == align_b:
            map_a[res_ids_a[running_idx_a]] = i
            map_b[res_ids_b[running_idx_b]] = i
        if align_a != "-":
            running_idx_a += 1
        if align_b != "-":
            running_idx_b += 1
    return map_a, map_b


def adjust_fingerprint_residues(fp: plf.Fingerprint, mapper: dict[str, int]) -> IFPType:
    """Replaces protein residues in the fingerprint based on the mapping."""

    def adjust_resid(res: plf.ResidueId) -> plf.ResidueId:
        try:
            number = mapper[f"{res.number}.{res.chain}"]
            chain = "Z"
        except KeyError:
            number = res.number
            chain = res.chain
        return plf.ResidueId(THREE_TO_ONE.get(res.name, "X"), number, chain)

    return {
        0: {(lres, adjust_resid(pres)): ifp for (lres, pres), ifp in fp.ifp[0].items()}
    }


def get_interaction_fp_per_complex(
    gen_ligs: list[Chem.Mol],
    pdb_file: str,
    add_optimize_lig_hs: bool = True,
    optimize_pocket_hs: bool = False,
    process_pocket: bool = False,
    optimization_method: str = "prolif_mmff",
    pocket_cutoff: float = 6.0,
    strip_invalid: bool = True,
):
    """
    Get the interactions between provided ligand(s) and the pocket of the provided complex.
    """
    if add_and_optimize_hs or process_pocket:
        optimizer = LigandPocketOptimization(
            pocket_cutoff=pocket_cutoff, strip_invalid=strip_invalid
        )
    complex_id = Path(pdb_file).stem
    if add_optimize_lig_hs:
        if isinstance(gen_ligs, Chem.Mol):
            gen_lig_mol = add_and_optimize_hs(
                gen_ligs,
                pdb_file,
                optimizer=optimizer,
                optimize_pocket_hs=optimize_pocket_hs,
                process_pocket=process_pocket,
            )
            if gen_lig_mol is None:
                print(
                    f"Could not find any ligand that could be optimized for complex: {complex_id}"
                )
                return
        elif isinstance(gen_ligs, list):
            gen_lig_mol = [
                add_and_optimize_hs(
                    lig,
                    pdb_file,
                    optimizer=optimizer,
                    optimize_pocket_hs=optimize_pocket_hs,
                    process_pocket=process_pocket,
                )
                for lig in gen_ligs
            ]
            gen_lig_mol = [lig for lig in gen_lig_mol if lig is not None]
            if len(gen_lig_mol) == 0:
                print(
                    f"Could not find any ligand that could be optimized for complex: {complex_id}"
                )
                return
        else:
            raise ValueError("Invalid ligand format")
    else:
        if isinstance(gen_ligs, Chem.Mol):
            gen_lig_mol = ligand_from_mol(gen_ligs, add_hydrogens=False)
            if gen_lig_mol is None:
                print("Provided ligand could not be processed")
                return
        elif isinstance(gen_ligs, list):
            gen_lig_mol = [
                ligand_from_mol(lig, add_hydrogens=False) for lig in gen_ligs
            ]
            gen_lig_mol = [lig for lig in gen_lig_mol if lig is not None]
            if len(gen_lig_mol) == 0:
                print("None of the provided ligands could be processed")
                return
        else:
            raise ValueError("Invalid ligand format")

    if process_pocket:
        ref_mol = gen_lig_mol if isinstance(gen_lig_mol, Chem.Mol) else gen_lig_mol[0]
        pocket_mol = optimizer.pocket_from_pdb(
            pdb_file, ligand_mol=ref_mol, process_pocket=True
        )
    else:
        pocket_mol = prepare_prolif_mols(pdb_file=pdb_file)

    interaction_fingerprint = InteractionFingerprints()
    fp = interaction_fingerprint(gen_lig_mol, pocket_mol)
    return fp

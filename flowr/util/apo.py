from pathlib import Path

import biotite.structure as struc
import numpy as np
import scipy as scipy
from biotite.sequence import ProteinSequence
from biotite.sequence.align import SubstitutionMatrix, align_optimal
from plinder.core import PlinderSystem

from flowr.util.plinder import load_structure, run_prep_wizard_protein
from flowr.util.pocket import ProteinPocket

DEFAULT_MIN_APO_ATOMS = 50


# *********************************
# ***** Apo exception classes *****
# *********************************


class ApoNotFound(Exception):
    """Used when we cannot find a suitable apo structure"""

    pass


class ApoPocket(Exception):
    """Used when an apo structure is available but no matching pockets are found"""

    pass


# **************************************************
# ***** Helper functions and loading functions *****
# **************************************************


def compute_rmsd(
    struc1: struc.AtomArray, struc2: struc.AtomArray, backbone: bool = False
) -> float:
    if len(struc1) != len(struc2):
        raise ValueError("Both arrays must have the same number of atoms.")

    if backbone:
        struc1_mask = struc.filter_peptide_backbone(struc1)
        struc2_mask = struc.filter_peptide_backbone(struc2)
        struc1 = struc1[struc1_mask]
        struc2 = struc2[struc2_mask]

        if len(struc1) != len(struc2):
            raise ValueError(
                "Both arrays must have the same number of backbone atoms if computing backbone RMSD."
            )

    rmsd = struc.rmsd(struc1, struc2)
    return rmsd


def align_apo_pocket(
    apo_pocket: struc.AtomArray, holo_pocket: struc.AtomArray
) -> struc.AtomArray:
    """Translate and rotate the apo pocket based on best alignment between apo and holo backbones"""

    backbone_mask = struc.filter_peptide_backbone(holo_pocket)
    aligned_apo, _ = struc.superimpose(apo_pocket, holo_pocket, atom_mask=backbone_mask)
    return aligned_apo


def _build_residue_stack(structure):
    residues = []
    for res_id in list(set(structure.res_id)):
        res_arr = structure[structure.res_id == res_id]
        res_name = res_arr.res_name[0]
        res_centre = res_arr.coord.mean(axis=0)
        residue = struc.Atom(res_centre, res_id=res_id, res_name=res_name)
        residues.append(residue)

    return struc.array(residues)


def load_apo_structure(
    system, apo_id, apo_type, model=1, include_hs=False, include_hetero=False
):
    apo_path = system.get_linked_structure(apo_type, apo_id)

    # Sometimes this randomly doesn't return anything, in this case just run it again
    if apo_path is None or apo_path == "":
        apo_path = system.get_linked_structure(apo_type, apo_id)

    linked_id = str(Path(apo_path).stem)
    chain_id = linked_id.split("_")[-1]
    file_type = str(Path(apo_path).suffix)[1:]

    apo_struct = load_structure(
        apo_path,
        chain_id,
        model=model,
        include_hs=include_hs,
        include_hetero=include_hetero,
        file_type=file_type,
    )

    if not include_hs:
        apo_struct = apo_struct[apo_struct.element != "H"]

    return apo_struct


# def load_apo_candidates(system):
#     """Produce a list of apo candidate structures in priority order"""

#     # sort_score corresponds to resolution for apo and plddt for pred
#     # Invert pred scores so that we can sort both apo and pred in the same direction
#     linked = system.linked_structures
#     pred_scores = linked[linked["kind"] == "pred"]["sort_score"]
#     linked.loc[pred_scores.index, "sort_score"] = 100 - pred_scores

#     # Sort the possible apo structures in priority order
#     # kind is either 'apo' or 'pred' so sort ascending since we prefer experimental structures
#     sort_by = ["kind", "pocket_fident", "pocket_lddt", "sort_score"]
#     ascending = [True, False, False, True]
#     linked = linked.sort_values(by=sort_by, ascending=ascending)

#     apo_candidates = []
#     for apo_id, apo_type in zip(linked["id"], linked["kind"]):
#         apo_struct = load_apo_structure(system, apo_id, apo_type, include_hs=False)
#         if len(apo_struct) >= MIN_APO_ATOMS:
#             apo_candidates.append((apo_type, apo_struct))

#     return apo_candidates


def _load_apo_candidates(system, min_n_atoms=DEFAULT_MIN_APO_ATOMS):
    # Just take AF predicted structures from plinder for now
    linked = system.linked_structures
    linked = linked[linked["kind"] == "pred"]

    # Sort the possible apo structures in priority order
    sort_by = ["pocket_fident", "pocket_lddt"]
    linked = linked.sort_values(by=sort_by, ascending=[False, False])

    apo_candidates = []
    for apo_id, apo_type in zip(linked["id"], linked["kind"]):
        apo_struct = load_apo_structure(system, apo_id, apo_type, include_hs=False)

        # Some plinder linked structures are empty, so make sure the structure is reasonable
        if len(apo_struct) >= min_n_atoms:
            apo_candidates.append((apo_type, apo_struct))

    return apo_candidates


# ***************************************
# ***** Sequence matching functions *****
# ***************************************


def _align_seqs(
    holo_res_names: list[str], apo_res_names: list[str], holo_pocket_idxs: list[int]
) -> list[list[int]]:
    """Performs a sequence alignment between full holo and apo structures and apo pocket candidates.

    Firstly finds all sequence optimal alignments between apo and holo sequences and returns the indices into the apo
    residue list of all apo pocket candidates where the residue type exactly match the holo pocket residue types.
    """

    holo_seq = ProteinSequence(holo_res_names)
    apo_seq = ProteinSequence(apo_res_names)
    holo_pocket_res_names = [holo_res_names[idx] for idx in holo_pocket_idxs]

    # TODO maybe other subs matrices work better?
    matrix = SubstitutionMatrix.std_protein_matrix()
    alignments = align_optimal(holo_seq, apo_seq, matrix)

    apo_pocket_candidate_idxs = []
    for alignment in alignments:
        # Exclude alignments where a holo pocket residue does not get mapped
        holo_apo_idx_map = {
            h_idx: a_idx for h_idx, a_idx in alignment.trace.tolist() if h_idx != -1
        }
        res_not_mapped = [idx not in holo_apo_idx_map for idx in holo_pocket_idxs]
        if any(res_not_mapped):
            continue

        # Exclude alignments where a holo pocket residue doesn't get mapped to an apo residue
        apo_pocket_idxs = [holo_apo_idx_map[idx] for idx in holo_pocket_idxs]
        if -1 in apo_pocket_idxs:
            continue

        apo_pocket_res_names = [apo_res_names[idx] for idx in apo_pocket_idxs]
        res_match = [
            h_res == a_res
            for h_res, a_res in zip(holo_pocket_res_names, apo_pocket_res_names)
        ]
        if all(res_match):
            apo_pocket_candidate_idxs.append(apo_pocket_idxs)

    return apo_pocket_candidate_idxs


def _sequence_matching(
    apo_structure: struc.AtomArray,
    holo_structure: struc.AtomArray,
    holo_pocket_res_ids: list[int],
) -> list[int]:
    """Find best apo pocket residue ids by matching apo and holo sequences"""

    holo_res_ids, holo_res_names = struc.get_residues(holo_structure)
    apo_res_ids, apo_res_names = struc.get_residues(apo_structure)

    holo_pocket_struc = holo_structure[
        np.isin(holo_structure.res_id, holo_pocket_res_ids)
    ]
    holo_id_idx_map = {res_id: idx for idx, res_id in enumerate(holo_res_ids)}
    holo_pocket_idxs = [holo_id_idx_map[res_id] for res_id in holo_pocket_res_ids]

    apo_candidate_idxs = _align_seqs(holo_res_names, apo_res_names, holo_pocket_idxs)

    alignment_rmsds = []
    apo_candidate_res_ids = []

    if len(apo_candidate_idxs) == 0:
        print("No successful alignments.")
        return None

    # Since the whole apo and holo structures should already be aligned we choose the best alignment by finding the
    # minimum RMSD between backbone atoms the holo pocket and the apo candidate pockets
    for apo_pocket_idxs in apo_candidate_idxs:
        apo_pocket_res_ids = [apo_res_ids[idx] for idx in apo_pocket_idxs]
        apo_pocket_struc = apo_structure[
            np.isin(apo_structure.res_id, apo_pocket_res_ids)
        ]
        apo_candidate_res_ids.append(apo_pocket_res_ids)

        # RMSD can fail if the number of atoms in the backbone is not the same
        # Catch an errors and try the next alignment

        # *** TODO align before calculating RMSD ***
        # RMSD doesnt make sense for AF structures since the alignment seems off.
        try:
            rmsd = compute_rmsd(holo_pocket_struc, apo_pocket_struc, backbone=True)
        except Exception as err:
            print(f"RMSD failed -- {type(err).__name__} -- {str(err)}")
            rmsd = None

        alignment_rmsds.append(rmsd)

    # Remove candidates which failed the RMSD calculation and take the minimum RMSD candidate
    apo_candidate_res_ids = [
        c
        for idx, c in enumerate(apo_candidate_res_ids)
        if alignment_rmsds[idx] is not None
    ]
    alignment_rmsds = [rmsd for rmsd in alignment_rmsds if rmsd is not None]

    if len(alignment_rmsds) == 0:
        return None

    opt_candidate_idx, _ = min(
        enumerate(alignment_rmsds), key=lambda idx_rmsd: idx_rmsd[1]
    )
    return apo_candidate_res_ids[opt_candidate_idx]


# *************************************************************
# ***** High-level functions for running the apo matching *****
# *************************************************************


def _find_apo_pocket(
    system_id: str,
    apo_candidates: list[tuple[str, struc.AtomArray]],
    holo_structure: struc.AtomArray,
    holo_pocket: struc.AtomArray,
    tmp_path: str,
    config_path: str,
    prepwizard: bool = False,
):
    """Takes a list of apo candidates and finds the first that is a good match with the holo.

    The matching algorithm produces match candidates in priority order and the first candidate which meets the
    matching criteria (matching lengths and matching atom names) is returned.
    """

    holo_pocket_without_hs = holo_pocket[holo_pocket.element != "H"]

    for apo_type, apo_struct in apo_candidates:
        try:
            # Note - this must be a list but not have any duplicates
            holo_res_ids = list(set(holo_pocket.res_id))
            apo_res_ids = _sequence_matching(apo_struct, holo_structure, holo_res_ids)
        except:
            continue

        if apo_res_ids is None:
            continue

        print(f"Got apo res ids for apo type {apo_type}!", apo_res_ids)
        apo_pocket_atoms = apo_struct[np.isin(apo_struct.res_id, apo_res_ids)]

        # Check that the number of atoms matches, otherwise continue to next candidate
        # Each apo should already have been loaded without Hs
        if len(apo_pocket_atoms) != len(holo_pocket_without_hs):
            print(
                f"Apo and holo length missmatch, apo {len(apo_pocket_atoms)} -- holo {len(holo_pocket)}"
            )
            continue

        # And check that all atom names match, otherwise continue to next candidate
        # TODO this could be too strict, could we soften this a bit?
        if not (apo_pocket_atoms.atom_name == holo_pocket_without_hs.atom_name).all():
            print("Unsuccesful apo and holo atom name match")
            continue

        print(f"Successfully matched apo and holo, apo type {apo_type}")

        if not prepwizard:
            return apo_pocket_atoms, apo_type

        print("Running prepwizard on apo structure...")
        apo_struct = run_prep_wizard_protein(
            system_id, apo_struct, tmp_path, config_path
        )
        apo_pocket_atoms = apo_struct[np.isin(apo_struct.res_id, apo_res_ids)]
        print("Done prepwizard.")

        # NOTE assumes prepwizard has also been applied to holo if being applied to apo, need explicit Hs in holo
        if len(apo_pocket_atoms) != len(holo_pocket):
            print("Length missmatch between apo and holo after applying prepwizard.")
            continue

        if not (apo_pocket_atoms.atom_name == holo_pocket.atom_name).all():
            print(
                "Unsuccessful apo and holo atom name match after applying prepwizard."
            )
            continue

        return apo_pocket_atoms, apo_type

    # If none of the apo candidates are successful then return None
    return None


def load_apo_pocket(
    system: PlinderSystem,
    holo_structure: struc.AtomArray,
    holo_pocket: struc.AtomArray,
    tmp_path: str,
    config_path: str,
    prepwizard: bool = False,
):
    """
    Load a list of apo candidate structures, apply a matching algorithm to find the best, extract the pocket
    and rotate align the apo and holo pockets based on the backbone atoms.
    """

    # Load apo candidates
    apo_candidates = _load_apo_candidates(system)
    if len(apo_candidates) == 0:
        raise ApoNotFound(
            f"No suitable apo candidate structures for system {system.system_id}"
        )

    # Find a pocket which best matches the holo pocket
    selected_apo_pocket = _find_apo_pocket(
        system.system_id,
        apo_candidates,
        holo_structure,
        holo_pocket,
        tmp_path,
        config_path,
        prepwizard=prepwizard,
    )

    if selected_apo_pocket is None:
        raise ApoPocket(
            f"No apo pockets were able to match the holo pocket for system {system.system_id}"
        )

    apo_pocket_atoms, apo_type = selected_apo_pocket
    apo_pocket_atoms = align_apo_pocket(apo_pocket_atoms, holo_pocket)
    apo_pocket = ProteinPocket.from_pocket_atoms(
        apo_pocket_atoms, infer_res_bonds=False
    )

    return apo_pocket, apo_type

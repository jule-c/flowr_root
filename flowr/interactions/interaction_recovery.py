import os
import tempfile
import warnings
from pathlib import Path

import prolif as plf
from rdkit import Chem
from tqdm import tqdm

from flowr.util.rdkit import write_sdf_file
from plif_utils.analysis import get_plif_recovery_rates, run
from plif_utils.file_prep import FilePrep, get_files
from plif_utils.system_prep import SystemPrep


def get_recovery_rates_given_target(
    target: str,
    ligands: list[Chem.Mol],
    system_prep: SystemPrep,
    data_dir: Path,
    ground_truth_plif: plf.Fingerprint,
    ground_truth_file_prep: FilePrep,
) -> dict[str, plf.Fingerprint]:
    """
    Given a target and a list of ligands, calculate the PLIFs for the target
    target: PDB file of the target
    ligands: list of RDKit molecules of N (sampled) ligands for the target
    system_prep: SystemPrep object
    data_dir: Path to the data directory
    return: ist of PLIF recovery rates
    """

    # loop over ligands given target and calculate plifs
    recovery_rates = []
    for ligand in tqdm(ligands, total=len(ligands), desc="Calculating PLIFs.."):
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            tmp_file = tmp.name
            write_sdf_file(tmp_file, [ligand])
            try:
                # get protein and ligand files
                file_prep = get_files(
                    target, Path(tmp_file), system_prep, data_dir=data_dir
                )
                assert os.path.exists(
                    file_prep.prepared_ligand_file
                ), "ligand file missing"
                assert os.path.exists(
                    file_prep.prepared_protein_file
                ), "protein file missing"
            except Exception:
                warnings.warn(f"Could not prepare system for target::{target.stem}")
                continue

            try:
                # construct plifs for this target
                _, _, sample_plif = run(
                    file_prep.prepared_ligand_file, file_prep.prepared_protein_file
                )
            except Exception:
                warnings.warn(f"Could not calculate PLIFs for target::{target}")
                continue

            # Get PLIF recovery rates
            sample_file_prep = get_files(target, Path(tmp_file), data_dir=data_dir)
            recovery_rates.append(
                get_plif_recovery_rates(
                    ground_truth_file_prep,
                    sample_file_prep,
                    ground_truth_plif,
                    sample_plif,
                ).count_recovery
            )

    return recovery_rates

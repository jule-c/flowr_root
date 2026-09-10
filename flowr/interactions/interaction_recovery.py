"""Legacy PLIF recovery helper built on the external ``plif_utils`` package.

``plif_utils`` is not distributed with FlowR. The maintained, self-contained
interaction-recovery implementation lives in
:func:`flowr.util.metrics.interaction_recovery_per_complex` (backed by
:class:`flowr.util.interaction_util.InteractionFingerprints`), which is what the
``--compute_interaction_recovery`` generation flag uses. This module is kept for
reference and imports ``plif_utils`` lazily so the package stays importable.
"""

import os
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import prolif as plf
from rdkit import Chem
from tqdm import tqdm

from flowr.util.rdkit import write_sdf_file

if TYPE_CHECKING:  # pragma: no cover - plif_utils is not shipped with FlowR
    from plif_utils.file_prep import FilePrep
    from plif_utils.system_prep import SystemPrep

_PLIF_UTILS_HINT = (
    "This function requires the external 'plif_utils' package, which is not "
    "distributed with FlowR. Use "
    "flowr.util.metrics.interaction_recovery_per_complex (the implementation "
    "behind --compute_interaction_recovery) instead."
)


def _import_plif_utils():
    try:
        from plif_utils.analysis import get_plif_recovery_rates, run
        from plif_utils.file_prep import get_files
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError(_PLIF_UTILS_HINT) from err

    return get_plif_recovery_rates, run, get_files


def get_recovery_rates_given_target(
    target: str,
    ligands: list[Chem.Mol],
    system_prep: "SystemPrep",
    data_dir: Path,
    ground_truth_plif: plf.Fingerprint,
    ground_truth_file_prep: "FilePrep",
) -> dict[str, plf.Fingerprint]:
    """
    Given a target and a list of ligands, calculate the PLIFs for the target
    target: PDB file of the target
    ligands: list of RDKit molecules of N (sampled) ligands for the target
    system_prep: SystemPrep object
    data_dir: Path to the data directory
    return: ist of PLIF recovery rates
    """

    get_plif_recovery_rates, run, get_files = _import_plif_utils()

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

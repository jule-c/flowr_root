import glob
import os
import tempfile
from typing import List

import datamol as dm
import rdkit.Chem as Chem
import wandb


def artifact_to_mols(
    wandb_artifact_name: str, type: str = "sdf_collection"
) -> List[Chem.Mol]:
    """Get molecules from a WandB artifact.

    Args:
        wandb_artifact_name: Full name of the artifact (e.g. 'username/project/artifact_name:version')
        type: Type of the artifact, defaults to "sdf_collection"

    Returns:
        List of RDKit molecules loaded from the SDF files in the artifact
    """

    api = wandb.Api()
    artifact = api.artifact(wandb_artifact_name, type=type)

    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_dir = artifact.download(tmp_dir)

        # list SDF files in the artifact directory
        sdf_files = glob.glob(os.path.join(artifact_dir, "*.sdf"))

        # load data
        mols = []
        for sdf_file in sdf_files:
            mol = dm.read_sdf(sdf_file)
            mols.extend(mol)

    return mols

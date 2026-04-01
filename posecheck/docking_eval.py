import argparse
import os

from rdkit import Chem
from tqdm import tqdm

import wandb
from posecheck.data.datasets import *
from posecheck.utils.docking import SMINA
from posecheck.utils.docking_utils import *


def benchmark_mols_for_target(
    mols: List[Chem.Mol], receptor_path: str
) -> Dict[str, Dict[str, float]]:
    """Benchmark a list of molecules for a target.

    Parameters
    ----------
    mols : list of rdkit.Chem.Mol
        The molecules to benchmark.
    target : str
        The target to benchmark against.

    Returns
    -------
    dict of str: dict of str: float
        A dictionary containing the results of the benchmark.
    """

    # Define the docking software and set the receptor
    smina = SMINA()
    smina.set_receptor(receptor_path)

    # Calculate the results for the raw molecules
    print("Calculating results for raw molecules")
    raw_results = [smina.calculate_all(mol) for mol in tqdm(mols) if mol is not None]

    # # Calculate the results for the relaxed molecules
    # print('Calculating results for relaxed molecules')
    # relaxed_results = []
    # for mol in mols:
    #     try:
    #         relaxed_mol = relax_mol(mol)
    #         relaxed_results.append(smina.calculate_all(relaxed_mol))
    #     except:
    #         relaxed_results.append(None)

    relaxed_results = [None] * len(raw_results)

    # relaxed_mols = [relax_mol(mol) for mol in mols]
    # relaxed_results = [smina.calculate_all(mol) for mol in tqdm(relaxed_mols)]

    results = {
        "raw": raw_results,
        "relaxed": relaxed_results,
        "receptor_path": receptor_path,
    }

    # Return the results
    return results


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_name", type=str, default="decompdiff")
    args.add_argument(
        "--data_dir", type=str, default="/home/cch57/projects/poses_benchmark/data"
    )
    args = args.parse_args()

    data_dir = args.data_dir
    dataset_name = args.dataset_name

    # Load the dataset
    if dataset_name == "diffsbdd":
        dataset = DiffSBDDSamples(docked=False)
    elif dataset_name == "flag":
        dataset = FLAGSamples()
    elif dataset_name == "decompdiff":
        dataset = DecompDiffSamples()
    else:
        dataset = DockedMolsDataset(dataset_name, docked=False)

    # Set up the output directory and logger
    out_dir = os.path.join(data_dir, "docking_results", dataset_name)
    logger = wandb.init(project="poses_benchmark", entity="cch1999", name=dataset_name)

    # Make the output directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Loop over the dataset and calculate the docking results
    for idx in range(len(dataset)):
        logger.log({"target": idx})
        print(f"Calculating docking results for target {idx}")

        out_path = os.path.join(out_dir, f"target_{idx}.pt")
        if os.path.exists(out_path):  # Skip if already done
            continue

        receptor_path = dataset.get_pdbqt_path(idx)
        mols = dataset.load_mols(idx)

        print(f"Loaded {len(mols)} molecules ****")

        # Calculate the docking results
        docking_results = benchmark_mols_for_target(mols, receptor_path)

        # Save the results
        torch.save(docking_results, out_path)

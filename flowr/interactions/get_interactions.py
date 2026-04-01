import argparse
from pathlib import Path

import dill as pkl

from plif_utils.analysis import get_plifs
from plif_utils.system_prep import SystemPrep


def args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data-path",
        type=str,
        help="Path to the data directory",
        required=True,
    )
    argparser.add_argument(
        "--state",
        type=str,
        help="train or test",
        required=True,
    )
    argparser.add_argument(
        "--optimize-hydrogens",
        action="store_true",
        help="Optimize hydrogens",
        default=False,
    )
    argparser.add_argument(
        "--add-hydrogens",
        action="store_true",
        help="Add hydrogens",
        default=False,
    )
    argparser.add_argument(
        "--sanitize",
        action="store_true",
        help="Sanitize",
        default=False,
    )
    args = argparser.parse_args()
    return args


def get_ground_truth_interactions(
    data_path: str,
    state: str = "train",
    system_prep: bool = False,
    optimize_hydrogens: bool = False,
    add_hydrogens: bool = False,
    sanitize: bool = False,
    pocket: str = "holo",
):
    """
    Get ground truth interactions for a given dataset
    The function expects the data directory to have the following structure:
        pdb_files are saved in a folder '{state}/pdbs/{pocket}' as system_id.pdb, where pocket can be apo or holo
        sdf_files are saved in a '{state}/sdfs' folder as system_id.sdf
            system_id: {pdb_id}__{id}__{chain-id}__{res_id}

        e.g.: ./data_path/train/pdbs/holo/6zc8__1__1.A__1.C.pdb
              ./data_path/train/sdfs/6zc8__1__1.A__1.C.sdf

    return (Tuple): A dictionary of PLIFs for every pocket-ligand complex
                    The path to all the PLIFs and the prepared protein-ligand files (protonated and optimized)

    """

    # Get data
    data_path = Path(data_path) / state
    sdf_path = data_path / "sdfs"
    sdf_files = list(sdf_path.glob("*.sdf"))

    # PLIFs for ground truth ligands
    ground_truth_plifs_path = (data_path / "ground_truth_plifs").resolve()
    ground_truth_plifs_pkl_path = (
        ground_truth_plifs_path / "plifs_dict_ground_truth.pkl"
    )
    if not ground_truth_plifs_path.exists():
        pdb_files = []
        for sdf_file in sdf_files:
            pdb_id = sdf_file.stem
            pdb_path = data_path / "pdbs" / pocket / f"{pdb_id}.pdb"
            pdb_files.append(pdb_path)

        system_prep = SystemPrep(
            pocket_cutoff=6.0, optimize_hydrogens=optimize_hydrogens, sanitize=sanitize
        )
        plifs_dict_ground_truth = get_plifs(
            pdb_files,
            sdf_files,
            system_prep,
            data_dir=ground_truth_plifs_path,
            add_hydrogens=add_hydrogens,
        )
        with open(ground_truth_plifs_pkl_path, "wb") as f:
            pkl.dump(plifs_dict_ground_truth, f)
    else:
        with open(ground_truth_plifs_pkl_path, "rb") as f:
            plifs_dict_ground_truth = pkl.load(f)

    return plifs_dict_ground_truth, ground_truth_plifs_path


if __name__ == "__main__":
    args = args()
    plifs_dict_ground_truth, ground_truth_plifs_path = get_ground_truth_interactions(
        args.data_path,
        args.state,
        args.optimize_hydrogens,
        args.add_hydrogens,
        args.sanitize,
    )

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from rdkit import Chem
from torch.utils.data import Subset
from tqdm import tqdm

from flowr.constants import ATOM_ENCODER as atom_encoder
from flowr.data.dataset import PocketComplexLMDBDataset
from flowr.data.util import compute_all_statistics, mol_to_dict
from flowr.scriptutil import make_splits
from flowr.util.molrepr import GeometricMol


def save_pickle(array, path, exist_ok=True):
    if exist_ok:
        with open(path, "wb") as f:
            pickle.dump(array, f)
    else:
        if not os.path.exists(path):
            with open(path, "wb") as f:
                pickle.dump(array, f)


def args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data_path",
        type=str,
        help="Path to the data directory",
        required=True,
    )
    argparser.add_argument(
        "--state",
        type=str,
        help="train, val or test, or random with seed",
        choices=["train", "val", "test", "all"],
        required=True,
    )
    argparser.add_argument(
        "--val_size",
        type=int,
        help="Validation set size (as a fraction of the dataset)",
        default=100,
    )
    argparser.add_argument(
        "--test_size",
        type=int,
        help="Test set size (as a fraction of the dataset)",
        default=225,
    )
    argparser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
        default=None,
    )
    argparser.add_argument(
        "--remove_hs",
        action="store_true",
        help="Remove hydrogens from the molecule",
    )
    argparser.add_argument(
        "--from_smol",
        action="store_true",
        help="Load data from .smol file",
    )
    argparser.add_argument(
        "--from_sdf",
        action="store_true",
        help="Load data from .sdf file",
    )
    argparser.add_argument(
        "--from_lmdb",
        action="store_true",
        help="Load data from LMDB",
    )
    args = argparser.parse_args()
    return args


def get_statistics(args):

    data_list = []
    failed = 0

    if args.from_smol:
        assert args.state in [
            "train",
            "val",
            "test",
        ], "State must be train, val or test when loading from .smol"
        print("Loading data from .smol file...")
        data_path = Path(f"{args.data_path}/{args.state}.smol")
        bytes_data = data_path.read_bytes()
        for mol_bytes in tqdm(pickle.loads(bytes_data)):
            obj = pickle.loads(mol_bytes)
            ligand = GeometricMol.from_bytes(obj["ligand"])
            meta_data = obj["metadata"]
            if ligand is None:
                failed += 1
                print(f"Failed to load molecule: {meta_data}")
                continue
            data_list.append(
                mol_to_dict(
                    ligand.to_rdkit(),
                    atom_encoder=atom_encoder,
                    remove_hs=args.remove_hs,
                )
            )
    elif args.from_sdf:
        print("Loading data from .sdf file...")
        sdf_files = Path(args.data_path).glob("*.sdf")
        for sdf in sdf_files:
            suppl = Chem.SDMolSupplier(str(sdf), removeHs=False)
            for mol in suppl:
                if mol is None:
                    failed += 1
                    continue
                ligand = GeometricMol.from_rdkit(mol)
                data_list.append(
                    mol_to_dict(
                        ligand.to_rdkit(),
                        atom_encoder=atom_encoder,
                        remove_hs=args.remove_hs,
                    )
                )
    elif args.from_lmdb:
        print("Loading data from LMDB...")
        dataset = PocketComplexLMDBDataset(
            root=args.data_path, transform=None, remove_hs=False, skip_non_valid=False
        )
        dataset_len = len(dataset)
        if not args.state == "all":
            splits_path = Path(args.data_path) / "splits.npz"
            if not splits_path.exists():
                print(f"Creating random splits with seed {args.seed}...")
                idx_train, idx_val, idx_test = make_splits(
                    dataset_len=dataset_len,
                    train_size=dataset_len - (args.val_size + args.test_size),
                    val_size=args.val_size,
                    test_size=args.test_size,
                    seed=args.seed,
                )
                np.savez(
                    str(splits_path),
                    idx_train=idx_train,
                    idx_val=idx_val,
                    idx_test=idx_test,
                )
            else:
                print(f"Loading splits from {splits_path}...")
                splits_path = Path(splits_path)
                idx_train, idx_val, idx_test = make_splits(
                    splits=splits_path,
                )
            dataset = (
                Subset(dataset, idx_train)
                if args.state == "train"
                else (
                    Subset(dataset, idx_val)
                    if args.state == "val"
                    else Subset(dataset, idx_test)
                )
            )
        for i in tqdm(range(len(dataset))):
            try:
                item = dataset[i]
                if item is None:
                    failed += 1
                    continue
                data_list.append(
                    mol_to_dict(
                        item.ligand.to_rdkit(),
                        atom_encoder=atom_encoder,
                        remove_hs=args.remove_hs,
                    )
                )
            except Exception as e:
                print(f"Failed to load molecule {i}: {e}")
                failed += 1
    print(f"Number of processed molecules: {len(data_list)}")
    print(f"Number of failed molecules: {failed}")
    print("Computing statistics...")
    statistics = compute_all_statistics(
        data_list,
        atom_encoder,
        charges_dic={-2: 0, -1: 1, 0: 2, 1: 3, 2: 4, 3: 5},
        additional_feats=True,
    )

    h = "noh" if args.remove_hs else "h"
    processed_paths = [
        f"{args.state}_{h}.pt",
        f"{args.state}_n_{h}.pickle",
        f"{args.state}_atom_types_{h}.npy",
        f"{args.state}_bond_types_{h}.npy",
        f"{args.state}_charges_{h}.npy",
        f"{args.state}_valency_{h}.pickle",
        f"{args.state}_bond_lengths_{h}.pickle",
        f"{args.state}_angles_{h}.npy",
        f"{args.state}_is_aromatic_{h}.npy",
        f"{args.state}_is_in_ring_{h}.npy",
        f"{args.state}_hybridization_{h}.npy",
        f"{args.state}_is_h_donor_{h}.npy",
        f"{args.state}_is_h_acceptor_{h}.npy",
        f"{args.state}_dihedrals_{h}.npy",
    ]
    if not Path(f"{args.data_path}/processed").exists():
        os.makedirs(f"{args.data_path}/processed")
    processed_paths = [f"{args.data_path}/processed/{p}" for p in processed_paths]
    save_pickle(statistics.num_nodes, processed_paths[1])
    np.save(processed_paths[2], statistics.atom_types)
    np.save(processed_paths[3], statistics.bond_types)
    np.save(processed_paths[4], statistics.charge_types)
    save_pickle(statistics.valencies, processed_paths[5])
    save_pickle(statistics.bond_lengths, processed_paths[6])
    np.save(processed_paths[7], statistics.bond_angles)
    np.save(processed_paths[8], statistics.is_aromatic)
    np.save(processed_paths[9], statistics.is_in_ring)
    np.save(processed_paths[10], statistics.hybridization)
    np.save(processed_paths[11], statistics.is_h_donor)
    np.save(processed_paths[12], statistics.is_h_acceptor)
    np.save(processed_paths[13], statistics.dihedrals)
    print("Statistics computed and saved.")


if __name__ == "__main__":
    args = args()
    get_statistics(args)

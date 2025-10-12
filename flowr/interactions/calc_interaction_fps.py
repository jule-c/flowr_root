import argparse
from collections import defaultdict
from pathlib import Path

import dill as pickle
from tqdm import tqdm

from flowr.util.molrepr import GeometricMol
from flowr.util.pocket import BindingInteractions, PocketComplex, ProteinPocket


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
    args = argparser.parse_args()
    return args


def retrieve_interactions(data_path: str, state: str = "train"):
    data_path = Path(data_path) / (state + ".smol")
    save_path = Path(data_path).parent / (state + "_interaction_fps.pickle")
    failed_path = Path(data_path).parent / (state + "interaction_fps_failed.pickle")
    bytes_data = data_path.read_bytes()

    failed = []
    interactions = defaultdict()
    for mol_bytes in tqdm(pickle.loads(bytes_data)):
        obj = pickle.loads(mol_bytes)
        holo = ProteinPocket.from_bytes(obj["holo"])
        ligand = GeometricMol.from_bytes(obj["ligand"])
        metadata = obj["metadata"]
        system = PocketComplex(
            ligand=ligand,
            holo=holo,
            apo=None,
            interactions=None,
            metadata=metadata,
        )
        try:
            interactions[metadata["system_id"]] = BindingInteractions.interaction_array(
                system
            )
        except SystemExit:
            print(f"Failed to process {metadata['system_id']}")
            failed.append(metadata["system_id"])
            continue

    with open(str(save_path), "wb") as f:
        pickle.dump(dict(interactions), f)
    with open(str(failed_path), "wb") as f:
        pickle.dump(failed, f)


if __name__ == "__main__":
    args = args()
    retrieve_interactions(args.data_path, args.state)

import argparse
import multiprocessing
from collections import defaultdict
from pathlib import Path
from typing import List

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
    argparser.add_argument(
        "--num-workers",
        type=int,
        help="Number of parallel processes to use",
        default=4,
    )
    args = argparser.parse_args()
    return args


def process_chunk(mol_bytes_chunk: List[bytes]) -> dict:
    """Process a chunk of molecule bytes, returning a dictionary of system_id -> interaction_fps."""
    interactions_chunk = defaultdict()
    for mol_bytes in mol_bytes_chunk:
        obj = pickle.loads(mol_bytes)
        holo = ProteinPocket.from_bytes(obj["holo"])
        ligand = GeometricMol.from_bytes(obj["ligand"])
        metadata = obj["metadata"]

        system = PocketComplex(
            ligand=ligand,
            holo=holo,
            interactions=None,
            metadata=metadata,
        )
        try:
            interactions_chunk[metadata["system_id"]] = (
                BindingInteractions.interaction_array(system)
            )
        except SystemExit:
            print(f"Failed to process {metadata['system_id']}")
            continue
    return interactions_chunk


def merge_interactions(results: List[dict]) -> defaultdict:
    """
    Merge the interaction dictionaries returned by each process into
    a single defaultdict(list).
    """
    merged = defaultdict(list)
    for result in results:
        for system_id, fps in result.items():
            merged[system_id].extend(fps)
    return merged


def retrieve_interactions(data_path: str, state: str = "train", num_workers: int = 4):
    """
    Retrieve interactions using multiprocessing.

    :param data_path: Path to the directory containing the <state>.smol file.
    :param state: State name (e.g., 'train', 'test').
    :param num_workers: Number of parallel processes to use.
    """
    data_path = Path(data_path) / f"{state}.smol"
    save_path = Path(data_path).parent / f"{state}_interaction_fps.pickle"

    bytes_data = data_path.read_bytes()
    mol_bytes_list = pickle.loads(bytes_data)

    chunk_size = max(1, len(mol_bytes_list) // num_workers)
    chunks = [
        mol_bytes_list[i : i + chunk_size]
        for i in range(0, len(mol_bytes_list), chunk_size)
    ]

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_chunk, chunks),
                total=len(chunks),
                desc="Processing chunks",
            )
        )

    interactions = merge_interactions(results)

    with open(save_path, "wb") as f:
        pickle.dump(interactions, f)


if __name__ == "__main__":
    args = args()
    retrieve_interactions(args.data_path, args.state, num_workers=args.num_workers)

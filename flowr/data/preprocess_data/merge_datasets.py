import argparse
import pickle
import shutil
import subprocess
from pathlib import Path

import lmdb
import numpy as np
from tqdm import tqdm

from flowr.constants import CORE_ATOMS
from flowr.data.datasets.complex_data.preprocess_util import check_ligand_atom_types
from flowr.util.pocket import PocketComplex


def estimate_required_size_datasets(dataset_paths):
    """Estimate the actual size needed by checking existing datasets"""
    total_size = 0
    for dataset_path in dataset_paths:
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            print(f"Warning: Dataset path {dataset_path} does not exist")
            continue

        try:
            result = subprocess.run(
                ["du", "-sb", str(dataset_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            dataset_size = int(result.stdout.split()[0])
            total_size += dataset_size
            print(f"Dataset {dataset_path.name}: {format_bytes(dataset_size)}")
        except Exception as e:
            print(f"Error estimating size for {dataset_path}: {e}")
            continue

    # Add 50% buffer for safety
    estimated_size = int(total_size * 1.5)
    return estimated_size, total_size


def format_bytes(bytes_size):
    """Format bytes into human readable format"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def load_pickle_dict(file_path):
    """Load a pickle dictionary file if it exists, return empty dict otherwise"""
    if file_path.exists():
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return {}


def load_pickle(file_path):
    """Load a pickle file if it exists, return None otherwise"""
    if file_path.exists():
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return None


def load_splits(dataset_path):
    """Load splits from a dataset's splits.npz file"""
    splits_file = dataset_path / "splits.npz"
    if splits_file.exists():
        return np.load(splits_file)
    return None


def merge_datasets(
    dataset_paths, output_path, remove_datasets=False, merge_splits=False
):
    """
    Merge multiple LMDB datasets into a single LMDB database.

    Args:
        dataset_paths (list): List of paths to dataset directories
        output_path (str): Path for the output merged LMDB database
        remove_datasets (bool): Whether to remove original datasets after merging
        merge_splits (bool): Whether to merge dataset splits
    """

    dataset_paths = [Path(p) for p in dataset_paths]
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # If the lmdb files in output directory exist, remove them
    lmdb_file = output_path / "data.mdb"
    lock_file = output_path / "lock.mdb"

    if lmdb_file.exists():
        lmdb_file.unlink()
    if lock_file.exists():
        lock_file.unlink()

    # Validate dataset paths
    valid_datasets = []
    for dataset_path in dataset_paths:
        if not dataset_path.exists():
            print(f"Warning: Dataset path {dataset_path} does not exist, skipping")
            continue

        # Check if it's an LMDB directory
        if not (dataset_path / "data.mdb").exists():
            print(
                f"Warning: {dataset_path} doesn't appear to be an LMDB database, skipping"
            )
            continue

        # If merge_splits is True, check for splits.npz
        if merge_splits and not (dataset_path / "splits.npz").exists():
            print(f"Warning: {dataset_path} doesn't have splits.npz file, skipping")
            continue

        valid_datasets.append(dataset_path)

    if not valid_datasets:
        raise ValueError("No valid LMDB datasets found")

    print(f"Found {len(valid_datasets)} valid datasets to merge")

    # Estimate required size
    print("Estimating required size...")
    estimated_size, current_total = estimate_required_size_datasets(valid_datasets)
    print(f"Current datasets total size: {format_bytes(current_total)}")
    print(f"Estimated size needed: {format_bytes(estimated_size)}")

    # Create output LMDB database with better settings
    output_env = lmdb.open(
        str(output_path),
        map_size=min(estimated_size, 10 * 1024**4),  # Cap at 10TB
        writemap=False,
        lock=True,
        readahead=False,
        max_dbs=1,
    )

    # Aggregate metadata
    total_count = 0
    filtered_count = 0  # Track how many were filtered out
    all_lengths_full = []
    all_lengths_no_ligand_pocket_hs = []
    all_lengths_no_pocket_hs = []
    all_system_ids = []
    rdkit_mols = []
    ligand_atom_types = {}
    protein_residue_types = {}
    protein_atom_names = {}
    protein_atom_types = {}

    # Split tracking for merge_splits
    merged_splits = {"train": [], "val": [], "test": []} if merge_splits else None

    core_atoms_set = set(CORE_ATOMS)

    output_txn = output_env.begin(write=True)
    commit_interval = 100

    try:
        for dataset_path in tqdm(valid_datasets, desc="Processing datasets"):
            print(f"\nProcessing dataset: {dataset_path.name}")

            # Load splits if merging splits
            dataset_splits = None
            if merge_splits:
                dataset_splits = load_splits(dataset_path)
                if dataset_splits is None:
                    print(
                        f"Warning: Could not load splits for {dataset_path}, skipping"
                    )
                    continue
                print(
                    f"Loaded splits: train={len(dataset_splits['train'])}, "
                    f"val={len(dataset_splits['val'])}, test={len(dataset_splits['test'])}"
                )

            # Open dataset database
            dataset_env = lmdb.open(str(dataset_path), readonly=True, lock=False)
            dataset_txn = dataset_env.begin()

            try:
                # First, get metadata from this dataset
                dataset_len_data = dataset_txn.get(b"__len__")
                dataset_len = pickle.loads(dataset_len_data) if dataset_len_data else 0
                print(f"Dataset contains {dataset_len} entries")

                # Get other metadata from LMDB
                metadata_keys = [
                    "lengths_full",
                    "lengths_no_ligand_pocket_hs",
                    "lengths_no_pocket_hs",
                    "system_ids",
                ]

                dataset_metadata = {}
                for key in metadata_keys:
                    data = dataset_txn.get(key.encode())
                    if data:
                        dataset_metadata[key] = pickle.loads(data)

                # Load pickle files for dictionaries
                dataset_ligand_atom_types = load_pickle_dict(
                    dataset_path / "ligand_atom_types.pkl"
                )
                dataset_protein_residue_types = load_pickle_dict(
                    dataset_path / "protein_residue_types.pkl"
                )
                dataset_protein_atom_names = load_pickle_dict(
                    dataset_path / "protein_atom_names.pkl"
                )
                dataset_protein_atom_types = load_pickle_dict(
                    dataset_path / "protein_atom_types.pkl"
                )
                dataset_rdkit_mols = load_pickle(dataset_path / "rdkit_mols.pkl")

                # Track mapping from original indices to new indices for this dataset
                original_to_new_mapping = {}

                # Copy data entries in order (0, 1, 2, ..., dataset_len-1)
                entries_copied = 0
                dataset_metadata_indices = []  # Track which indices actually exist

                for i in range(dataset_len):
                    key = str(i).encode("utf-8")
                    value = dataset_txn.get(key)

                    if value is None:
                        continue

                    # Check the atom types of the complex
                    try:
                        complex_data = PocketComplex.from_bytes(value, remove_hs=False)
                        if not check_ligand_atom_types(complex_data, core_atoms_set):
                            filtered_count += 1
                            continue
                    except Exception as e:
                        print(f"Error processing entry {i}: {e}")
                        filtered_count += 1
                        continue

                    # Convert dataset index to global index
                    new_key = str(total_count).encode("utf-8")
                    output_txn.put(new_key, value)
                    original_to_new_mapping[i] = total_count
                    dataset_metadata_indices.append(i)  # Track this index
                    total_count += 1
                    entries_copied += 1

                    # Commit at intervals
                    if total_count % commit_interval == 0:
                        output_txn.commit()
                        output_txn = output_env.begin(write=True)

                # Update splits with new indices if merging splits
                if merge_splits and dataset_splits is not None:
                    for split_name in ["train", "val", "test"]:
                        original_indices = dataset_splits[split_name]
                        # Map original indices to new indices, only including valid entries
                        new_indices = [
                            original_to_new_mapping[idx]
                            for idx in original_indices
                            if idx in original_to_new_mapping
                        ]
                        merged_splits[split_name].extend(new_indices)
                        print(
                            f"  {split_name}: {len(original_indices)} -> {len(new_indices)} indices"
                        )

                # Aggregate metadata only for entries that actually existed
                if "lengths_full" in dataset_metadata:
                    dataset_lengths = dataset_metadata["lengths_full"]
                    all_lengths_full.extend(
                        [
                            dataset_lengths[i]
                            for i in dataset_metadata_indices
                            if i < len(dataset_lengths)
                        ]
                    )

                if "lengths_no_ligand_pocket_hs" in dataset_metadata:
                    dataset_lengths = dataset_metadata["lengths_no_ligand_pocket_hs"]
                    all_lengths_no_ligand_pocket_hs.extend(
                        [
                            dataset_lengths[i]
                            for i in dataset_metadata_indices
                            if i < len(dataset_lengths)
                        ]
                    )

                if "lengths_no_pocket_hs" in dataset_metadata:
                    dataset_lengths = dataset_metadata["lengths_no_pocket_hs"]
                    all_lengths_no_pocket_hs.extend(
                        [
                            dataset_lengths[i]
                            for i in dataset_metadata_indices
                            if i < len(dataset_lengths)
                        ]
                    )

                if "system_ids" in dataset_metadata:
                    dataset_system_ids = dataset_metadata["system_ids"]
                    all_system_ids.extend(
                        [
                            dataset_system_ids[i]
                            for i in dataset_metadata_indices
                            if i < len(dataset_system_ids)
                        ]
                    )

                # Add RDKit mols
                rdkit_mols.extend(
                    [
                        dataset_rdkit_mols[i]
                        for i in dataset_metadata_indices
                        if i < len(dataset_rdkit_mols)
                    ]
                )

                # Aggregate dictionary counts only for valid entries
                # Only count ligand atom types that are actually in CORE_ATOMS
                for atom_type, count in dataset_ligand_atom_types.items():
                    if atom_type in core_atoms_set:
                        ligand_atom_types[atom_type] = (
                            ligand_atom_types.get(atom_type, 0) + count
                        )

                for dict_name, source_dict, target_dict in [
                    (
                        "protein_residue_types",
                        dataset_protein_residue_types,
                        protein_residue_types,
                    ),
                    (
                        "protein_atom_names",
                        dataset_protein_atom_names,
                        protein_atom_names,
                    ),
                    (
                        "protein_atom_types",
                        dataset_protein_atom_types,
                        protein_atom_types,
                    ),
                ]:
                    for key, count in source_dict.items():
                        target_dict[key] = target_dict.get(key, 0) + count

                print(f"Copied {entries_copied} entries from {dataset_path.name}")

            except Exception as e:
                print(f"Error processing dataset {dataset_path.name}: {e}")
                raise
            finally:
                dataset_txn.abort()
                dataset_env.close()

        # Store aggregated metadata
        final_metadata = {
            "__len__": total_count,
            "lengths_full": all_lengths_full,
            "lengths_no_ligand_pocket_hs": all_lengths_no_ligand_pocket_hs,
            "lengths_no_pocket_hs": all_lengths_no_pocket_hs,
            "system_ids": all_system_ids,
        }

        for key, value in final_metadata.items():
            output_txn.put(
                key.encode() if isinstance(key, str) else key, pickle.dumps(value)
            )

    except Exception as e:
        print(f"Merging failed: {e}")
        raise
    finally:
        # Final commit and cleanup
        try:
            output_txn.commit()
        except Exception:
            print("Final commit failed")
        output_env.sync()
        output_env.close()

    # Save dictionaries as pickle files
    for filename, data in [
        ("ligand_atom_types.pkl", ligand_atom_types),
        ("protein_residue_types.pkl", protein_residue_types),
        ("protein_atom_names.pkl", protein_atom_names),
        ("protein_atom_types.pkl", protein_atom_types),
        ("rdkit_mols.pkl", rdkit_mols),
    ]:
        with open(output_path / filename, "wb") as f:
            pickle.dump(data, f)

    # Save merged splits if requested
    if merge_splits and merged_splits is not None:
        splits_output = {}
        for split_name, indices in merged_splits.items():
            splits_output[split_name] = np.array(indices, dtype=np.int64)

        np.savez(output_path / "splits.npz", **splits_output)

        print("\n=== Split Summary ===")
        for split_name, indices in merged_splits.items():
            print(f"{split_name}: {len(indices)} samples")
        print(
            f"Total samples in splits: {sum(len(indices) for indices in merged_splits.values())}"
        )

    # Final size reporting
    try:
        result = subprocess.run(
            ["du", "-sh", str(output_path)], capture_output=True, text=True, check=True
        )
        actual_size = result.stdout.split()[0]
    except Exception:
        actual_size = "Unknown"

    print("\n=== Merging Summary ===")
    print(f"Total entries merged: {total_count}")
    print(f"Total entries filtered out: {filtered_count}")
    print(f"Total system IDs: {len(all_system_ids)}")
    print(f"Output database: {output_path}")
    print(f"Estimated size: {format_bytes(estimated_size)}")
    print(f"Actual final size: {actual_size}")

    # Optionally remove original datasets
    if remove_datasets:
        print("Removing original datasets...")
        for dataset_path in valid_datasets:
            try:
                shutil.rmtree(dataset_path)
                print(f"Removed: {dataset_path.name}")
            except Exception as e:
                print(f"Failed to remove {dataset_path.name}: {e}")

    return output_path


def main():
    """Main function for merging multiple LMDB datasets"""
    parser = argparse.ArgumentParser(
        description="Merge multiple LMDB datasets into single database"
    )

    parser.add_argument(
        "--dataset_paths",
        type=str,
        nargs="+",
        required=True,
        help="List of paths to dataset directories to merge",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path for output merged LMDB database",
    )
    parser.add_argument(
        "--merge_splits",
        action="store_true",
        help="Merge dataset splits (requires splits.npz in each dataset)",
    )
    parser.add_argument(
        "--remove_datasets",
        action="store_true",
        help="Remove original datasets after merging",
    )

    args = parser.parse_args()

    merge_datasets(
        dataset_paths=args.dataset_paths,
        output_path=args.output_path,
        remove_datasets=args.remove_datasets,
        merge_splits=args.merge_splits,
    )


if __name__ == "__main__":
    main()

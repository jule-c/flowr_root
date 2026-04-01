import argparse
import pickle
import shutil
import subprocess
from pathlib import Path

import lmdb
from tqdm import tqdm

from flowr.constants import CORE_ATOMS, MAX_LIGAND_SIZE
from flowr.data.datasets.complex_data.preprocess_util import (
    check_ligand_atom_types,
    check_ligand_size,
)
from flowr.util.pocket import PocketComplex


def estimate_required_size(chunks_dir):
    """Estimate the actual size needed by checking existing chunks"""
    chunks_dir = Path(chunks_dir)
    chunk_dirs = sorted(chunks_dir.glob("chunk_*"))

    total_size = 0
    for chunk_dir in chunk_dirs:
        try:
            result = subprocess.run(
                ["du", "-sb", str(chunk_dir)],
                capture_output=True,
                text=True,
                check=True,
            )
            chunk_size = int(result.stdout.split()[0])
            total_size += chunk_size
        except Exception:
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


def fuse_lmdb_chunks(chunks_dir, output_path, remove_chunks=False):
    """
    Fuse multiple LMDB chunk databases into a single LMDB database.

    Args:
        chunks_dir (str): Directory containing chunk LMDB databases
        output_path (str): Path for the output fused LMDB database
        remove_chunks (bool): Whether to remove chunk databases after fusion
    """

    chunks_dir = Path(chunks_dir)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # If the lmdb files in output directory exist, remove them
    lmdb_file = output_path / "data.mdb"
    lock_file = output_path / "lock.mdb"

    if lmdb_file.exists():
        lmdb_file.unlink()
    if lock_file.exists():
        lock_file.unlink()

    # Find all chunk LMDB databases
    chunk_pattern = "chunk_*"
    chunk_dirs = sorted(chunks_dir.glob(chunk_pattern))

    if not chunk_dirs:
        raise ValueError(f"No chunk databases found matching pattern: {chunk_pattern}")

    print(f"Found {len(chunk_dirs)} chunk databases to fuse")

    # Estimate required size
    print("Estimating required size...")
    estimated_size, current_total = estimate_required_size(chunks_dir)
    print(f"Current chunks total size: {format_bytes(current_total)}")
    print(f"Estimated size needed: {format_bytes(estimated_size)}")

    # Create output LMDB database
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

    core_atoms_set = set(CORE_ATOMS)
    max_ligand_size = MAX_LIGAND_SIZE

    output_txn = output_env.begin(write=True)
    commit_interval = 100

    try:
        for chunk_dir in tqdm(chunk_dirs, desc="Processing chunks"):
            print(f"Processing chunk: {chunk_dir.name}")

            # Open chunk database
            chunk_env = lmdb.open(str(chunk_dir), readonly=True, lock=False)
            chunk_txn = chunk_env.begin()

            try:
                # First, get metadata from this chunk
                chunk_len_data = chunk_txn.get(b"__len__")
                chunk_len = pickle.loads(chunk_len_data) if chunk_len_data else 0

                # Get other metadata
                metadata_keys = [
                    "lengths_full",
                    "lengths_no_ligand_pocket_hs",
                    "lengths_no_pocket_hs",
                    "system_ids",
                    "rdkit_mols",
                    "ligand_atom_types",
                    "protein_residue_types",
                    "protein_atom_names",
                    "protein_atom_types",
                ]

                chunk_metadata = {}
                for key in metadata_keys:
                    try:
                        data = chunk_txn.get(key.encode())
                    except Exception:
                        data = None
                    if data:
                        chunk_metadata[key] = pickle.loads(data)

                # Copy data entries in order (0, 1, 2, ..., chunk_len-1)
                entries_copied = 0
                chunk_metadata_indices = []  # Track which indices actually exist

                for i in range(chunk_len):
                    key = str(i).encode("utf-8")
                    value = chunk_txn.get(key)

                    # check the atom types of the complex
                    complex_data = PocketComplex.from_bytes(value, remove_hs=False)
                    if not check_ligand_atom_types(complex_data, core_atoms_set):
                        filtered_count += 1
                        continue
                    if not check_ligand_size(complex_data, max_ligand_size):
                        filtered_count += 1
                        continue

                    if (
                        "rdkit_mols" not in chunk_metadata
                    ):  # backward compatibility for preprocessed data without rdkit_mols saved
                        rdkit_mol = complex_data.ligand.to_rdkit()
                        rdkit_mol_bytes = rdkit_mol.ToBinary()
                        rdkit_mols.append(rdkit_mol_bytes)

                    # Convert chunk index to global index
                    new_key = str(total_count).encode("utf-8")
                    output_txn.put(new_key, value)
                    chunk_metadata_indices.append(i)  # Track this index
                    total_count += 1
                    entries_copied += 1

                    # Commit at intervals
                    if total_count % commit_interval == 0:
                        output_txn.commit()
                        output_txn = output_env.begin(write=True)

                # Aggregate metadata only for entries that actually existed
                if "lengths_full" in chunk_metadata:
                    chunk_lengths = chunk_metadata["lengths_full"]
                    all_lengths_full.extend(
                        [chunk_lengths[i] for i in chunk_metadata_indices]
                    )
                if "lengths_no_ligand_pocket_hs" in chunk_metadata:
                    chunk_lengths = chunk_metadata["lengths_no_ligand_pocket_hs"]
                    all_lengths_no_ligand_pocket_hs.extend(
                        [chunk_lengths[i] for i in chunk_metadata_indices]
                    )
                if "lengths_no_pocket_hs" in chunk_metadata:
                    chunk_lengths = chunk_metadata["lengths_no_pocket_hs"]
                    all_lengths_no_pocket_hs.extend(
                        [chunk_lengths[i] for i in chunk_metadata_indices]
                    )
                if "system_ids" in chunk_metadata:
                    chunk_system_ids = chunk_metadata["system_ids"]
                    all_system_ids.extend(
                        [chunk_system_ids[i] for i in chunk_metadata_indices]
                    )
                if "rdkit_mols" in chunk_metadata:
                    mols = chunk_metadata["rdkit_mols"]
                    rdkit_mols.extend([mols[i] for i in chunk_metadata_indices])

                # Aggregate dictionary counts only for valid entries
                # Note: Only count ligand atom types that are actually in CORE_ATOMS
                if "ligand_atom_types" in chunk_metadata:
                    for atom_type, count in chunk_metadata["ligand_atom_types"].items():
                        if atom_type in core_atoms_set:
                            ligand_atom_types[atom_type] = (
                                ligand_atom_types.get(atom_type, 0) + count
                            )

                for dict_name, target_dict in [
                    ("protein_residue_types", protein_residue_types),
                    ("protein_atom_names", protein_atom_names),
                    ("protein_atom_types", protein_atom_types),
                ]:
                    if dict_name in chunk_metadata:
                        for key, count in chunk_metadata[dict_name].items():
                            target_dict[key] = target_dict.get(key, 0) + count

                print(f"Copied {entries_copied} entries from {chunk_dir.name}")

            except Exception as e:
                print(f"Error processing chunk {chunk_dir.name}: {e}")
                raise
            finally:
                chunk_txn.abort()
                chunk_env.close()

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
        print(f"Fusion failed: {e}")
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

    # Final size reporting
    try:
        result = subprocess.run(
            ["du", "-sh", str(output_path)], capture_output=True, text=True, check=True
        )
        actual_size = result.stdout.split()[0]
    except Exception:
        actual_size = "Unknown"

    print("\n=== Fusion Summary ===")
    print(f"Total entries fused: {total_count}")
    print(f"Total entries filtered out: {filtered_count}")
    print(f"Total system IDs: {len(all_system_ids)}")
    print(f"Output database: {output_path}")
    print(f"Estimated size: {format_bytes(estimated_size)}")
    print(f"Actual final size: {actual_size}")

    # Optionally remove chunk databases
    if remove_chunks:
        print("Removing chunk databases...")
        for chunk_dir in chunk_dirs:
            try:
                shutil.rmtree(chunk_dir)
                print(f"Removed: {chunk_dir.name}")
            except Exception as e:
                print(f"Failed to remove {chunk_dir.name}: {e}")

    return output_path


def main():
    """Example main function for fusing LMDB chunks"""
    parser = argparse.ArgumentParser(
        description="Fuse LMDB chunks into single database"
    )

    parser.add_argument(
        "--chunks_dir",
        type=str,
        required=True,
        help="Directory containing chunk LMDB databases",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path for output fused LMDB database",
    )
    parser.add_argument(
        "--remove_chunks",
        action="store_true",
        help="Remove chunk databases after fusion",
    )

    args = parser.parse_args()

    fuse_lmdb_chunks(
        chunks_dir=args.chunks_dir,
        output_path=args.output_path,
        remove_chunks=args.remove_chunks,
    )


if __name__ == "__main__":
    main()

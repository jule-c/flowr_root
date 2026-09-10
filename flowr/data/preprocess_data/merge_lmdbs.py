import argparse
import os
import pickle
import re
import shutil
import subprocess
from pathlib import Path

import lmdb
from tqdm import tqdm

from flowr.constants import CORE_ATOMS, MAX_LIGAND_SIZE
from flowr.data.preprocess_data.preprocess_util import (
    check_ligand_atom_types,
    check_ligand_size,
)
from flowr.util.pocket import PocketComplex


def _tree_apparent_size(path: Path) -> int:
    """Sum the apparent size of everything under ``path``, like ``du -sb``.

    Apparent size (``st_size``), not allocated blocks: LMDB's ``data.mdb`` is sparse,
    so block usage would under-report it badly and we would size the output map from a
    number far smaller than the data about to be copied in.
    """
    total = 0
    stack = [path]
    while stack:
        current = stack.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                # follow_symlinks=False matches du, which measures the link not the target.
                total += entry.stat(follow_symlinks=False).st_size
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
    # du counts the directory inode itself too.
    return total + path.stat().st_size


def estimate_required_size(chunks_dir):
    """Estimate the actual size needed by checking existing chunks

    This measurement used to shell out to ``du -sb``. ``-b`` is a GNU extension that
    BSD/macOS ``du`` does not have, so on macOS the call raised, the bare ``except``
    below swallowed it, and the estimate silently stayed 0 -- which left ``lmdb.open``
    on its ~10 MB default and killed the merge with ``MDB_MAP_FULL`` on any real
    dataset. Measuring in Python is portable and needs no subprocess.
    """
    chunks_dir = Path(chunks_dir)
    chunk_dirs = sorted(chunks_dir.glob("chunk_*"))

    total_size = 0
    for chunk_dir in chunk_dirs:
        try:
            total_size += _tree_apparent_size(chunk_dir)
        except OSError as err:
            # Was ``except Exception: continue``. Skipping a chunk here does not skip it
            # during the merge -- it is still copied in -- so a swallowed error means we
            # size the map for less data than we then write and LMDB fails later with an
            # opaque MDB_MAP_FULL. Be loud instead.
            raise RuntimeError(
                f"Could not measure chunk '{chunk_dir}' while sizing the output "
                f"database: {err}"
            ) from err

    if total_size == 0:
        raise RuntimeError(
            f"Measured 0 bytes across {len(chunk_dirs)} chunk database(s) under "
            f"'{chunks_dir}'. Sizing the output map from this would leave LMDB on its "
            f"~10 MB default and fail with MDB_MAP_FULL part-way through the merge."
        )

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


def _chunk_index(chunk_dir):
    """Return the 1-based job index encoded in a ``chunk_<index>`` directory name.

    ``None`` for anything that does not follow the scheme ``preprocess.py`` writes
    (``chunk_{job_index:04d}``).
    """
    match = re.fullmatch(r"chunk_(\d+)", chunk_dir.name)
    return int(match.group(1)) if match is not None else None


def _read_chunk_system_ids(chunk_dir):
    """Read only the ``system_ids`` metadata key of a chunk LMDB.

    Cheap enough to run over every chunk before any data is copied, so an inconsistent
    set of chunks is rejected before a corrupt database is written.
    """
    env = lmdb.open(str(chunk_dir), readonly=True, lock=False)
    try:
        with env.begin() as txn:
            data = txn.get(b"system_ids")
    finally:
        env.close()
    return pickle.loads(data) if data else None


def _format_indices(indices):
    """Render a sorted index list compactly, e.g. [1,2,3,5,6,9] -> '1-3, 5-6, 9'."""
    indices = sorted(indices)
    if not indices:
        return "none"
    groups = []
    start = prev = indices[0]
    for index in indices[1:]:
        if index == prev + 1:
            prev = index
            continue
        groups.append((start, prev))
        start = prev = index
    groups.append((start, prev))
    return ", ".join(str(lo) if lo == hi else f"{lo}-{hi}" for lo, hi in groups)


def check_chunks_consistent(chunks_dir, chunk_dirs):
    """Refuse to fuse a set of chunk databases that cannot come from one clean run.

    ``fuse_lmdb_chunks`` globs ``chunk_*`` unconditionally, so re-running preprocessing
    with a *smaller* ``--num_jobs`` into the same ``--save_path`` leaves the previous
    run's higher-numbered chunks on disk and merges them in as well. That is not merely
    a wrong entry count: the duplicated systems are handed to the random train/val/test
    split as independent entries, so the same complex can land in train *and* in
    val/test -- silent train/test leakage that quietly invalidates the results.

    Two things are checked, both before a single entry is copied:

    * no ``system_id`` may appear in more than one chunk (hard failure -- this is never
      legitimate), and
    * the chunk indices must run ``1..N`` with no holes. Only *trailing* jobs may be
      missing, which is the legitimate oversized-array case: with ``--num_jobs 20`` on
      12 systems, jobs 13-20 are assigned an empty chunk and exit without creating a
      directory at all, so a short contiguous ``1..12`` is valid and is not flagged.
    """
    chunks_dir = Path(chunks_dir)

    # --- chunk indices must be contiguous from 1 -----------------------------------
    indices = {}
    for chunk_dir in chunk_dirs:
        index = _chunk_index(chunk_dir)
        if index is None:
            print(
                f"WARNING: '{chunk_dir.name}' does not follow the 'chunk_<job_index>' "
                "naming scheme, skipping the chunk-index contiguity check."
            )
            indices = None
            break
        indices[index] = chunk_dir.name

    if indices:
        missing = sorted(set(range(1, len(indices) + 1)) - set(indices))
        if missing:
            raise ValueError(
                f"Missing chunk databases in {chunks_dir}.\n"
                f"  Chunk indices found:   {_format_indices(indices)}\n"
                f"  Chunk indices missing: {_format_indices(missing)}\n"
                "Each preprocessing job writes chunk_<job_index>, so the indices must run "
                "1..N without gaps. Only the trailing jobs may be absent: when --num_jobs "
                "exceeds the number of systems the surplus jobs are assigned an empty "
                "chunk and exit without creating a directory. A gap in the middle means "
                "one of the array tasks never finished, and the systems assigned to it "
                "would be silently missing from the merged database.\n"
                "Re-run the missing job index(es) (sbatch --array=<missing indices>) "
                "before merging, or point --chunks_dir at a directory holding only the "
                "chunks you intend to merge."
            )

    # --- no system_id may appear in more than one chunk ----------------------------
    seen = {}
    for chunk_dir in chunk_dirs:
        try:
            system_ids = _read_chunk_system_ids(chunk_dir)
        except Exception as e:
            print(
                f"WARNING: could not read system_ids from {chunk_dir.name} ({e}), "
                "skipping it in the duplicate check."
            )
            continue
        if system_ids is None:
            print(
                f"WARNING: {chunk_dir.name} stores no 'system_ids' metadata, skipping "
                "it in the duplicate check."
            )
            continue
        for system_id in system_ids:
            seen.setdefault(system_id, []).append(chunk_dir.name)

    duplicates = {
        system_id: names for system_id, names in seen.items() if len(names) > 1
    }
    if duplicates:
        shown = sorted(duplicates)[:20]
        listing = "\n".join(
            f"    {system_id}: {', '.join(duplicates[system_id])}" for system_id in shown
        )
        if len(duplicates) > len(shown):
            listing += f"\n    ... and {len(duplicates) - len(shown)} more"
        raise ValueError(
            f"Duplicate system_ids across the chunk databases in {chunks_dir}: "
            f"{len(duplicates)} system id(s) appear in more than one chunk.\n"
            f"{listing}\n"
            "This almost always means the directory still holds chunk_* directories from "
            "an earlier preprocessing run that used a different --num_jobs. A re-run only "
            "overwrites chunk_0001..chunk_<num_jobs>; every higher-numbered chunk of the "
            "previous run is left behind and merged in as well, because the merge globs "
            "chunk_* unconditionally.\n"
            "Merging them would store each of those complexes several times, and the "
            "random train/val/test split treats the copies as independent systems -- so "
            "the same complex can end up in train AND in val/test (train/test leakage).\n"
            "Fix it by either:\n"
            "  * removing the stale chunks and re-running preprocessing "
            f"(rm -rf {chunks_dir}/chunk_*, then re-submit the array with the --num_jobs "
            "you actually want), or\n"
            "  * pointing --chunks_dir at a directory that contains the chunks of a "
            "single run only."
        )


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

    # Find all chunk LMDB databases
    chunk_pattern = "chunk_*"
    chunk_dirs = sorted(chunks_dir.glob(chunk_pattern))

    if not chunk_dirs:
        raise ValueError(f"No chunk databases found matching pattern: {chunk_pattern}")

    print(f"Found {len(chunk_dirs)} chunk databases to fuse")

    # Reject stale/incomplete chunk sets before touching the output directory: see
    # check_chunks_consistent for why merging them silently corrupts the dataset.
    # Validating first also means a failed check leaves any previously fused database
    # intact instead of deleting it on the way to an abort.
    check_chunks_consistent(chunks_dir, chunk_dirs)

    output_path.mkdir(parents=True, exist_ok=True)

    # If the lmdb files in output directory exist, remove them
    lmdb_file = output_path / "data.mdb"
    lock_file = output_path / "lock.mdb"

    if lmdb_file.exists():
        lmdb_file.unlink()
    if lock_file.exists():
        lock_file.unlink()

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

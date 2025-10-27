import argparse
import pickle
import shutil
import time
from collections import defaultdict
from pathlib import Path

import lmdb
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

import flowr.util.rdkit as smolRD
from flowr.data.datasets.complex_data.preprocess_util import (
    calculate_docking_score,
    extract_affinity_data_from_csv,
    extract_affinity_data_from_mol,
)
from flowr.data.preprocess_pdbs import (
    process_complex,
)


def main():
    parser = argparse.ArgumentParser(description="Process data complexes in chunks")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        required=True,
        help="Directory containing PDB and SDF files",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./processed",
        required=True,
        help="Directory to save processed systems",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        help="Path to metadata CSV file that contains affinity labels",
    )
    parser.add_argument(
        "--file_type",
        type=str,
        choices=["pdb", "cif"],
        default="pdb",
        required=True,
        help="File type of protein structures",
    )
    parser.add_argument(
        "--calc_gnina_score",
        action="store_true",
        help="Calculate GNINA docking scores",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=12,
        help="Number of workers for parallel processing",
    )
    parser.add_argument(
        "--num_jobs", type=int, required=True, help="Total number of jobs"
    )
    parser.add_argument(
        "--job_index", type=int, required=True, help="Current job index (1-indexed)"
    )
    parser.add_argument(
        "--commit_interval", type=int, default=50, help="Commit interval for LMDB"
    )

    # Processing parameters
    parser.add_argument(
        "--add_hs", action="store_true", help="Add hydrogens to ligands"
    )
    parser.add_argument(
        "--add_hs_and_optimize",
        action="store_true",
        help="Add hydrogens and optimize ligands",
    )
    parser.add_argument(
        "--remove_hs", action="store_true", help="Remove hydrogens from ligands"
    )
    parser.add_argument("--kekulize", action="store_true", help="Kekulize ligands")
    parser.add_argument(
        "--add_bonds_to_protein", action="store_true", help="Add bonds to protein"
    )
    parser.add_argument(
        "--add_hs_to_protein",
        action="store_true",
        help="Add hydrogens to protein (will be optimized using Hydride)",
    )
    parser.add_argument(
        "--use_pdbfixer", action="store_true", help="Use PDBFixer for processing"
    )
    parser.add_argument(
        "--pocket_cutoff",
        type=float,
        default=7.0,
        help="Cutoff distance for pocket extraction",
    )
    parser.add_argument(
        "--cut_pocket", action="store_true", help="Cut out pocket from protein"
    )
    parser.add_argument(
        "--max_pocket_size",
        type=int,
        default=800,
        help="Maximum pocket size threshold",
    )
    parser.add_argument(
        "--min_pocket_size", type=int, default=10, help="Minimum pocket size threshold"
    )
    parser.add_argument(
        "--compute_interactions",
        action="store_true",
        help="Compute protein-ligand interactions",
    )
    parser.add_argument(
        "--pocket_type",
        type=str,
        default="holo",
        help='Type of pocket ("holo" or "apo")',
    )

    args = parser.parse_args()

    data_path = Path(args.data_dir)
    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Find all PDB or CIF files and get system IDs
    pdb_files = list(data_path.glob(f"*.{args.file_type}"))
    system_ids = [f.stem for f in pdb_files]

    print(f"Found {len(system_ids)} total systems")

    # Chunk the systems
    chunk_size = (len(system_ids) + args.num_jobs - 1) // args.num_jobs
    chunks = [
        system_ids[i : i + chunk_size] for i in range(0, len(system_ids), chunk_size)
    ]
    chunk_systems = chunks[args.job_index - 1]  # 1-indexed

    print(
        f"Job {args.job_index}/{args.num_jobs}: Processing {len(chunk_systems)} systems"
    )

    # Create LMDB database for this chunk
    lmdb_path = save_dir / f"chunk_{args.job_index:04d}"
    if lmdb_path.exists():
        shutil.rmtree(lmdb_path)

    lmdb_env = lmdb.open(
        str(lmdb_path),
        map_size=500 * 1024**3,  # 500 GB
        max_dbs=1,
        writemap=False,
        lock=True,
        readahead=False,  # Better for sequential writes
    )

    # Processing parameters
    processing_params = {
        "remove_hs": args.remove_hs,
        "add_hs": args.add_hs,
        "add_hs_and_optimize": args.add_hs_and_optimize,
        "kekulize": args.kekulize,
        "use_pdbfixer": args.use_pdbfixer,
        "add_bonds_to_protein": args.add_bonds_to_protein,
        "add_hs_to_protein": args.add_hs_to_protein and not args.use_pdbfixer,
        "pocket_cutoff": args.pocket_cutoff,
        "cut_pocket": args.cut_pocket,
        "max_pocket_size": args.max_pocket_size,
        "min_pocket_size": args.min_pocket_size,
        "compute_interactions": args.compute_interactions,
        "pocket_type": args.pocket_type,
    }

    start_time = time.time()
    processed_count = 0
    failed_count = 0
    missing_files = 0
    lengths_full = []
    lengths_no_ligand_pocket_hs = []
    lengths_no_pocket_hs = []
    processed_system_ids = []
    rdkit_mols = []
    ligand_atom_types = defaultdict(int)
    protein_residue_types = defaultdict(int)
    protein_atom_names = defaultdict(int)
    protein_atom_types = defaultdict(int)

    # Metadata
    if args.metadata_file is not None:
        index_df = pd.read_csv(args.metadata_file)

    # Initialize LMDB transaction
    txn = lmdb_env.begin(write=True)

    try:
        for system_id in tqdm(chunk_systems, desc=f"Processing job {args.job_index}"):
            pdb_file = data_path / f"{system_id}.{args.file_type}"
            sdf_file = data_path / f"{system_id}.sdf"

            if not (pdb_file.exists() and sdf_file.exists()):
                missing_files += 1
                print(f"Missing files for {system_id}: {pdb_file}, {sdf_file}")
                continue

            # Extract affinity data if available
            if args.metadata_file is not None:
                affinity = extract_affinity_data_from_csv(index_df, system_id)
            else:
                mol = Chem.SDMolSupplier(str(sdf_file), removeHs=False)[0]
                affinity = extract_affinity_data_from_mol(mol)

            # Process docking data, if not already present in affinity
            docking_data = calculate_docking_score(
                sdf_file=str(sdf_file),
                pdb_file=str(pdb_file),
                system_id=system_id,
                calc_vina_score=args.calc_vina_score and "vina_score" not in affinity,
                calc_gnina_score=args.calc_gnina_score
                and "gnina_score" not in affinity,
                num_workers=args.num_workers,
            )

            # Add GNINA scores to affinity data if available
            if "gnina_score" not in affinity:
                affinity["gnina_score"] = docking_data.get("gnina_score", None)

            try:
                # Process the complex
                complex_system = process_complex(
                    pdb_path=str(pdb_file),
                    ligand_sdf_path=str(sdf_file),
                    affinity=affinity,
                    **processing_params,
                )
            except Exception as e:
                print(f"Error processing system {system_id}: {e}")
                failed_count += 1
                continue

            if complex_system is not None:
                try:
                    # NOTE: At training time pocket won't have hydrogens (default)!
                    length_full = complex_system.seq_length
                    length_no_pocket_hs = complex_system.remove_hs(
                        include_ligand=False
                    ).seq_length
                    length_no_ligand_pocket_hs = complex_system.remove_hs(
                        include_ligand=True
                    ).seq_length

                    rdkit_mol = complex_system.ligand.to_rdkit()
                    rdkit_mol_bytes = rdkit_mol.ToBinary()
                    rdkit_mols.append(rdkit_mol_bytes)

                except Exception as e:
                    print(f"Error calculating sequence length for {system_id}: {e}")
                    failed_count += 1
                    continue
                try:
                    # Serialize and save to LMDB
                    serialized_system = complex_system.to_bytes()
                    key = str(processed_count).encode("utf-8")
                    txn.put(key, serialized_system)
                except Exception as e:
                    print(f"Error serializing system {system_id}: {e}")
                    failed_count += 1
                    continue

                processed_count += 1

                # Store sequence length metadata
                lengths_full.append(length_full)
                lengths_no_pocket_hs.append(length_no_pocket_hs)
                lengths_no_ligand_pocket_hs.append(length_no_ligand_pocket_hs)
                processed_system_ids.append(system_id)

                # Store atom and residue types
                for atom in complex_system.ligand.atomics:
                    ligand_atom_types[smolRD.PT.symbol_from_atomic(int(atom))] += 1
                for row in complex_system.holo.atoms:
                    protein_atom_types[row.element] += 1
                    protein_atom_names[row.atom_name] += 1
                    protein_residue_types[row.res_name] += 1

                # Commit at intervals
                if processed_count % args.commit_interval == 0:
                    txn.commit()
                    txn = lmdb_env.begin(write=True)
            else:
                print(f"Complex system is None for {system_id}")
                failed_count += 1

        # Store metadata in LMDB
        metadata = {
            "__len__": processed_count,
            "lengths_full": lengths_full,
            "lengths_no_ligand_pocket_hs": lengths_no_ligand_pocket_hs,
            "lengths_no_pocket_hs": lengths_no_pocket_hs,
            "system_ids": processed_system_ids,
            "ligand_atom_types": dict(ligand_atom_types),
            "protein_residue_types": dict(protein_residue_types),
            "protein_atom_names": dict(protein_atom_names),
            "protein_atom_types": dict(protein_atom_types),
            "rdkit_mols": rdkit_mols,
        }

        for key, value in metadata.items():
            txn.put(key.encode() if isinstance(key, str) else key, pickle.dumps(value))

    except Exception as e:
        print(f"Processing failed: {e}")
    finally:
        # Final commit and cleanup
        try:
            txn.commit()
        except Exception:
            print("Final commit failed")
            pass
        lmdb_env.sync()
        lmdb_env.close()

    end_time = time.time()

    # Print summary
    print(f"\n=== Job {args.job_index} Summary ===")
    print(f"Total systems in chunk: {len(chunk_systems)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed: {failed_count}")
    print(f"Missing files: {missing_files}")
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print(f"LMDB saved to: {lmdb_path}")


if __name__ == "__main__":
    main()

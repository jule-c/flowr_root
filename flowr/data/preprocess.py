import argparse
import os
import resource
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import scipy as scipy
from plinder.core import PlinderSystem
from tqdm import tqdm

from flowr.util.apo import load_apo_pocket
from flowr.util.plinder import (
    cleanup_group,
    filter_systems,
    group_systems,
    holo_pocket_from_structure,
    load_complex_structure,
    load_holo_structure,
    load_ligand,
    load_system_metadata,
    run_prep_wizard,
    setup_plinder_env,
)
from flowr.util.pocket import PocketComplex, PocketComplexBatch

PLINDER_RELEASE = "2024-06"
PLINDER_ITERATION = "v2"


# Setup environment before importing plinder
os.environ["PLINDER_RELEASE"] = PLINDER_RELEASE
os.environ["PLINDER_ITERATION"] = PLINDER_ITERATION
os.environ["PLINDER_REPO"] = str(Path.home() / "plinder-org/plinder")
os.environ["PLINDER_LOCAL_DIR"] = str(Path.home() / ".local/share/plinder")
os.environ["GCLOUD_PROJECT"] = "plinder"
os.environ["PLINDER_LOG_LEVEL"] = "30"

warnings.filterwarnings("ignore", "Attribute .*", UserWarning)


DEFAULT_N_WORKERS = 8
DEFAULT_TMP_PATH = os.environ.get("TMP", ".tmp")
DEFAULT_CONFIG_PATH = "config.yaml"

GROUP_CODES = ["0"] + [chr(i) for i in range(97, 123)]
GROUP_IDX_CODE_MAP = {idx: code for idx, code in enumerate(GROUP_CODES)}


# Need to ensure the limits are large enough for running multiple processes reading files simultaneously
def configure_fs(limit=4096):
    """
    Try to increase the limit on open file descriptors
    """

    n_file_resource = resource.RLIMIT_NOFILE
    soft_limit, hard_limit = resource.getrlimit(n_file_resource)

    print(f"Current limits (soft, hard): {(soft_limit, hard_limit)}")

    if limit > soft_limit:
        try:
            print(f"Attempting to increase open file limit to {limit}...")
            resource.setrlimit(n_file_resource, (limit, hard_limit))
            print("Limit changed successfully!")

        except:
            print("Limit change unsuccessful.")

    else:
        print("Open file limit already sufficiently large.")


def process_complex(args, system):
    # Load ligand structure in smol format
    ligand = load_ligand(system)

    if args.prep_wizard:
        # Load the full complex (protein and ligand) and apply protein prep wizard
        complex_structure = load_complex_structure(
            system, include_hs=False, include_hetero=False
        )
        holo_structure, ligand = run_prep_wizard(
            system.system_id, complex_structure, args.tmp_path, args.config_path
        )

    else:
        # Load holo protein without hydrogens (if there are any) and without hetero residues
        holo_structure = load_holo_structure(
            system, include_hs=False, include_hetero=False
        )

    return holo_structure, ligand


def process_apo(args, system, holo_structure, holo_pocket):
    # Load apo pocket
    if args.include_apo:
        apo_pocket, apo_type = load_apo_pocket(
            system,
            holo_structure,
            holo_pocket.atoms,
            args.tmp_path,
            args.config_path,
            prepwizard=args.prepwizard,
        )

    else:
        apo_pocket = None
        apo_type = None

    return apo_pocket, apo_type


# Assumes the system has a single protein chain and a single ligand
def process_system(args, system_id, split):
    system = PlinderSystem(system_id=system_id)

    if len(system.system["ligands"]) > 1:
        print(f"WARNING -- multiple ligands in system {system_id}")

    # Collect holo pocket and ligand, convert both to smol format
    holo_structure, ligand = process_complex(args, system)
    holo_pocket = holo_pocket_from_structure(system, holo_structure)

    # Sort out apo structure, if needed
    apo_pocket, apo_type = process_apo(args, system, holo_structure, holo_pocket)

    # Load the metadata from the plinder system object
    metadata = load_system_metadata(system)
    metadata["apo_type"] = apo_type
    metadata["split"] = split

    complex = PocketComplex(
        holo=holo_pocket, ligand=ligand, apo=apo_pocket, metadata=metadata
    )
    complex.store_metrics_()

    return complex


def process_system_wrapper(args, system_id, split):
    error = None
    system = None

    try:
        system = process_system(args, system_id, split)
    except Exception as e:
        error = e

    return system, error


# Assumes that all system_ids share the same pdb_code
def process_group(args, pdb_code, system_ids, splits):
    systems = []
    errors = []

    for system_id, split in zip(system_ids, splits):
        system, error = process_system_wrapper(args, system_id, split)
        systems.append(system)
        errors.append(error)

    if args.delete_tmp:
        cleanup_group(pdb_code, PLINDER_RELEASE, PLINDER_ITERATION)

    return systems, errors


def save_systems_(args, systems, group_code):
    batch = PocketComplexBatch(systems)
    save_dir = Path(args.save_path) / "intermediate"
    save_dir.mkdir(exist_ok=True, parents=True)
    save_file = save_dir / f"{group_code}.smol"
    bytes_data = batch.to_bytes()
    save_file.write_bytes(bytes_data)
    print(f"Saved {len(systems)} systems to {save_file.resolve()}")


def process_systems_sequential(args, grouped_ids, grouped_splits):
    print("Processing sequentially...")

    system_ids = []
    ordered_systems = []
    errors = []

    n_systems = sum([len(systems) for systems in grouped_ids.values()])

    with tqdm(total=n_systems) as pbar:
        for group_code, group_ids in list(grouped_ids.items()):
            splits = grouped_splits[group_code]
            result_systems, result_errors = process_group(
                args, group_code, group_ids, splits
            )

            system_ids.extend(group_ids)
            ordered_systems.extend(result_systems)
            errors.extend(result_errors)

            pbar.update(len(result_systems))

    return system_ids, ordered_systems, errors


def process_systems_parallel(args, grouped_ids, grouped_splits):
    print(f"Processing in parallel using {args.n_workers} worker processes...")

    executor = ProcessPoolExecutor(max_workers=args.n_workers)

    futures = []
    group_codes = []

    for group_code, group_ids in list(grouped_ids.items()):
        splits = grouped_splits[group_code]
        future = executor.submit(process_group, args, group_code, group_ids, splits)
        futures.append(future)
        group_codes.append(group_code)

    system_ids = []
    ordered_systems = []
    errors = []

    n_systems = sum([len(systems) for systems in grouped_ids.values()])

    with tqdm(total=n_systems) as pbar:
        for future, group_code in zip(futures, group_codes):
            result_systems, result_errors = future.result()

            system_ids.extend(grouped_ids[group_code])
            ordered_systems.extend(result_systems)
            errors.extend(result_errors)

            pbar.update(len(result_systems))

    executor.shutdown()

    return system_ids, ordered_systems, errors


def process_systems_(args, systems_dict, splits_dict, group_code):
    if args.n_workers < 0:
        raise ValueError(f"n_workers must be >= 0, got {args.n_workers}")
    elif args.n_workers == 0:
        process_fn = process_systems_sequential
    else:
        process_fn = process_systems_parallel

    system_ids, processed_systems, errors = process_fn(args, systems_dict, splits_dict)

    result_count = {"success": 0}
    for error in errors:
        if error is None:
            result_count["success"] += 1
            continue

        error_name = type(error).__name__
        if result_count.get(error_name) is None:
            result_count[error_name] = 0

        result_count[error_name] += 1

    print("Processing complete. Outcome:")
    print(result_count)

    complete_systems = [system for system in processed_systems if system is not None]
    error_indices = [idx for idx, err in enumerate(errors) if err is not None]
    failed_system_ids = [system_ids[idx] for idx in error_indices]
    error_objs = [errors[idx] for idx in error_indices]

    print()
    print(f"Saving {len(complete_systems)} systems to {args.save_path}")
    save_systems_(args, complete_systems, group_code)

    return failed_system_ids, error_objs


def print_dataset_info(system_ids, splits):
    split_counts = {}
    for split in ["train", "val", "test"]:
        count = len([spl for spl in splits if spl == split])
        split_counts[split] = count

    print(
        f"Got {len(system_ids)} systems in full dataset after filtering, split into the following:"
    )
    for split, count in split_counts.items():
        print(f"{split:<8}{count}")


def main(args):
    print("Running preprocess script...")

    setup_plinder_env(args.data_path, PLINDER_RELEASE, PLINDER_ITERATION)
    configure_fs()

    print("Loading index files and filtering systems...")
    system_ids, system_splits = filter_systems(
        args.data_path, PLINDER_RELEASE, PLINDER_ITERATION, keep_covalent=False
    )
    print("Filtering complete.\n")

    # process_system(args, system_ids[50], system_splits[50])

    # # Randomly select a small number of systems for now
    # import numpy as np
    # n_systems = 5000
    # idxs = np.random.choice(len(system_ids), size=n_systems, replace=False)
    # system_ids = [system_ids[idx] for idx in idxs]
    # system_splits = [system_splits[idx] for idx in idxs]

    grouped_systems, grouped_splits = group_systems(system_ids, system_splits)
    group_code = GROUP_IDX_CODE_MAP.get(args.group_index)

    if group_code is None:
        min_idx, max_idx = (
            min(GROUP_IDX_CODE_MAP.keys()),
            max(GROUP_IDX_CODE_MAP.keys()),
        )
        raise ValueError(
            f"Group index must be between {min_idx} and {max_idx}, got {args.group_index}"
        )

    print_dataset_info(system_ids, system_splits)

    systems_dict = grouped_systems[group_code]
    splits_dict = grouped_splits[group_code]

    # # Just take some pdb code groups to test
    # pdb_group_codes = list(systems_dict.keys())[:8]
    # systems_dict = {code: systems_dict[code] for code in pdb_group_codes}
    # splits_dict = {code: splits_dict[code] for code in pdb_group_codes}

    n_systems = sum([len(system_ids) for system_ids in systems_dict.values()])

    print()
    print(f"Processing {n_systems} systems with group code {group_code}...")
    failed_ids, error_types = process_systems_(
        args, systems_dict, splits_dict, group_code
    )
    print("Processing complete.\n")

    if len(failed_ids) > 0:
        print("The following system ids failed:")
        for sys_id, err in zip(failed_ids, error_types):
            print(f"{sys_id} -- {type(err).__name__} -- {str(err)}")

    print("Preprocess script complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Plinder dataset preprocess script")

    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--group_index", type=int)

    parser.add_argument("--n_workers", type=int, default=DEFAULT_N_WORKERS)
    parser.add_argument("--tmp_path", type=str, default=DEFAULT_TMP_PATH)
    parser.add_argument("--config_path", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--include_apo", action="store_true")
    parser.add_argument("--no_seq_match", action="store_false", dest="seq_match")
    parser.add_argument("--no_delete_tmp", action="store_false", dest="delete_tmp")
    parser.add_argument("--no_prep_wizard", action="store_false", dest="prep_wizard")

    parser.set_defaults(
        include_apo=False, seq_match=True, delete_tmp=True, prep_wizard=True
    )

    args = parser.parse_args()
    main(args)

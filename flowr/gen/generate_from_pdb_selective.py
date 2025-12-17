import argparse
import signal
import time
import warnings
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from tqdm import tqdm

import flowr.gen.utils as util
import flowr.util.rdkit as smolRD
from flowr.data.dataset import GeometricDataset
from flowr.gen.generate import generate_ligands_per_target_selective
from flowr.scriptutil import (
    load_model,
)
from flowr.util.device import get_device
from flowr.util.functional import (
    LigandPocketOptimization,
)
from flowr.util.metrics import evaluate_pb_validity
from flowr.util.pocket import PocketComplexBatch
from flowr.util.rdkit import write_sdf_file

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Default script arguments
DEFAULT_BUCKET_COST_SCALE = "quadratic"
DEFAULT_INTEGRATION_STEPS = 100
DEFAULT_CAT_SAMPLING_NOISE_LEVEL = 1
DEFAULT_ODE_SAMPLING_STRATEGY = "linear"
DEFAULT_CATEGORICAL_STRATEGY = "uniform-sample"


def create_list_defaultdict():
    return defaultdict(list)


def get_dataset(system, transform, vocab, interpolant, args, hparams):
    systems = PocketComplexBatch([system])
    dataset = GeometricDataset(
        systems, data_cls=PocketComplexBatch, transform=transform
    )
    dataset = dataset.sample_n_molecules_per_target(args.sample_n_molecules_per_target)
    if args.gpus > 1:
        dataset = dataset.split(idx=args.mp_index, n_chunks=args.gpus)
    return dataset


def evaluate(args):
    # set seeds
    torch.random.manual_seed(args.mp_index + args.seed)
    np.random.seed(args.mp_index + args.seed)
    # Set precision
    torch.set_float32_matmul_precision("high")

    # Load hyperparameter
    print(f"Using model stored at {args.ckpt_path}")

    print("Loading model...")
    (
        model,
        hparams,
        vocab,
        vocab_charges,
        vocab_hybridization,
        vocab_aromatic,
        vocab_pocket_atoms,
        vocab_pocket_res,
    ) = load_model(
        args,
    )
    model = model.to(get_device())
    model.eval()
    print("Model complete.")

    # load util
    transform, interpolant = util.load_util(
        args,
        hparams,
        vocab,
        vocab_charges,
        vocab_hybridization,
        vocab_aromatic,
    )

    # Load guidance parameters
    guidance_params = util.get_guidance_params(args)

    # Load the data
    system_target, system_untarget = util.load_data_from_pdb_selective(
        args,
        remove_hs=hparams["remove_hs"],
        remove_aromaticity=hparams["remove_aromaticity"],
    )
    dataset_target = get_dataset(
        system_target, transform, vocab, interpolant, args, hparams
    )
    dataset_untarget = get_dataset(
        system_untarget, transform, vocab, interpolant, args, hparams
    )

    if args.gpus > 1:
        sample_n_molecules_per_target = dataset_target.__len__()
    else:
        sample_n_molecules_per_target = args.sample_n_molecules_per_target

    print("\nStarting sampling...\n")
    k = 0
    num_ligands = 0
    all_gen_ligs = []
    all_gen_pdbs = []
    gen_pdbs = None
    start = time.time()
    out_dict = defaultdict(list)
    while num_ligands < sample_n_molecules_per_target and k <= args.max_sample_iter:
        print(
            f"...Sampling iteration {k + 1}...",
            end="\r",
        )
        dataloader_target = util.get_dataloader(
            args, dataset_target, interpolant, iter=k
        )
        dataloader_untarget = util.get_dataloader(
            args, dataset_untarget, interpolant, iter=k
        )

        dataloader = zip(dataloader_target, dataloader_untarget)
        for i, (batch_target, batch_untarget) in enumerate(
            tqdm(dataloader, desc="Sampling", leave=False)
        ):
            prior, data_target, _, _ = batch_target
            _, data_untarget, _, _ = batch_untarget
            if args.arch == "pocket_flex":
                gen_ligs, gen_pdbs = generate_ligands_per_target_selective(
                    args,
                    model,
                    prior=prior,
                    posterior_target=data_target,
                    posterior_untarget=data_untarget,
                    pocket_noise=args.pocket_noise,
                    save_traj=False,
                    iter=f"{k}_{i}",
                    guidance_params=guidance_params,
                )
            else:
                gen_ligs = generate_ligands_per_target_selective(
                    args,
                    model,
                    prior=prior,
                    posterior_target=data_target,
                    posterior_untarget=data_untarget,
                    pocket_noise=args.pocket_noise,
                    save_traj=False,
                    iter=f"{k}_{i}",
                    guidance_params=guidance_params,
                )
            if args.filter_valid_unique:
                if gen_pdbs:
                    gen_ligs, gen_pdbs = smolRD.sanitize_list(
                        gen_ligs,
                        pdbs=gen_pdbs,
                        filter_uniqueness=True,
                        filter_pdb=True,
                        sanitize=True,
                    )
                else:
                    gen_ligs = smolRD.sanitize_list(
                        gen_ligs,
                        filter_uniqueness=True,
                        sanitize=True,
                    )
            all_gen_ligs.extend(gen_ligs)
            if gen_pdbs:
                all_gen_pdbs.extend(gen_pdbs)
            num_ligands += len(gen_ligs)
        if args.filter_valid_unique:
            print(
                f"Validity rate: {round(len(all_gen_ligs) / sample_n_molecules_per_target, 2)}"
            )
        k += 1

    run_time = time.time() - start
    if num_ligands == 0:
        raise (
            f"Reached {args.max_sample_iter} sampling iterations, but could not find any ligands."
        )
    elif num_ligands < args.sample_n_molecules_per_target:
        print(
            f"FYI: Reached {args.max_sample_iter} sampling iterations, but could only find {num_ligands} ligands."
        )
    elif num_ligands > args.sample_n_molecules_per_target:
        all_gen_ligs = all_gen_ligs[: args.sample_n_molecules_per_target]
        if all_gen_pdbs:
            all_gen_pdbs = all_gen_pdbs[: args.sample_n_molecules_per_target]

    if not args.filter_valid_unique:
        # Remove all Nones from the generated ligands
        if gen_pdbs:
            all_gen_ligs, all_gen_pdbs = smolRD.sanitize_list(
                all_gen_ligs,
                pdbs=all_gen_pdbs,
                filter_uniqueness=False,
                filter_pdb=True,
                sanitize=True,
            )
        else:
            all_gen_ligs = smolRD.sanitize_list(
                all_gen_ligs,
                filter_uniqueness=False,
                sanitize=True,
            )

    ref_ligs = model._generate_ligs(
        data_target, lig_mask=data_target["lig_mask"].bool(), scale=model.coord_scale
    )[0]
    ref_lig_with_hs = model.retrieve_ligs_with_hs(data_target, save_idx=0)
    ref_pdb = model.retrieve_pdbs(
        data_target, save_dir=Path(args.save_dir) / "ref_pdbs", save_idx=0
    )
    ref_pdb_with_hs = model.retrieve_pdbs_with_hs(
        data_target, save_dir=Path(args.save_dir) / "ref_pdbs", save_idx=0
    )

    out_dict["gen_ligs"] = all_gen_ligs
    if args.arch == "pocket_flex":
        assert len(all_gen_pdbs) == len(all_gen_ligs)
        out_dict["gen_pdbs"] = all_gen_pdbs
    out_dict["ref_lig"] = ref_ligs
    out_dict["ref_lig_with_hs"] = ref_lig_with_hs
    out_dict["ref_pdb"] = ref_pdb
    out_dict["ref_pdb_with_hs"] = ref_pdb_with_hs
    out_dict["run_time"] = run_time

    print(
        f"\n Run time={round(run_time, 2)}s for {len(out_dict['gen_ligs'])} molecules \n"
    )

    # Filter by diversity
    print("Filtering ligands by diversity...")
    if args.arch == "pocket_flex":
        all_gen_ligs, all_gen_pdbs = util.filter_diverse_ligands_bulk(
            all_gen_ligs, all_gen_pdbs, threshold=0.9
        )
    else:
        all_gen_ligs = util.filter_diverse_ligands_bulk(all_gen_ligs, threshold=0.995)
    print(
        f"Number of ligands after filtering by diversity: {len(all_gen_ligs)} ligands ({args.sample_n_molecules_per_target - len(all_gen_ligs)} removed)"
    )

    # PoseBusters validity
    if args.filter_pb_valid:
        pb_valid = evaluate_pb_validity(
            all_gen_ligs,
            pdb_file=ref_pdb_with_hs,
            return_list=True,
        )
        all_gen_ligs = [lig for lig, valid in zip(all_gen_ligs, pb_valid) if valid]
        print(
            f"PB-validity (mean): {np.mean(pb_valid)}, PB-validity (std): {np.std(pb_valid)}"
        )

    # Protonate generated ligands:
    if args.add_hs_and_optimize_gen_ligs:
        assert hparams[
            "remove_hs"
        ], "The model outputs protonated ligands, no need for additional protonation."
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        print("Protonating ligands...")
        optimizer = LigandPocketOptimization(
            pocket_cutoff=args.pocket_cutoff, strip_invalid=True
        )
        # Create a partial function binding pdb_file and optimizer
        if args.arch == "pocket_flex":
            process_lig_partial = partial(
                util.process_lig_wrapper,
                optimizer=optimizer,
                optimize_pocket_hs=True,
                process_pocket=True,
            )
            with Pool(processes=args.num_workers) as pool:
                all_gen_ligs_hs = pool.map(
                    process_lig_partial, zip(all_gen_ligs, all_gen_pdbs)
                )
        else:
            process_lig_partial = partial(
                util.process_lig,
                pdb_file=ref_pdb_with_hs,
                optimizer=optimizer,
                process_pocket=True,
            )
            with Pool(processes=args.num_workers) as pool:
                all_gen_ligs_hs = pool.map(process_lig_partial, all_gen_ligs)

        all_gen_ligs_hs = [Chem.Mol(lig) for lig in all_gen_ligs_hs]
        out_dict["gen_ligs_hs"] = all_gen_ligs_hs
        print("Done!")

    # Save out_dict as pickle file
    target_name = (
        Path(args.pdb_file_target).stem if args.pdb_file_target else args.pdb_id
    )
    if args.filter_valid_unique:
        predictions = (
            Path(args.save_dir) / f"samples_{target_name}_batch_{args.mp_index}.pt"
        )
    else:
        predictions = (
            Path(args.save_dir)
            / f"samples_unfiltered_{target_name}_batch_{args.mp_index}.pt"
        )
    torch.save(out_dict, str(predictions))

    # Save ligands as SDF
    sdf_dir = Path(args.save_dir) / f"samples_{target_name}_batch_{args.mp_index}.sdf"
    write_sdf_file(sdf_dir, all_gen_ligs, name=target_name)
    if args.add_hs_and_optimize_gen_ligs:
        sdf_dir = (
            Path(args.save_dir)
            / f"samples_{target_name}_batch_{args.mp_index}_protonated.sdf"
        )
        write_sdf_file(sdf_dir, all_gen_ligs_hs, name=target_name)

    # Save ligand-pocket complexes
    if args.arch == "pocket_flex":
        gen_complexes_dir = Path(args.save_dir) / "gen_complexes_protonated"
        if not gen_complexes_dir.exists():
            gen_complexes_dir.mkdir(parents=True, exist_ok=True)
        if args.add_hs_and_optimize_gen_ligs:
            util.write_ligand_pocket_complex_pdb(
                all_gen_ligs_hs,
                all_gen_pdbs,
                gen_complexes_dir,
                complex_name=target_name,
            )
        util.write_ligand_pocket_complex_pdb(
            all_gen_ligs,
            all_gen_pdbs,
            gen_complexes_dir,
            complex_name=target_name,
        )

    print(f"Samples saved to {str(args.save_dir)}")
    print("Sampling finished.")

    if args.compute_interaction_recovery:
        print("Computing interaction recovery...")
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        # Split the list into chunks of size args.num_workers
        chunks = [
            all_gen_ligs_hs[i : i + args.num_workers]
            for i in range(0, len(all_gen_ligs_hs), args.num_workers)
        ]
        process_interaction_partial = partial(
            util.process_interaction,
            ref_lig=ref_lig_with_hs,
            pdb_file=ref_pdb_with_hs,
            pocket_cutoff=args.pocket_cutoff,
            save_dir=args.save_dir,
            remove_hs=hparams["remove_hs"],
            add_hs_and_optimize_gen_ligs=args.add_hs_and_optimize_gen_ligs,
        )
        recovery_rates = []
        tanimoto_sims = []
        with Pool(processes=args.num_workers) as pool:
            for chunk in chunks:
                # Process each chunk in parallel.
                chunk_results = pool.map(process_interaction_partial, chunk)
                recovery_rates.extend([res["recovery_rate"] for res in chunk_results])
                tanimoto_sims.extend([res["tanimoto_sim"] for res in chunk_results])

        recovery_rates = [result for result in recovery_rates if result is not None]
        tanimoto_sims = [result for result in tanimoto_sims if result is not None]
        out_dict["interaction_recovery"] = recovery_rates
        out_dict["tanimoto_sims"] = tanimoto_sims
        print(f"Interaction recovery rate: {np.nanmean(recovery_rates)}")
        print(f"Interaction Tanimoto similarity: {np.nanmean(tanimoto_sims)}")


def parse_substructure(value):
    """
    Parse substructure argument as either a SMILES/SMARTS string or an integer.

    Args:
        value: String input from argparse

    Returns:
        str or int: SMILES/SMARTS string or atom index as integer
    """
    # Try to parse as integer
    try:
        return int(value)
    except ValueError:
        # If not an integer, treat as SMILES/SMARTS string
        return value


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--pdb_id', type=str, default=None)
    parser.add_argument('--ligand_id', type=str, default=None)
    parser.add_argument('--pdb_file_target', type=str, default=None)
    parser.add_argument('--ligand_file_target', type=str, default=None)
    parser.add_argument('--pdb_file_untarget', type=str, default=None)
    parser.add_argument('--ligand_file_untarget', type=str, default=None)
    parser.add_argument('--res_txt_file', type=str, default=None)

    parser.add_argument('--pocket_noise', type=str, choices=["apo", "random", "fix"], default="fix")
    parser.add_argument('--cut_pocket', action='store_true')
    parser.add_argument('--pocket_cutoff', type=float, default=6.0)
    parser.add_argument('--protonate_pocket', action='store_true')
    parser.add_argument('--compute_interactions', action='store_true')
    parser.add_argument('--compute_interaction_recovery', action='store_true')
    parser.add_argument('--add_hs', action='store_true')
    parser.add_argument('--add_hs_and_optimize', action='store_true')
    parser.add_argument('--add_hs_and_optimize_gen_ligs', action='store_true')
    parser.add_argument('--kekulize', action='store_true')
    parser.add_argument('--use_pdbfixer', action='store_true')
    parser.add_argument('--add_bonds_to_protein', action='store_true')
    parser.add_argument('--add_hs_to_protein', action='store_true')
    parser.add_argument('--max_pocket_size', type=int, default=1000)
    parser.add_argument('--min_pocket_size', type=int, default=10)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--gpus", default=8, type=int)
    parser.add_argument('--mp_index', default=0, type=int)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--arch", type=str, choices=["pocket", "pocket_flex"], required=True)
    parser.add_argument(
        "--pocket_type", type=str, choices=["holo", "apo"], required=True
    )
    parser.add_argument(
        "--pocket_coord_noise_std", type=float, default=0.0,
        help="Standard deviation of the pocket coordinate noise"
    )
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--lora_finetuned", action="store_true")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--save_file", type=str)

    parser.add_argument("--coord_noise_scale", type=float, default=0.0)

    parser.add_argument("--max_sample_iter", type=int, default=20)
    parser.add_argument("--sample_n_molecules_per_target", type=int, default=1)
    parser.add_argument("--sample_mol_sizes", action="store_true")
    parser.add_argument("--corrector_iters", type=int, default=0)
    parser.add_argument("--rotation_alignment", action="store_true")
    parser.add_argument("--permutation_alignment", action="store_true")

    parser.add_argument("--filter_valid_unique", action="store_true")
    parser.add_argument("--filter_pb_valid", action="store_true")
    parser.add_argument("--filter_cond_substructure", action="store_true")
    parser.add_argument("--calc_strain", action="store_true")

    parser.add_argument("--batch_cost", type=int)
    parser.add_argument("--dataset_split", type=str, default=None)
    parser.add_argument("--ligand_time", type=float, default=None)
    parser.add_argument("--pocket_time", type=float, default=None)
    parser.add_argument("--interaction_time", type=float, default=None)
    parser.add_argument("--fixed_interactions", action="store_true")
    parser.add_argument("--interaction_inpainting", action="store_true")
    parser.add_argument("--scaffold_inpainting", action="store_true")
    parser.add_argument("--func_group_inpainting", action="store_true")
    parser.add_argument("--linker_inpainting", action="store_true")
    parser.add_argument("--fragment_inpainting", action="store_true")
    parser.add_argument("--max_fragment_cuts", type=int, default=3)
    parser.add_argument("--core_inpainting", action="store_true")
    parser.add_argument("--substructure_inpainting", action="store_true")
    parser.add_argument(
        "--substructure", 
        type=parse_substructure,
        nargs='+',  # This allows multiple space-separated values
        default=None,
        help="SMILES/SMARTS string or space-separated atom indices (e.g., '10 11 12 13' or 'c1ccccc1')"
    )
    parser.add_argument(
        "--graph_inpainting",
        default=None,
        type=str,
        choices=["conformer", "random", "harmonic"],
    )
    parser.add_argument("--separate_pocket_interpolation", action="store_true")
    parser.add_argument("--separate_interaction_interpolation", action="store_true")
    parser.add_argument(
        "--integration_steps", type=int, default=DEFAULT_INTEGRATION_STEPS
    )
    parser.add_argument(
        "--cat_sampling_noise_level", type=int, default=DEFAULT_CAT_SAMPLING_NOISE_LEVEL
    )
    parser.add_argument(
        "--ode_sampling_strategy", type=str, default=DEFAULT_ODE_SAMPLING_STRATEGY
    )
    parser.add_argument(
        "--solver", type=str, default="euler", choices=["euler", "midpoint"]
    )
    parser.add_argument(
        "--categorical_strategy", type=str, default=DEFAULT_CATEGORICAL_STRATEGY
    )
    parser.add_argument("--use_sde_simulation", action="store_true")
    parser.add_argument("--use_cosine_scheduler", action="store_true")
    parser.add_argument(
        "--bucket_cost_scale", type=str, default=DEFAULT_BUCKET_COST_SCALE
    )
    parser.add_argument("--guidance_config", default=None)


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    evaluate(args)

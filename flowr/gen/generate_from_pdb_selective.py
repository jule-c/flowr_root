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

import flowr.util.rdkit as smolRD
from flowr.data.dataset import GeometricDataset
from flowr.gen.generate_from_lmdb import get_guidance_params
from flowr.gen.utils import (
    filter_diverse_ligands_bulk,
    get_dataloader,
    load_data_from_pdb_selective,
    load_util,
    process_interaction,
    process_lig,
    process_lig_wrapper,
    write_ligand_pocket_complex_pdb,
)
from flowr.scriptutil import (
    load_model,
)
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


def generate_ligands_per_target_selective(
    args,
    model,
    prior,
    posterior_target,
    posterior_untarget,
    pocket_noise="fix",
    device="cuda",
    save_traj=False,
    iter="",
    guidance_params: dict | None = None,
):
    """
    Generate ligands for a specific protein target using flow-based molecular generation.

    This function performs molecular generation by flowing from a prior distribution to the
    target data distribution, conditioning on protein pocket information. It supports both
    rigid and flexible pocket conformations depending on the model architecture.
    If inpainting is enabled, the function will also handle masked regions in the ligand
    during generation, where the prior is modified accordingly (masked prior).

    Args:
        args: Configuration object containing generation parameters including:
            - arch: Model architecture ("pocket_flex" for flexible pocket models)
            - integration_steps: Number of ODE integration steps for generation
            - ode_sampling_strategy: Strategy for ODE sampling ("euler", "heun", etc.)
            - solver: ODE solver type for numerical integration
            - corrector_iters: Number of corrector iterations for predictor-corrector methods
            - save_dir: Directory path for saving generated structures
        model: Trained flow-based molecular generation model with methods:
            - builder.extract_ligand_from_complex(): Extract ligand data from complex
            - builder.extract_pocket_from_complex(): Extract pocket data from complex
            - _generate(): Core generation method using flow matching
            - _generate_mols(): Convert generated coordinates to RDKit molecules
            - retrieve_pdbs(): Save generated structures as PDB files (for flexible models)
        prior: Prior distribution samples (noise) for ligand generation
        posterior: Target data containing ground truth ligand-pocket complexes
        pocket_noise: Type of pocket noise ("apo", "holo", "random")
        device (str, optional): PyTorch device for computation. Defaults to "cuda".
        save_traj (bool, optional): Whether to save generation trajectory. Defaults to False.
        iter (str, optional): Iteration identifier for file naming. Defaults to "".

    Returns:
        For rigid pocket models:
            list[rdkit.Chem.Mol]: Generated ligand molecules as RDKit Mol objects

        For flexible pocket models ("pocket_flex"):
            tuple: (generated_ligands, generated_pdbs) where:
                - generated_ligands: list[rdkit.Chem.Mol] - Generated ligand molecules
                - generated_pdbs: list[str] - Paths to generated PDB files with pocket conformations
    """

    assert (
        len(set([cmpl.metadata["system_id"] for cmpl in posterior_target["complex"]]))
        == 1
    ), "Sampling N ligands per target, but found mixed targets"

    # Get ligand and pocket data
    lig_prior = model.builder.extract_ligand_from_complex(prior)
    lig_prior["interactions"] = prior["interactions"]
    lig_prior["fragment_mask"] = prior["fragment_mask"]
    lig_prior = {k: v.cuda() if torch.is_tensor(v) else v for k, v in lig_prior.items()}

    # pocket data
    if args.arch == "pocket_flex" and pocket_noise == "apo":
        pocket_data_target = model.builder.extract_pocket_from_complex(prior)
    else:
        pocket_data_target = model.builder.extract_pocket_from_complex(posterior_target)
    pocket_data_target["interactions"] = posterior_target["interactions"]
    pocket_data_target["complex"] = posterior_target["complex"]
    pocket_data_target = {
        k: v.cuda() if torch.is_tensor(v) else v for k, v in pocket_data_target.items()
    }

    if args.arch == "pocket_flex" and pocket_noise == "apo":
        pocket_data_untarget = model.builder.extract_pocket_from_complex(prior)
    else:
        pocket_data_untarget = model.builder.extract_pocket_from_complex(
            posterior_untarget
        )
    pocket_data_untarget["interactions"] = posterior_untarget["interactions"]
    pocket_data_untarget["complex"] = posterior_untarget["complex"]
    pocket_data_untarget = {
        k: v.cuda() if torch.is_tensor(v) else v
        for k, v in pocket_data_untarget.items()
    }

    # Build starting times for the integrator
    lig_times_cont = torch.zeros(prior["coords"].size(0), device=device)
    lig_times_disc = torch.zeros(prior["coords"].size(0), device=device)
    pocket_times = torch.zeros(pocket_data_target["coords"].size(0), device=device)
    prior_times = [lig_times_cont, lig_times_disc, pocket_times]

    # Run generation N times
    if guidance_params is None:
        guidance_params = {}
        guidance_params["apply_guidance"] = False
        guidance_params["window_start"] = 0.0
        guidance_params["window_end"] = 0.4
        guidance_params["value_key"] = "affinity"
        guidance_params["subvalue_key"] = "pic50"
        guidance_params["mu"] = 8.0
        guidance_params["sigma"] = 2.0
        guidance_params["maximize"] = True
        guidance_params["coord_noise_level"] = 0.2

    output = model._generate_selective(
        prior=lig_prior,
        pocket_data_target=pocket_data_target,
        pocket_data_untarget=pocket_data_untarget,
        times=prior_times,
        steps=args.integration_steps,
        strategy=args.ode_sampling_strategy,
        solver=args.solver,
        corr_iters=args.corrector_iters,
        save_traj=save_traj,
        iter=iter,
        apply_guidance=guidance_params["apply_guidance"],
        guidance_window_start=guidance_params["window_start"],
        guidance_window_end=guidance_params["window_end"],
        value_key=guidance_params["value_key"],
        mu=guidance_params["mu"],
        sigma=guidance_params["sigma"],
        maximize=guidance_params["maximize"],
        coord_noise_level=guidance_params["coord_noise_level"],
    )

    # Generate RDKit molecules
    gen_ligs = model._generate_mols(output)

    # Attach affinity predictions as properties if present
    if "affinity" in output:
        affinity = output["affinity"]
        # Ensure affinity is a dict-like object with keys: pic50, pki, pkd, pec50
        for mol, idx in zip(gen_ligs, range(len(gen_ligs))):
            if mol is not None:
                metadata = posterior_target["complex"][0].metadata
                system_id = metadata["system_id"]
                mol.SetProp("_Name", str(system_id))
                if "pic50" in metadata:
                    exp_pic50 = metadata["pic50"]
                    mol.SetProp("exp_pic50", str(exp_pic50))
                if "pkd" in metadata:
                    exp_pkd = metadata["pkd"]
                    mol.SetProp("exp_pkd", str(exp_pkd))
                if "pki" in metadata:
                    exp_pki = metadata["pki"]
                    mol.SetProp("exp_pki", str(exp_pki))
                if "pec50" in metadata:
                    exp_pec50 = metadata["pec50"]
                    mol.SetProp("exp_pec50", str(exp_pec50))
                for key in ["pic50", "pki", "pkd", "pec50"] + [
                    "pic50_untarget",
                    "pki_untarget",
                    "pkd_untarget",
                    "pec50_untarget",
                ]:
                    value = affinity[key]
                    # If value is a tensor, get the scalar for this molecule
                    if hasattr(value, "detach"):
                        val = value[idx].item() if value.ndim > 0 else value.item()
                    else:
                        val = value[idx] if isinstance(value, (list, tuple)) else value
                    mol.SetProp(key, str(val))

    if args.arch == "pocket_flex":
        pocket_data_target = {
            k: v.cpu().detach() if torch.is_tensor(v) else v
            for k, v in pocket_data_target.items()
        }
        gen_pdbs = model.retrieve_pdbs(
            pocket_data_target,
            coords=output["pocket_coords"],
            save_dir=Path(args.save_dir) / "gen_pdbs",
            iter=iter,
        )
        return gen_ligs, gen_pdbs
    return gen_ligs


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
    model = model.to("cuda")
    model.eval()
    print("Model complete.")

    # load util
    transform, interpolant = load_util(
        args,
        hparams,
        vocab,
        vocab_charges,
        vocab_hybridization,
        vocab_aromatic,
    )

    guidance_params = get_guidance_params(args)

    # Load the data
    system_target, system_untarget = load_data_from_pdb_selective(
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
        dataloader_target = get_dataloader(args, dataset_target, interpolant, iter=k)
        dataloader_untarget = get_dataloader(
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
        all_gen_ligs, all_gen_pdbs = filter_diverse_ligands_bulk(
            all_gen_ligs, all_gen_pdbs, threshold=0.9
        )
    else:
        all_gen_ligs = filter_diverse_ligands_bulk(all_gen_ligs, threshold=0.995)
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
                process_lig_wrapper,
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
                process_lig,
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
            write_ligand_pocket_complex_pdb(
                all_gen_ligs_hs,
                all_gen_pdbs,
                gen_complexes_dir,
                complex_name=target_name,
            )
        write_ligand_pocket_complex_pdb(
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
            process_interaction,
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
        "--pocket_type", type=str, choices=["holo", "apo"],
        default="holo",
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
    parser.add_argument("--substructure", type=str, default=None)
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

import argparse
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from tqdm import tqdm

import flowr.gen.utils as util
import flowr.util.rdkit as smolRD
from flowr.gen.generate import generate_molecules
from flowr.scriptutil import (
    load_mol_model,
)
from flowr.util.device import get_device
from flowr.util.metrics import evaluate_pb_validity_mol, evaluate_strain
from flowr.util.molrepr import GeometricMolBatch
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


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def create_list_defaultdict():
    return defaultdict(list)


def evaluate(args):
    # Set precision
    torch.set_float32_matmul_precision("high")

    # Load hyperparameter
    print(f"Using model stored at {args.ckpt_path}")

    # Load the model
    print("Loading model...")
    (
        model,
        hparams,
        vocab,
        vocab_charges,
        vocab_hybridization,
        vocab_aromatic,
    ) = load_mol_model(
        args,
    )
    model = model.to(get_device())
    model.eval()
    print("Model complete.")

    # load util
    transform, interpolant = util.load_util_mol(
        args,
        hparams,
        vocab,
        vocab_charges,
        vocab_hybridization,
        vocab_aromatic,
    )

    # Load the data
    molecules = util.load_data_from_sdf_mol(
        args,
        remove_hs=hparams["remove_hs"],
        remove_aromaticity=hparams["remove_aromaticity"],
        transform=transform,
        sample=True,
        sample_n_molecules=args.sample_n_molecules,
        sample_n_molecules_per_mol=args.sample_n_molecules_per_mol,
    )
    dataset = GeometricMolBatch(molecules)

    # Initialize variables for sampling
    k = 0
    num_molecules = 0
    all_gen_mols = []
    times = []
    global_start = time.time()
    inpainting_mode = util.get_conditional_mode(args)

    # Sampling loop
    print("\nStarting sampling...\n")
    ## Determine the number of molecules to sample given GPU count
    sample_n_molecules = len(molecules)
    while num_molecules < sample_n_molecules and k <= args.max_sample_iter:
        print(
            f"...Sampling iteration {k + 1}...",
            end="\r",
        )
        dataloader = util.get_dataloader(args, dataset, interpolant, iter=k)
        for batch in tqdm(dataloader, desc="Sampling", leave=False):
            prior, posterior, _, _ = batch
            ref_mols = model._generate_mols(posterior)
            batch_start = time.time()
            gen_mols = generate_molecules(
                args,
                model=model,
                prior=prior,
            )
            num_sampled = len(gen_mols)

            # Get the time for one batch iteration
            batch_end = time.time()
            times.append((batch_end - batch_start) / args.batch_cost)

            # Filter ligands by validity and uniqueness
            if args.filter_valid_unique:
                gen_mols, ref_mols = smolRD.sanitize_list(
                    gen_mols,
                    ref_mols=ref_mols,
                    filter_uniqueness=False,
                    sanitize=True,
                )
                print(f"Validity rate: {round(len(gen_mols) / num_sampled, 2)}")
                gen_mols, ref_mols = smolRD.sanitize_list(
                    gen_mols,
                    ref_mols=ref_mols,
                    filter_uniqueness=True,
                    sanitize=False,
                )
                print(f"Uniqueness rate: {round(len(gen_mols) / num_sampled, 2)}")

            # Filter by conditional substructure
            if args.filter_cond_substructure:
                assert (
                    args.filter_valid_unique
                ), "filter_cond_substructure requires filter_valid_unique to be set"
                assert (
                    inpainting_mode is not None
                ), "filter_cond_substructure expected to be set in combination with inpainting mode"
                num_sampled = len(gen_mols)
                gen_mols = util.filter_substructure(
                    gen_mols,
                    ref_mols,
                    inpainting_mode=inpainting_mode,
                    substructure_query=args.substructure,
                    max_fragment_cuts=3,
                )
                print(
                    f"Substructure match rate: {round(len(gen_mols) / num_sampled, 2)}"
                )
            # Add to global molecule list
            all_gen_mols.extend(gen_mols)

            # Filter by diversity
            if args.filter_diversity:
                n_molecules = len(all_gen_mols)
                all_gen_mols = util.filter_diverse_ligands_bulk(
                    all_gen_mols, threshold=args.diversity_threshold
                )
                print(f"Diversity rate: {round(len(all_gen_mols) / n_molecules, 2)}")

            # Update number of generated molecules
            num_molecules = len(all_gen_mols)

        # Increment iteration counter
        k += 1

        # Save timings
        global_run_time = time.time() - global_start
        print(
            f"Mean time per batch: {np.mean(times):.3f} \pm {np.std(times):.2f} seconds"
        )
        print(
            f"\n Sampling took {round(global_run_time, 2)}s for {len(all_gen_mols)} molecules"
        )

        # Empty the cache
        torch.cuda.empty_cache()

    # Check how many molecules were generated
    if num_molecules == 0:
        raise (
            f"Reached {args.max_sample_iter} sampling iterations, but could not find any molecules."
        )
    elif num_molecules < sample_n_molecules:
        print(
            f"FYI: Reached {args.max_sample_iter} sampling iterations, but could only find {num_molecules} molecules."
        )
    elif num_molecules > sample_n_molecules:
        all_gen_mols = all_gen_mols[:sample_n_molecules]

    # Get reference molecules with hydrogens for strain calculations
    ref_mols_with_hs = [mol.orig_mol.to_rdkit() for mol in molecules]

    # Prepare output dictionary
    out_dict = {
        "gen_ligs": all_gen_mols,
        "ref_ligs_hs": ref_mols_with_hs,
        "sampling_time_per_mol": np.mean(times),
        "sampling_time_per_mol_std": np.std(times),
    }

    # Protonate generated ligands:
    if args.add_hs_gen_mols:
        assert hparams[
            "remove_hs"
        ], "The model outputs protonated ligands, no need for additional protonation."
        print("Protonating generated ligands...")
        all_gen_mols_hs = [Chem.AddHs(mol, addCoords=True) for mol in all_gen_mols]
        all_gen_mols = all_gen_mols_hs  # Update generated mols to protonated versions
        out_dict["gen_mols_hs"] = all_gen_mols_hs
        print("Done!")

    if args.optimize_gen_mols_xtb or args.optimize_gen_mols_rdkit:
        assert (
            args.add_hs_gen_mols
        ), "Please add hydrogens to generated ligands before optimization setting add_hs_gen_mols."
        print("Optimizing generated ligands...")
        # Create a temporary directory for xTB calculations
        import tempfile

        if args.optimize_gen_mols_xtb:
            print("Running xTB for optimization...")
            with tempfile.TemporaryDirectory() as temp_dir:
                all_gen_mols_hs_optim, energy_gains, rmsds = [], [], []
                for i, mol in enumerate(
                    tqdm(all_gen_mols_hs, desc="Optimizing ligands")
                ):
                    optimized_mol, energy_gain, rmsd = util.optimize_molecule_xtb(
                        mol, temp_dir
                    )
                    if optimized_mol is not None:
                        all_gen_mols_hs_optim.append(optimized_mol)
                        energy_gains.append(
                            energy_gain if energy_gain is not None else np.nan
                        )
                        rmsds.append(rmsd if rmsd is not None else np.nan)
                    else:
                        all_gen_mols_hs_optim.append(
                            mol
                        )  # If optimization failed, keep original
                        energy_gains.append(np.nan)
                        rmsds.append(np.nan)
        else:
            print("Running RDKit for optimization...")
            all_gen_mols_hs_optim, energy_gains, rmsds = [], [], []
            for i, mol in enumerate(tqdm(all_gen_mols_hs, desc="Optimizing ligands")):
                optimized_mol, energy_gain, rmsd = util.optimize_molecule_rdkit(mol)
                if optimized_mol is not None:
                    all_gen_mols_hs_optim.append(optimized_mol)
                    energy_gains.append(
                        energy_gain if energy_gain is not None else np.nan
                    )
                    rmsds.append(rmsd if rmsd is not None else np.nan)
                else:
                    all_gen_mols_hs_optim.append(
                        mol
                    )  # If optimization failed, keep original
                    energy_gains.append(np.nan)
                    rmsds.append(np.nan)
        all_gen_mols_hs_optim = smolRD.sanitize_list(
            all_gen_mols_hs_optim,
            filter_uniqueness=False,
            sanitize=True,
        )
        out_dict["gen_mols_hs_optim"] = all_gen_mols_hs_optim
        out_dict["gen_mols_optim_energy_gains"] = energy_gains
        out_dict["gen_mols_optim_rmsds"] = rmsds
        print(
            f"Average energy gain from optimization: {np.nanmean(energy_gains)} kcal/mol"
        )
        print(f"Average RMSD from optimization: {np.nanmean(rmsds)} Å")
        print("Done!")

    if args.calculate_strain_energies:
        # Calculate strain energies
        print("Calculating strain energies...")
        ref_strain_energies = evaluate_strain(
            ref_mols_with_hs,
            add_hs=False,
        )
        gen_strain_energies = evaluate_strain(
            (
                all_gen_mols_hs_optim
                if args.optimize_gen_mols_xtb or args.optimize_gen_mols_rdkit
                else all_gen_mols_hs_optim if args.add_hs_gen_mols else all_gen_mols
            ),
            add_hs=(
                False
                if args.add_hs_gen_mols
                or args.optimize_gen_mols_rdkit
                or args.optimize_gen_mols_xtb
                else True if hparams["remove_hs"] else False
            ),
        )
        print(f"Reference ligand strain energy: {ref_strain_energies}")
        print(f"Generated ligands strain energy: {gen_strain_energies}")

    # PoseBusters validity
    if args.filter_pb_valid:
        pb_valid = evaluate_pb_validity_mol(
            (
                all_gen_mols_hs_optim
                if args.optimize_gen_mols_xtb or args.optimize_gen_mols_rdkit
                else all_gen_mols_hs if args.add_hs_gen_mols else all_gen_mols
            ),
            return_list=True,
        )
        if args.optimize_gen_mols_xtb or args.optimize_gen_mols_rdkit:
            all_gen_mols_hs_optim = [
                mol for mol, valid in zip(all_gen_mols_hs_optim, pb_valid) if valid
            ]
            out_dict["gen_mols_hs_optim"] = all_gen_mols_hs_optim
        elif args.add_hs_gen_mols:
            all_gen_mols_hs = [
                mol for mol, valid in zip(all_gen_mols_hs, pb_valid) if valid
            ]
            out_dict["gen_mols_hs"] = all_gen_mols_hs
        else:
            all_gen_mols = [mol for mol, valid in zip(all_gen_mols, pb_valid) if valid]
        print(
            f"PB-validity (mean): {np.mean(pb_valid)}, PB-validity (std): {np.std(pb_valid)}"
        )

    # Save out_dict as pickle file
    target_name = Path(args.sdf_path).stem
    if args.filter_valid_unique:
        predictions = Path(args.save_dir) / f"samples_{target_name}.pt"
    else:
        predictions = Path(args.save_dir) / f"samples_unfiltered_{target_name}.pt"
    torch.save(out_dict, str(predictions))

    # Save ligands as SDF file
    sdf_dir = Path(args.save_dir) / f"samples_{target_name}_unprocessed.sdf"
    write_sdf_file(sdf_dir, all_gen_mols, name=target_name)
    if args.add_hs_gen_mols:
        sdf_dir = Path(args.save_dir) / f"samples_{target_name}_protonated.sdf"
        write_sdf_file(sdf_dir, all_gen_mols_hs, name=target_name)
    if args.optimize_gen_mols_rdkit or args.optimize_gen_mols_xtb:
        sdf_dir = Path(args.save_dir) / f"samples_{target_name}_optimized.sdf"
        write_sdf_file(sdf_dir, all_gen_mols_hs_optim, name=target_name)

    print(f"\nGenerated molecules saved to {sdf_dir}")
    print("\nSampling complete!\n")


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
    # General arguments
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mp_index', default=0, type=int)
    parser.add_argument("--gpus", default=8, type=int)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--arch", type=str, choices=["flowr", "transformer"], required=True)

    # Data paths
    parser.add_argument("--sdf_path", type=str, required=True, default=None, help="Path to the sdf file containing molecules")
    parser.add_argument("--ligand_idx", type=int, default=None, help="Index of the ligand in the sdf file to be used for generation. None or -1 to use all ligands.")
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--save_file", type=str)

    # Model paths
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--lora_finetuned", action="store_true")

    # Sampling parameters
    parser.add_argument("--max_sample_iter", type=int, default=5)
    parser.add_argument("--sample_n_molecules", type=int, default=None)
    parser.add_argument("--sample_n_molecules_per_mol", type=int, default=None)
    parser.add_argument("--sample_mol_sizes", action="store_true")

    # Filtering parameters
    parser.add_argument("--filter_valid_unique", action="store_true")
    parser.add_argument("--filter_diversity", action="store_true")
    parser.add_argument("--diversity_threshold", type=float, default=0.9)
    parser.add_argument("--filter_pb_valid", action="store_true")
    parser.add_argument("--filter_cond_substructure", action="store_true")
    parser.add_argument("--calculate_strain_energies", action="store_true")
    parser.add_argument("--add_hs_gen_mols", action="store_true")
    parser.add_argument("--optimize_gen_mols_rdkit", action="store_true")
    parser.add_argument("--optimize_gen_mols_xtb", action="store_true")

    # Generation parameters
    parser.add_argument("--batch_cost", type=int)
    parser.add_argument("--ligand_time", type=float, default=None)
    parser.add_argument("--resampling_steps", type=int, default=None)
    parser.add_argument("--scaffold_inpainting", action="store_true")
    parser.add_argument("--func_group_inpainting", action="store_true")
    parser.add_argument("--linker_inpainting", action="store_true")
    parser.add_argument("--core_inpainting", action="store_true")
    parser.add_argument("--fragment_inpainting", action="store_true")
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
        choices=["random", "harmonic"],
    )
    parser.add_argument("--max_fragment_cuts", type=int, default=3)
    parser.add_argument("--rotation_alignment", action="store_true")
    parser.add_argument("--permutation_alignment", action="store_true")
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
    parser.add_argument("--use_sde_simulation", action="store_true")
    parser.add_argument("--use_cosine_scheduler", action="store_true")
    parser.add_argument(
        "--categorical_strategy", type=str, default=DEFAULT_CATEGORICAL_STRATEGY
    )
    parser.add_argument(
        "--bucket_cost_scale", type=str, default=DEFAULT_BUCKET_COST_SCALE
    )
    parser.add_argument("--coord_noise_scale", type=float, default=0.0)
    parser.add_argument("--corrector_iters", type=int, default=0)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    evaluate(args)

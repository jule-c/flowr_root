import argparse
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import flowr.util.rdkit as smolRD
from flowr.gen.generate import generate_molecules
from flowr.gen.utils import (
    get_dataloader,
    load_data_from_lmdb_mol,
    load_util_mol,
)
from flowr.scriptutil import (
    load_mol_model,
)
from flowr.util.device import get_device
from flowr.util.molrepr import GeometricMolBatch

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
    transform, interpolant = load_util_mol(
        args,
        hparams,
        vocab,
        vocab_charges,
        vocab_hybridization,
        vocab_aromatic,
    )

    # Load the data
    molecules = load_data_from_lmdb_mol(
        args,
        remove_hs=hparams["remove_hs"],
        remove_aromaticity=hparams["remove_aromaticity"],
        transform=transform,
        sample=True,
        sample_n_molecules=args.sample_n_molecules,
    )
    dataset = GeometricMolBatch(molecules)

    # Initialize variables for sampling
    k = 0
    num_molecules = 0
    validities = []
    all_gen_mols = []
    times = []
    global_start = time.time()
    out_list = []

    # Sampling loop
    print("\nStarting sampling...\n")
    ## Determine the number of molecules to sample given GPU count
    sample_n_molecules = args.sample_n_molecules // args.gpus
    while num_molecules < sample_n_molecules and k <= args.max_sample_iter:
        print(
            f"...Sampling iteration {k + 1}...",
            end="\r",
        )
        dataloader = get_dataloader(args, dataset, interpolant, iter=k)
        for batch in tqdm(dataloader, desc="Sampling", leave=False):
            prior, _, _, _ = batch
            batch_start = time.time()
            gen_mols = generate_molecules(
                args,
                model=model,
                prior=prior,
            )

            # Get the time for one batch iteration
            batch_end = time.time()
            times.append((batch_end - batch_start) / args.batch_cost)

            # validity of generated ligands
            validity = np.mean(
                [smolRD.mol_is_valid(mol, connected=True) for mol in gen_mols]
            )
            validities.append(validity)

            # filter ligands if specified
            if args.filter_valid_unique:
                gen_mols = smolRD.sanitize_list(
                    gen_mols,
                    filter_uniqueness=True,
                )
            all_gen_mols.extend(gen_mols)
            num_molecules += len(gen_mols)
        k += 1

        # Save timings
        global_run_time = time.time() - global_start

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

        ## Save the generated molecules
        out_list.extend(all_gen_mols)

        print(
            f"Mean time per batch: {np.mean(times):.3f} \pm {np.std(times):.2f} seconds"
        )
        print(
            f"\n Sampling took {round(global_run_time, 2)}s for {len(all_gen_mols)} molecules"
        )
        print(f"Validity of generated molecules: {np.mean(validities):.3f}\n")

        # Empty the cache
        torch.cuda.empty_cache()

    # Save out_dict as pickle file
    if args.filter_valid_unique:
        predictions = (
            Path(args.save_dir) / f"predictions_multi_valid_unique_{args.mp_index}.pt"
        )
    else:
        predictions = Path(args.save_dir) / f"predictions_multi_{args.mp_index}.pt"
    torch.save(out_list, str(predictions))
    print(f"Samples saved as {str(predictions)}")
    print("Sampling finished.")


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mp_index', default=0, type=int)
    parser.add_argument("--gpus", default=8, type=int)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--arch", type=str, choices=["flowr", "transformer"], required=True)

    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--lora_finetuned", action="store_true")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--save_file", type=str)

    parser.add_argument("--coord_noise_scale", type=float, default=0.0)

    parser.add_argument("--max_sample_iter", type=int, default=5)
    parser.add_argument("--sample_n_molecules", type=int, default=1000)
    parser.add_argument("--sample_mol_sizes", action="store_true")
    parser.add_argument("--corrector_iters", type=int, default=0)

    parser.add_argument("--filter_valid_unique", action="store_true")

    parser.add_argument("--batch_cost", type=int)
    parser.add_argument("--dataset_split", type=str, default="test", choices=["train", "val", "test"], required=True)
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
        type=str,
        default=None,
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

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    evaluate(args)

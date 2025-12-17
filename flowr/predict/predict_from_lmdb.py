import argparse
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import flowr.util.rdkit as smolRD
from flowr.gen.utils import (
    get_dataloader,
    load_data_from_lmdb,
    load_util,
)
from flowr.predict.predict import predict_affinity_batch
from flowr.scriptutil import (
    load_model,
)
from flowr.util.device import get_device

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
    transform, interpolant = load_util(
        args,
        hparams,
        vocab,
        vocab_charges,
        vocab_hybridization,
        vocab_aromatic,
    )

    # Load the data
    dataset = load_data_from_lmdb(
        args,
        remove_hs=hparams["remove_hs"],
        remove_aromaticity=hparams["remove_aromaticity"],
        transform=transform,
    )

    # Initialize tracking variables
    out_dict = defaultdict(list)
    validities = []
    all_gen_ligs_with_aff = []
    times = []
    global_start = time.time()

    # Start affinity prediction
    print("\nStarting affinity prediction...\n")
    dataloader = get_dataloader(args, dataset, interpolant)
    for i, batch in tqdm(enumerate(dataloader), desc="Predicting affinity..."):
        prior, posterior, _, _ = batch
        gen_ligs_with_aff = predict_affinity_batch(
            args,
            model=model,
            prior=prior,
            posterior=posterior,
            noise_scale=args.coord_noise_scale,
            eps=1e-4,
            seed=args.seed + i,
        )

        # validity of generated ligands
        validity = np.mean(
            [smolRD.mol_is_valid(mol, connected=True) for mol in gen_ligs_with_aff]
        )
        validities.append(validity)
        all_gen_ligs_with_aff.extend(gen_ligs_with_aff)

    # Total run time
    global_run_time = time.time() - global_start
    out_dict["time_total"].append(global_run_time)

    print(
        f"\n Mean run time={round(global_run_time, 2)}s for {len(all_gen_ligs_with_aff)} molecules"
    )
    print(f"Mean time per batch={np.mean(times):.3f} \pm {np.std(times):.2f} seconds")
    print(f"Validity of generated ligands: {np.mean(validities):.3f}\n")

    # Save ligands as SDF
    sdf_path = Path(args.save_dir) / "gen_ligs_with_aff.sdf"
    sdf_path.parent.mkdir(parents=True, exist_ok=True)
    smolRD.write_sdf_file(str(sdf_path), all_gen_ligs_with_aff, name=False)
    print(f"Samples saved as to {str(sdf_path)}")
    print("Sampling finished.")


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--arch", type=str, choices=["pocket", "pocket_flex"], required=True)
    parser.add_argument(
        "--pocket_noise", type=str, choices=["fix", "random", "apo"], required=True
    )
    parser.add_argument("--pocket_type", default="holo", type=str)
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

    parser.add_argument("--sample_mol_sizes", action="store_true")
    parser.add_argument("--corrector_iters", type=int, default=0)

    parser.add_argument("--batch_cost", type=int)
    parser.add_argument("--dataset_split", type=str, default="test", choices=["train", "val", "test", "all"], required=True)
    parser.add_argument("--ligand_time", type=float, default=None)
    parser.add_argument("--pocket_time", type=float, default=None)
    parser.add_argument("--interaction_time", type=float, default=None)
    parser.add_argument("--resampling_steps", type=int, default=None)
    parser.add_argument("--interaction_inpainting", action="store_true")
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
        choices=["conformer", "random", "harmonic"],
    )
    parser.add_argument("--max_fragment_cuts", type=int, default=3)
    parser.add_argument("--rotation_alignment", action="store_true")
    parser.add_argument("--permutation_alignment", action="store_true")
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

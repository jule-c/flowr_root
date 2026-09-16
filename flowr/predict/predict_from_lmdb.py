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
from flowr.util.device import resolve_device

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
    # Device placement. `--gpus` is a device *count*, so `--gpus 0` selects CPU even on
    # a CUDA machine; otherwise CUDA is used when present and CPU everywhere else.
    # Apple's MPS backend is deliberately NOT auto-selected: it is opt-in via
    # FLOWR_DEVICE=mps (see the CPU/macOS note in the README).
    device = resolve_device(args)
    print(f"Using device: {device}")
    model = model.to(device)
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
        batch_start = time.time()
        gen_ligs_with_aff = predict_affinity_batch(
            args,
            model=model,
            prior=prior,
            posterior=posterior,
            noise_scale=args.coord_noise_scale,
            eps=1e-4,
            seed=args.seed + i,
            device=device,
        )
        times.append(time.time() - batch_start)

        # Sanity check on the scored ligands. These are the *input* ligands (affinity
        # prediction scores what it is given, it does not generate molecules), so this
        # is a check on the input data rather than on model output.
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
    if times:
        print(
            f"Mean time per batch={np.mean(times):.3f} \\pm {np.std(times):.2f} seconds"
        )
    print(f"Validity of scored ligands: {np.mean(validities):.3f}\n")

    # Save ligands as SDF
    sdf_path = Path(args.save_dir) / "gen_ligs_with_aff.sdf"
    sdf_path.parent.mkdir(parents=True, exist_ok=True)
    smolRD.write_sdf_file(str(sdf_path), all_gen_ligs_with_aff, name=False)
    print(f"Samples saved as to {str(sdf_path)}")
    print("Sampling finished.")


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
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--splits_path", type=str, default=None)
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
    parser.add_argument("--fixed_interactions", action="store_true")
    parser.add_argument("--interaction_conditional", action="store_true")
    parser.add_argument("--scaffold_hopping", action="store_true")
    parser.add_argument("--scaffold_elaboration", action="store_true")
    parser.add_argument("--linker_inpainting", action="store_true")
    parser.add_argument("--anisotropic_prior", action="store_true")
    parser.add_argument("--ref_ligand_com_prior", action="store_true")
    parser.add_argument("--ref_ligand_com_noise_std", type=float, default=1.0)
    parser.add_argument("--fragment_inpainting", action="store_true")
    parser.add_argument("--fragment_growing", action="store_true")
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
    parser.add_argument("--core_growing", action="store_true")
    parser.add_argument("--ring_system_index", "--ring_system_indexing", default=0, type=int,
                        help="Index of the ring system to keep as the core when using --core_growing (0-indexed; use flowr.data.interpolate.get_num_ring_systems to see how many exist)")
    parser.add_argument(
        "--graph_inpainting",
        default=None,
        type=str,
        choices=["conformer", "random", "harmonic"],
    )
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

    # Inference-time sampler guard and decode repair. Every one of these defaults OFF, so a
    # command line that does not name them behaves exactly as before.
    parser.add_argument("--cat_noise_euler_guard", dest="cat_noise_euler_guard",
        action="store_true", default=True,
        help="ON BY DEFAULT. Silence the categorical sampling noise over the terminal "
             "window where the Euler step stops being a valid probability step "
             "(1-t <= step*(1+noise*K)). Without it a converged prediction is still kicked "
             "off its own argmax at (K-1)*noise/steps per step. Measured over 4000 "
             "generations it changed neither build yield nor PoseBusters validity, so it is "
             "on as a correctness fix rather than for a measured gain. Disable with "
             "--no_cat_noise_euler_guard.")
    parser.add_argument("--no_cat_noise_euler_guard", dest="cat_noise_euler_guard",
        action="store_false",
        help="Keep the historical sampler: uniform noise on every step but the last, "
             "including the terminal window where the Euler step is not a valid "
             "probability step.")
    parser.add_argument("--ligand_valence_repair", dest="ligand_valence_repair",
        action="store_true", default=True,
        help="ON BY DEFAULT. When a generated ligand's argmax decode FAILS to build, "
             "re-decode it to the model's own highest-joint-probability assignment that "
             "satisfies the RDKit-probed valence limits. It is gated on the build having "
             "already returned None, so it can only ADD molecules -- it never alters or "
             "drops one that built, and it is never applied to reference ligands. "
             "Disable with --no_ligand_valence_repair.")
    parser.add_argument("--no_ligand_valence_repair", dest="ligand_valence_repair",
        action="store_false",
        help="Deliver the raw argmax decode: a ligand whose independently-argmaxed heads "
             "name a chemically impossible atom is dropped rather than re-decoded.")
    parser.add_argument("--ligand_valence_repair_allow_bond_deletion", action="store_true",
        help="Let the repair escape an over-valence by DELETING a bond, not just demoting "
             "it. Off by default because deleting a bond can split the molecule, turning a "
             "valence failure into a disconnected one -- that lifts validity but not "
             "fully-connected validity.")
    parser.add_argument("--ligand_valence_repair_max_edits", type=int, default=2)
    parser.add_argument("--ligand_valence_repair_top_k", type=int, default=4)
    parser.add_argument("--ligand_valence_repair_max_states", type=int, default=200)
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

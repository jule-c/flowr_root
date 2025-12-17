import argparse
import time
import warnings
from collections import defaultdict
from pathlib import Path

import torch
from rdkit import Chem
from tqdm import tqdm

import flowr.util.rdkit as smolRD
from flowr.data.dataset import GeometricDataset
from flowr.gen.utils import (
    get_dataloader,
    load_data_from_pdb,
    load_util,
)
from flowr.predict.predict import predict_affinity_batch
from flowr.scriptutil import (
    load_model,
)
from flowr.util.device import get_device
from flowr.util.pocket import PocketComplexBatch

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
    if isinstance(system, list):
        systems = PocketComplexBatch(system)
    else:
        systems = PocketComplexBatch([system])
    dataset = GeometricDataset(
        systems, data_cls=PocketComplexBatch, transform=transform
    )
    return dataset


def predict(args):
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
    if args.multiple_ligands:
        try:
            assert (
                args.ligand_file is not None
            ), "Please provide a ligand SDF file with --ligand_file."
            n_ligands = len(Chem.SDMolSupplier(str(args.ligand_file)))
        except Exception:
            raise ValueError("Could not read ligands from SDF file.")
        systems = []
        for ligand_idx in range(n_ligands):
            system = load_data_from_pdb(
                args,
                remove_hs=hparams["remove_hs"],
                remove_aromaticity=hparams["remove_aromaticity"],
                ligand_idx=ligand_idx,
            )
            systems.append(system)
        dataset = get_dataset(systems, transform, vocab, interpolant, args, hparams)

    else:
        system = load_data_from_pdb(
            args,
            remove_hs=hparams["remove_hs"],
            remove_aromaticity=hparams["remove_aromaticity"],
        )
        dataset = get_dataset(system, transform, vocab, interpolant, args, hparams)
    dataloader = get_dataloader(args, dataset, interpolant)

    global_start = time.time()

    if args.multiple_ligands:
        for batch in tqdm(
            dataloader,
            desc=f"Predicting affinity for all {n_ligands} ligands in the SDF...",
        ):
            prior, posterior, _, _ = batch
            gen_lig_with_aff = predict_affinity_batch(
                args,
                model=model,
                prior=prior,
                posterior=posterior,
                noise_scale=args.coord_noise_scale,
                eps=1e-4,
            )
    else:
        print("\nPredicting affinity...\n")
        batch = next(iter(dataloader))
        gen_lig_with_aff = predict_affinity_batch(
            args,
            model=model,
            prior=batch[0],
            posterior=batch[1],
            noise_scale=args.coord_noise_scale,
            eps=1e-4,
            seed=args.seed,
        )

    # Total run time
    global_run_time = time.time() - global_start

    print(f"\n Run time={round(global_run_time, 2)}s")
    # Save ligands as SDF
    sdf_path = Path(args.save_dir) / "gen_lig_with_aff.sdf"
    sdf_path.parent.mkdir(parents=True, exist_ok=True)
    smolRD.write_sdf_file(str(sdf_path), gen_lig_with_aff, name=False)
    print(f"Sample saved as {str(sdf_path)}")
    print("Prediction finished.")


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--pdb_id', type=str, default=None)
    parser.add_argument('--ligand_id', type=str, default=None)
    parser.add_argument('--pdb_file', type=str, default=None)
    parser.add_argument('--ligand_file', type=str, default=None)
    parser.add_argument('--multiple_ligands', action='store_true')
    parser.add_argument('--res_txt_file', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)

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
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--save_file", type=str)

    parser.add_argument("--coord_noise_scale", type=float, default=0.0)

    parser.add_argument("--sample_mol_sizes", action="store_true")
    parser.add_argument("--corrector_iters", type=int, default=0)
    parser.add_argument("--rotation_alignment", action="store_true")
    parser.add_argument("--permutation_alignment", action="store_true")

    parser.add_argument("--batch_cost", type=int, default=1)
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

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    predict(args)

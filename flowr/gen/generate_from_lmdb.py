import argparse
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import flowr.gen.utils as util
import flowr.util.rdkit as smolRD
from flowr.data.dataset import (
    GeometricDataset,
)
from flowr.gen.generate import generate_ligands_per_target
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
    transform, interpolant = util.load_util(
        args,
        hparams,
        vocab,
        vocab_charges,
        vocab_hybridization,
        vocab_aromatic,
    )

    # Load the data
    systems = util.load_data_from_lmdb(
        args,
        remove_hs=hparams["remove_hs"],
        remove_aromaticity=hparams["remove_aromaticity"],
        transform=transform,
        sample=args.max_samples is not None,
        sample_n_molecules=args.max_samples,  # Limit to max samples e.g. for quicker testing
    )

    guidance_params = util.get_guidance_params(args)
    print("\nStarting sampling...\n")
    out_dict = defaultdict(list)
    for system in tqdm(systems, desc="Sampling ligands"):
        system = PocketComplexBatch([system])
        dataset = GeometricDataset(system, data_cls=PocketComplexBatch)

        k = 0
        num_ligands = 0
        validity_rate = 1.0
        validities = []
        all_gen_ligs = []
        times = []
        global_start = time.time()
        while (
            num_ligands < args.sample_n_molecules_per_target
            and k <= args.max_sample_iter
        ):
            sample_n_molecules_per_target = int(
                (args.sample_n_molecules_per_target - num_ligands) * validity_rate
            )
            data = dataset.sample_n_molecules_per_target(sample_n_molecules_per_target)
            print(
                f"...Sampling iteration {k + 1}...",
                end="\r",
            )
            dataloader = util.get_dataloader(args, data, interpolant, iter=k)
            for batch in tqdm(dataloader, desc="Sampling", leave=False):
                prior, posterior, _, _ = batch
                batch_start = time.time()
                gen_ligs = generate_ligands_per_target(
                    args,
                    model=model,
                    prior=prior,
                    posterior=posterior,
                    pocket_noise=args.pocket_noise,
                    guidance_params=guidance_params,
                )

                # Get the time for one batch iteration
                batch_end = time.time()
                times.append((batch_end - batch_start) / args.batch_cost)

                # validity of generated ligands
                validity = np.mean(
                    [smolRD.mol_is_valid(mol, connected=True) for mol in gen_ligs]
                )
                validities.append(validity)

                # filter ligands if specified
                if args.filter_valid_unique:
                    validity_rate = (1 - validity) + 1
                    gen_ligs = smolRD.sanitize_list(
                        gen_ligs,
                        filter_uniqueness=True,
                    )
                all_gen_ligs.extend(gen_ligs)
                num_ligands += len(gen_ligs)
            k += 1

        # Save timings
        time_per_complex = np.mean(times)
        global_run_time = time.time() - global_start

        # Check how many ligands were generated
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

        # Retrieve and save reference ligands per batch
        ref_ligs = model._generate_ligs(
            posterior, lig_mask=posterior["lig_mask"].bool(), scale=model.coord_scale
        )[0]
        ref_ligs_with_hs = model.retrieve_ligs_with_hs(posterior, save_idx=0)
        ref_pdbs = model.retrieve_pdbs(
            posterior, save_dir=Path(args.save_dir) / "ref_pdbs", save_idx=0
        )
        ref_pdbs_with_hs = model.retrieve_pdbs_with_hs(
            posterior, save_dir=Path(args.save_dir) / "ref_pdbs", save_idx=0
        )
        ## Save the generated ligands
        out_dict["gen_ligs"].append(all_gen_ligs)
        out_dict["ref_ligs"].append(ref_ligs)
        out_dict["ref_ligs_with_hs"].append(ref_ligs_with_hs)
        out_dict["ref_pdbs"].append(ref_pdbs)
        out_dict["ref_pdbs_with_hs"].append(ref_pdbs_with_hs)
        # Time for the sampling process
        out_dict["time_per_complex"].append(time_per_complex)
        out_dict["time_per_pocket"].append(global_run_time)

        # save properties separately as torch.save somehow does not save the property within the rdkit molecule
        properties = []
        for mol in all_gen_ligs:
            if mol is None:
                props = {}
            else:
                props = mol.GetPropsAsDict()
            properties.append(props)
        out_dict["properties"].append(properties)
        # Save the ligands as SDF file as well to preserve properties
        system_id = posterior["complex"][0].metadata["system_id"]
        sdf_path = Path(args.save_dir) / "gen_sdfs" / f"gen_ligs_{system_id}.sdf"
        sdf_path.parent.mkdir(parents=True, exist_ok=True)
        smolRD.write_sdf_file(str(sdf_path), all_gen_ligs, name=False)

        print(
            f"\n Mean time per pocket={round(global_run_time, 2)}s for {len(all_gen_ligs)} molecules"
        )
        print(
            f"Mean time per complex: {np.mean(times):.3f} \pm {np.std(times):.2f} seconds"
        )
        print(f"Validity of generated ligands: {np.mean(validities):.3f}\n")

        # Empty the cache
        torch.cuda.empty_cache()

    # Save out_dict as pickle file
    if args.filter_valid_unique:
        predictions = (
            Path(args.save_dir) / f"predictions_multi_valid_unique_{args.mp_index}.pt"
        )
    else:
        predictions = Path(args.save_dir) / f"predictions_multi_{args.mp_index}.pt"
    torch.save(out_dict, str(predictions))
    print(f"Samples saved as {str(predictions)}")

    print(
        f"Time per pocket: {np.mean(out_dict['time_per_pocket']):.3f} \pm "
        f"{np.std(out_dict['time_per_pocket']):.2f}"
    )
    print("Sampling finished.")


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mp_index', default=0, type=int)
    parser.add_argument("--gpus", default=8, type=int)
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

    parser.add_argument("--max_sample_iter", type=int, default=20)
    parser.add_argument("--sample_n_molecules_per_target", type=int, default=1)
    parser.add_argument("--sample_mol_sizes", action="store_true")
    parser.add_argument("--corrector_iters", type=int, default=0)

    parser.add_argument("--filter_valid_unique", action="store_true")

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
    parser.add_argument("--guidance_config", default=None)
    parser.add_argument("--max_samples", default=None, help="Max samples to process from the corresponding dataset. Performs slicing up to this value.", type=int)


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    evaluate(args)

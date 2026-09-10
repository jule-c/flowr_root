import argparse
import time
import warnings
from collections import defaultdict
from functools import partial
from pathlib import Path

import lightning as L
import numpy as np
import torch
from tqdm import tqdm

import flowr.scriptutil as util
import flowr.util.rdkit as smolRD
from flowr import constants
from flowr.data.datamodules import GeometricInterpolantDM
from flowr.data.dataset import GeometricDataset
from flowr.data.interpolate import (
    ComplexInterpolant,
    GeometricNoiseSampler,
)
from flowr.gen.generate import generate_ligands_per_target
from flowr.scriptutil import load_model
from flowr.util.device import clear_cache, resolve_device
from flowr.util.pocket import PROLIF_INTERACTIONS, PocketComplexBatch
from flowr.util.rdkit import ConformerGenerator

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


def split_list(data, num_chunks):
    chunk_size = len(data) // num_chunks
    remainder = len(data) % num_chunks
    chunks = []
    start = 0
    for i in range(num_chunks):
        chunk_end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(data[start:chunk_end])
        start = chunk_end
    return chunks


def load_util(
    args,
    hparams,
    vocab,
    vocab_charges,
    vocab_hybridization=None,
    vocab_aromatic=None,
):
    """Load utility functions and interpolant for evaluation."""

    if hparams["coord_scale"] == 1.0:
        coord_std = 1.0

    n_bond_types = util.get_n_bond_types(args.categorical_strategy)
    n_charge_types = vocab_charges.size
    n_hybridization_types = (
        vocab_hybridization.size if vocab_hybridization is not None else 0
    )
    n_aromatic_types = vocab_aromatic.size if vocab_aromatic is not None else 0
    transform = partial(
        util.complex_transform,
        vocab=vocab,
        vocab_charges=vocab_charges,
        vocab_hybridization=vocab_hybridization,
        vocab_aromatic=vocab_aromatic,
        n_bonds=n_bond_types,
        coord_std=coord_std,
        pocket_noise=args.pocket_noise,
        pocket_noise_std=args.pocket_coord_noise_std,
    )

    type_mask_index = None
    bond_mask_index = None
    conformer_generator = (
        ConformerGenerator(
            cache_dir=Path(args.data_path) / "conformers",
            max_conformers=10,
            max_iters=200,
            enable_caching=True,
            vocab=vocab,
        )
        if args.graph_inpainting is not None and args.graph_inpainting == "conformer"
        else None
    )
    prior_sampler = GeometricNoiseSampler(
        vocab.size,
        n_bond_types,
        n_charge_types,
        n_hybridization_types=n_hybridization_types,
        n_aromatic_types=n_aromatic_types,
        coord_noise="gaussian",
        type_noise=hparams["val-ligand-prior-type-noise"],
        bond_noise=hparams["val-ligand-prior-bond-noise"],
        zero_com=True,  # args.pocket_noise in ["fix", "random"],
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        conformer_generator=conformer_generator,
    )
    if args.pocket_time is not None:
        assert (
            args.separate_pocket_interpolation
        ), "Setting a pocket time requires a separate pocket interpolation"
    if args.interaction_time is not None:
        assert (
            args.separate_interaction_interpolation
        ), "Setting an interaction time requires a separate interaction interpolation"

    ## Determine the categorical sampling strategy
    if args.categorical_strategy == "mask":
        assert hparams["val-ligand-type-interpolation"] == "unmask"
        type_mask_index = vocab.indices_from_tokens(["<MASK>"])[0]
        bond_mask_index = util.BOND_MASK_INDEX
        categorical_interpolation = "unmask"
    elif args.categorical_strategy == "uniform-sample":
        assert hparams["val-ligand-type-interpolation"] == "unmask"
        categorical_interpolation = "unmask"
    elif args.categorical_strategy == "prior-sample":
        assert hparams["val-ligand-type-interpolation"] == "unmask"
        categorical_interpolation = "unmask"
    elif args.categorical_strategy == "velocity-sample":
        assert hparams["val-ligand-type-interpolation"] == "sample"
        categorical_interpolation = "sample"
    else:
        raise ValueError(
            f"Interpolation '{args.categorical_strategy}' is not supported."
        )

    eval_interpolant = ComplexInterpolant(
        prior_sampler,
        ligand_coord_interpolation="linear",
        ligand_type_interpolation=categorical_interpolation,
        ligand_bond_interpolation=categorical_interpolation,
        use_cosine_scheduler=args.use_cosine_scheduler,
        pocket_noise=args.pocket_noise,
        separate_pocket_interpolation=args.separate_pocket_interpolation,
        separate_interaction_interpolation=args.separate_interaction_interpolation,
        n_interaction_types=(
            len(PROLIF_INTERACTIONS)
            if hparams["flow_interactions"]
            or hparams["predict_interactions"]
            or hparams["interaction_conditional"]
            else None
        ),
        flow_interactions=hparams["flow_interactions"],
        interaction_conditional=args.interaction_conditional,
        scaffold_hopping=args.scaffold_hopping,
        scaffold_elaboration=args.scaffold_elaboration,
        linker_inpainting=args.linker_inpainting,
        fragment_inpainting=args.fragment_inpainting,
        fragment_growing=getattr(args, "fragment_growing", False),
        max_fragment_cuts=args.max_fragment_cuts,
        substructure_inpainting=args.substructure_inpainting,
        substructure=args.substructure,
        graph_inpainting=args.graph_inpainting,
        equivariant_ot=False,
        batch_ot=False,
        dataset=args.dataset,
        sample_mol_sizes=False,
        virtual_atom_p=getattr(args, "virtual_atom_p", 0.0),
        virtual_atom_noise_std=getattr(args, "virtual_atom_noise_std", 0.5),
        noatom_index=(
            vocab.token_idx_map.get(constants.NOATOM_TOKEN)
            if getattr(args, "virtual_atom_p", 0.0) > 0
            else None
        ),
        inference=True,
        vocab=vocab,
        vocab_charges=vocab_charges,
        vocab_hybridization=vocab_hybridization,
        anisotropic_prior=getattr(args, "anisotropic_prior", False),
        ref_ligand_com_prior=getattr(args, "ref_ligand_com_prior", False),
        ref_ligand_com_noise_std=getattr(args, "ref_ligand_com_noise_std", 1.0),
    )
    return transform, eval_interpolant


def get_dataloader(args, dataset, interpolant, iter=0):
    L.seed_everything(args.seed + iter)
    util.configure_fs()

    dm = GeometricInterpolantDM(
        None,
        None,
        dataset,
        args.batch_cost,
        val_batch_size=args.batch_cost,
        test_interpolant=interpolant,
        bucket_cost_scale=args.bucket_cost_scale,
        pad_to_bucket=False,
        num_workers=args.num_workers,
    )

    test_dl = dm.test_dataloader()
    return test_dl


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
    data_path = Path(args.data_path) / f"{args.dataset_split}.smol"
    bytes_data = data_path.read_bytes()
    systems = PocketComplexBatch.from_bytes(bytes_data, remove_hs=hparams["remove_hs"])
    # ``--gpus`` is a device *count* and ``--gpus 0`` selects CPU, so taking it
    # literally as a shard count raises ZeroDivisionError. CPU is one shard.
    systems = split_list(systems, max(1, args.gpus))[args.mp_index - 1]

    print("\nStarting sampling...\n")
    out_dict = defaultdict(list)
    for system in tqdm(systems, desc="Sampling ligands"):
        system = PocketComplexBatch([system])
        dataset = GeometricDataset(
            system, data_cls=PocketComplexBatch, transform=transform
        )

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
            dataloader = get_dataloader(args, data, interpolant, iter=k)
            for batch in tqdm(dataloader, desc="Sampling", leave=False):
                prior, data, _, _ = batch
                batch_start = time.time()
                gen_ligs = generate_ligands_per_target(
                    args,
                    model,
                    prior=prior,
                    posterior=data,
                    pocket_noise=args.pocket_noise,
                    device=device,
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

        time_per_complex = np.mean(times)
        global_run_time = time.time() - global_start
        if num_ligands == 0:
            # NB: `raise <str>` here raised TypeError: exceptions must derive from
            # BaseException, destroying the diagnostic it was written to deliver.
            raise RuntimeError(
                f"Reached {args.max_sample_iter} sampling iterations, but could not "
                "find any ligands."
            )
        elif num_ligands < args.sample_n_molecules_per_target:
            print(
                f"FYI: Reached {args.max_sample_iter} sampling iterations, but could only find {num_ligands} ligands."
            )
        elif num_ligands > args.sample_n_molecules_per_target:
            all_gen_ligs = all_gen_ligs[: args.sample_n_molecules_per_target]

        # Retrieve and save reference ligands per batch
        ref_ligs = model._generate_ligs(
            data, lig_mask=data["lig_mask"].bool(), scale=model.coord_scale
        )[0]
        ref_ligs_with_hs = model.retrieve_ligs_with_hs(data, save_idx=0)
        ref_pdbs = model.retrieve_pdbs(
            data, save_dir=Path(args.save_dir) / "ref_pdbs", save_idx=0
        )
        ref_pdbs_with_hs = model.retrieve_pdbs_with_hs(
            data, save_dir=Path(args.save_dir) / "ref_pdbs", save_idx=0
        )

        # Empty the cache
        clear_cache()

        # Save the generated ligands
        out_dict["gen_ligs"].append(all_gen_ligs)
        out_dict["ref_ligs"].append(ref_ligs)
        out_dict["ref_ligs_with_hs"].append(ref_ligs_with_hs)
        out_dict["ref_pdbs"].append(ref_pdbs)
        out_dict["ref_pdbs_with_hs"].append(ref_pdbs_with_hs)
        # Time for the sampling process
        out_dict["time_per_complex"].append(time_per_complex)
        out_dict["time_per_pocket"].append(global_run_time)

        print(
            f"\n Mean time per pocket={round(global_run_time, 2)}s for {len(all_gen_ligs)} molecules"
        )
        print(
            f"Mean time per complex: {np.mean(times):.3f} \\pm {np.std(times):.2f} seconds"
        )
        print(f"Validity of generated ligands: {np.mean(validities):.3f}\n")

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
        f"Time per pocket: {np.mean(out_dict['time_per_pocket']):.3f} \\pm "
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
    parser.add_argument("--pocket_type", type=str, choices=["holo", "apo"], default="holo")
    parser.add_argument(
        "--pocket_noise", type=str, choices=["fix", "random", "apo"], required=True
    )
    parser.add_argument(
        "--pocket_coord_noise_std", type=float, default=0.0,
        help="Standard deviation of the pocket coordinate noise"
    )
    parser.add_argument("--ckpt_path", type=str)
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
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--ligand_time", type=float, default=None)
    parser.add_argument("--pocket_time", type=float, default=None)
    parser.add_argument("--interaction_time", type=float, default=None)
    parser.add_argument("--resampling_steps", type=int, default=None)
    parser.add_argument("--interaction_conditional", action="store_true")
    parser.add_argument("--scaffold_hopping", action="store_true")
    parser.add_argument("--scaffold_elaboration", action="store_true")
    parser.add_argument("--linker_inpainting", action="store_true")
    parser.add_argument("--anisotropic_prior", action="store_true")
    parser.add_argument("--ref_ligand_com_prior", action="store_true")
    parser.add_argument("--ref_ligand_com_noise_std", type=float, default=1.0)
    parser.add_argument("--core_growing", action="store_true")
    parser.add_argument("--fragment_inpainting", action="store_true")
    parser.add_argument("--fragment_growing", action="store_true")
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

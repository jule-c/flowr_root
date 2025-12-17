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
from flowr.data.datamodules import GeometricInterpolantDM
from flowr.data.dataset import GeometricDataset
from flowr.data.interpolate import (
    ComplexInterpolant,
    GeometricNoiseSampler,
)
from flowr.gen.generate import generate_ligands_per_target
from flowr.models.fm_pocket import LigandPocketCFM
from flowr.models.integrator import Integrator
from flowr.models.pocket import LigandGenerator, PocketEncoder
from flowr.util.device import get_device, get_map_location
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


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


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


def load_model(args):
    checkpoint = torch.load(args.ckpt_path, map_location=get_map_location())
    hparams = dotdict(checkpoint["hyper_parameters"])
    hparams["compile_model"] = False
    hparams["integration-steps"] = args.integration_steps
    hparams["sampling_strategy"] = args.ode_sampling_strategy
    hparams["interaction_inpainting"] = args.interaction_inpainting
    hparams["scaffold_inpainting"] = args.scaffold_inpainting
    hparams["func_group_inpainting"] = args.func_group_inpainting
    hparams["linker_inpainting"] = args.linker_inpainting
    hparams["data_path"] = args.data_path
    hparams["save_dir"] = args.save_dir
    hparams["predict_affinity"] = hparams.get("predict_affinity", False)
    hparams["predict_docking_score"] = hparams.get("predict_docking_score", False)

    # Number of corrector iterations
    if args.corrector_iters > 0:
        assert (
            args.categorical_strategy == "velocity-sample"
        ), "Only velocity sampling supported for corrector iterations."
        hparams["corrector_iters"] = args.corrector_iters

    print("Building model vocabs...")
    # vocab = util.build_vocab(remove_hs=args.remove_hs)
    vocab = util._build_vocab()
    vocab_charges = util._build_vocab_charges()
    vocab_pocket_atoms = util._build_vocab_pocket_atoms()
    vocab_pocket_res = util._build_vocab_pocket_res()
    if hparams["add_feats"]:
        print("Including hybridization features...")
        vocab_hybridization = util._build_vocab_hybridization()
        vocab_aromatic = None  # util._build_vocab_aromatic()
    else:
        vocab_hybridization = None
        vocab_aromatic = None
    print("Vocabs complete.")

    n_atom_types = vocab.size
    n_bond_types = util.get_n_bond_types(args.categorical_strategy)
    n_charge_types = vocab_charges.size
    n_hybridization_types = (
        vocab_hybridization.size if vocab_hybridization is not None else 0
    )
    n_aromatic_types = vocab_aromatic.size if vocab_aromatic is not None else 0
    n_interaction_types = (
        len(PROLIF_INTERACTIONS) + 1
        if hparams["flow_interactions"] or hparams["predict_interactions"]
        else None
    )

    fixed_equi = hparams["pocket-fixed_equi"]
    pocket_enc = PocketEncoder(
        hparams["pocket-d_equi"],
        hparams["pocket-d_inv"],
        hparams["d_message"],
        hparams["pocket-n_layers"],
        hparams["n_attn_heads"],
        hparams["d_message_ff"],
        hparams["d_edge"],
        vocab_pocket_atoms.size,
        n_bond_types,
        vocab_pocket_res.size,
        fixed_equi=fixed_equi,
    )

    egnn_gen = LigandGenerator(
        hparams["d_equi"],
        hparams["d_inv"],
        hparams["d_message"],
        hparams["n_layers"],
        hparams["n_attn_heads"],
        hparams["d_message_ff"],
        hparams["d_edge"],
        emb_size=hparams["emb_size"],
        n_atom_types=n_atom_types,
        n_charge_types=n_charge_types,
        n_bond_types=n_bond_types,
        n_extra_atom_feats=(
            n_hybridization_types + n_aromatic_types if hparams["add_feats"] else 0
        ),
        predict_interactions=hparams["predict_interactions"],
        flow_interactions=hparams["flow_interactions"],
        predict_affinity=hparams["predict_affinity"],
        predict_docking_score=hparams["predict_docking_score"],
        use_lig_pocket_rbf=hparams["use_lig_pocket_rbf"],
        use_rbf=hparams["use_rbf"],
        use_sphcs=hparams["use_sphcs"],
        n_interaction_types=n_interaction_types,
        self_cond=hparams["self_cond"],
        pocket_enc=pocket_enc,
        coord_skip_connect=hparams["coord_skip_connect"],
    )

    if args.lora_finetuned:
        from flowr.models.lora import LinearWithLoRA
        from flowr.models.pocket import SemlaCondAttention

        lora_rank, lora_alpha = 8, 16

        def _inject_lora(mod):
            for name, child in mod.named_children():
                # skip the entire cross-attention blocks
                if isinstance(child, SemlaCondAttention):
                    continue
                # wrap any pure Linear
                if isinstance(child, torch.nn.Linear):
                    setattr(
                        mod,
                        name,
                        LinearWithLoRA(child, rank=lora_rank, alpha=lora_alpha),
                    )
                else:
                    _inject_lora(child)

        # inject LoRA into the ligand generator
        _inject_lora(egnn_gen.ligand_dec)

    CFM = LigandPocketCFM
    type_mask_index = None
    bond_mask_index = None
    integrator = Integrator(
        args.integration_steps,
        use_sde_simulation=args.use_sde_simulation,
        type_strategy=args.categorical_strategy,
        bond_strategy=args.categorical_strategy,
        coord_strategy="continuous",
        pocket_noise=hparams["pocket_noise"],
        cat_noise_level=args.cat_sampling_noise_level,
        coord_noise_std=args.coord_noise_scale,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        use_cosine_scheduler=args.use_cosine_scheduler,
    )
    fm_model = CFM.load_from_checkpoint(
        args.ckpt_path,
        gen=egnn_gen,
        vocab=vocab,
        vocab_charges=vocab_charges,
        vocab_hybridization=vocab_hybridization,
        vocab_aromatic=vocab_aromatic,
        integrator=integrator,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        graph_inpainting=args.graph_inpainting is not None,
        **hparams,
    )
    return (
        fm_model,
        hparams,
        vocab,
        vocab_charges,
        vocab_hybridization,
        vocab_aromatic,
        vocab_pocket_atoms,
        vocab_pocket_res,
    )


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
            or hparams["interaction_inpainting"]
            else None
        ),
        flow_interactions=hparams["flow_interactions"],
        interaction_inpainting=args.interaction_inpainting,
        scaffold_inpainting=args.scaffold_inpainting,
        func_group_inpainting=args.func_group_inpainting,
        linker_inpainting=args.linker_inpainting,
        fragment_inpainting=args.fragment_inpainting,
        max_fragment_cuts=args.max_fragment_cuts,
        substructure_inpainting=args.substructure_inpainting,
        substructure=args.substructure,
        graph_inpainting=args.graph_inpainting,
        equivariant_ot=False,
        batch_ot=False,
        dataset=args.dataset,
        sample_mol_sizes=False,
        inference=True,
        vocab=vocab,
        vocab_charges=vocab_charges,
        vocab_hybridization=vocab_hybridization,
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
    data_path = Path(args.data_path) / f"{args.dataset_split}.smol"
    bytes_data = data_path.read_bytes()
    systems = PocketComplexBatch.from_bytes(bytes_data, remove_hs=hparams["remove_hs"])
    systems = split_list(systems, args.gpus)[args.mp_index - 1]

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
        torch.cuda.empty_cache()

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
            f"Mean time per complex: {np.mean(times):.3f} \pm {np.std(times):.2f} seconds"
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
    parser.add_argument("--arch", type=str, choices=["pocket", "semla"], required=True)
    parser.add_argument(
        "--pocket_noise", type=str, choices=["fix", "random", "apo"], required=True
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

    parser.add_argument("--filter_valid_unique", action="store_true")

    parser.add_argument("--batch_cost", type=int)
    parser.add_argument("--dataset_split", type=str, default="test")
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

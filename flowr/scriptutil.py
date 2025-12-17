"""Main util file for all scripts"""

import copy
import datetime
import math
import os
import pickle
import resource
from argparse import Namespace
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional

import lightning.pytorch as pl
import numpy as np
import rdkit
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import MLFlowLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from openbabel import openbabel as ob
from rdkit import Chem, RDLogger
from torch.utils.data import ConcatDataset
from torchmetrics import MetricCollection

import flowr.constants as constants
import flowr.scriptutil as util
import flowr.util.functional as smolF
import flowr.util.rdkit as smolRD
from flowr.callbacks import EMA, EMAModelCheckpoint
from flowr.data.datamodules import GeometricInterpolantDM
from flowr.data.dataset import (
    DatasetSubset,
    GeometricDataset,
    GeometricMolLMDBDataset,
    PocketComplexLMDBDataset,
)
from flowr.data.interpolate import (
    ComplexInterpolant,
    GeometricInterpolant,
    GeometricNoiseSampler,
)
from flowr.data.util import Statistics
from flowr.models.integrator import Integrator
from flowr.models.pocket import LigandGenerator, PocketEncoder
from flowr.util.device import get_device, get_map_location
from flowr.util.pocket import PROLIF_INTERACTIONS
from flowr.util.tokeniser import (
    Vocabulary,
    pocket_atom_names,
    pocket_residue_names,
)

# from flowr.models.fm_mol import LigandCFM
# from flowr.models.fm_pocket_flex import LigandPocketFlexCFM
LigandCFM = LigandPocketFlexCFM = None


PROJECT_PREFIX = "flowr"
BOND_MASK_INDEX = 5
COMPILER_CACHE_SIZE = 128


# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# ******************************* UTILS ***************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************


def disable_lib_stdout():
    ob.obErrorLog.StopLogging()
    RDLogger.DisableLog("rdApp.*")


# bfloat16 training produced significantly worse models than full so use default 16-bit instead
def get_precision(args):
    return "32"
    # return "16-mixed" if args.mixed_precision else "32"


def get_warm_up_steps(args, train_steps):
    return min(train_steps // 20, 5000)  # 5% of training


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# Need to ensure the limits are large enough when using OT since lots of preprocessing needs to be done on the batches
# OT seems to cause a problem when there are not enough allowed open FDs
def configure_fs(limit=4096):
    """
    Try to increase the limit on open file descriptors
    If not possible use a different strategy for sharing files in torch
    """

    n_file_resource = resource.RLIMIT_NOFILE
    soft_limit, hard_limit = resource.getrlimit(n_file_resource)

    print(f"Current limits (soft, hard): {(soft_limit, hard_limit)}")

    if limit > soft_limit:
        try:
            print(f"Attempting to increase open file limit to {limit}...")
            resource.setrlimit(n_file_resource, (limit, hard_limit))
            print("Limit changed successfully!")

        except Exception:
            print(
                "Limit change unsuccessful. Using torch file_system file sharing strategy instead."
            )

            import torch.multiprocessing

            torch.multiprocessing.set_sharing_strategy("file_system")

    else:
        print("Open file limit already sufficiently large.")


def mol_transform(
    molecule,
    vocab,
    vocab_charges,
    n_bonds,
    vocab_hybridization=None,
    vocab_aromatic=None,
    coord_std=1.0,
    rotate=False,
    zero_com=False,
):
    """
    Transform a molecule into a format suitable for model input.
    # Applies the following optional transformations to a molecule:
        # 1. Scales coordinate values by 1 / coord_std (so that they are standard normal)
        # 2. Applies a random rotation to the coordinates
        # 3. Removes the centre of mass of the molecule
        # 4. Creates a one-hot vector for the atomic numbers of each atom
        # 5. Creates a one-hot vector for the bond type for every possible bond
        # 6. Encodes charges as non-negative numbers according to encoding map
    """
    if zero_com:
        molecule = molecule.zero_com()
    if coord_std != 1.0:
        molecule = molecule.scale(1.0 / coord_std)
    if rotate:
        from scipy.spatial.transform import Rotation

        rotation = Rotation.random()
        molecule = molecule.rotate(rotation)

    atomic_nums = [int(atomic) for atomic in molecule.atomics.tolist()]
    tokens = [smolRD.PT.symbol_from_atomic(atomic) for atomic in atomic_nums]
    atomics = torch.tensor(vocab.indices_from_tokens(tokens, one_hot=True))

    bond_types = smolF.one_hot_encode_tensor(molecule.bond_types, n_bonds)

    charges = [int(charge) for charge in molecule.charges.tolist()]
    charges = torch.tensor(vocab_charges.indices_from_tokens(charges, one_hot=True))

    if vocab_hybridization is not None:
        hybridization = [
            smolRD.IDX_ADD_FEAT_MAP["hybridization"][int(hybrid)]
            for hybrid in molecule.hybridization.tolist()
        ]
        hybridization = torch.tensor(
            vocab_hybridization.indices_from_tokens(hybridization, one_hot=True)
        )
    else:
        hybridization = None
    if vocab_aromatic is not None:
        is_aromatic = [
            smolRD.IDX_ADD_FEAT_MAP["is_aromatic"][int(aromatic)]
            for aromatic in molecule.aromaticity.tolist()
        ]
        is_aromatic = torch.tensor(
            vocab_aromatic.indices_from_tokens(is_aromatic, one_hot=True)
        )
    else:
        is_aromatic = None

    transformed = molecule._copy_with(
        atomics=atomics, bond_types=bond_types, charges=charges
    )
    if hybridization is not None:
        transformed = transformed._copy_with(hybridization=hybridization)
    else:
        transformed = transformed._copy_with(hybridization=None)
    if vocab_aromatic is not None:
        transformed = transformed._copy_with(is_aromatic=is_aromatic)
    else:
        transformed = transformed._copy_with(is_aromatic=None)

    return transformed


def complex_transform(
    pocket_complex,
    vocab,
    vocab_charges,
    n_bonds,
    vocab_hybridization=None,
    vocab_aromatic=None,
    coord_std: float = 1.0,
    pocket_noise: str = "apo",
    pocket_noise_std: float = 0.02,
    use_interactions: bool = False,
    rotate_complex: bool = False,
):
    assert coord_std == 1.0, "coord_std must be 1.0 for complex transform for now"

    holo_pocket = pocket_complex.holo
    apo_pocket = pocket_complex.apo
    ligand = pocket_complex.ligand

    # *** Transform LIGAND *** #
    lig_trans = mol_transform(
        ligand,
        vocab,
        vocab_charges,
        n_bonds,
        vocab_hybridization=vocab_hybridization,
        vocab_aromatic=vocab_aromatic,
        coord_std=coord_std,
        rotate=False,
        zero_com=False,
    )

    # *** Transform HOLO *** #
    if holo_pocket is not None:
        holo_mol_trans = mol_transform(
            holo_pocket.mol,
            vocab,
            vocab_charges,
            n_bonds,
            coord_std=coord_std,
            rotate=False,
            zero_com=False,
        )
        holo_pocket = holo_pocket._copy_with(mol=holo_mol_trans)

    # *** Transform APO *** #
    if apo_pocket is not None:
        apo_mol_trans = mol_transform(
            apo_pocket.mol,
            vocab,
            vocab_charges,
            n_bonds,
            coord_std=coord_std,
            rotate=False,
            zero_com=False,
        )
        apo_pocket = apo_pocket._copy_with(mol=apo_mol_trans)

    # *** Transform COMPLEX *** #
    if pocket_noise == "fix":
        assert holo_pocket is not None, "Holo must be provided for rigid flow matching"
        trans_complex = pocket_complex._copy_with(lig_trans, holo=holo_pocket)
        trans_complex = trans_complex.move_holo_and_lig_to_holo_com()
    elif pocket_noise == "random":
        assert (
            holo_pocket is not None
        ), "Holo must be provided for random pocket flow matching"
        coords = (
            holo_pocket.mol.coords
            + torch.randn_like(holo_pocket.mol.coords) * pocket_noise_std
        )
        apo_pocket_mol = holo_pocket.mol._copy_with(coords=coords)
        apo_pocket = holo_pocket._copy_with(mol=apo_pocket_mol)
        trans_complex = pocket_complex._copy_with(
            lig_trans, holo=holo_pocket, apo=apo_pocket
        )
        trans_complex = trans_complex.move_apo_and_holo_and_lig_to_apo_com()
    elif pocket_noise == "apo":
        assert apo_pocket is not None, "apo must be provided for apo-holo flow matching"
        if holo_pocket is None:
            # NOTE: This should only happen at inference when just a apo structure is provided
            holo_pocket = apo_pocket._copy_with()
        trans_complex = pocket_complex._copy_with(
            lig_trans, holo=holo_pocket, apo=apo_pocket
        )
        trans_complex = trans_complex.move_apo_and_holo_and_lig_to_apo_com()
    else:
        raise ValueError(
            f"Invalid pocket noise type {pocket_noise}. Must be one of ['fix', 'apo', 'random']"
        )

    # *** Transform INTERACTIONS *** #
    if use_interactions:
        # Add a one-hot vector for the interaction type (N_pocket, N_lig, num_interactions + 1)
        # where no interaction is encoded as the first index
        interactions = trans_complex.interactions
        if interactions is not None:
            n_pocket, n_lig, n_interactions = interactions.shape
            interactions_arr = np.zeros((n_pocket, n_lig, n_interactions + 1))
            interactions_arr[:, :, 1:] = interactions
            interactions_flat = interactions_arr.reshape(
                n_pocket * n_lig, n_interactions + 1
            )
            interactions_flat = np.argmax(
                interactions_flat, axis=-1
            )  # to get no interaction class at index 0
            interactions_arr = smolF.one_hot_encode_tensor(
                torch.from_numpy(interactions_flat), n_interactions + 1
            )
            interactions_arr = interactions_arr.reshape(
                n_pocket, n_lig, -1
            )  # (N_pocket, N_lig, num_interactions + 1)
            trans_complex.interactions = interactions_arr

    if rotate_complex:
        trans_complex = trans_complex.rotate()

    return trans_complex


def get_n_bond_types(cat_strategy):
    n_bond_types = len(smolRD.BOND_IDX_MAP.keys()) + 1
    n_bond_types = n_bond_types + 1 if cat_strategy == "mask" else n_bond_types
    return n_bond_types


def _build_vocab():
    # Need to make sure PAD has index 0
    special_tokens = ["<PAD>"]
    tokens = special_tokens + constants.CORE_ATOMS
    return Vocabulary(tokens)


def _build_vocab_charges():
    # Need to make sure PAD has index 0
    special_tokens = ["<PAD>"]
    charge_tokens = [0, 1, 2, 3, -1, -2, -3]
    tokens = special_tokens + charge_tokens
    return Vocabulary(tokens)


def _build_vocab_hybridization():
    # Need to make sure PAD has index 0
    special_tokens = ["<PAD>"]
    hybridization_tokens = [
        rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED,
        rdkit.Chem.rdchem.HybridizationType.S,
        rdkit.Chem.rdchem.HybridizationType.SP,
        rdkit.Chem.rdchem.HybridizationType.SP2,
        rdkit.Chem.rdchem.HybridizationType.SP3,
        rdkit.Chem.rdchem.HybridizationType.SP2D,
        rdkit.Chem.rdchem.HybridizationType.SP3D,
        rdkit.Chem.rdchem.HybridizationType.SP3D2,
        rdkit.Chem.rdchem.HybridizationType.OTHER,
    ]
    tokens = special_tokens + hybridization_tokens
    return Vocabulary(tokens)


def _build_vocab_aromaticity():
    # Need to make sure PAD has index 0
    special_tokens = ["<PAD>"]
    aromaticity_tokens = [False, True]
    tokens = special_tokens + aromaticity_tokens
    return Vocabulary(tokens)


def _build_vocab_pocket_atoms():
    special_token = ["<PAD>"]
    ligand = ["LIG"]
    tokens = special_token + ligand + pocket_atom_names
    return Vocabulary(tokens)


def _build_vocab_pocket_res(pocket_noise="apo"):
    special_token = ["<PAD>"]
    ligand = ["LIG"]
    tokens = special_token + ligand + pocket_residue_names
    return Vocabulary(tokens)


# TODO support multi gpus
def calc_train_steps(dm, epochs, acc_batches):
    dm.setup("train")
    steps_per_epoch = math.ceil(len(dm.train_dataloader()) / acc_batches)
    return steps_per_epoch * epochs


def calc_metrics_(
    gen_mols: list[Chem.Mol],
    metrics: MetricCollection,
    ref_mols: list[Chem.Mol] = None,
    ref_pdbs: list[str] = None,
    stab_metrics: MetricCollection = None,
    mol_stabs: list = None,
):
    metrics.reset()
    if ref_mols is not None and ref_pdbs is not None:
        metrics.update(gen_mols, ref_mols, ref_pdbs)
    else:
        metrics.update(gen_mols)
    results = metrics.compute()

    if stab_metrics is None:
        return results

    stab_metrics.reset()
    stab_metrics.update(mol_stabs)
    stab_results = stab_metrics.compute()

    results = {**results, **stab_results}
    return results


def print_results(results, std_results=None):
    print()
    print(f"{'Metric':<22}Result")
    print("-" * 30)

    for metric, value in results.items():
        if isinstance(value, dict):
            for k, v in value.items():
                print(f"{metric} ({k}): {v:.5f}")
        else:
            result_str = f"{metric:<22}{value:.5f}"
            if std_results is not None:
                std = std_results[metric]
                result_str = f"{result_str} +- {std:.7f}"

        print(result_str)
    print()


def train_val_test_split(dset_len, train_size, val_size, test_size, seed, order=None):
    assert (train_size is None) + (val_size is None) + (
        test_size is None
    ) <= 1, "Only one of train_size, val_size, test_size is allowed to be None."
    is_float = (
        isinstance(train_size, float),
        isinstance(val_size, float),
        isinstance(test_size, float),
    )

    train_size = round(dset_len * train_size) if is_float[0] else train_size
    val_size = round(dset_len * val_size) if is_float[1] else val_size
    test_size = round(dset_len * test_size) if is_float[2] else test_size

    if train_size is None:
        train_size = dset_len - val_size - test_size
    elif val_size is None:
        val_size = dset_len - train_size - test_size
    elif test_size is None:
        test_size = dset_len - train_size - val_size

    if train_size + val_size + test_size > dset_len:
        if is_float[2]:
            test_size -= 1
        elif is_float[1]:
            val_size -= 1
        elif is_float[0]:
            train_size -= 1

    assert train_size >= 0 and val_size >= 0 and test_size >= 0, (
        f"One of training ({train_size}), validation ({val_size}) or "
        f"testing ({test_size}) splits ended up with a negative size."
    )

    total = train_size + val_size + test_size
    assert dset_len >= total, (
        f"The dataset ({dset_len}) is smaller than the "
        f"combined split sizes ({total})."
    )
    if total < dset_len:
        print(f"{dset_len - total} samples were excluded from the dataset")

    idxs = np.arange(dset_len, dtype=np.int64)
    if order is None:
        idxs = np.random.default_rng(seed).permutation(idxs)

    idx_train = idxs[:train_size]
    idx_val = idxs[train_size : train_size + val_size]
    idx_test = idxs[train_size + val_size : total]

    if order is not None:
        idx_train = [order[i] for i in idx_train]
        idx_val = [order[i] for i in idx_val]
        idx_test = [order[i] for i in idx_test]

    return np.array(idx_train), np.array(idx_val), np.array(idx_test)


def make_splits(
    dataset_len: int = None,
    train_size: float = None,
    val_size: float = None,
    test_size: float = None,
    seed: int = None,
    filename: str = None,
    splits: str = None,
    order: list = None,
):
    """
    Create train, validation, and test splits for a dataset.
    If `splits` is provided, it will load the splits from the file.
        No need to provide `dataset_len`, `train_size`, `val_size`, `test_size`, or `seed`.
    If `splits` is None, it will create splits based on the provided parameters.
    """

    if splits is not None:
        splits = np.load(splits)
        idx_train = splits["idx_train"]
        idx_val = splits["idx_val"]
        idx_test = splits["idx_test"]
    else:
        idx_train, idx_val, idx_test = train_val_test_split(
            dataset_len, train_size, val_size, test_size, seed, order
        )

    if filename is not None:
        np.savez(filename, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

    return (
        torch.from_numpy(idx_train),
        torch.from_numpy(idx_val),
        torch.from_numpy(idx_test),
    )


# Function to recursively inject LoRA layers
def _inject_lora(lora_rank: int, lora_alpha: float, mod: torch.nn.Module):
    from flowr.models.lora import LinearWithLoRA

    for name, child in mod.named_children():
        # wrap any pure Linear
        if isinstance(child, torch.nn.Linear):
            setattr(
                mod,
                name,
                LinearWithLoRA(child, rank=lora_rank, alpha=lora_alpha),
            )
        else:
            _inject_lora(lora_rank, lora_alpha, child)


# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# **************************** BUILD TRAINER **********************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************


def build_trainer(args, model=None):
    epochs = 1 if args.trial_run else args.epochs

    project_name = f"{PROJECT_PREFIX}-{args.dataset}"
    precision = util.get_precision(args)
    print(f"Using precision '{precision}'")

    lr_logger = LearningRateMonitor(logging_interval="step")
    mllogger = MLFlowLogger(
        experiment_name=args.dataset + "_" + args.exp_name,
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
        run_id=os.environ.get("MLFLOW_RUN_ID"),
        run_name=args.run_name if args.run_name else None,
        log_model="best",
    )
    if args.wandb:
        wdblogger = WandbLogger(project=project_name, log_model="all", offline=True)
        wdblogger.watch(model, log="all")
        loggers = [wdblogger, mllogger]
    else:
        loggers = mllogger
    # tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.save_dir)

    if args.use_ema:
        ema_callback = EMA(
            decay=args.ema_decay,
            apply_ema_every_n_steps=1,
            start_step=0,
            save_ema_weights_in_callback_state=True,
            evaluate_ema_weights_instead=True,
        )
        checkpoint_callback = EMAModelCheckpoint(
            dirpath=args.save_dir,
            save_top_k=3,
            # monitor="val-fc-validity",
            monitor="val-pb_validity",
            mode="max",
            save_last=True,
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.save_dir,
            save_top_k=3,
            # monitor="val-fc-validity",
            monitor="val-pb_validity",
            mode="max",
            save_last=True,
        )
    callbacks = [
        lr_logger,
        # ASCIIProgressBar(refresh_rate=5),
        TQDMProgressBar(refresh_rate=5),
        ModelSummary(max_depth=2),
        checkpoint_callback,
    ]
    if args.use_ema:
        callbacks.append(ema_callback)

    # When to do validation ckpt
    if args.val_check_epochs is None:
        val_check_epochs = 1
        val_check_interval = args.val_check_interval
    else:
        val_check_epochs = args.val_check_epochs
        val_check_interval = None

    from lightning.pytorch.plugins.environments import (
        LightningEnvironment,
        SLURMEnvironment,
    )

    strategy = DDPStrategy(
        timeout=datetime.timedelta(seconds=1800 * args.gpus)
    )  # "ddp" if args.gpus > 1 else "auto"
    trainer = pl.Trainer(
        accelerator="gpu" if args.gpus else "cpu",
        devices=args.gpus if args.gpus else 1,
        strategy=strategy,
        plugins=SLURMEnvironment() if args.num_nodes > 1 else LightningEnvironment(),
        num_nodes=args.num_nodes,
        enable_checkpointing=True,
        accumulate_grad_batches=args.acc_batches,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=val_check_epochs,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        logger=loggers,
        precision=precision,
        max_epochs=epochs,
        use_distributed_sampler=True,  # not args.use_bucket_sampler,
    )

    pl.seed_everything(seed=args.seed, workers=args.gpus > 1)
    return trainer


# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# ******************************* BUILD MODEL *********************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************


def build_model(
    args,
    dm,
    dataset_info,
    train_mols,
    vocab,
    vocab_charges,
    vocab_hybridization=None,
    vocab_aromatic=None,
    vocab_pocket_atoms=None,
    vocab_pocket_res=None,
):
    # Get hyperparameters from the datamodule, pass these into the model to be saved
    hparams = {
        "epochs": args.epochs,
        "gradient_clip_val": args.gradient_clip_val,
        "dataset": args.dataset,
        "precision": get_precision(args),
        "architecture": args.arch,
        **dm.hparams,
    }

    n_atom_types = vocab.size
    n_bond_types = get_n_bond_types(args.categorical_strategy)
    n_charge_types = vocab_charges.size
    n_res_names = vocab_pocket_res.size if vocab_pocket_res is not None else 0
    n_hybridization_types = (
        vocab_hybridization.size if vocab_hybridization is not None else 0
    )
    n_aromatic_types = vocab_aromatic.size if vocab_aromatic is not None else 0
    n_interaction_types = (
        len(PROLIF_INTERACTIONS) + 1
        if args.flow_interactions or args.predict_interactions
        else None
    )

    fixed_equi = args.pocket_fixed_equi
    pocket_d_equi = 1 if fixed_equi else args.n_coord_sets
    pocket_d_inv = args.pocket_d_model
    pocket_n_layers = args.pocket_n_layers
    pocket_enc = PocketEncoder(
        pocket_d_equi,
        pocket_d_inv,
        args.d_message,
        pocket_n_layers,
        args.n_attn_heads,
        args.d_message_hidden,
        args.d_edge,
        vocab_pocket_atoms.size,
        n_bond_types,
        n_res_names,
        emb_size=args.emb_size,
        fixed_equi=fixed_equi,
        use_rbf=args.use_rbf,
        use_distances=args.use_distances,
        use_crossproducts=args.use_crossproducts,
    )
    gen = LigandGenerator(
        args.n_coord_sets,
        args.d_model,
        args.d_message,
        args.n_layers,
        args.n_attn_heads,
        args.d_message_hidden,
        args.d_edge,
        emb_size=args.emb_size,
        n_atom_types=n_atom_types,
        n_charge_types=n_charge_types,
        n_bond_types=n_bond_types,
        n_extra_atom_feats=(
            (n_hybridization_types + n_aromatic_types) if args.add_feats else None
        ),
        flow_interactions=args.flow_interactions,
        predict_interactions=args.predict_interactions,
        predict_affinity=args.predict_affinity,
        predict_docking_score=args.predict_docking_score,
        n_interaction_types=n_interaction_types,
        use_rbf=args.use_rbf,
        use_sphcs=args.use_sphcs,
        use_distances=args.use_distances,
        use_crossproducts=args.use_crossproducts,
        use_lig_pocket_rbf=args.use_lig_pocket_rbf,
        use_fourier_time_embed=args.use_fourier_time_embed,
        graph_inpainting=args.graph_inpainting,
        use_inpaint_mode_embed=args.scaffold_inpainting
        or args.functional_group_inpainting
        or args.interaction_inpainting
        or args.core_inpainting
        or args.linker_inpainting
        or args.fragment_inpainting
        or args.substructure_inpainting,
        self_cond=args.self_condition,
        coord_skip_connect=not args.no_coord_skip_connect,
        coord_update_every_n=getattr(args, "coord_update_every_n", None),
        pocket_enc=pocket_enc,
    )

    if args.load_pretrained_ckpt:
        print(f"Loading pretrained checkpoint from {args.load_pretrained_ckpt}...")
        state_dict = torch.load(
            args.load_pretrained_ckpt, map_location=get_map_location()
        )["state_dict"]
        # Remove the prefix from the state dict keys
        state_dict = {k.replace("gen.", ""): v for k, v in state_dict.items()}
        gen.load_state_dict(state_dict, strict=False)

        if args.lora_finetuning:
            from flowr.models.lora import LinearWithLoRA
            from flowr.models.pocket import SemlaCondAttention

            lora_rank, lora_alpha = 32, 64

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
            _inject_lora(gen.ligand_dec)
            # freeze all parameters except LoRA and conditional attention
            for n, p in gen.ligand_dec.named_parameters():
                if "lora" in n:
                    p.requires_grad = True
                elif "cond_attention" in n or "equi_attn" in n or "inv_attn" in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            # fully train pocket encoding
            if gen.pocket_enc is not None:
                for p in gen.pocket_enc.parameters():
                    p.requires_grad = True
        print("Done.")

    coord_scale = 1.0
    type_mask_index = None
    bond_mask_index = None

    if args.categorical_strategy == "mask":
        type_mask_index = vocab.indices_from_tokens(["<MASK>"])[0]
        bond_mask_index = BOND_MASK_INDEX
        train_strategy = "mask"
        sampling_strategy = "mask"

    elif args.categorical_strategy == "uniform-sample":
        train_strategy = "ce"
        sampling_strategy = "uniform-sample"

    elif args.categorical_strategy == "prior-sample":
        train_strategy = "ce"
        sampling_strategy = "uniform-sample"

    elif args.categorical_strategy == "velocity-sample":
        train_strategy = "ce"
        sampling_strategy = "velocity-sample"

    else:
        raise ValueError(
            f"Interpolation '{args.categorical_strategy}' is not supported."
        )

    # Training steps
    train_steps = calc_train_steps(dm, args.epochs, args.acc_batches) // args.gpus
    print(f"Total training steps {train_steps}")
    if args.lr_schedule in ["cosine", "constant"]:
        warm_up_steps = get_warm_up_steps(args, train_steps)
        print(f"Warmup steps {warm_up_steps}")
    else:
        warm_up_steps = 0

    if args.arch == "pocket":
        from flowr.models.integrator import Integrator
    elif args.arch == "pocket_flex":
        from flowr.models.fm_pocket_flex import Integrator
    else:
        from flowr.models.integrator import Integrator
    integrator = Integrator(
        args.num_inference_steps,
        use_sde_simulation=args.use_sde_simulation,
        type_strategy=sampling_strategy,
        bond_strategy=sampling_strategy,
        coord_strategy=args.coord_sampling_strategy,
        pocket_noise=args.pocket_noise,
        cat_noise_level=args.cat_sampling_noise_level,
        coord_noise_std=args.coord_noise_scale,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        use_cosine_scheduler=args.use_cosine_scheduler,
    )

    if args.arch == "pocket":
        from flowr.models.fm_pocket import LigandPocketCFM
    elif args.arch == "transformer":
        from flowr.models.fm_transformer import LigandPocketCFM

    CFM = (
        LigandPocketCFM
        if args.arch in ["pocket", "transformer"]
        else LigandPocketFlexCFM if args.arch == "pocket_flex" else LigandCFM
    )
    fm_model = CFM(
        gen,
        vocab,
        vocab_charges,
        args.lr,
        integrator,
        add_feats=args.add_feats,
        vocab_hybridization=vocab_hybridization,
        vocab_aromatic=vocab_aromatic,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        coord_scale=coord_scale,
        sampling_strategy=args.sample_schedule,
        type_strategy=train_strategy,
        bond_strategy=train_strategy,
        coord_loss_weight=args.coord_loss_weight,
        type_loss_weight=args.type_loss_weight,
        bond_loss_weight=args.bond_loss_weight,
        interaction_loss_weight=args.interaction_loss_weight,
        charge_loss_weight=args.charge_loss_weight,
        hybridization_loss_weight=args.hybridization_loss_weight,
        distance_loss_weight_lig=args.distance_loss_weight_lig,
        distance_loss_weight_lig_pocket=args.distance_loss_weight_lig_pocket,
        smooth_distance_loss_weight_lig=args.smooth_distance_loss_weight_lig,
        smooth_distance_loss_weight_lig_pocket=args.smooth_distance_loss_weight_lig_pocket,
        plddt_confidence_loss_weight=args.plddt_confidence_loss_weight,
        train_confidence=args.train_confidence,
        confidence_gen_steps=args.confidence_gen_steps,
        affinity_loss_weight=args.affinity_loss_weight,
        docking_loss_weight=args.docking_loss_weight,
        pocket_noise=args.pocket_noise,
        pairwise_metrics=False,
        use_ema=args.use_ema,
        compile_model=False,
        self_condition=args.self_condition,
        inpaint_self_condition=args.inpaint_self_condition,
        distill=False,
        lr_schedule=args.lr_schedule,
        lr_gamma=args.lr_gamma,
        warm_up_steps=warm_up_steps,
        total_steps=train_steps,
        train_mols=train_mols,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        remove_hs=args.remove_hs,
        remove_aromaticity=args.remove_aromaticity,
        save_dir=args.save_dir,
        dataset_info=dataset_info,
        data_path=args.data_path,
        flow_interactions=args.flow_interactions,
        predict_interactions=args.predict_interactions,
        interaction_inpainting=args.interaction_inpainting,
        scaffold_inpainting=args.scaffold_inpainting,
        func_group_inpainting=args.func_group_inpainting,
        linker_inpainting=args.linker_inpainting,
        core_inpainting=args.core_inpainting,
        fragment_inpainting=args.fragment_inpainting,
        substructure_inpainting=args.substructure_inpainting,
        graph_inpainting=args.graph_inpainting is not None,
        mixed_uncond_inpaint=args.mixed_uncond_inpaint,
        use_t_loss_weights=args.use_t_loss_weights,
        corrector_iters=args.corrector_iters,
        pretrained_weights=args.load_pretrained_ckpt is not None,
        **hparams,
    )
    return fm_model


def load_model(
    args,
    ckpt_path: str = None,
    return_info: bool = True,
    dataset_info: Optional[dict] = None,
):
    checkpoint = torch.load(
        args.ckpt_path if ckpt_path is None else ckpt_path,
        map_location=get_map_location(),
    )
    hparams = dotdict(checkpoint["hyper_parameters"])
    hparams["compile_model"] = False
    hparams["integration-steps"] = args.integration_steps
    hparams["sampling_strategy"] = args.ode_sampling_strategy
    hparams["use_inpaint_mode_embed"] = (
        hparams.get("scaffold_inpainting", False)
        or hparams.get("func_group_inpainting", False)
        or hparams.get("interaction_inpainting", False)
        or hparams.get("core_inpainting", False)
        or hparams.get("linker_inpainting", False)
        or hparams.get("fragment_inpainting", False)
        or hparams.get("substructure_inpainting", False)
    )
    hparams["interaction_inpainting"] = args.interaction_inpainting
    hparams["scaffold_inpainting"] = args.scaffold_inpainting
    hparams["func_group_inpainting"] = args.func_group_inpainting
    hparams["linker_inpainting"] = args.linker_inpainting
    hparams["core_inpainting"] = args.core_inpainting
    hparams["fragment_inpainting"] = args.fragment_inpainting
    hparams["substructure_inpainting"] = args.substructure_inpainting
    hparams["substructure"] = args.substructure
    hparams["data_path"] = args.data_path
    hparams["save_dir"] = args.save_dir
    hparams["predict_affinity"] = hparams.get("predict_affinity", False)
    hparams["predict_docking_score"] = hparams.get("predict_docking_score", False)
    hparams["add_feats"] = hparams.get("add_feats", False)
    hparams["use_fourier_time_embed"] = hparams.get("use_fourier_time_embed", False)
    hparams["use_lig_pocket_rbf"] = hparams.get("use_lig_pocket_rbf", False)
    hparams["use_rbf"] = hparams.get("use_rbf", False)
    hparams["use_distances"] = hparams.get("use_distances", False)
    hparams["use_crossproducts"] = hparams.get("use_crossproducts", False)
    hparams["lr"] = args.lr if getattr(args, "lr", None) else hparams.get("lr", 1e-4)
    hparams["lr_schedule"] = (
        args.lr_schedule
        if getattr(args, "lr_schedule", None) is not None
        else hparams.get("lr_schedule", "exponential")
    )
    hparams["lr_gamma"] = (
        args.lr_gamma
        if getattr(args, "lr_gamma", None) is not None
        else hparams.get("lr_gamma", 0.995)
    )
    hparams["weight_decay"] = (
        args.weight_decay
        if getattr(args, "weight_decay", None) is not None
        else hparams.get("weight_decay", 1e-12)
    )
    hparams["beta1"] = (
        args.beta1
        if getattr(args, "beta1", None) is not None
        else hparams.get("beta1", 0.9)
    )
    hparams["beta2"] = (
        args.beta2
        if getattr(args, "beta2", None) is not None
        else hparams.get("beta2", 0.95)
    )

    # Number of corrector iterations
    if args.corrector_iters > 0:
        assert (
            args.categorical_strategy == "velocity-sample"
        ), "Only velocity sampling supported for corrector iterations."
        hparams["corrector_iters"] = args.corrector_iters

    print("Building model vocabs...")
    vocab = _build_vocab()
    vocab_charges = _build_vocab_charges()
    vocab_pocket_atoms = _build_vocab_pocket_atoms()
    vocab_pocket_res = _build_vocab_pocket_res()
    if hparams["add_feats"]:
        print("Including hybridization features...")
        vocab_hybridization = _build_vocab_hybridization()
        vocab_aromatic = None  # _build_vocab_aromatic()
    else:
        vocab_hybridization = None
        vocab_aromatic = None
    print("Vocabs complete.")

    if hparams["pocket_noise"] in ["fix", "random"]:
        assert (
            args.arch == "pocket"
        ), "Model trained on rigid pocket flow matching. Change arch to pocket."
        assert (
            args.pocket_type == "holo"
        ), "Model trained on rigid pocket flow matching. Change pocket_type to holo."
    if hparams["pocket_noise"] == "apo":
        assert (
            args.arch == "pocket_flex"
        ), "Model trained on apo pocket flow matching. Change arch to pocket_flex."
        assert (
            args.pocket_type == "apo"
        ), "Model trained on apo pocket flow matching. Change pocket_type to apo."

    n_atom_types = vocab.size
    n_bond_types = get_n_bond_types(args.categorical_strategy)
    n_charge_types = vocab_charges.size
    n_hybridization_types = (
        vocab_hybridization.size if vocab_hybridization is not None else None
    )
    # n_aromatic_types = vocab_aromatic.size if vocab_aromatic is not None else None
    n_interaction_types = (
        len(PROLIF_INTERACTIONS) + 1
        if hparams["flow_interactions"] or hparams["predict_interactions"]
        else None
    )

    # Build the EGNN generator
    if args.arch == "pocket":
        from flowr.models.fm_pocket import LigandPocketCFM
        from flowr.models.pocket import LigandGenerator, PocketEncoder

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
            emb_size=hparams["emb_size"],
            use_rbf=hparams["use_rbf"],
            use_distances=hparams["use_distances"],
            use_crossproducts=hparams["use_crossproducts"],
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
                n_hybridization_types  # + n_aromatic_types
                if hparams["add_feats"]
                else None
            ),
            predict_interactions=hparams["predict_interactions"],
            flow_interactions=hparams["flow_interactions"],
            n_interaction_types=n_interaction_types,
            predict_affinity=hparams["predict_affinity"],
            predict_docking_score=hparams["predict_docking_score"],
            use_rbf=hparams["use_rbf"],
            use_sphcs=hparams["use_sphcs"],
            use_distances=hparams["use_distances"],
            use_crossproducts=hparams["use_crossproducts"],
            use_fourier_time_embed=hparams["use_fourier_time_embed"],
            use_lig_pocket_rbf=hparams["use_lig_pocket_rbf"],
            use_inpaint_mode_embed=hparams["use_inpaint_mode_embed"],
            self_cond=hparams["self_cond"],
            coord_skip_connect=hparams["coord_skip_connect"],
            coord_update_every_n=hparams.get("coord_update_every_n", None),
            pocket_enc=pocket_enc,
        )
    elif args.arch == "pocket_flex":
        from flowr.models.complex import SemlaEncoder, SemlaLayer
        from flowr.models.fm_complex import LigandPocketCFM

        n_res_types = vocab_pocket_res.size
        layer = SemlaLayer(
            d_equi=hparams["d_equi"],
            d_inv=hparams["d_inv"],
            d_message=hparams["d_message"],
            n_heads=hparams["n_attn_heads"],
            d_attn_ff=hparams["d_attn_ff"],
            d_edge=hparams["d_edge"],
        )
        egnn_gen = SemlaEncoder(
            layer=layer,
            n_layers=hparams["n_layers"],
            n_atom_names=n_atom_types,
            n_res_types=n_res_types,
            n_charge_types=n_charge_types,
            n_bond_types=n_bond_types,
            self_cond=hparams["self_cond"],
            n_rbf=hparams["num_rbf"],
            emb_size=hparams["size_emb"],
            equi_diff=True,
        )
    else:
        raise ValueError(f"Unknown architecture {args.arch}")

    # Check if the model has been LoRA finetuned
    if hparams.get("lora_finetuning", False):
        # Apply LoRA to ligand decoder
        _inject_lora(
            lora_rank=hparams["lora_rank"],
            lora_alpha=hparams["lora_alpha"],
            mod=egnn_gen.ligand_dec,
        )
        # Apply LoRA to pocket encoder if exists
        if egnn_gen.pocket_enc is not None:
            _inject_lora(
                lora_rank=hparams["lora_rank"],
                lora_alpha=hparams["lora_alpha"],
                mod=egnn_gen.pocket_enc,
            )

    print(f"Loading pretrained checkpoint from {args.ckpt_path}...")
    # Initialize the ligand-pocket conditional flow model
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
        dataset_info=dataset_info,
        graph_inpainting=args.graph_inpainting is not None,
        **hparams,
    )

    if getattr(args, "lora_finetuning", None):
        print("Applying LoRA finetuning...")
        _hparams = fm_model.hparams
        _hparams["lora_finetuning"] = True
        _hparams["lora_rank"] = args.lora_rank
        _hparams["lora_alpha"] = args.lora_alpha

        # Load the pretrained weights
        state_dict = torch.load(args.ckpt_path, map_location=get_map_location())[
            "state_dict"
        ]
        state_dict = {k.replace("gen.", ""): v for k, v in state_dict.items()}
        egnn = copy.deepcopy(egnn_gen)
        egnn.load_state_dict(state_dict, strict=True)
        assert (
            not args.affinity_finetuning
        ), "Cannot use both LoRA and affinity_finetune."
        assert not args.freeze_layers, "Cannot use both LoRA and freeze_layers."

        # Apply LoRA to ligand decoder
        _inject_lora(
            lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, mod=egnn.ligand_dec
        )
        # Apply LoRA to pocket encoder if exists
        if egnn.pocket_enc is not None:
            _inject_lora(
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                mod=egnn.pocket_enc,
            )

        # Freeze all parameters except LoRA
        trainable_params = 0
        total_params = 0

        for n, p in egnn.ligand_dec.named_parameters():
            total_params += p.numel()
            if "lora" in n:
                p.requires_grad = True
                trainable_params += p.numel()
            else:
                p.requires_grad = False

        # Keep pocket encoder trainable if exists
        if egnn.pocket_enc is not None:
            for n, p in egnn.pocket_enc.named_parameters():
                total_params += p.numel()
                if "lora" in n:
                    p.requires_grad = True
                    trainable_params += p.numel()
                else:
                    p.requires_grad = False

        print(
            f"LoRA: {trainable_params}/{total_params} parameters trainable ({100*trainable_params/total_params:.2f}%)"
        )
        # Set the modified generator back to the model
        fm_model.gen = egnn
        fm_model.save_hyperparameters(_hparams)

    elif getattr(args, "freeze_layers", None):
        print("Applying freeze_layers finetuning...")
        state_dict = torch.load(args.ckpt_path, map_location=get_map_location())[
            "state_dict"
        ]
        state_dict = {k.replace("gen.", ""): v for k, v in state_dict.items()}
        egnn = copy.deepcopy(egnn_gen)
        egnn.load_state_dict(state_dict, strict=True)
        assert (
            not args.affinity_finetuning
        ), "Cannot use both freeze_layers and affinity_finetune."

        def _freeze_bottom_layers(args, egnn_gen):
            """Freeze bottom layers for gentle fine-tuning"""

            n_layers_to_train = getattr(args, "n_top_layers_to_retrain", 3)
            n_layers_to_train_pocket = getattr(
                args, "n_top_layers_to_retrain_pocket", 2
            )

            trainable_params = 0
            total_params = 0

            # For LigandGenerator with pocket conditioning (args.arch == "pocket")
            if hasattr(egnn_gen, "ligand_dec") and hasattr(
                egnn_gen.ligand_dec, "layers"
            ):
                layers = egnn_gen.ligand_dec.layers
                n_layers = len(layers)

                print(
                    f"Found {n_layers} ligand decoder layers, training top {n_layers_to_train}"
                )

                for i, layer in enumerate(layers):
                    layer_name = f"ligand_dec.layers.{i}"
                    if i < n_layers - n_layers_to_train:
                        # Freeze bottom layers
                        for name, param in layer.named_parameters():
                            param.requires_grad = False
                            total_params += param.numel()
                            print(f"  Frozen: {layer_name}.{name}")
                    else:
                        # Train top layers
                        for name, param in layer.named_parameters():
                            param.requires_grad = True
                            trainable_params += param.numel()
                            total_params += param.numel()
                            print(f"  Trainable: {layer_name}.{name}")
            else:
                raise ValueError("Ligand decoder missing in initiated model!")

            # Here in this func, always keep pocket encoder trainable (if it exists)
            if hasattr(egnn_gen, "pocket_enc") and egnn_gen.pocket_enc is not None:
                if n_layers_to_train_pocket is not None:
                    egnn_gen.pocket_enc.freeze_bottom_layers(n_layers_to_train_pocket)
                    print(
                        f"  Pocket encoder: training top {n_layers_to_train_pocket} layers"
                    )
                else:
                    for name, param in egnn_gen.pocket_enc.named_parameters():
                        param.requires_grad = True
                        trainable_params += param.numel()
                        total_params += param.numel()
                    print("  Pocket encoder: kept trainable")
            else:
                raise ValueError("Pocket encoder missing in initiated model!")

            # Keep output projections trainable - these are direct attributes of ligand_dec
            output_module_names = [
                "coord_out_proj",
                "atom_type_proj",
                "atom_charge_proj",
                "bond_proj",
                "bond_refine",
            ]
            if hparams["predict_affinity"]:
                output_module_names += [
                    "pic50_head",
                    "pkd_head",
                    "pki_head",
                    "pec50_head",
                ]
            if hparams["predict_docking_score"]:
                output_module_names += ["vina_head", "gnina_head"]

            for module_name in output_module_names:
                if hasattr(egnn_gen.ligand_dec, module_name):
                    module = getattr(egnn_gen.ligand_dec, module_name)
                    for name, param in module.named_parameters():
                        param.requires_grad = True
                        trainable_params += param.numel()
                        total_params += param.numel()
                    print(f"  Output module ligand_dec.{module_name}: kept trainable")

            # Handle final normalization layers
            norm_modules = ["final_coord_norm", "final_inv_norm", "final_bond_norm"]
            for module_name in norm_modules:
                if hasattr(egnn_gen.ligand_dec, module_name):
                    module = getattr(egnn_gen.ligand_dec, module_name)
                    for name, param in module.named_parameters():
                        param.requires_grad = True
                        trainable_params += param.numel()
                        total_params += param.numel()
                    print(f"  Norm module ligand_dec.{module_name}: kept trainable")

            print(
                f"Layer freezing: {trainable_params}/{total_params} parameters trainable ({100*trainable_params/total_params:.2f}%)"
            )
            return egnn_gen

        egnn = _freeze_bottom_layers(args, egnn)
        fm_model.gen = egnn

    elif getattr(args, "affinity_finetuning", None):
        print("Applying affinity_finetuning...")
        state_dict = torch.load(args.ckpt_path, map_location=get_map_location())[
            "state_dict"
        ]
        state_dict = {k.replace("gen.", ""): v for k, v in state_dict.items()}
        egnn = copy.deepcopy(egnn_gen)
        egnn.load_state_dict(state_dict, strict=True)
        assert (
            not args.freeze_layers
        ), "Cannot use both freeze_layers and affinity_finetune."

        def _freeze_all_except_affinity_heads(args, egnn_gen):
            """
            Freeze all parameters except the selected affinity head(s).
            """
            affinity_heads = args.affinity_finetuning
            if isinstance(affinity_heads, str):
                affinity_heads = [affinity_heads]
            trainable_params = 0
            total_params = 0

            # Freeze all parameters
            for name, param in egnn_gen.named_parameters():
                param.requires_grad = False
                total_params += param.numel()

            # Unfreeze only the selected affinity head(s)
            for head in affinity_heads:
                head_name = f"{head}_head"
                if hasattr(egnn_gen.ligand_dec, head_name):
                    module = getattr(egnn_gen.ligand_dec, head_name)
                    for n, p in module.named_parameters():
                        p.requires_grad = True
                        trainable_params += p.numel()
                        print(f"  Affinity head ligand_dec.{head_name}.{n}: trainable")
                else:
                    print(
                        f"  Warning: Affinity head ligand_dec.{head_name} not found in model."
                    )

            print(
                f"Affinity finetuning: {trainable_params}/{total_params} parameters trainable ({100*trainable_params/total_params:.2f}%)"
            )
            return egnn_gen

        egnn = _freeze_all_except_affinity_heads(args, egnn)
        fm_model.gen = egnn

    print("Done.")
    if return_info:
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
    return fm_model


def load_mol_model(
    args,
    ckpt_path: str = None,
    return_info: bool = True,
    dataset_info: Optional[dict] = None,
):
    checkpoint = torch.load(
        args.ckpt_path if ckpt_path is None else ckpt_path,
        map_location=get_map_location(),
    )
    hparams = dotdict(checkpoint["hyper_parameters"])
    hparams["compile_model"] = False
    # Set dataset and save paths
    hparams["data_path"] = getattr(args, "data_path", None)
    hparams["save_dir"] = args.save_dir
    # Set sampling params
    hparams["integration-steps"] = args.integration_steps
    hparams["sampling_strategy"] = args.ode_sampling_strategy
    hparams["use_inpaint_mode_embed"] = (
        hparams.get("scaffold_inpainting", False)
        or hparams.get("func_group_inpainting", False)
        or hparams.get("core_inpainting", False)
        or hparams.get("linker_inpainting", False)
        or hparams.get("fragment_inpainting", False)
        or hparams.get("substructure_inpainting", False)
    )
    hparams["scaffold_inpainting"] = args.scaffold_inpainting
    hparams["func_group_inpainting"] = args.func_group_inpainting
    hparams["linker_inpainting"] = args.linker_inpainting
    hparams["core_inpainting"] = args.core_inpainting
    hparams["fragment_inpainting"] = args.fragment_inpainting
    hparams["substructure_inpainting"] = args.substructure_inpainting
    hparams["substructure"] = args.substructure
    # Learning rate and optimizer params
    hparams["lr"] = args.lr if getattr(args, "lr", None) else hparams.get("lr", 1e-4)
    hparams["lr_schedule"] = (
        args.lr_schedule
        if getattr(args, "lr_schedule", None) is not None
        else hparams.get("lr_schedule", "exponential")
    )
    hparams["lr_gamma"] = (
        args.lr_gamma
        if getattr(args, "lr_gamma", None) is not None
        else hparams.get("lr_gamma", 0.995)
    )
    hparams["weight_decay"] = (
        args.weight_decay
        if getattr(args, "weight_decay", None) is not None
        else hparams.get("weight_decay", 1e-12)
    )
    hparams["beta1"] = (
        args.beta1
        if getattr(args, "beta1", None) is not None
        else hparams.get("beta1", 0.9)
    )
    hparams["beta2"] = (
        args.beta2
        if getattr(args, "beta2", None) is not None
        else hparams.get("beta2", 0.95)
    )
    # Loss weights
    hparams["coord_loss_weight"] = getattr(args, "coord_loss_weight", None)
    hparams["type_loss_weight"] = getattr(args, "type_loss_weight", None)
    hparams["bond_loss_weight"] = getattr(args, "bond_loss_weight", None)
    hparams["charge_loss_weight"] = getattr(args, "charge_loss_weight", None)
    hparams["hybridization_loss_weight"] = getattr(
        args, "hybridization_loss_weight", None
    )
    hparams["distance_loss_weight_lig"] = getattr(
        args, "distance_loss_weight_lig", None
    )
    hparams["distance_loss_weight_lig_pocket"] = getattr(
        args, "distance_loss_weight_lig_pocket", None
    )
    hparams["angle_loss_weight"] = getattr(args, "angle_loss_weight", None)
    hparams["angle_huber_delta"] = getattr(args, "angle_huber_delta", None)
    hparams["affinity_loss_weight"] = getattr(args, "affinity_loss_weight", None)
    hparams["docking_loss_weight"] = getattr(args, "docking_loss_weight", None)

    # Number of corrector iterations
    if args.corrector_iters > 0:
        assert (
            args.categorical_strategy == "velocity-sample"
        ), "Only velocity sampling supported for corrector iterations."
        hparams["corrector_iters"] = args.corrector_iters

    print("Building model vocabs...")
    vocab = _build_vocab()
    vocab_charges = _build_vocab_charges()
    if hparams["add_feats"]:
        print("Including hybridization features...")
        vocab_hybridization = _build_vocab_hybridization()
        vocab_aromatic = None  # _build_vocab_aromatic()
    else:
        vocab_hybridization = None
        vocab_aromatic = None
    print("Vocabs complete.")

    n_atom_types = vocab.size
    n_bond_types = get_n_bond_types(args.categorical_strategy)
    n_charge_types = vocab_charges.size
    n_hybridization_types = (
        vocab_hybridization.size if vocab_hybridization is not None else None
    )
    # n_aromatic_types = vocab_aromatic.size if vocab_aromatic is not None else None

    if args.arch == "flowr":
        from flowr.models.fm_mol import LigandCFM

        gen = LigandGenerator(
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
                n_hybridization_types  # + n_aromatic_types
                if hparams["add_feats"]
                else None
            ),
            use_rbf=hparams["use_rbf"],
            use_sphcs=hparams["use_sphcs"],
            use_distances=hparams["use_distances"],
            use_crossproducts=hparams["use_crossproducts"],
            use_fourier_time_embed=hparams["use_fourier_time_embed"],
            use_inpaint_mode_embed=hparams["use_inpaint_mode_embed"],
            self_cond=hparams["self_cond"],
            coord_skip_connect=hparams["coord_skip_connect"],
            coord_update_every_n=hparams.get("coord_update_every_n", None),
        )
    elif args.arch == "transformer":
        from flowr.models.fm_mol_transformer import LigandCFM
        from flowr.models.transformer.components import TransformerModule

        gen = TransformerModule(
            spatial_dim=3,
            n_atom_types=n_atom_types,
            n_charge_types=n_charge_types,
            n_hybridization_types=n_hybridization_types,
            predict_charges=True,
            num_heads=hparams["n_attn_heads"],
            num_layers=hparams["n_layers"],
            hidden_dim=hparams["d_model"],
            activation="SiLU",
            implementation="reimplemented",
            cross_attention=True,
            add_sinusoid_posenc=True,
            concat_combine_input=False,
            custom_weight_init=None,
        )
    else:
        raise ValueError(f"Unknown architecture {args.arch}")

    # Check if the model has been LoRA finetuned
    if hparams.get("lora_finetuning", False):
        # Apply LoRA to ligand decoder
        _inject_lora(
            lora_rank=hparams["lora_rank"],
            lora_alpha=hparams["lora_alpha"],
            mod=gen.ligand_dec,
        )

    # Initialize the integrator
    type_mask_index = None
    bond_mask_index = None
    integrator = Integrator(
        args.integration_steps,
        use_sde_simulation=args.use_sde_simulation,
        type_strategy=args.categorical_strategy,
        bond_strategy=args.categorical_strategy,
        coord_strategy="continuous",
        cat_noise_level=args.cat_sampling_noise_level,
        coord_noise_std=args.coord_noise_scale,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        use_cosine_scheduler=args.use_cosine_scheduler,
    )

    # Initialize the ligand flow model
    CFM = LigandCFM
    fm_model = CFM.load_from_checkpoint(
        args.ckpt_path,
        gen=gen,
        vocab=vocab,
        vocab_charges=vocab_charges,
        vocab_hybridization=vocab_hybridization,
        vocab_aromatic=vocab_aromatic,
        integrator=integrator,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        dataset_info=dataset_info,
        graph_inpainting=args.graph_inpainting is not None,
        **hparams,
    )

    if getattr(args, "lora_finetuning", None):
        print("Applying LoRA finetuning...")
        # Save the LoRA hyperparameters
        _hparams = fm_model.hparams
        _hparams["lora_finetuning"] = True
        _hparams["lora_rank"] = args.lora_rank
        _hparams["lora_alpha"] = args.lora_alpha

        # Load the pretrained weights
        state_dict = torch.load(args.ckpt_path, map_location=get_map_location())[
            "state_dict"
        ]
        state_dict = {k.replace("gen.", ""): v for k, v in state_dict.items()}
        egnn = copy.deepcopy(gen)
        egnn.load_state_dict(state_dict, strict=True)
        assert (
            not args.affinity_finetuning
        ), "Cannot use both LoRA and affinity_finetune."
        assert not args.freeze_layers, "Cannot use both LoRA and freeze_layers."

        # Apply LoRA to ligand decoder
        _inject_lora(
            lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, mod=egnn.ligand_dec
        )

        # Freeze all parameters except LoRA
        trainable_params = 0
        total_params = 0

        for n, p in egnn.ligand_dec.named_parameters():
            total_params += p.numel()
            if "lora" in n:
                p.requires_grad = True
                trainable_params += p.numel()
            else:
                p.requires_grad = False

        print(
            f"LoRA: {trainable_params}/{total_params} parameters trainable ({100*trainable_params/total_params:.2f}%)"
        )
        # Set the modified generator back to the model
        fm_model.gen = egnn
        fm_model.save_hyperparameters(_hparams)

    if return_info:
        return (
            fm_model,
            hparams,
            vocab,
            vocab_charges,
            vocab_hybridization,
            vocab_aromatic,
        )
    return fm_model


def build_mol_model(
    args,
    dm,
    dataset_info,
    train_mols,
    vocab,
    vocab_charges,
    vocab_hybridization=None,
    vocab_aromatic=None,
    coord_scale=1.0,
):
    # Get hyperparameters from the datamodule, pass these into the model to be saved
    hparams = {
        "epochs": args.epochs,
        "gradient_clip_val": args.gradient_clip_val,
        "dataset": args.dataset,
        "precision": get_precision(args),
        "architecture": args.arch,
        **dm.hparams,
    }

    n_atom_types = vocab.size
    n_bond_types = get_n_bond_types(args.categorical_strategy)
    n_charge_types = vocab_charges.size
    n_hybridization_types = (
        vocab_hybridization.size if vocab_hybridization is not None else 0
    )
    n_aromatic_types = vocab_aromatic.size if vocab_aromatic is not None else 0

    if args.arch == "transformer":
        from flowr.models.fm_mol_transformer import LigandCFM
        from flowr.models.transformer.components import TransformerModule

        gen = TransformerModule(
            spatial_dim=3,
            n_atom_types=n_atom_types,
            n_charge_types=n_charge_types,
            n_hybridization_types=n_hybridization_types,
            predict_charges=True,
            num_heads=args.n_attn_heads,
            num_layers=args.n_layers,
            hidden_dim=args.d_model,
            activation="SiLU",
            implementation="reimplemented",
            cross_attention=True,
            add_sinusoid_posenc=True,
            concat_combine_input=False,
            custom_weight_init=None,
        )
    else:
        from flowr.models.fm_mol import LigandCFM

        gen = LigandGenerator(
            args.n_coord_sets,
            args.d_model,
            args.d_message,
            args.n_layers,
            args.n_attn_heads,
            args.d_message_hidden,
            args.d_edge,
            emb_size=args.emb_size,
            n_atom_types=n_atom_types,
            n_charge_types=n_charge_types,
            n_bond_types=n_bond_types,
            n_extra_atom_feats=(
                (n_hybridization_types + n_aromatic_types) if args.add_feats else None
            ),
            use_rbf=args.use_rbf,
            use_sphcs=args.use_sphcs,
            use_distances=args.use_distances,
            use_crossproducts=args.use_crossproducts,
            use_fourier_time_embed=args.use_fourier_time_embed,
            graph_inpainting=args.graph_inpainting,
            use_inpaint_mode_embed=args.scaffold_inpainting
            or args.func_group_inpainting
            or args.core_inpainting
            or args.linker_inpainting
            or args.fragment_inpainting
            or args.substructure_inpainting,
            self_cond=args.self_condition,
            coord_skip_connect=not args.no_coord_skip_connect,
            coord_update_every_n=getattr(args, "coord_update_every_n", None),
        )

    type_mask_index = None
    bond_mask_index = None

    if args.categorical_strategy == "mask":
        type_mask_index = vocab.indices_from_tokens(["<MASK>"])[0]
        bond_mask_index = util.BOND_MASK_INDEX
        train_strategy = "mask"
        sampling_strategy = "mask"

    elif args.categorical_strategy == "uniform-sample":
        train_strategy = "ce"
        sampling_strategy = "uniform-sample"

    elif args.categorical_strategy == "prior-sample":
        train_strategy = "ce"
        sampling_strategy = "uniform-sample"

    elif args.categorical_strategy == "velocity-sample":
        train_strategy = "ce"
        sampling_strategy = "velocity-sample"

    else:
        raise ValueError(
            f"Interpolation '{args.categorical_strategy}' is not supported."
        )

    # Training steps
    train_steps = util.calc_train_steps(dm, args.epochs, args.acc_batches) // args.gpus
    print(f"Total training steps {train_steps}")
    if args.lr_schedule in ["cosine", "constant"]:
        warm_up_steps = get_warm_up_steps(args, train_steps)
        print(f"Warmup steps {warm_up_steps}")
    else:
        warm_up_steps = 0

    from flowr.models.integrator import Integrator

    integrator = Integrator(
        args.num_inference_steps,
        use_sde_simulation=args.use_sde_simulation,
        type_strategy=sampling_strategy,
        bond_strategy=sampling_strategy,
        coord_strategy=args.coord_sampling_strategy,
        cat_noise_level=args.cat_sampling_noise_level,
        coord_noise_std=args.coord_noise_scale,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        use_cosine_scheduler=args.use_cosine_scheduler,
    )

    fm_model = LigandCFM(
        gen,
        vocab,
        vocab_charges,
        args.lr,
        integrator,
        add_feats=args.add_feats,
        vocab_hybridization=vocab_hybridization,
        vocab_aromatic=vocab_aromatic,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        coord_scale=coord_scale,
        sampling_strategy=args.sample_schedule,
        type_strategy=train_strategy,
        bond_strategy=train_strategy,
        coord_loss_weight=args.coord_loss_weight,
        type_loss_weight=args.type_loss_weight,
        bond_loss_weight=args.bond_loss_weight,
        charge_loss_weight=args.charge_loss_weight,
        hybridization_loss_weight=args.hybridization_loss_weight,
        distance_loss_weight_lig=args.distance_loss_weight_lig,
        pairwise_metrics=False,
        use_ema=args.use_ema,
        compile_model=False,
        self_condition=args.self_condition,
        distill=False,
        lr_schedule=args.lr_schedule,
        lr_gamma=args.lr_gamma,
        warm_up_steps=warm_up_steps,
        total_steps=train_steps,
        train_mols=train_mols,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        remove_hs=args.remove_hs,
        remove_aromaticity=args.remove_aromaticity,
        save_dir=args.save_dir,
        dataset_info=dataset_info,
        data_path=args.data_path,
        scaffold_inpainting=args.scaffold_inpainting,
        func_group_inpainting=args.func_group_inpainting,
        linker_inpainting=args.linker_inpainting,
        core_inpainting=args.core_inpainting,
        fragment_inpainting=args.fragment_inpainting,
        substructure_inpainting=args.substructure_inpainting,
        graph_inpainting=args.graph_inpainting is not None,
        mixed_uncond_inpaint=args.mixed_uncond_inpaint,
        use_t_loss_weights=args.use_t_loss_weights,
        corrector_iters=args.corrector_iters,
        pretrained_weights=args.load_pretrained_ckpt is not None,
        **hparams,
    )
    return fm_model


# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# ******************************* LOAD DATA ***********************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************


def load_rdkit_mols(data_path, train_ids, val_ids, test_ids):
    if isinstance(data_path, list):
        # Get all rdkit mols from all data_paths
        rdkit_mols = []
        for path in data_path:
            if Path(os.path.join(path, "rdkit_mols.pkl")).exists():
                with open(os.path.join(path, "rdkit_mols.pkl"), "rb") as f:
                    rdkit_mols.extend(pickle.load(f))
            else:
                raise FileNotFoundError(
                    "RDKit mols not found. Must be provided as rdkit_mols.pkl in data_path."
                )
    else:
        if Path(os.path.join(data_path, "rdkit_mols.pkl")).exists():
            print("Loading RDKit mols...")
            with open(os.path.join(data_path, "rdkit_mols.pkl"), "rb") as f:
                rdkit_mols = pickle.load(f)
            print("Done.")
        else:
            raise FileNotFoundError(
                "RDKit mols not found. Must be provided as rdkit_mols.pkl in data_path."
            )

    train_mols = [Chem.Mol(rdkit_mols[i]) for i in train_ids]
    val_mols = [Chem.Mol(rdkit_mols[i]) for i in val_ids]
    test_mols = [Chem.Mol(rdkit_mols[i]) for i in test_ids]
    return train_mols, val_mols, test_mols


def build_data_statistic(args):
    if args.data_path is None:
        assert (
            args.data_paths is not None
        ), "If data_path is not provided, data_paths must be provided."
        print(
            f"Using multiple datasets at once. Extracting dataset statistics from first dataset in the list: {args.data_paths[0]}"
        )
        data_path = args.data_paths[0]
    else:
        assert args.data_paths is None, "Both data_path and data_paths provided."
        data_path = args.data_path

    train_statistics = Statistics.get_statistics(
        os.path.join(data_path, "processed"),
        "train",
        dataset=args.dataset,
        remove_hs=args.remove_hs,
    )
    val_statistics = Statistics.get_statistics(
        os.path.join(data_path, "processed"),
        "val",
        dataset=args.dataset,
        remove_hs=args.remove_hs,
    )
    test_statistics = Statistics.get_statistics(
        os.path.join(data_path, "processed"),
        "test",
        dataset=args.dataset,
        remove_hs=args.remove_hs,
    )
    return {"train": train_statistics, "val": val_statistics, "test": test_statistics}


def build_dm(
    args: Namespace,
    vocab: Vocabulary,
    vocab_charges: Vocabulary,
    vocab_hybridization: Optional[Vocabulary] = None,
    vocab_aromatic: Optional[Vocabulary] = None,
    vocab_pocket_atoms: Optional[Vocabulary] = None,
    vocab_pocket_res: Optional[Vocabulary] = None,
    atom_types_distribution: Optional[Dict[str, float]] = None,
    bond_types_distribution: Optional[Dict[str, float]] = None,
    same_train_test_split: Optional[bool] = False,
    train_mols: Optional[List[Chem.Mol]] = None,
    val_mols: Optional[List[Chem.Mol]] = None,
    test_mols: Optional[List[Chem.Mol]] = None,
    train_dataset: Optional[Callable] = None,
    val_dataset: Optional[Callable] = None,
    test_dataset: Optional[Callable] = None,
):
    """
    Build a data module for training, validation, and testing.
    Args:
        args: The command-line arguments.
        vocab: The vocabulary for the dataset.
        vocab_charges: The vocabulary for atomic charges.
        vocab_hybridization: The vocabulary for hybridization states.
        vocab_aromatic: The vocabulary for aromaticity.
        vocab_pocket_atoms: The vocabulary for pocket atoms.
        vocab_pocket_res: The vocabulary for pocket residues.
        atom_types_distribution: The distribution of atom types.
        bond_types_distribution: The distribution of bond types.
        train_mols: The training RDKit molecules.
        val_mols: The validation RDKit molecules.
        test_mols: The test RDKit molecules.
        train_dataset: A pre-processed training dataset.
        val_dataset: A pre-processed validation dataset.
        test_dataset: A pre-processed test dataset.
    """

    # Check if any dataset is provided
    if (
        train_dataset is not None or val_dataset is not None or test_dataset is not None
    ) and not args.use_smol:
        assert (
            train_dataset is not None
            and val_dataset is not None
            and test_dataset is not None
        ), "If one dataset is provided, all datasets must be provided."
    # Check if any RDKit molecule list is provided
    if train_mols is not None or val_mols is not None or test_mols is not None:
        assert (
            train_mols is not None and val_mols is not None and test_mols is not None
        ), "If a train/val/test RDKit molecule list is provided, all must be provided."

    # Load the bucket sizes for the dataloader
    if args.use_bucket_sampler:
        if args.dataset == "qm9":
            coord_std = constants.QM9_COORDS_STD_DEV
            padded_sizes = constants.QM9_BUCKET_LIMITS
        elif args.dataset in ["geom-drugs", "zinc3d", "enamine", "pubchem3d", "omol25"]:
            coord_std = constants.GEOM_COORDS_STD_DEV
            padded_sizes = constants.GEOM_DRUGS_BUCKET_LIMITS
        elif args.dataset == "spindr" or args.dataset == "spindr_kinase":
            coord_std = constants.SPINDR_COORDS_STD_DEV
            padded_sizes = constants.SPINDR_BUCKET_LIMITS
        elif args.dataset == "crossdocked":
            coord_std = constants.CROSSDOCKED_COORDS_STD_DEV
            padded_sizes = constants.CROSSDOCKED_BUCKET_LIMITS
        elif args.dataset == "kinodata":
            coord_std = constants.KINODATA_COORDS_STD_DEV
            padded_sizes = constants.KINODATA_BUCKET_LIMITS
        elif args.dataset == "bindingmoad":
            coord_std = constants.BINDINGMOAD_COORDS_STD_DEV
            padded_sizes = constants.BINDINGMOAD_BUCKET_LIMITS
        elif args.dataset == "adds":
            coord_std = constants.ADDS_COORDS_STD_DEV
            padded_sizes = constants.ADDS_BUCKET_LIMITS
        elif args.dataset == "hiqbind":
            coord_std = constants.HIQBIND_COORDS_STD_DEV
            padded_sizes = constants.HIQBIND_BUCKET_LIMITS
        elif args.dataset == "sair":
            coord_std = constants.SAIR_COORDS_STD_DEV
            padded_sizes = constants.SAIR_BUCKET_LIMITS
        elif args.dataset == "bindingmoad":
            coord_std = constants.BINDINGMOAD_COORDS_STD_DEV
            padded_sizes = constants.BINDINGMOAD_BUCKET_LIMITS
        elif args.dataset.startswith("bindingnet"):
            coord_std = constants.BINDINGNET_COORDS_STD_DEV
            padded_sizes = constants.BINDINGNET_BUCKET_LIMITS
        elif args.dataset in ["dataset_low-mid-q", "dataset_high-q"]:
            coord_std = constants.DEFAULT_COORDS_STD_DEV
            padded_sizes = constants.DEFAULT_BUCKET_LIMITS
        else:
            raise ValueError(f"Unknown dataset {args.dataset}")
    else:
        padded_sizes = None

    if not args.scale_coords:
        coord_std = 1.0

    # Molecule features
    n_bond_types = get_n_bond_types(args.categorical_strategy)
    n_charge_types = vocab_charges.size
    n_hybridization_types = (
        vocab_hybridization.size if vocab_hybridization is not None else None
    )
    n_aromatic_types = vocab_aromatic.size if vocab_aromatic is not None else None

    # Build transform function for individual molecules (takes care of zero-com, one-hot, opt. rotation/translation/scaling)
    transform = partial(
        complex_transform,
        vocab=vocab,
        vocab_charges=vocab_charges,
        vocab_hybridization=vocab_hybridization,
        vocab_aromatic=vocab_aromatic,
        n_bonds=n_bond_types,
        coord_std=coord_std,
        pocket_noise=args.pocket_noise,
        pocket_noise_std=args.pocket_coord_noise_std,
        use_interactions=args.flow_interactions
        or args.predict_interactions
        or args.interaction_inpainting,
        rotate_complex=args.arch == "transformer",
    )
    # Build datasets
    if args.dataset == "spindr" and args.use_smol:
        data_path = Path(args.data_path)
        train_dataset = (
            GeometricDataset.load(
                data_path / "train.smol",
                dataset=args.dataset,
                transform=transform,
                remove_hs=args.remove_hs,
                remove_aromaticity=args.remove_aromaticity,
                skip_non_valid=True,
            )
            if train_dataset is None
            else train_dataset
        )
        val_dataset = (
            GeometricDataset.load(
                data_path / "val.smol",
                dataset=args.dataset,
                transform=transform,
                remove_hs=args.remove_hs,
                remove_aromaticity=args.remove_aromaticity,
                skip_non_valid=True,
            )
            if val_dataset is None
            else val_dataset
        )
        if test_dataset is None:
            test_dataset = GeometricDataset.load(
                data_path / "test.smol",
                dataset=args.dataset,
                transform=transform,
                remove_hs=args.remove_hs,
                remove_aromaticity=args.remove_aromaticity,
                skip_non_valid=True,
            )
            if getattr(args, "sample_n_molecules_per_target", 0) > 1:
                test_dataset = test_dataset.sample_n_molecules_per_target(
                    args.sample_n_molecules_per_target
                )
    else:
        if train_dataset is None and val_dataset is None and test_dataset is None:
            # Handle multiple data paths for concatenated datasets
            # Handle multiple data paths for concatenated datasets
            if hasattr(args, "data_paths") and args.data_paths is not None:
                print(
                    "Multiple data paths detected. Using concatenated datasets and weighted sampling."
                )
                print(f"Dataset weights: {args.dataset_weights}")
                print(f"Validation and test split taken from {args.data_paths[0]}")

                assert (
                    args.data_path is None
                ), "If data_paths is provided, data_path must be None."
                assert (
                    hasattr(args, "dataset_weights")
                    and args.dataset_weights is not None
                ), "Dataset weights must be provided when using multiple data paths."
                assert len(args.dataset_weights) == len(
                    args.data_paths
                ), "Dataset weights must match data paths length."
                assert (
                    abs(sum(args.dataset_weights) - 1.0) < 1e-6
                ), "Dataset weight probabilities must sum to 1.0"

                # Load individual datasets
                datasets = []
                dataset_sizes = []
                all_lengths = []
                dataset_offsets = [
                    0
                ]  # Track where each dataset starts in concatenated version

                # Load splits for each dataset
                all_train_indices = []
                val_indices = []
                test_indices = []

                for i, data_path in enumerate(args.data_paths):
                    dataset = PocketComplexLMDBDataset(
                        root=data_path,
                        transform=transform,
                        remove_hs=args.remove_hs,
                        remove_aromaticity=args.remove_aromaticity,
                        skip_non_valid=False,
                    )
                    datasets.append(dataset)
                    dataset_sizes.append(len(dataset))
                    all_lengths.extend(dataset.lengths)

                    # Load splits for this dataset
                    splits_path = os.path.join(data_path, "splits.npz")
                    if not os.path.exists(splits_path):
                        raise ValueError(
                            f"Splits file {splits_path} not found. Please create it using create_data_statistics.py."
                        )

                    splits = np.load(splits_path)
                    dataset_train_idx = splits["idx_train"]
                    if i == 0:
                        val_indices = splits["idx_val"]
                        test_indices = splits["idx_test"]

                    # Adjust indices to work with concatenated dataset
                    offset = dataset_offsets[-1]
                    adjusted_train_idx = dataset_train_idx + offset
                    all_train_indices.extend(adjusted_train_idx)

                    # Update offset for next dataset
                    dataset_offsets.append(offset + len(dataset))

                # Create concatenated dataset
                dataset = ConcatDataset(datasets)
                dataset_lengths = torch.tensor(all_lengths)

                # Convert probabilities to per-sample weights
                sample_weights = []
                for prob, size in zip(args.dataset_weights, dataset_sizes):
                    weight_per_sample = (
                        prob / size
                    )  # This ensures the probability is achieved
                    sample_weights.extend([weight_per_sample] * size)

                # Store weights for the datamodule
                dataset.sample_weights = np.array(sample_weights)

                # Convert indices to numpy arrays
                idx_train = np.array(all_train_indices)
                idx_val = np.array(val_indices)
                idx_test = np.array(test_indices)
            else:
                dataset = PocketComplexLMDBDataset(
                    root=args.data_path,
                    transform=transform,
                    remove_hs=args.remove_hs,
                    remove_aromaticity=args.remove_aromaticity,
                    skip_non_valid=False,
                )
                dataset_lengths = torch.tensor(dataset.lengths)

                # Get dataset splits
                splits_path = os.path.join(args.data_path, "splits.npz")
                if os.path.exists(splits_path):
                    idx_train, idx_val, idx_test = make_splits(splits=splits_path)
                else:
                    raise ValueError(
                        f"Splits file {splits_path} not found. Please create it using create_data_statistics.py."
                    )

            if same_train_test_split:
                # NOTE: Necessary when running active learning as predict runs over test set!
                idx_test = idx_train
                print("Warning: Using the same split for train and test!")
            print(f"train {len(idx_train)}, val {len(idx_val)}, test {len(idx_test)}")
            train_dataset = DatasetSubset(
                dataset, idx_train, lengths=dataset_lengths[idx_train].tolist()
            )
            val_dataset = DatasetSubset(
                dataset, idx_val, lengths=dataset_lengths[idx_val].tolist()
            )
            test_dataset = DatasetSubset(
                dataset, idx_test, lengths=dataset_lengths[idx_test].tolist()
            )
            # Slice the sample weights for training indices
            if hasattr(dataset, "sample_weights"):
                train_dataset.sample_weights = dataset.sample_weights[
                    idx_train
                ].tolist()

            if getattr(args, "sample_n_molecules_per_target", 0) > 1:
                test_dataset = test_dataset.sample_n_molecules_per_target(
                    args.sample_n_molecules_per_target
                )

    # Load training molecules
    if train_mols is None:
        data_path = args.data_path if args.data_path is not None else args.data_paths
        train_mols, val_mols, test_mols = load_rdkit_mols(
            data_path,
            train_ids=idx_train,
            val_ids=idx_val,
            test_ids=idx_test,
        )

    # Build the interpolants
    type_mask_index, bond_mask_index = None, None
    if args.categorical_strategy == "mask":
        type_mask_index = vocab.indices_from_tokens(["<MASK>"])[0]
        bond_mask_index = BOND_MASK_INDEX
        categorical_interpolation = "unmask"
        categorical_noise = "mask"
    elif args.categorical_strategy == "uniform-sample":
        categorical_interpolation = "unmask"
        categorical_noise = "uniform-sample"
    elif args.categorical_strategy == "prior-sample":
        categorical_interpolation = "unmask"
        categorical_noise = "prior-sample"
    elif args.categorical_strategy == "velocity-sample":
        categorical_interpolation = "sample"
        categorical_noise = "uniform-sample"
    else:
        raise ValueError(
            f"Interpolation '{args.categorical_strategy}' is not supported."
        )

    ## Create conformer generator if graph inpainting is set to conformer
    conformer_generator = (
        smolRD.ConformerGenerator(
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
        type_noise=categorical_noise,
        bond_noise=categorical_noise,
        zero_com=True,  # args.pocket_noise in ["fix", "random"],
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        atom_types_distribution=atom_types_distribution,
        bond_types_distribution=bond_types_distribution,
        conformer_generator=conformer_generator,
    )

    train_fixed_time = None
    train_interpolant = ComplexInterpolant(
        prior_sampler,
        ligand_coord_interpolation=(
            "linear" if not args.use_cosine_scheduler else "cosine"
        ),
        ligand_coord_noise_std=args.coord_noise_std_dev,
        ligand_coord_noise_schedule=args.coord_noise_schedule,
        ligand_type_interpolation=categorical_interpolation,
        ligand_bond_interpolation=categorical_interpolation,
        ligand_time_alpha=args.time_alpha,
        ligand_time_beta=args.time_beta,
        ligand_fixed_time=train_fixed_time,
        split_continuous_discrete_time=args.split_continuous_discrete_time,
        pocket_time_alpha=args.time_alpha,
        pocket_time_beta=args.time_beta,
        pocket_fixed_time=train_fixed_time,
        pocket_coord_noise_std=args.pocket_coord_noise_std,
        pocket_noise=args.pocket_noise,
        separate_pocket_interpolation=args.separate_pocket_interpolation,
        separate_interaction_interpolation=args.separate_interaction_interpolation,
        interaction_fixed_time=args.interaction_fixed_time,
        interaction_time_alpha=args.time_alpha,
        interaction_time_beta=args.time_beta,
        flow_interactions=args.flow_interactions,
        interaction_inpainting=args.interaction_inpainting,
        scaffold_inpainting=args.scaffold_inpainting,
        func_group_inpainting=args.func_group_inpainting,
        substructure_inpainting=args.substructure_inpainting,
        substructure=args.substructure,
        linker_inpainting=args.linker_inpainting,
        core_inpainting=args.core_inpainting,
        fragment_inpainting=args.fragment_inpainting,
        max_fragment_cuts=args.max_fragment_cuts,
        graph_inpainting=args.graph_inpainting,
        mixed_uncond_inpaint=args.mixed_uncond_inpaint,
        mixed_uniform_beta_time=args.mixed_uniform_beta_time,
        n_interaction_types=(
            len(PROLIF_INTERACTIONS) + 1
            if args.flow_interactions
            or args.predict_interactions
            or args.interaction_inpainting
            else None
        ),
        dataset=args.dataset,
        vocab=vocab,
        vocab_charges=vocab_charges,
        vocab_hybridization=vocab_hybridization,
        rotation_alignment=args.rotation_alignment,
        permutation_alignment=args.permutation_alignment,
        sample_mol_sizes=False,
        inference=False,
    )
    eval_interpolant = ComplexInterpolant(
        prior_sampler,
        dataset=args.dataset,
        vocab=vocab,
        vocab_charges=vocab_charges,
        vocab_hybridization=vocab_hybridization,
        ligand_coord_interpolation=(
            "linear" if not args.use_cosine_scheduler else "cosine"
        ),
        ligand_type_interpolation=categorical_interpolation,
        ligand_bond_interpolation=categorical_interpolation,
        ligand_fixed_time=0.9,
        pocket_fixed_time=0.9,
        interaction_fixed_time=0.9,
        split_continuous_discrete_time=args.split_continuous_discrete_time,
        pocket_noise=args.pocket_noise,
        separate_pocket_interpolation=args.separate_pocket_interpolation,
        separate_interaction_interpolation=args.separate_interaction_interpolation,
        n_interaction_types=(
            len(PROLIF_INTERACTIONS) + 1
            if args.flow_interactions
            or args.predict_interactions
            or args.interaction_inpainting
            else None
        ),
        flow_interactions=args.flow_interactions,
        interaction_inpainting=args.interaction_inpainting,
        scaffold_inpainting=args.scaffold_inpainting,
        func_group_inpainting=args.func_group_inpainting,
        linker_inpainting=args.linker_inpainting,
        core_inpainting=args.core_inpainting,
        fragment_inpainting=args.fragment_inpainting,
        max_fragment_cuts=args.max_fragment_cuts,
        substructure_inpainting=args.substructure_inpainting,
        substructure=args.substructure,
        graph_inpainting=args.graph_inpainting,
        batch_ot=False,
        rotation_alignment=(
            args.linker_inpainting
            or args.fragment_inpainting
            or args.func_group_inpainting
            or args.substructure_inpainting
            or args.scaffold_inpainting
            or args.interaction_inpainting
            or args.core_inpainting
        )
        and args.rotation_alignment,
        permutation_alignment=(
            args.linker_inpainting
            or args.fragment_inpainting
            or args.func_group_inpainting
            or args.substructure_inpainting
            or args.scaffold_inpainting
            or args.interaction_inpainting
            or args.core_inpainting
        )
        and args.permutation_alignment,
        sample_mol_sizes=getattr(args, "sample_mol_sizes", False),
        inference=True,
    )

    # Build datamodule
    dm = GeometricInterpolantDM(
        train_dataset,
        val_dataset,
        test_dataset,
        args.batch_cost,
        val_batch_size=args.val_batch_cost,
        train_interpolant=train_interpolant,
        val_interpolant=eval_interpolant,
        test_interpolant=eval_interpolant,
        use_bucket_sampler=args.use_bucket_sampler,
        use_adaptive_sampler=args.use_adaptive_sampler,
        use_weighted_sampler=args.use_weighted_sampler,
        bucket_limits=padded_sizes,
        bucket_cost_scale=args.bucket_cost_scale,
        pad_to_bucket=False,
        num_workers=args.num_workers,
        train_mols=train_mols,
        val_mols=val_mols,
        test_mols=test_mols,
    )

    return dm


def load_dm(
    args,
    hparams,
    vocab,
    vocab_charges,
    vocab_hybridization=None,
    vocab_aromatic=None,
    atom_types_distribution=None,
    bond_types_distribution=None,
    train_mode: bool = False,
):
    """Load utility functions and interpolant for evaluation."""

    coord_std = hparams["coord_scale"]

    n_bond_types = util.get_n_bond_types(args.categorical_strategy)
    n_charge_types = vocab_charges.size
    n_hybridization_types = (
        vocab_hybridization.size if vocab_hybridization is not None else None
    )
    n_aromatic_types = vocab_aromatic.size if vocab_aromatic is not None else None
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
        use_interactions=args.flow_interactions
        or args.predict_interactions
        or args.interaction_inpainting,
    )
    # Initialize conformer generator if graph inpainting is enabled and set to conformer
    conformer_generator = (
        smolRD.ConformerGenerator(
            cache_dir=Path(args.data_path) / "conformers",
            max_conformers=10,
            max_iters=200,
            enable_caching=True,
            vocab=vocab,
        )
        if args.graph_inpainting is not None and args.graph_inpainting == "conformer"
        else None
    )
    type_mask_index = None
    bond_mask_index = None
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
        atom_types_distribution=atom_types_distribution,
        bond_types_distribution=bond_types_distribution,
    )

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

    train_interpolant = None
    if train_mode:
        train_fixed_time = None
        train_interpolant = ComplexInterpolant(
            prior_sampler,
            ligand_coord_interpolation=(
                "linear" if not args.use_cosine_scheduler else "cosine"
            ),
            ligand_coord_noise_std=args.coord_noise_std_dev,
            ligand_coord_noise_schedule=args.coord_noise_schedule,
            ligand_type_interpolation=categorical_interpolation,
            ligand_bond_interpolation=categorical_interpolation,
            ligand_time_alpha=args.time_alpha,
            ligand_time_beta=args.time_beta,
            ligand_fixed_time=train_fixed_time,
            split_continuous_discrete_time=args.split_continuous_discrete_time,
            pocket_time_alpha=args.time_alpha,
            pocket_time_beta=args.time_beta,
            pocket_fixed_time=train_fixed_time,
            pocket_coord_noise_std=args.pocket_coord_noise_std,
            pocket_noise=args.pocket_noise,
            separate_pocket_interpolation=args.separate_pocket_interpolation,
            separate_interaction_interpolation=args.separate_interaction_interpolation,
            interaction_fixed_time=args.interaction_fixed_time,
            interaction_time_alpha=args.time_alpha,
            interaction_time_beta=args.time_beta,
            interaction_inpainting=args.interaction_inpainting,
            scaffold_inpainting=args.scaffold_inpainting,
            func_group_inpainting=args.func_group_inpainting,
            substructure_inpainting=args.substructure_inpainting,
            substructure=args.substructure,
            linker_inpainting=args.linker_inpainting,
            core_inpainting=args.core_inpainting,
            fragment_inpainting=args.fragment_inpainting,
            max_fragment_cuts=args.max_fragment_cuts,
            graph_inpainting=args.graph_inpainting,
            mixed_uncond_inpaint=args.mixed_uncond_inpaint,
            mixed_uniform_beta_time=args.mixed_uniform_beta_time,
            n_interaction_types=(
                len(PROLIF_INTERACTIONS)
                if hparams["flow_interactions"]
                or hparams["predict_interactions"]
                or hparams["interaction_inpainting"]
                else None
            ),
            flow_interactions=hparams["flow_interactions"],
            dataset=args.dataset,
            vocab=vocab,
            vocab_charges=vocab_charges,
            vocab_hybridization=vocab_hybridization,
            rotation_alignment=args.rotation_alignment,
            permutation_alignment=args.permutation_alignment,
            sample_mol_sizes=False,
            inference=False,
        )
    eval_interpolant = ComplexInterpolant(
        prior_sampler,
        ligand_coord_interpolation=(
            "linear" if not args.use_cosine_scheduler else "cosine"
        ),
        ligand_type_interpolation=categorical_interpolation,
        ligand_bond_interpolation=categorical_interpolation,
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
        core_inpainting=args.core_inpainting,
        fragment_inpainting=args.fragment_inpainting,
        max_fragment_cuts=args.max_fragment_cuts,
        substructure_inpainting=args.substructure_inpainting,
        substructure=args.substructure,
        graph_inpainting=args.graph_inpainting,
        dataset=args.dataset,
        sample_mol_sizes=False,
        vocab=vocab,
        vocab_charges=vocab_charges,
        vocab_hybridization=vocab_hybridization,
        batch_ot=False,
        rotation_alignment=(
            args.linker_inpainting
            or args.fragment_inpainting
            or args.func_group_inpainting
            or args.substructure_inpainting
            or args.scaffold_inpainting
            or args.core_inpainting
        )
        and args.rotation_alignment,
        permutation_alignment=(
            args.linker_inpainting
            or args.fragment_inpainting
            or args.func_group_inpainting
            or args.substructure_inpainting
            or args.scaffold_inpainting
            or args.core_inpainting
        )
        and args.permutation_alignment,
        inference=True,
    )

    # Get dataset
    dataset = PocketComplexLMDBDataset(
        root=args.data_path,
        transform=transform,
        remove_hs=hparams["remove_hs"],
        remove_aromaticity=hparams["remove_aromaticity"],
        skip_non_valid=False,
    )
    dataset_lengths = torch.tensor(dataset.lengths)

    # Get dataset splits
    splits_path = os.path.join(args.data_path, "splits.npz")
    if os.path.exists(splits_path):
        idx_train, idx_val, idx_test = make_splits(splits=splits_path)
    else:
        raise ValueError(
            f"Splits file {splits_path} not found. Please create it using create_data_statistics.py."
        )
    print(f"train {len(idx_train)}, val {len(idx_val)}, test {len(idx_test)}")
    train_dataset = DatasetSubset(
        dataset, idx_train, lengths=dataset_lengths[idx_train].tolist()
    )
    val_dataset = DatasetSubset(
        dataset, idx_val, lengths=dataset_lengths[idx_val].tolist()
    )
    if args.sample_n_molecules_val:
        val_dataset = val_dataset.sample_n_molecules(
            args.sample_n_molecules_val, seed=args.seed
        )
    test_dataset = DatasetSubset(
        dataset, idx_test, lengths=dataset_lengths[idx_test].tolist()
    )
    # Load training molecules
    train_mols, val_mols, test_mols = load_rdkit_mols(
        args.data_path,
        train_ids=idx_train,
        val_ids=idx_val,
        test_ids=idx_test,
    )

    # Build datamodule
    dm = GeometricInterpolantDM(
        train_dataset,
        val_dataset,
        test_dataset,
        args.batch_cost,
        val_batch_size=args.val_batch_cost,
        train_interpolant=train_interpolant,
        val_interpolant=eval_interpolant,
        test_interpolant=eval_interpolant,
        use_bucket_sampler=args.use_bucket_sampler,
        use_adaptive_sampler=args.use_adaptive_sampler,
        use_weighted_sampler=args.use_weighted_sampler,
        bucket_limits=None,
        bucket_cost_scale=args.bucket_cost_scale,
        pad_to_bucket=False,
        num_workers=args.num_workers,
        train_mols=train_mols,
        val_mols=val_mols,
        test_mols=test_mols,
    )
    return dm


def build_mol_dm(
    args,
    vocab,
    vocab_charges,
    vocab_hybridization=None,
    vocab_aromatic=None,
    atom_types_distribution=None,
    bond_types_distribution=None,
):

    if args.use_bucket_sampler:
        padded_sizes = constants.QM9_BUCKET_LIMITS
        if args.dataset in [
            "geom_drugs",
            "geom_drugs_full",
            "zinc3d",
            "enamine",
            "pubchem3d",
            "omol25",
        ]:
            padded_sizes = constants.GEOM_DRUGS_BUCKET_LIMITS
        elif args.dataset == "combined":
            padded_sizes = constants.COMBINED_BUCKET_LIMITS
        else:
            raise ValueError(f"Dataset '{args.dataset}' is not supported for training.")
    else:
        padded_sizes = None

    if not args.scale_coords:
        coord_std = 1.0
    else:
        if args.dataset == "qm9":
            coord_std = constants.QM9_COORDS_STD_DEV
        elif args.dataset in [
            "geom_drugs",
            "geom_drugs_full",
            "zinc3d",
            "enamine",
            "pubchem3d",
            "omol25",
        ]:
            coord_std = constants.GEOM_COORDS_STD_DEV
        elif args.dataset == "combined":
            coord_std = constants.COMBINED_COORDS_STD_DEV

    # Molecule features
    n_bond_types = get_n_bond_types(args.categorical_strategy)
    n_charge_types = vocab_charges.size
    n_hybridization_types = (
        vocab_hybridization.size if vocab_hybridization is not None else None
    )
    n_aromatic_types = vocab_aromatic.size if vocab_aromatic is not None else None

    # Build transform function for individual molecules (takes care of zero-com, one-hot, opt. rotation/translation/scaling)
    transform = partial(
        util.mol_transform,
        vocab=vocab,
        vocab_charges=vocab_charges,
        vocab_hybridization=vocab_hybridization,
        vocab_aromatic=vocab_aromatic,
        n_bonds=n_bond_types,
        coord_std=coord_std,
        zero_com=True,
        rotate=True,  # args.arch == "transformer",
    )

    if args.dataset == "geom_drugs":
        train_dataset = GeometricMolLMDBDataset(
            root=Path(args.data_path) / "train",
            transform=transform,
            remove_hs=args.remove_hs,
            remove_aromaticity=args.remove_aromaticity,
            skip_non_valid=False,
        )
        val_dataset = GeometricMolLMDBDataset(
            root=Path(args.data_path) / "val",
            transform=transform,
            remove_hs=args.remove_hs,
            remove_aromaticity=args.remove_aromaticity,
            skip_non_valid=False,
        )
        train_len, val_len = len(train_dataset), len(val_dataset)
        print(f"train {train_len}, val {val_len}")
    else:
        if hasattr(args, "data_paths") and args.data_paths:
            assert (
                args.data_path is None
            ), "data_path must be None when using data_paths"
            datasets = []
            all_lengths = []
            for data_path in args.data_paths:
                dataset = GeometricMolLMDBDataset(
                    root=data_path,
                    transform=transform,
                    remove_hs=args.remove_hs,
                    remove_aromaticity=args.remove_aromaticity,
                    skip_non_valid=False,
                )
                datasets.append(dataset)
                all_lengths.extend(dataset.lengths)
            # Create concatenated dataset
            dataset = ConcatDataset(datasets)
            dataset_len = len(dataset)
            dataset_lengths = torch.tensor(all_lengths)
        else:
            dataset = GeometricMolLMDBDataset(
                root=args.data_path,
                transform=transform,
                remove_hs=args.remove_hs,
                remove_aromaticity=args.remove_aromaticity,
                skip_non_valid=False,
            )
            dataset_len = len(dataset)
            dataset_lengths = torch.tensor(dataset.lengths)

        # Get data splits
        if args.data_path is not None:
            splits_path = os.path.join(args.data_path, "splits.npz")
            idx_train, idx_val, idx_test = make_splits(splits=splits_path)
        else:
            print("Creating random splits.")
            idx_train, idx_val, idx_test = util.make_splits(
                dataset_len=dataset_len,
                train_size=dataset_len - 600,
                val_size=500,
                test_size=100,
                splits=None,
            )
        print(f"train {len(idx_train)}, val {len(idx_val)}, test {len(idx_test)}")
        train_dataset = DatasetSubset(
            dataset, idx_train, lengths=dataset_lengths[idx_train].tolist()
        )
        val_dataset = DatasetSubset(
            dataset, idx_val, lengths=dataset_lengths[idx_val].tolist()
        )
        test_dataset = DatasetSubset(
            dataset, idx_test, lengths=dataset_lengths[idx_test].tolist()
        )

    # Load training molecules
    data_path = args.data_path if args.data_path is not None else args.data_paths
    try:
        train_mols, val_mols, test_mols = load_rdkit_mols(
            data_path,
            train_ids=idx_train,
            val_ids=idx_val,
            test_ids=idx_test,
        )
    except Exception as e:
        print(f"Could not load molecules with rdkit: {e}")
        train_mols, val_mols, test_mols = None, None, None

    # Build the interpolants
    type_mask_index = None
    bond_mask_index = None
    if args.categorical_strategy == "mask":
        type_mask_index = vocab.indices_from_tokens(["<MASK>"])[0]
        bond_mask_index = util.BOND_MASK_INDEX
        categorical_interpolation = "unmask"
        categorical_noise = "mask"
    elif args.categorical_strategy == "uniform-sample":
        categorical_interpolation = "unmask"
        categorical_noise = "uniform-sample"
    elif args.categorical_strategy == "prior-sample":
        categorical_interpolation = "unmask"
        categorical_noise = "prior-sample"
    elif args.categorical_strategy == "velocity-sample":
        categorical_interpolation = "sample"
        categorical_noise = "uniform-sample"
    else:
        raise ValueError(
            f"Interpolation '{args.categorical_strategy}' is not supported."
        )

    ## Create conformer generator if graph inpainting is set to conformer
    conformer_generator = (
        smolRD.ConformerGenerator(
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
        type_noise=categorical_noise,
        bond_noise=categorical_noise,
        zero_com=True,  # args.pocket_noise in ["fix", "random"],
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        atom_types_distribution=atom_types_distribution,
        bond_types_distribution=bond_types_distribution,
        conformer_generator=conformer_generator,
    )
    train_interpolant = GeometricInterpolant(
        prior_sampler,
        coord_interpolation=("linear" if not args.use_cosine_scheduler else "cosine"),
        coord_noise_std=args.coord_noise_std_dev,
        coord_noise_schedule=args.coord_noise_schedule,
        type_interpolation=categorical_interpolation,
        bond_interpolation=categorical_interpolation,
        split_continuous_discrete_time=args.split_continuous_discrete_time,
        type_dist_temp=args.type_dist_temp,
        time_alpha=args.time_alpha,
        time_beta=args.time_beta,
        fixed_time=None,
        scaffold_inpainting=args.scaffold_inpainting,
        func_group_inpainting=args.func_group_inpainting,
        substructure_inpainting=args.substructure_inpainting,
        substructure=args.substructure,
        linker_inpainting=args.linker_inpainting,
        fragment_inpainting=args.fragment_inpainting,
        graph_inpainting=args.graph_inpainting,
        core_inpainting=args.core_inpainting,
        max_fragment_cuts=args.max_fragment_cuts,
        mixed_uncond_inpaint=args.mixed_uncond_inpaint,
        mixed_uniform_beta_time=args.mixed_uniform_beta_time,
        vocab=vocab,
        vocab_charges=vocab_charges,
        vocab_hybridization=vocab_hybridization,
        vocab_aromatic=vocab_aromatic,
        rotation_alignment=args.rotation_alignment,
        permutation_alignment=args.permutation_alignment,
        sample_mol_sizes=False,
        inference=False,
    )
    eval_interpolant = GeometricInterpolant(
        prior_sampler,
        coord_interpolation=("linear" if not args.use_cosine_scheduler else "cosine"),
        type_interpolation=categorical_interpolation,
        bond_interpolation=categorical_interpolation,
        scaffold_inpainting=args.scaffold_inpainting,
        func_group_inpainting=args.func_group_inpainting,
        substructure_inpainting=args.substructure_inpainting,
        substructure=args.substructure,
        linker_inpainting=args.linker_inpainting,
        fragment_inpainting=args.fragment_inpainting,
        graph_inpainting=args.graph_inpainting,
        core_inpainting=args.core_inpainting,
        max_fragment_cuts=args.max_fragment_cuts,
        vocab=vocab,
        vocab_charges=vocab_charges,
        vocab_hybridization=vocab_hybridization,
        vocab_aromatic=vocab_aromatic,
        rotation_alignment=False,
        permutation_alignment=(
            args.linker_inpainting
            or args.fragment_inpainting
            or args.func_group_inpainting
            or args.substructure_inpainting
            or args.scaffold_inpainting
        )
        and args.permutation_alignment,
        fixed_time=None,
        sample_mol_sizes=getattr(args, "sample_mol_sizes", False),
        inference=True,
    )

    # Build datamodule
    dm = GeometricInterpolantDM(
        train_dataset,
        val_dataset,
        test_dataset,
        args.batch_cost,
        val_batch_size=args.val_batch_cost,
        train_interpolant=train_interpolant,
        val_interpolant=eval_interpolant,
        test_interpolant=eval_interpolant,
        use_bucket_sampler=args.use_bucket_sampler,
        use_adaptive_sampler=args.use_adaptive_sampler,
        use_weighted_sampler=args.use_weighted_sampler,
        bucket_limits=padded_sizes,
        bucket_cost_scale=args.bucket_cost_scale,
        pad_to_bucket=False,
        num_workers=args.num_workers,
        train_mols=train_mols,
        val_mols=val_mols,
        test_mols=test_mols,
    )
    return dm, coord_std

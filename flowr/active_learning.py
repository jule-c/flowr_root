"""
Active Learning LoRA Finetuning Module
=======================================
Streamlined LoRA finetuning pipeline for the active learning loop in the
FLOWR visualization UI. Takes user-selected generated ligands + protein
and runs a short LoRA finetuning to adapt the model to the preferred
chemical space.

This module is designed to be called from the GPU worker service and
handles the entire pipeline: data preparation → LMDB creation →
statistics computation → LoRA finetuning → checkpoint saving.
"""

import os
import pickle
import tempfile
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Callable, Dict, List, Optional

# Need to import pl at module level for the callback base class
import lightning as pl
import lmdb
import numpy as np
import torch
from rdkit import Chem

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def _compute_dynamic_epochs(n_ligands: int) -> int:
    """Compute number of finetuning epochs based on dataset size.

    Fewer ligands → fewer epochs to avoid overfitting.
    More ligands → more epochs for better adaptation.
    """
    if n_ligands <= 100:
        return 2
    elif n_ligands <= 300:
        return 3
    elif n_ligands <= 500:
        return 5
    elif n_ligands <= 1000:
        return 8
    else:
        return 10


def _compute_dynamic_acc_batches(n_ligands: int) -> int:
    """Scale gradient accumulation with dataset size."""
    if n_ligands <= 100:
        return 2
    elif n_ligands <= 300:
        return 3
    elif n_ligands <= 500:
        return 5
    elif n_ligands <= 1000:
        return 8
    else:
        return 12


def prepare_al_data(
    protein_pdb_path: str,
    ligand_sdf_strings: List[str],
    output_dir: str,
    pocket_cutoff: float = 7.0,
    remove_hs: bool = True,
    progress_callback: Optional[Callable[[int, str, str], None]] = None,
) -> Dict[str, str]:
    """Prepare LMDB dataset + splits + statistics from selected ligands.

    Args:
        protein_pdb_path: Path to the protein PDB file.
        ligand_sdf_strings: List of SDF-format strings for selected ligands.
        output_dir: Directory to write LMDB, splits, and statistics.
        pocket_cutoff: Distance cutoff for pocket extraction.
        remove_hs: Whether to remove hydrogens (must match base model).
        progress_callback: Optional callback(progress_pct, message, phase).

    Returns:
        Dict with keys: data_path, splits_path, statistics_path
    """
    from flowr.constants import ATOM_ENCODER as atom_encoder
    from flowr.data.preprocess_pdbs import process_complex
    from flowr.data.util import compute_all_statistics, mol_to_dict

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = output_dir / "lmdb"
    data_path.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(0, "Processing ligand-protein complexes...", "preparing")

    # Process each ligand with the protein into PocketComplex objects
    pocket_complexes = []
    rdkit_mols = []
    failed = 0

    for i, sdf_str in enumerate(ligand_sdf_strings):
        try:
            # Write SDF to temp file
            tmp_sdf = tempfile.NamedTemporaryFile(suffix=".sdf", delete=False, mode="w")
            tmp_sdf.write(sdf_str)
            tmp_sdf.close()

            complex_data = process_complex(
                pdb_path=protein_pdb_path,
                ligand_sdf_path=tmp_sdf.name,
                remove_hs=remove_hs,
                add_bonds_to_protein=True,
                pocket_cutoff=pocket_cutoff,
                cut_pocket=True,
                max_pocket_size=800,
                min_pocket_size=10,
                pocket_type="holo",
            )

            if complex_data is not None:
                pocket_complexes.append(complex_data)
                # Extract RDKit mol for statistics
                mol = Chem.SDMolSupplier(tmp_sdf.name, removeHs=remove_hs)
                if mol and mol[0] is not None:
                    rdkit_mols.append(mol[0].ToBinary())
                else:
                    rdkit_mols.append(None)
            else:
                failed += 1

            os.unlink(tmp_sdf.name)

        except Exception as e:
            failed += 1
            print(f"Failed to process ligand {i}: {e}")
            try:
                os.unlink(tmp_sdf.name)
            except Exception:
                pass

        if progress_callback and (i + 1) % max(1, len(ligand_sdf_strings) // 10) == 0:
            pct = int(60 * (i + 1) / len(ligand_sdf_strings))
            progress_callback(
                pct, f"Processed {i + 1}/{len(ligand_sdf_strings)} ligands", "preparing"
            )

    if not pocket_complexes:
        raise ValueError("No ligands could be processed successfully.")

    print(
        f"AL data prep: {len(pocket_complexes)} processed, {failed} failed "
        f"out of {len(ligand_sdf_strings)}"
    )

    if progress_callback:
        progress_callback(60, "Creating LMDB dataset...", "preparing")

    # Create LMDB database
    lmdb_env = lmdb.open(
        str(data_path),
        map_size=1 * 1024**3,  # 1 GB (plenty for ~100-1000 ligands)
        max_dbs=1,
        writemap=False,
        lock=True,
        readahead=False,
    )

    lengths_full = []
    lengths_no_pocket_hs = []
    system_ids = []

    with lmdb_env.begin(write=True) as txn:
        for idx, pc in enumerate(pocket_complexes):
            key = str(idx).encode("utf-8")
            txn.put(key, pc.to_bytes())

            # Compute lengths
            lengths_full.append(pc.seq_length)
            pc_no_hs = pc.remove_hs(include_ligand=False)
            lengths_no_pocket_hs.append(pc_no_hs.seq_length)
            system_ids.append(f"al_ligand_{idx}")

        # Store metadata
        txn.put(b"__len__", pickle.dumps(len(pocket_complexes)))
        txn.put(b"lengths_full", pickle.dumps(lengths_full))
        txn.put(b"lengths_no_pocket_hs", pickle.dumps(lengths_no_pocket_hs))
        txn.put(b"lengths_no_ligand_pocket_hs", pickle.dumps(lengths_no_pocket_hs))
        txn.put(b"system_ids", pickle.dumps(system_ids))

    lmdb_env.close()

    # Save rdkit_mols.pkl
    with open(data_path / "rdkit_mols.pkl", "wb") as f:
        pickle.dump(rdkit_mols, f)

    if progress_callback:
        progress_callback(70, "Creating data splits...", "preparing")

    # Create splits: all train, empty val/test
    n = len(pocket_complexes)
    idx_train = np.arange(n, dtype=np.int64)
    idx_val = np.array([], dtype=np.int64)
    idx_test = np.array([], dtype=np.int64)

    splits_path = data_path / "splits.npz"
    np.savez(splits_path, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

    if progress_callback:
        progress_callback(75, "Computing dataset statistics...", "preparing")

    # Compute statistics from training data
    data_list = []
    for mol_bytes in rdkit_mols:
        if mol_bytes is not None:
            mol = Chem.Mol(mol_bytes)
            try:
                data_list.append(
                    mol_to_dict(mol, atom_encoder=atom_encoder, remove_hs=remove_hs)
                )
            except Exception:
                pass

    if not data_list:
        raise ValueError("Could not compute statistics from processed ligands.")

    statistics = compute_all_statistics(
        data_list,
        atom_encoder,
        charges_dic={-2: 0, -1: 1, 0: 2, 1: 3, 2: 4, 3: 5},
        additional_feats=True,
    )

    # Save statistics files
    h = "noh" if remove_hs else "h"
    processed_dir = data_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Save for train split (used by build_data_statistic)
    _save_statistics(statistics, processed_dir, "train", h)
    # Also save as val and test (even though they're empty, the loader expects them)
    _save_statistics(statistics, processed_dir, "val", h)
    _save_statistics(statistics, processed_dir, "test", h)

    if progress_callback:
        progress_callback(80, "Data preparation complete.", "preparing")

    return {
        "data_path": str(data_path),
        "splits_path": str(splits_path),
        "statistics_path": str(processed_dir),
    }


def _save_statistics(statistics, processed_dir: Path, split: str, h: str):
    """Save statistics files for a given split."""
    processed_dir = Path(processed_dir)

    with open(processed_dir / f"{split}_n_{h}.pickle", "wb") as f:
        pickle.dump(statistics.num_nodes, f)
    np.save(processed_dir / f"{split}_atom_types_{h}.npy", statistics.atom_types)
    np.save(processed_dir / f"{split}_bond_types_{h}.npy", statistics.bond_types)
    np.save(processed_dir / f"{split}_charges_{h}.npy", statistics.charge_types)
    with open(processed_dir / f"{split}_valency_{h}.pickle", "wb") as f:
        pickle.dump(statistics.valencies, f)
    with open(processed_dir / f"{split}_bond_lengths_{h}.pickle", "wb") as f:
        pickle.dump(statistics.bond_lengths, f)
    np.save(processed_dir / f"{split}_angles_{h}.npy", statistics.bond_angles)
    np.save(processed_dir / f"{split}_is_aromatic_{h}.npy", statistics.is_aromatic)
    np.save(processed_dir / f"{split}_is_in_ring_{h}.npy", statistics.is_in_ring)
    np.save(processed_dir / f"{split}_hybridization_{h}.npy", statistics.hybridization)
    np.save(processed_dir / f"{split}_is_h_donor_{h}.npy", statistics.is_h_donor)
    np.save(processed_dir / f"{split}_is_h_acceptor_{h}.npy", statistics.is_h_acceptor)
    np.save(processed_dir / f"{split}_dihedrals_{h}.npy", statistics.dihedrals)


def run_al_finetuning(
    ckpt_path: str,
    data_path: str,
    splits_path: str,
    statistics_path: str,
    save_dir: str,
    n_ligands: int,
    epochs: Optional[int] = None,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lr: float = 5e-4,
    batch_cost: int = 4,
    acc_batches: Optional[int] = None,
    seed: int = 42,
    is_continuation: bool = False,
    progress_callback: Optional[Callable[[int, str, str], None]] = None,
) -> str:
    """Run LoRA finetuning on the prepared active learning dataset.

    Args:
        ckpt_path: Path to the model checkpoint (base or previously finetuned).
        data_path: Path to the LMDB dataset.
        splits_path: Path to splits.npz file.
        statistics_path: Path to folder with statistics files.
        save_dir: Directory to save the finetuned checkpoint.
        n_ligands: Number of ligands in the dataset (for dynamic epochs).
        epochs: Override number of epochs (if None, computed dynamically).
        lora_rank: LoRA rank parameter.
        lora_alpha: LoRA alpha parameter.
        lr: Learning rate.
        batch_cost: Batch cost for dataloader.
        acc_batches: Gradient accumulation batches.
        seed: Random seed.
        is_continuation: True if loading a previously finetuned LoRA checkpoint.
        progress_callback: Optional callback(progress_pct, message, phase).

    Returns:
        Path to the finetuned checkpoint file.
    """
    import lightning as pl
    from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar

    import flowr.scriptutil as util
    from flowr.callbacks import EMA, EMAModelCheckpoint
    from flowr.data.data_info import GeneralInfos as DataInfos

    if epochs is None:
        epochs = _compute_dynamic_epochs(n_ligands)
    if acc_batches is None:
        acc_batches = _compute_dynamic_acc_batches(n_ligands)

    print(f"AL finetuning: {n_ligands} ligands, {epochs} epochs, lr={lr}")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        _msg = (
            "Loading previous LoRA model..."
            if is_continuation
            else "Loading base model..."
        )
        progress_callback(85, _msg, "preparing")

    # Build args namespace matching finetune.py expectations
    args = _build_al_args(
        ckpt_path=ckpt_path,
        data_path=data_path,
        splits_path=splits_path,
        statistics_path=statistics_path,
        save_dir=str(save_dir),
        epochs=epochs,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lr=lr,
        batch_cost=batch_cost,
        acc_batches=acc_batches,
        seed=seed,
    )

    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.cache_size_limit = util.COMPILER_CACHE_SIZE
    util.disable_lib_stdout()
    util.configure_fs()

    # Load model with LoRA injection
    (
        model,
        hparams,
        vocab,
        vocab_charges,
        vocab_hybridization,
        vocab_aromatic,
        vocab_pocket_atoms,
        vocab_pocket_res,
    ) = util.load_model(args)

    if progress_callback:
        progress_callback(95, "Loading dataset...", "preparing")

    args.remove_hs = hparams["remove_hs"]
    statistics = util.build_data_statistic(args)
    dataset_info = DataInfos(statistics, vocab, hparams)
    atom_types_distribution = dataset_info.atom_types.float()
    bond_types_distribution = dataset_info.edge_types.float()

    # Load datamodule with custom handling for empty val/test
    dm = _load_al_dm(
        args,
        hparams,
        vocab,
        vocab_charges,
        atom_types_distribution=atom_types_distribution,
        bond_types_distribution=bond_types_distribution,
    )

    if progress_callback:
        progress_callback(100, "Preparation complete.", "preparing")
        progress_callback(
            0, f"Starting LoRA finetuning ({epochs} epochs)...", "training"
        )

    # Build a simplified trainer (no validation, no wandb, just train + save)
    trainer = _build_al_trainer(args, model, progress_callback)

    try:
        trainer.fit(model, datamodule=dm)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA out of memory during finetuning.\n"
                "Please reduce the batch size."
            ) from e
        raise

    if progress_callback:
        progress_callback(95, "Saving finetuned checkpoint...", "training")

    # Find the saved checkpoint
    # Prefer EMA-smoothed weights when available (better quality)
    ckpt_file = save_dir / "last-EMA.ckpt"
    if not ckpt_file.exists():
        ckpt_file = save_dir / "last.ckpt"
    if not ckpt_file.exists():
        # Try to find any checkpoint
        ckpts = list(save_dir.glob("*.ckpt"))
        if ckpts:
            ckpt_file = ckpts[0]
        else:
            raise RuntimeError("No checkpoint was saved during finetuning.")

    if progress_callback:
        progress_callback(100, "Finetuning complete.", "training")

    print(f"AL finetuning complete. Checkpoint saved to {ckpt_file}")
    return str(ckpt_file)


def _build_al_args(
    ckpt_path: str,
    data_path: str,
    splits_path: str,
    statistics_path: str,
    save_dir: str,
    epochs: int,
    lora_rank: int,
    lora_alpha: int,
    lr: float,
    batch_cost: int,
    acc_batches: int,
    seed: int,
) -> Namespace:
    """Build a Namespace with all args needed for LoRA finetuning."""
    return Namespace(
        # Setup
        arch="pocket",
        exp_name="active_learning",
        run_name="al_lora",
        seed=seed,
        gpus=1,
        num_nodes=1,
        num_workers=2,
        load_ckpt=None,
        ckpt_path=ckpt_path,
        load_pretrained_ckpt=True,
        save_dir=save_dir,
        val_check_epochs=epochs + 1,  # Never validate (will exceed max_epochs)
        val_check_interval=None,
        wandb=False,
        trial_run=False,
        # Finetuning
        lora_finetuning=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        freeze_layers=False,
        affinity_finetuning=None,
        n_top_layers_to_retrain=3,
        n_top_layers_to_retrain_pocket=2,
        # Data
        data_path=data_path,
        data_paths=None,
        splits_path=splits_path,
        splits_paths=None,
        statistics_path=statistics_path,
        dataset_weights=None,
        sample_n_molecules_val=None,
        dataset="active_learning",
        use_smol=False,
        batch_cost=batch_cost,
        val_batch_cost=batch_cost,
        use_bucket_sampler=False,
        bucket_cost_scale="linear",
        use_adaptive_sampler=False,
        use_weighted_sampler=False,
        acc_batches=acc_batches,
        # Training
        train_confidence=False,
        confidence_loss_weight=0.0,
        confidence_gen_steps=20,
        epochs=epochs,
        lr=lr,
        weight_decay=1e-4,
        beta1=0.9,
        beta2=0.95,
        ligand_lr=None,
        pocket_lr=None,
        gradient_clip_val=1.0,
        coord_loss_weight=3.0,
        type_loss_weight=1.0,
        bond_loss_weight=3.0,
        charge_loss_weight=2.0,
        hybridization_loss_weight=1.0,
        distance_loss_weight_lig=0.0,
        affinity_loss_weight=0.0,
        docking_loss_weight=0.0,
        plddt_confidence_loss_weight=0.0,
        distance_loss_weight_lig_pocket=0.0,
        smooth_distance_loss_weight_lig=0.0,
        smooth_distance_loss_weight_lig_pocket=0.0,
        interaction_loss_weight=10.0,
        bond_length_loss_weight=0.0,
        bond_angle_loss_weight=0.0,
        bond_angle_huber_delta=0.0,
        energy_loss_weight=0.0,
        energy_loss_weighting="constant",
        energy_loss_decay_rate=1.0,
        use_t_loss_weights=False,
        lr_schedule="constant",
        lr_gamma=0.995,
        cosine_decay_fraction=1.0,
        warm_up_steps=0,
        use_ema=True,
        ema_decay=0.998,
        # Interpolation
        pocket_noise="fix",
        pocket_type="holo",
        separate_pocket_interpolation=False,
        separate_interaction_interpolation=False,
        integration_steps=100,
        interaction_fixed_time=None,
        flow_interactions=False,
        predict_interactions=False,
        predict_affinity=False,
        predict_docking_score=False,
        interaction_conditional=False,
        scaffold_hopping=True,
        graph_inpainting=None,
        mixed_uncond_inpaint=True,
        scaffold_elaboration=True,
        fragment_inpainting=True,
        fragment_growing=True,
        max_fragment_cuts=3,
        substructure_inpainting=False,
        substructure=None,
        linker_inpainting=False,
        core_growing=False,
        use_cosine_scheduler=False,
        categorical_strategy="uniform-sample",
        ode_sampling_strategy="linear",
        split_continuous_discrete_time=False,
        # Sampling
        n_validation_mols=0,
        num_inference_steps=100,
        cat_sampling_noise_level=1.0,
        coord_noise_std_dev=0.2,
        coord_noise_schedule="constant_decay",
        coord_noise_scale=0.0,
        coord_sampling_strategy="continuous",
        use_sde_simulation=False,
        sample_schedule="linear",
        pocket_coord_noise_std=0.0,
        type_dist_temp=1.0,
        time_alpha=1.8,
        time_beta=1.0,
        mixed_uniform_beta_time=False,
        rotation_alignment=False,
        permutation_alignment=True,
        anisotropic_prior=False,
        corrector_iters=0,
    )


def _load_al_dm(
    args,
    hparams,
    vocab,
    vocab_charges,
    atom_types_distribution=None,
    bond_types_distribution=None,
):
    """Load datamodule for active learning with empty val/test handling."""
    from functools import partial

    from flowr.data.datamodules import GeometricInterpolantDM
    from flowr.data.dataset import DatasetSubset, PocketComplexLMDBDataset
    from flowr.data.interpolate import (
        ComplexInterpolant,
        GeometricNoiseSampler,
    )
    from flowr.scriptutil import (
        BOND_MASK_INDEX,
        complex_transform,
        get_n_bond_types,
        make_splits,
    )
    from flowr.util.pocket import PROLIF_INTERACTIONS

    coord_std = hparams["coord_scale"]
    n_bond_types = get_n_bond_types(args.categorical_strategy)
    n_charge_types = vocab_charges.size

    transform = partial(
        complex_transform,
        vocab=vocab,
        vocab_charges=vocab_charges,
        vocab_hybridization=None,
        vocab_aromatic=None,
        n_bonds=n_bond_types,
        coord_std=coord_std,
        pocket_noise=args.pocket_noise,
        pocket_noise_std=args.pocket_coord_noise_std,
        use_interactions=False,
    )

    prior_sampler = GeometricNoiseSampler(
        vocab.size,
        n_bond_types,
        n_charge_types,
        n_hybridization_types=None,
        n_aromatic_types=None,
        coord_noise="gaussian",
        type_noise=hparams["val-ligand-prior-type-noise"],
        bond_noise=hparams["val-ligand-prior-bond-noise"],
        zero_com=True,
        type_mask_index=None,
        bond_mask_index=None,
        conformer_generator=None,
        atom_types_distribution=atom_types_distribution,
        bond_types_distribution=bond_types_distribution,
    )

    # Build train interpolant
    train_interpolant = ComplexInterpolant(
        prior_sampler,
        ligand_coord_interpolation="linear",
        ligand_coord_noise_std=args.coord_noise_std_dev,
        ligand_coord_noise_schedule=args.coord_noise_schedule,
        ligand_type_interpolation="unmask",
        ligand_bond_interpolation="unmask",
        ligand_time_alpha=args.time_alpha,
        ligand_time_beta=args.time_beta,
        ligand_fixed_time=None,
        split_continuous_discrete_time=False,
        pocket_time_alpha=args.time_alpha,
        pocket_time_beta=args.time_beta,
        pocket_fixed_time=None,
        pocket_coord_noise_std=args.pocket_coord_noise_std,
        pocket_noise=args.pocket_noise,
        separate_pocket_interpolation=False,
        separate_interaction_interpolation=False,
        interaction_fixed_time=None,
        interaction_time_alpha=args.time_alpha,
        interaction_time_beta=args.time_beta,
        interaction_conditional=False,
        scaffold_hopping=args.scaffold_hopping,
        scaffold_elaboration=args.scaffold_elaboration,
        substructure_inpainting=False,
        substructure=None,
        linker_inpainting=False,
        core_growing=False,
        fragment_inpainting=args.fragment_inpainting,
        fragment_growing=args.fragment_growing,
        max_fragment_cuts=args.max_fragment_cuts,
        graph_inpainting=None,
        mixed_uncond_inpaint=args.mixed_uncond_inpaint,
        mixed_uniform_beta_time=False,
        n_interaction_types=None,
        flow_interactions=False,
        dataset=args.dataset,
        vocab=vocab,
        vocab_charges=vocab_charges,
        vocab_hybridization=None,
        rotation_alignment=False,
        permutation_alignment=args.permutation_alignment,
        anisotropic_prior=False,
        ref_ligand_com_prior=False,
        ref_ligand_com_noise_std=1.0,
        sample_mol_sizes=False,
        inference=False,
    )

    # Eval interpolant (needed by DM even if not used)
    eval_interpolant = ComplexInterpolant(
        prior_sampler,
        ligand_coord_interpolation="linear",
        ligand_type_interpolation="unmask",
        ligand_bond_interpolation="unmask",
        pocket_noise=args.pocket_noise,
        separate_pocket_interpolation=False,
        separate_interaction_interpolation=False,
        n_interaction_types=None,
        flow_interactions=False,
        interaction_conditional=False,
        scaffold_hopping=args.scaffold_hopping,
        scaffold_elaboration=args.scaffold_elaboration,
        linker_inpainting=False,
        core_growing=False,
        fragment_inpainting=args.fragment_inpainting,
        fragment_growing=args.fragment_growing,
        max_fragment_cuts=args.max_fragment_cuts,
        substructure_inpainting=False,
        substructure=None,
        graph_inpainting=None,
        dataset=args.dataset,
        sample_mol_sizes=False,
        vocab=vocab,
        vocab_charges=None,
        vocab_hybridization=None,
        batch_ot=False,
        rotation_alignment=False,
        permutation_alignment=args.permutation_alignment,
        anisotropic_prior=False,
        ref_ligand_com_prior=False,
        ref_ligand_com_noise_std=1.0,
        inference=True,
    )

    # Load dataset
    dataset = PocketComplexLMDBDataset(
        root=args.data_path,
        transform=transform,
        remove_hs=hparams["remove_hs"],
        remove_aromaticity=hparams.get("remove_aromaticity", False),
        skip_non_valid=False,
        read_only=True,
    )
    dataset_lengths = torch.tensor(dataset.lengths)

    # Load splits
    idx_train, idx_val, idx_test = make_splits(splits=args.splits_path)

    print(f"AL DM: train {len(idx_train)}, val {len(idx_val)}, test {len(idx_test)}")

    # Create subsets - val/test are empty but still valid DatasetSubset
    train_dataset = DatasetSubset(
        dataset, idx_train, lengths=dataset_lengths[idx_train].tolist()
    )
    val_dataset = DatasetSubset(dataset, idx_val, lengths=[])
    test_dataset = DatasetSubset(dataset, idx_test, lengths=[])

    # Load RDKit mols for training (needed by the DM)
    rdkit_mols_path = Path(args.data_path) / "rdkit_mols.pkl"
    if rdkit_mols_path.exists():
        with open(rdkit_mols_path, "rb") as f:
            rdkit_mols = pickle.load(f)
        train_mols = []
        for i in idx_train:
            mol_bytes = rdkit_mols[int(i)]
            if mol_bytes is not None:
                train_mols.append(Chem.Mol(mol_bytes))
            else:
                train_mols.append(None)
    else:
        train_mols = None

    dm = GeometricInterpolantDM(
        train_dataset,
        val_dataset,
        test_dataset,
        args.batch_cost,
        val_batch_size=args.val_batch_cost,
        train_interpolant=train_interpolant,
        val_interpolant=eval_interpolant,
        test_interpolant=eval_interpolant,
        use_bucket_sampler=False,
        use_adaptive_sampler=False,
        use_weighted_sampler=False,
        bucket_limits=None,
        bucket_cost_scale="linear",
        pad_to_bucket=False,
        num_workers=0,  # Small dataset; avoids persistent-worker issues
        train_mols=train_mols,
        val_mols=[],
        test_mols=[],
    )

    # Prevent Lightning from creating a CombinedLoader for the empty val set.
    # Even with limit_val_batches=0, some Lightning versions still initialise
    # the val dataloader and crash on an empty dataset.
    dm.val_dataloader = lambda *_a, **_kw: []

    return dm


def _build_al_trainer(args, model, progress_callback=None):
    """Build a Lightning trainer for AL finetuning.

    Mirrors scriptutil.build_trainer / finetune.py but uses strategy="auto"
    so it works on both CUDA (single-GPU) and MPS without forcing DDP.
    """
    import lightning as pl
    from lightning.pytorch.callbacks import (
        LearningRateMonitor,
        ModelCheckpoint,
        ModelSummary,
        TQDMProgressBar,
    )
    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch.plugins.environments import LightningEnvironment

    import flowr.scriptutil as util
    from flowr.callbacks import EMA, EMAModelCheckpoint

    precision = util.get_precision(args)
    epochs = 1 if args.trial_run else args.epochs

    # Callbacks – same set as build_trainer
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=5),
        ModelSummary(max_depth=2),
    ]

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
            save_top_k=0,
            monitor=None,
            save_last=True,
        )
        callbacks.append(ema_callback)
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.save_dir,
            save_top_k=0,
            monitor=None,
            save_last=True,
        )
    callbacks.append(checkpoint_callback)

    if progress_callback:
        callbacks.append(_ALProgressCallback(progress_callback, epochs))

    logger = CSVLogger(save_dir=args.save_dir, name="al_logs")

    trainer = pl.Trainer(
        accelerator="gpu" if args.gpus else "cpu",
        devices=args.gpus if args.gpus else 1,
        strategy="auto",  # Works on both CUDA and MPS (no DDP for single-GPU)
        plugins=LightningEnvironment(),
        num_nodes=1,
        enable_checkpointing=True,
        accumulate_grad_batches=args.acc_batches,
        val_check_interval=None,
        check_val_every_n_epoch=args.val_check_epochs,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        logger=logger,
        precision=precision,
        max_epochs=epochs,
        limit_val_batches=0,
        use_distributed_sampler=True,
    )

    pl.seed_everything(seed=args.seed, workers=False)
    return trainer


class _ALProgressCallback(pl.Callback):
    """Lightning callback to report training progress back to the worker."""

    def __init__(self, progress_callback, total_epochs):
        super().__init__()
        self.progress_callback = progress_callback
        self.total_epochs = total_epochs
        self._total_batches = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._total_batches is None:
            self._total_batches = trainer.num_training_batches
        total_batches = self._total_batches or 1
        epoch = trainer.current_epoch
        total_steps = self.total_epochs * total_batches
        current_step = epoch * total_batches + (batch_idx + 1)
        pct = int(current_step / total_steps * 95)
        self.progress_callback(
            min(pct, 95),
            f"Training epoch {epoch + 1}/{self.total_epochs}",
            "training",
        )

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        pct = int(epoch / self.total_epochs * 95)
        self.progress_callback(
            min(pct, 95),
            f"Completed epoch {epoch}/{self.total_epochs}",
            "training",
        )

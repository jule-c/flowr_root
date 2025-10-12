import os
import tempfile
import warnings
from argparse import Namespace
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, Optional

import lightning as L
import numpy as np
import torch
from pymol import cmd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

import flowr.scriptutil as util
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
from flowr.data.preprocess_pdbs import process_complex
from flowr.util.functional import (
    add_and_optimize_hs,
)
from flowr.util.metrics import interaction_recovery_per_complex
from flowr.util.pocket import PROLIF_INTERACTIONS
from flowr.util.rdkit import ConformerGenerator, write_sdf_file

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
warnings.filterwarnings("ignore", category=DeprecationWarning)


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
    )
    # Initialize conformer generator if graph inpainting is enabled and set to conformer
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
    return transform, eval_interpolant


def load_util_mol(
    args,
    hparams,
    vocab,
    vocab_charges,
    vocab_hybridization=None,
    vocab_aromatic=None,
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
        util.mol_transform,
        vocab=vocab,
        vocab_charges=vocab_charges,
        vocab_hybridization=vocab_hybridization,
        vocab_aromatic=vocab_aromatic,
        n_bonds=n_bond_types,
        coord_std=coord_std,
        zero_com=True,
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
        type_noise=hparams["val-prior-type-noise"],
        bond_noise=hparams["val-prior-bond-noise"],
        zero_com=True,  # args.pocket_noise in ["fix", "random"],
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
    )

    ## Determine the categorical sampling strategy
    if args.categorical_strategy == "mask":
        assert hparams["val-type-interpolation"] == "unmask"
        type_mask_index = vocab.indices_from_tokens(["<MASK>"])[0]
        bond_mask_index = util.BOND_MASK_INDEX
        categorical_interpolation = "unmask"
    elif args.categorical_strategy == "uniform-sample":
        assert hparams["val-type-interpolation"] == "unmask"
        categorical_interpolation = "unmask"
    elif args.categorical_strategy == "prior-sample":
        assert hparams["val-type-interpolation"] == "unmask"
        categorical_interpolation = "unmask"
    elif args.categorical_strategy == "velocity-sample":
        assert hparams["val-type-interpolation"] == "sample"
        categorical_interpolation = "sample"
    else:
        raise ValueError(
            f"Interpolation '{args.categorical_strategy}' is not supported."
        )

    eval_interpolant = GeometricInterpolant(
        prior_sampler,
        coord_interpolation=("linear" if not args.use_cosine_scheduler else "cosine"),
        type_interpolation=categorical_interpolation,
        bond_interpolation=categorical_interpolation,
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
    return transform, eval_interpolant


def load_data_from_lmdb(
    args: Namespace,
    remove_hs: bool,
    remove_aromaticity: bool,
    transform: Optional[Callable] = None,
    sample: bool = False,
    sample_n_molecules: int = None,
):
    """Load the dataset from an LMDB file.
    Args:
        args: Command line arguments containing the data path and other parameters.
        transform: Optional transformation function to apply to the dataset.
    Returns:
        A list of systems (ligand-pocket complexes) from the dataset.
    """
    dataset = PocketComplexLMDBDataset(
        root=args.data_path,
        transform=transform,
        remove_hs=remove_hs,
        remove_aromaticity=remove_aromaticity,
        skip_non_valid=False,
    )
    dataset_lengths = torch.tensor(dataset.lengths)

    # Split the data - if "all", all data will be used, no train/val/test split
    if args.dataset_split in ["train", "val", "test"]:
        idx_train, idx_val, idx_test = util.make_splits(
            splits=os.path.join(args.data_path, "splits.npz"),
        )
        idx = (
            idx_train
            if args.dataset_split == "train"
            else idx_val if args.dataset_split == "val" else idx_test
        )
        dataset = DatasetSubset(
            dataset, idx, lengths=dataset_lengths[idx_test].tolist()
        )
    if sample:
        dataset = dataset.sample_n_molecules(sample_n_molecules, seed=args.seed)
    print(
        f"Dataset split is set to {args.dataset_split}. Number of systems: {len(dataset)}"
    )

    # Split the dataset for multi-processing via job arrays
    if hasattr(args, "mp_index"):
        systems = [system for system in dataset if system is not None]
        systems = split_list(systems, args.gpus)[args.mp_index - 1]
        return systems

    return dataset


def load_data_from_pdb(args, remove_hs: bool, remove_aromaticity: bool):
    # Load the data
    processing_params = {
        "add_hs": args.add_hs,
        "add_hs_and_optimize": args.add_hs_and_optimize,
        "kekulize": args.kekulize,
        "use_pdbfixer": args.use_pdbfixer,
        "add_bonds_to_protein": True,
        "add_hs_to_protein": args.protonate_pocket and not args.use_pdbfixer,
        "pocket_cutoff": args.pocket_cutoff,
        "cut_pocket": args.cut_pocket,
        "max_pocket_size": args.max_pocket_size,
        "min_pocket_size": args.min_pocket_size,
        "compute_interactions": args.compute_interactions,
        "pocket_type": args.pocket_type,
    }
    system = process_complex(
        pdb_id=args.pdb_id,
        ligand_id=args.ligand_id,
        pdb_path=args.pdb_file,
        ligand_sdf_path=args.ligand_file,
        **processing_params,
    )
    system = system.remove_hs(include_ligand=remove_hs)
    return system


def load_data_from_pdb_selective(args, remove_hs: bool, remove_aromaticity: bool):
    # Load the data
    processing_params = {
        "add_hs": args.add_hs,
        "add_hs_and_optimize": args.add_hs_and_optimize,
        "kekulize": args.kekulize,
        "use_pdbfixer": args.use_pdbfixer,
        "add_bonds_to_protein": True,
        "add_hs_to_protein": args.protonate_pocket and not args.use_pdbfixer,
        "pocket_cutoff": args.pocket_cutoff,
        "cut_pocket": args.cut_pocket,
        "max_pocket_size": args.max_pocket_size,
        "min_pocket_size": args.min_pocket_size,
        "compute_interactions": args.compute_interactions,
        "pocket_type": args.pocket_type,
    }
    system_target = process_complex(
        pdb_id=args.pdb_id,
        ligand_id=args.ligand_id,
        pdb_path=args.pdb_file_target,
        ligand_sdf_path=args.ligand_file_target,
        **processing_params,
    )

    system_untarget = process_complex(
        pdb_id=args.pdb_id,
        ligand_id=args.ligand_id,
        pdb_path=args.pdb_file_untarget,
        ligand_sdf_path=args.ligand_file_untarget,
        **processing_params,
    )
    system_target = system_target.remove_hs(include_ligand=remove_hs)
    system_untarget = system_untarget.remove_hs(include_ligand=remove_hs)
    return system_target, system_untarget


def load_data_from_lmdb_mol(
    args: Namespace,
    remove_hs: bool,
    remove_aromaticity: bool,
    transform: Optional[Callable] = None,
    sample: bool = False,
    sample_n_molecules: int = None,
):
    """Load the dataset from an LMDB file.
    Args:
        args: Command line arguments containing the data path and other parameters.
        transform: Optional transformation function to apply to the dataset.
    Returns:
        A list of molecules from the dataset.
    """
    dataset = GeometricMolLMDBDataset(
        root=args.data_path,
        transform=transform,
        remove_hs=remove_hs,
        remove_aromaticity=remove_aromaticity,
        skip_non_valid=False,
    )
    dataset_lengths = torch.tensor(dataset.lengths)
    idx_train, idx_val, idx_test = util.make_splits(
        splits=os.path.join(args.data_path, "splits.npz"),
    )
    idx = (
        idx_train
        if args.dataset_split == "train"
        else idx_val if args.dataset_split == "val" else idx_test
    )
    dataset = DatasetSubset(dataset, idx, lengths=dataset_lengths[idx_test].tolist())
    if sample:
        dataset = dataset.sample_n_molecules(sample_n_molecules, seed=args.seed)
    print(
        f"Dataset split is set to {args.dataset_split}. Number of systems: {len(dataset)}"
    )
    molecules = [molecule for molecule in dataset if molecule is not None]
    molecules = split_list(molecules, args.gpus)[args.mp_index - 1]
    return molecules


def get_dataloader(
    args: Namespace,
    dataset: GeometricDataset,
    interpolant: ComplexInterpolant,
    sample: bool = False,
    sample_n_molecules: int = None,
    iter: int = 0,
):
    """
    Get the data loader for the given dataset and interpolant.
    Sets the random seed for reproducibility and accounting for different workers.
    Args:
        args: Command line arguments.
        dataset: The dataset to load.
        interpolant: The interpolant to use.
        iter: The current iteration (default: 0).
    Returns:
        The data loader for the given dataset and interpolant.
    """

    L.seed_everything(args.seed + iter)
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
    test_dl = dm.test_dataloader(subset=sample_n_molecules if sample else None)
    return test_dl


def write_ligand_pocket_complex_pdb(
    all_gen_ligs, all_gen_pdbs, output_path, complex_name="complex"
):
    """
    Combines ligand and pocket (PDB file) into a ligand-pocket complex and writes them to an SDF file.

    Args:
        all_gen_ligs (list[Chem.Mol]): List of RDKit ligand molecules.
        all_gen_pdbs (list[str]): List of file paths (or paths as strings) for the pocket PDBs.
        output_path (str): File path to write the combined complexes (PDB format).
        complex_name (str): Name prefix for each complex. Default is "Complex".

    Notes:
        This function assumes that the PDB files can be parsed by RDKit via Chem.MolFromPDBFile.
    """
    if not all_gen_ligs:
        raise ValueError("No ligand molecules provided.")
    if not all_gen_pdbs:
        raise ValueError("No pocket PDB files provided.")

    for i, (lig, pdb_file) in enumerate(zip(all_gen_ligs, all_gen_pdbs)):
        out_path = Path(output_path) / f"{complex_name}_{i}.pdb"
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            lig_sdf = tmp.name
            write_sdf_file(lig_sdf, [lig])
            cmd.reinitialize()
            cmd.load(pdb_file, "protein")
            cmd.load(lig_sdf, "ligand")
            cmd.save(out_path, "protein or ligand")
            cmd.reinitialize()


def write_ligand_pocket_complex_sdf(
    all_gen_ligs, all_gen_pdbs, output_path, complex_name="complex"
):
    """
    Combines ligand and pocket (PDB file) into a ligand-pocket complex and writes them to an SDF file.

    Args:
        all_gen_ligs (list[Chem.Mol]): List of RDKit ligand molecules.
        all_gen_pdbs (list[str]): List of file paths (or paths as strings) for the pocket PDBs.
        output_path (str): File path to write the combined complexes (SDF format).
        complex_name (str): Name prefix for each complex. Default is "Complex".

    Notes:
        This function assumes that the PDB files can be parsed by RDKit via Chem.MolFromPDBFile.
    """
    if not all_gen_ligs:
        raise ValueError("No ligand molecules provided.")
    if not all_gen_pdbs:
        raise ValueError("No pocket PDB files provided.")

    complexes = []
    for i, (lig, pdb_file) in enumerate(zip(all_gen_ligs, all_gen_pdbs)):

        # Parse the pocket molecule from the PDB file.
        pocket_mol = Chem.MolFromPDBFile(pdb_file, removeHs=False)
        if pocket_mol is None:
            print(
                f"Warning: Could not parse pocket from {pdb_file}; skipping entry {i}."
            )
            continue

        # Combine the pocket and ligand.
        # Note: CombineMols doesn't merge conformers. The output will have separate conformers.
        complex_mol = Chem.CombineMols(pocket_mol, lig)
        complex_mol.SetProp("Name", f"{complex_name}_{i}")
        complexes.append(complex_mol)

    if not complexes:
        raise RuntimeError("No valid complexes were created.")

    # Write complexes to SDF
    sdf_path = Path(output_path) / f"{complex_name}.sdf"
    writer = Chem.SDWriter(sdf_path)
    for mol in complexes:
        writer.write(mol)
    writer.close()
    print(f"Wrote {len(complexes)} ligand-pocket complexes to {output_path}")


def process_lig_wrapper(
    pair, optimizer, optimize_pocket_hs=False, process_pocket=False
):
    """
    Wrapper for process_lig to enable pickling.

    Args:
        pair (tuple): A tuple containing (lig, pdb_file).
        optimizer: The optimizer to pass to process_lig.

    Returns:
        The processed ligand.
    """
    lig, pdb_file = pair
    return process_lig(
        lig,
        pdb_file=pdb_file,
        optimizer=optimizer,
        optimize_pocket_hs=optimize_pocket_hs,
        process_pocket=process_pocket,
    )


def process_lig(
    lig, pdb_file, optimizer, optimize_pocket_hs=False, process_pocket=False
):
    """
    Add hydrogens to a ligand and optimize it.
    """
    return add_and_optimize_hs(
        lig,
        pdb_file,
        optimizer=optimizer,
        optimize_pocket_hs=optimize_pocket_hs,
        process_pocket=process_pocket,
    )


def process_interaction(
    gen_lig,
    ref_lig,
    pdb_file,
    pocket_cutoff,
    save_dir,
    remove_hs,
    add_hs_and_optimize_gen_ligs,
):
    recovery_rate, tanimoto_sim = interaction_recovery_per_complex(
        gen_ligs=gen_lig,
        native_lig=ref_lig,
        pdb_file=pdb_file,
        add_optimize_gen_lig_hs=remove_hs and not add_hs_and_optimize_gen_ligs,
        add_optimize_ref_lig_hs=False,
        optimize_pocket_hs=False,
        process_pocket=False,
        optimization_method="prolif_mmff",
        pocket_cutoff=pocket_cutoff,
        strip_invalid=True,
        save_dir=save_dir,
        return_list=False,
    )
    return {
        "recovery_rate": recovery_rate,
        "tanimoto_sim": tanimoto_sim,
    }


def get_fingerprints(
    mols: Iterable[Chem.Mol], radius=2, length=4096, chiral=True, sanitize=False
):
    """
    Converts molecules to ECFP bitvectors.

    Args:
        mols: RDKit molecules
        radius: ECFP fingerprint radius
        length: number of bits

    Returns: a list of fingerprints
    """
    if sanitize:
        fps = []
        for mol in mols:
            Chem.SanitizeMol(mol)
            fps.append(
                AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius, length, useChirality=chiral
                )
            )
    else:
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius, length) for m in mols]
    return fps


def filter_diverse_ligands(ligands, threshold=0.9):
    """
    Filters ligands by diversity based on Tanimoto similarity.
    Molecules with Tanimoto similarity greater than `threshold` to
    any already kept molecule are filtered out.

    Args:
        ligands (list): List of RDKit Chem.Mol objects.
        threshold (float): Tanimoto similarity threshold, default 0.9.

    Returns:
        list: Filtered list of diverse Chem.Mol objects.
    """
    selected_ligs = []
    selected_fps = []
    for lig in ligands:
        # Compute the Morgan fingerprint with radius 2 (default bit vector size)
        fp = AllChem.GetMorganFingerprint(lig, radius=2)
        # Check if fingerprint is too similar to any already selected one.
        if any(
            DataStructs.TanimotoSimilarity(fp, sel_fp) > threshold
            for sel_fp in selected_fps
        ):
            continue
        selected_ligs.append(lig)
        selected_fps.append(fp)
    return selected_ligs


def filter_diverse_ligands_bulk(ligands, pdbs=None, threshold=0.9):
    """
    Filters ligands by diversity using a bulk computation of Tanimoto similarities.
    The function computes Morgan fingerprints for all ligands, builds a symmetric
    pairwise similarity matrix using BulkTanimotoSimilarity (with the diagonal set to zero),
    and then greedily selects a set of ligands such that any ligand added has a Tanimoto similarity
    of at most `threshold` (default 0.9) with every ligand already selected.

    Args:
        ligands (list): List of RDKit Chem.Mol objects.
        threshold (float): Tanimoto similarity threshold. Default: 0.9.

    Returns:
        list: Filtered list of diverse Chem.Mol objects.
    """
    # Compute Morgan fingerprints (with radius 2)
    fps = get_fingerprints(ligands, radius=2, length=4096, chiral=True, sanitize=False)
    n = len(fps)
    if n == 0:
        return []

    # Compute pairwise similarity matrix; initialize with zeros.
    sim_matrix = np.zeros((n, n))
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        sim_matrix[i, :i] = sims
        sim_matrix[:i, i] = sims

    # Greedy selection: add a ligand only if its similarity with all already selected ligands is <= threshold.
    selected_indices = []
    for i in range(n):
        if selected_indices:
            max_sim = np.max(sim_matrix[i, selected_indices])
            if max_sim > threshold:
                continue
        selected_indices.append(i)

    # Return the molecules corresponding to the selected indices.
    ligands = [ligands[i] for i in selected_indices]
    if pdbs is not None:
        pdbs = [pdbs[i] for i in selected_indices]
        return ligands, pdbs
    return ligands

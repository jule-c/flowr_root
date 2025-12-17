import json
import os
import tempfile
import warnings
from argparse import Namespace
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import lightning as L
import numpy as np
import torch
import yaml
from pymol import cmd
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator

import flowr.scriptutil as util
from flowr.data.datamodules import GeometricInterpolantDM
from flowr.data.dataset import (
    DatasetSubset,
    GeometricDataset,
    GeometricMolLMDBDataset,
    GeometricMolSDFDataset,
    PocketComplexLMDBDataset,
)
from flowr.data.interpolate import (
    ComplexInterpolant,
    GeometricInterpolant,
    GeometricNoiseSampler,
    extract_cores,
    extract_fragments,
    extract_func_groups,
    extract_linkers,
    extract_scaffolds,
    extract_substructure,
)
from flowr.data.preprocess_pdbs import process_complex
from flowr.util.functional import (
    optimize_ligand_in_pocket,
)
from flowr.util.metrics import (
    calculate_minimization_metrics,
    interaction_recovery_per_complex,
)
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


def get_conditional_mode(args):
    return (
        "scaffold"
        if args.scaffold_inpainting
        else (
            "func_group"
            if args.func_group_inpainting
            else (
                "linker"
                if args.linker_inpainting
                else (
                    "core"
                    if args.core_inpainting
                    else (
                        "fragment"
                        if args.fragment_inpainting
                        else (
                            "substructure"
                            if args.substructure_inpainting
                            else "interaction" if args.interaction_inpainting else None
                        )
                    )
                )
            )
        )
    )


def filter_substructure(
    gen_ligs: list[Chem.Mol],
    ref_mols: list[Chem.Mol],
    inpainting_mode: str,
    substructure_query: Optional[str] = None,
    max_fragment_cuts: int = 3,
    canonicalize_conformer: Optional[bool] = False,
):
    """
    Filter a list of generated molecules to only those that match the required substructure.

    Args:
        gen_ligs: List of generated RDKit molecules
        ref_mols: List of reference RDKit molecules
        inpainting_mode: One of ['scaffold', 'func_group', 'linker', 'core',
                        'fragment', 'substructure', 'interaction']
        substructure_query: SMILES/SMARTS string for substructure mode or list of atom IDs
        max_fragment_cuts: Maximum cuts for fragment mode
    Returns:
        Filtered list of generated molecules
    """
    filtered_ligs = []
    for ref_mol, gen_mol in zip(ref_mols, gen_ligs):
        if check_substructure_match(
            gen_mol,
            ref_mol,
            inpainting_mode,
            substructure_query=substructure_query,
            max_fragment_cuts=max_fragment_cuts,
            canonicalize_conformer=canonicalize_conformer,
        ):
            filtered_ligs.append(gen_mol)
    return filtered_ligs


def check_substructure_match(
    gen_mol: Chem.Mol,
    ref_mol: Chem.Mol,
    inpainting_mode: str,
    substructure_query: Optional[str] = None,
    max_fragment_cuts: int = 3,
    canonicalize_conformer: Optional[bool] = False,
) -> bool:
    """
    Check if a generated molecule contains the required substructure based on the inpainting mode.

    Args:
        gen_mol: Generated RDKit molecule
        ref_mol: Reference RDKit molecule
        inpainting_mode: One of ['scaffold', 'func_group', 'linker', 'core',
                        'fragment', 'substructure', 'interaction']
        substructure_query: SMILES/SMARTS string for substructure mode or list of atom IDs
        max_fragment_cuts: Maximum cuts for fragment mode

    Returns:
        True if the generated molecule contains the required substructure, False otherwise
    """

    # Extract the expected substructure mask based on mode
    if inpainting_mode == "scaffold":
        expected_mask = extract_scaffolds([ref_mol])[0]
    elif inpainting_mode == "func_group":
        expected_mask = extract_func_groups([ref_mol], includeHs=True)[0]
    elif inpainting_mode == "linker":
        expected_mask = extract_linkers([ref_mol])[0]
    elif inpainting_mode == "core":
        expected_mask = extract_cores([ref_mol])[0]
    elif inpainting_mode == "fragment":
        expected_mask = extract_fragments([ref_mol], maxCuts=max_fragment_cuts)[0]
    elif inpainting_mode == "substructure":
        if substructure_query is None:
            raise ValueError(
                "substructure_query must be provided for substructure mode"
            )
        expected_mask = extract_substructure(
            [ref_mol], substructure_query=substructure_query, invert_mask=True
        )[0]
    elif inpainting_mode == "interaction":
        # For interaction mode, we can't check from ref_mol alone
        print("Warning: Interaction mode validation not implemented")
        return True
    else:
        raise ValueError(f"Unknown inpainting mode: {inpainting_mode}")

    # If no atoms should be fixed, accept any molecule
    if not expected_mask.any():
        raise ValueError(
            f"No substructure found for inpainting mode: {inpainting_mode}"
        )

    # Get the atom indices that should be preserved
    expected_indices = expected_mask.nonzero(as_tuple=False).squeeze().tolist()

    if isinstance(expected_indices, int):
        expected_indices = [expected_indices]

    # Create substructure molecule from reference
    expected_submol = Chem.RWMol(ref_mol)
    atoms_to_remove = [
        i for i in range(ref_mol.GetNumAtoms()) if i not in expected_indices
    ]
    for idx in sorted(atoms_to_remove, reverse=True):
        expected_submol.RemoveAtom(idx)
    expected_submol = expected_submol.GetMol()

    # Check if generated molecule contains the substructure
    return gen_mol.HasSubstructMatch(expected_submol)


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
        use_interactions=args.interaction_inpainting,
        rotate_complex=args.arch == "transformer",
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
        dataset=getattr(args, "dataset", None),
        sample_mol_sizes=args.sample_mol_sizes,
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
        dataset=getattr(args, "dataset", None),
        sample_mol_sizes=args.sample_mol_sizes,
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


def load_data_from_pdb(
    args,
    remove_hs: bool,
    remove_aromaticity: bool,
    ligand_idx: int = 0,
    chain_id: Optional[str] = None,
    canonicalize_conformer: Optional[bool] = False,
):
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
        ligand_idx=ligand_idx,
        canonicalize_conformer=canonicalize_conformer,
        chain_id=chain_id,
        **processing_params,
    )
    system = system.remove_hs(include_ligand=remove_hs)
    return system


def load_data_from_pdb_selective(args, remove_hs: bool, remove_aromaticity: bool):
    # Load the data

    args.pdb_file = args.pdb_file_target
    args.ligand_file = args.ligand_file_target
    system_target = load_data_from_pdb(
        args,
        remove_hs=remove_hs,
        remove_aromaticity=remove_aromaticity,
    )
    args.pdb_file = args.pdb_file_untarget
    args.ligand_file = args.ligand_file_untarget
    system_offtarget = load_data_from_pdb(
        args,
        remove_hs=remove_hs,
        remove_aromaticity=remove_aromaticity,
    )
    return system_target, system_offtarget


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
        f"Dataset split is set to {args.dataset_split}. Number of molecules: {len(dataset)}"
    )
    molecules = [molecule for molecule in dataset if molecule is not None]
    molecules = split_list(molecules, args.gpus)[args.mp_index - 1]
    return molecules


def load_data_from_sdf_mol(
    args: Namespace,
    remove_hs: bool,
    remove_aromaticity: bool,
    transform: Optional[Callable] = None,
    sample: bool = False,
    sample_n_molecules: int = None,
    sample_n_molecules_per_mol: int = None,
):
    """Load the dataset from an LMDB file.
    Args:
        args: Command line arguments containing the data path and other parameters.
        transform: Optional transformation function to apply to the dataset.
        sample: Whether to sample molecules from the dataset.
        sample_n_molecules: Number of molecules to subsample from the dataset.
        sample_n_molecules_per_mol: Number of molecules to sample per input molecule.

    Returns:
        A list of molecules from the dataset.
    """
    dataset = GeometricMolSDFDataset(
        sdf_path=args.sdf_path,
        ligand_idx=args.ligand_idx,
        transform=transform,
        remove_hs=remove_hs,
        remove_aromaticity=remove_aromaticity,
        skip_non_valid=False,
    )
    if sample:
        assert (
            sample_n_molecules is not None or sample_n_molecules_per_mol is not None
        ), "Must specify sample_n_molecules or sample_n_molecules_per_mol when sampling."
        assert not (
            sample_n_molecules is not None and sample_n_molecules_per_mol is not None
        ), "Cannot specify both sample_n_molecules and sample_n_molecules_per_mol."
        if sample_n_molecules_per_mol is not None:
            dataset = dataset.sample_n_molecules_per_mol(sample_n_molecules_per_mol)
        elif sample_n_molecules is not None:
            dataset = dataset.sample_n_molecules(sample_n_molecules, seed=args.seed)

    print(f"Number of molecules: {len(dataset)}")
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
    pair: tuple,
    optimizer: Callable,
    optimize_pocket_hs: bool = False,
    add_ligand_hs: bool = True,
    only_ligand_hs: bool = False,
    process_pocket: bool = False,
):
    """
    Wrapper for process_lig to enable pickling.

    Args:
        pair (tuple): A tuple containing (lig, pdb_file).
        optimizer: The optimizer to pass to process_lig.
        optimize_pocket_hs: Whether to optimize pocket hydrogens.
        add_ligand_hs: Whether to add hydrogens to the ligand.
        only_ligand_hs: Whether to only optimize ligand hydrogens.
        process_pocket: Whether to process the pocket.

    Returns:
        The processed ligand.
    """
    lig, pdb_file = pair
    return process_lig(
        lig,
        pdb_file=pdb_file,
        optimizer=optimizer,
        add_ligand_hs=add_ligand_hs,
        only_ligand_hs=only_ligand_hs,
        optimize_pocket_hs=optimize_pocket_hs,
        process_pocket=process_pocket,
    )


def process_lig(
    lig: Chem.Mol,
    pdb_file: str,
    optimizer: Callable,
    optimize_pocket_hs: bool = False,
    add_ligand_hs: bool = True,
    process_pocket: bool = False,
    only_ligand_hs: bool = False,
):
    """
    Optimize the ligand in the pocket using the specified optimizer.
    Args:
        lig: RDKit ligand molecule.
        pdb_file: Path to the pocket PDB file.
        optimizer: The optimizer function to use.
        optimize_pocket_hs: Whether to optimize pocket hydrogens.
        add_ligand_hs: Whether to add hydrogens to the ligand.
        process_pocket: Whether to process the pocket.
        only_ligand_hs: Whether to only optimize ligand hydrogens.
    Returns:
        The optimized ligand.
    """
    # Optimize the ligand in the pocket
    lig_optim = optimize_ligand_in_pocket(
        lig,
        pdb_file,
        optimizer=optimizer,
        add_ligand_hs=add_ligand_hs,
        optimize_pocket_hs=optimize_pocket_hs,
        process_pocket=process_pocket,
        only_ligand_hs=only_ligand_hs,
    )
    return lig_optim


def optimize_lig(
    gen_lig: Chem.Mol,
    ref_lig: Chem.Mol,
    pdb_file: str,
    add_ligand_hs: bool = True,
    pocket_distance: float = 5.0,
    n_steps: int = 200,
    distance_constraint: float = 1.0,
):
    """
    Optimize the ligand in the pocket using MMFF optimization.

    Args:
        gen_lig: Generated RDKit ligand molecule.
        ref_lig: Reference RDKit ligand molecule.
        pdb_file: Path to the pocket PDB file.
        pocket_distance: Distance cutoff for pocket residues.
        n_steps: Number of optimization steps.
        distance_constraint: Distance constraint for optimization.
    Returns:
        The optimized ligand.
    """
    from flowr.util.functional import setup_minimize

    lig_optim = setup_minimize(
        gen_lig,
        ref_lig=ref_lig,
        pdb_file=pdb_file,
        add_ligand_hs=add_ligand_hs,
        pocket_distance=pocket_distance,
        n_steps=n_steps,
        distance_constraint=distance_constraint,
    )
    return lig_optim


def optimize_ligs(
    gen_ligs: List[Chem.Mol],
    ref_lig: Chem.Mol,
    pdb_file: str,
    add_ligand_hs: bool = True,
    pocket_distance: float = 5.0,
    n_steps: int = 200,
    distance_constraint: float = 1.0,
):
    """
    Optimize the ligand in the pocket using MMFF optimization.

    Args:
        gen_ligs: List of generated RDKit ligand molecules.
        ref_lig: Reference RDKit ligand molecule.
        pdb_file: Path to the pocket PDB file.
        pocket_distance: Distance cutoff for pocket residues.
        n_steps: Number of optimization steps.
        distance_constraint: Distance constraint for optimization.
    Returns:
        The optimized ligand.
    """
    from flowr.util.functional import setup_minimize_list

    ligs_optim = setup_minimize_list(
        gen_ligs,
        ref_lig=ref_lig,
        pdb_file=pdb_file,
        add_ligand_hs=add_ligand_hs,
        pocket_distance=pocket_distance,
        n_steps=n_steps,
        distance_constraint=distance_constraint,
    )
    return ligs_optim


def evaluate_optimization(
    ligs: List[Chem.Mol],
    optim_ligs: List[Chem.Mol],
):
    # Calculate RMSD and strains for each pair
    strains_before_minimization = []
    strains_after_minimization = []
    rmsds_before_after = []
    for mol_before, mol_after in zip(ligs, optim_ligs):
        strain_before, strain_after, rmsd_value = calculate_minimization_metrics(
            mol_before, mol_after
        )
        strains_before_minimization.append(strain_before)
        strains_after_minimization.append(strain_after)
        rmsds_before_after.append(rmsd_value)
    print(
        f"Strain before minimization - Mean: {np.nanmean(strains_before_minimization):.3f}, "
        f"Std: {np.nanstd(strains_before_minimization):.3f}"
    )
    print(
        f"Strain after minimization - Mean: {np.nanmean(strains_after_minimization):.3f}, "
        f"Std: {np.nanstd(strains_after_minimization):.3f}"
    )
    print(
        f"RMSD (before -> after) - Mean: {np.nanmean(rmsds_before_after):.3f}, "
        f"Std: {np.nanstd(rmsds_before_after):.3f}"
    )


def process_interaction(
    gen_lig: Chem.Mol,
    ref_lig: Chem.Mol,
    pdb_file: str,
    pocket_cutoff: float,
    save_dir: str,
    add_hs_and_optimize_gen_ligs: bool,
):
    recovery_rate, tanimoto_sim = interaction_recovery_per_complex(
        gen_ligs=gen_lig,
        native_lig=ref_lig,
        pdb_file=pdb_file,
        add_optimize_gen_lig_hs=add_hs_and_optimize_gen_ligs,
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
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        fpSize=length, radius=radius, includeChirality=chiral
    )
    if sanitize:
        fps = []
        for mol in mols:
            Chem.SanitizeMol(mol)
            fps.append(morgan_gen.GetFingerprint(mol))
    else:
        fps = [morgan_gen.GetFingerprint(m) for m in mols]
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
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    for lig in ligands:
        # Compute the Morgan fingerprint with radius 2 (default bit vector size)
        fp = morgan_gen.GetFingerprint(lig)
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


def load_config_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a JSON or YAML configuration file and return as dictionary.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    file_extension = config_path.suffix.lower()

    try:
        with open(config_path, "r") as f:
            if file_extension == ".json":
                config = json.load(f)
            elif file_extension in [".yaml", ".yml"]:
                config = yaml.safe_load(f)
            else:
                raise ValueError(
                    f"Unsupported file format: {file_extension}. Use .json, .yaml, or .yml"
                )

        return config

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {e}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML file: {e}")
    except Exception as e:
        raise ValueError(f"Error reading config file: {e}")


def get_guidance_params(args) -> Dict[str, Any]:
    """
    Get guidance parameters either from config file or use defaults.
    """
    if args.guidance_config is not None:
        print(f"Loading guidance config from: {args.guidance_config}")
        guidance_params = load_config_file(args.guidance_config)
    else:
        # Default guidance parameters
        guidance_params = {
            "apply_guidance": False,
            "window_start": 0.0,
            "window_end": 0.3,
            "value_key": "affinity",
            "subvalue_key": "pic50",
            "mu": 8.0,
            "sigma": 2.0,
            "maximize": True,
            "coord_noise_level": 0.2,
        }

    return guidance_params


def optimize_molecule_rdkit(mol):
    """
    Optimize a molecule using RDKit MMFF force field.

    Args:
        mol: RDKit Mol object with 3D conformer

    Returns:
        tuple: (optimized_mol, energy_diff, rmsd)
            - optimized_mol: RDKit Mol with optimized conformer
            - energy_diff: Energy change in kcal/mol (negative = stabilization)
            - rmsd: RMSD between initial and optimized structure in Angstroms
    """
    from rdkit.Chem import AllChem, rdMolAlign

    if mol is None:
        return None, None, None

    # Make a copy to preserve the original
    mol_copy = Chem.Mol(mol)

    try:
        # Store initial positions for RMSD calculation
        initial_mol = Chem.Mol(mol_copy)

        # Set up MMFF force field
        ff_props = AllChem.MMFFGetMoleculeProperties(mol_copy)
        if ff_props is None:
            # Fallback to UFF if MMFF fails
            print("MMFF failed, using UFF instead")
            initial_energy = AllChem.UFFGetMoleculeForceField(mol_copy).CalcEnergy()
            AllChem.UFFOptimizeMolecule(mol_copy, maxIters=2000)
            final_energy = AllChem.UFFGetMoleculeForceField(mol_copy).CalcEnergy()
        else:
            # Use MMFF
            ff = AllChem.MMFFGetMoleculeForceField(mol_copy, ff_props)
            initial_energy = ff.CalcEnergy()

            # Optimize
            ff.Minimize(maxIts=2000)

            # Recalculate energy after optimization
            ff = AllChem.MMFFGetMoleculeForceField(mol_copy, ff_props)
            final_energy = ff.CalcEnergy()

        # Calculate energy difference (in kcal/mol)
        energy_diff = final_energy - initial_energy

        # Calculate RMSD between initial and optimized structures
        rmsd = rdMolAlign.GetBestRMS(initial_mol, mol_copy)

        return mol_copy, energy_diff, rmsd

    except Exception as e:
        print(f"Error optimizing molecule: {e}")
        return None, None, None

#!/usr/bin/env python
"""
Utility functions for the Flowr Data Processing Pipeline.

This module contains shared functions used across the receptor preparation,
docking, full workflow, and PBSA calculation scripts.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import biotite.database.rcsb as rcsb
import biotite.sequence as seq  # sequence information
import biotite.structure as struc  # structure manipulation
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx  # read and write PDB and CIF files
import numpy as np  # array and matrix manipulation
import openeye
import pandas as pd
from biotite.interface import rdkit  # interface with the RDKit
from joblib import Parallel, delayed
from numpy import argmin as npargmin
from numpy import ix_ as npix
from numpy import sum as npsum
from numpy import where as npwhere
from numpy import zeros as npz
from openeye import (
    oechem,
    oedocking,
    oegrid,
    oeomega,
    oequacpac,
    oespruce,
    oeszybki,
    oezap,
)
from openeye.oechem import (
    OEAddExplicitHydrogens,
    OEGetSDData,
    OEMol,
    OESetSDData,
    OESuppressHydrogens,
)
from openeye.oedocking import (
    OEDock,
    OEDockingReturnCode_Success,
    OEDockMethod_Hybrid,
    OEDockOptions,
    OEShapeFit,
    OEShapeFitOptions,
)
from openeye.oeomega import (
    OEFlipper,
    OEFlipperOptions,
    OEGetOmegaError,
    OEOmega,
    OEOmegaOptions,
    OEOmegaReturnCode_Success,
)
from openeye.oequacpac import OEGetReasonableProtomer
from openeye.oeshape import (
    OEBestOverlayScore,
    OEOverlapPrep,
    OEOverlay,
    OEOverlayOptions,
    OERemoveColorAtoms,
)
from openeye.oeszybki import (
    OEFixedProteinLigandOptimizer,
    OEFlexProteinLigandOptimizer,
    OEProteinFlexOptions,
    OEProteinLigandOptOptions,
    OEProteinLigandOptResults,
    OESzybkiReturnCode_Success,
)
from rdkit import Chem  # for reading SDF/PDB ligand files
from sklearn.cluster import DBSCAN

# =============================================================================
# Constants
# =============================================================================

# Valid charge types for ligand charge assignment
CHARGE_TYPE_MMFF94 = "mmff94"
CHARGE_TYPE_AM1BCCELF10 = "am1bccelf10"
VALID_CHARGE_TYPES = [CHARGE_TYPE_MMFF94, CHARGE_TYPE_AM1BCCELF10]

# Szybki error code mapping (common codes)
SZYBKI_ERROR_CODES = {
    0: "Success",
    1: "UnspecifiedError",
    2: "MoleculeError",
    3: "ProteinError",
    4: "LigandError",
    200: "ForceFieldError",
    201: "ForceFieldMissingParameters",
    202: "ForceFieldAtomTyping",
    203: "ForceFieldChargeAssignment",
    204: "ForceFieldSetupFailed",
    205: "OptimizationFailed",
    206: "MemoryError",
}


def get_szybki_error_name(code: int) -> str:
    """Get human-readable name for Szybki error code."""
    return SZYBKI_ERROR_CODES.get(code, f"UnknownError({code})")


# =============================================================================
# License Setup
# =============================================================================

LICENSE_FILENAME = os.environ.get(
    "OE_LICENSE",
    str(Path(__file__).resolve().parent / "tools" / "oe_license.txt"),
)


def setup_openeye_license(license_filename: str = LICENSE_FILENAME) -> None:
    """
    Set up the OpenEye license.

    Args:
        license_filename: Path to the OpenEye license file
    """
    if os.path.isfile(license_filename):
        with open(license_filename, "r") as license_file:
            openeye.OEAddLicenseData(license_file.read())
        assert openeye.oechem.OEChemIsLicensed(), "OpenEye license is not valid"
    else:
        raise Exception(
            f"Error: OpenEye license file not found at {license_filename}. "
            "Please check the path."
        )


# Initialize license on module import
try:
    setup_openeye_license()
except Exception as e:
    print(f"WARNING: OpenEye license setup failed: {e}")


# =============================================================================
# Charge Handling Functions
# =============================================================================


def get_ligand_charge_type(
    mol: oechem.OEMolBase, charge_type: str = CHARGE_TYPE_MMFF94
) -> int:
    """
    Determine the appropriate OELigandChargeType based on molecule state and user preference.

    If the molecule already has partial charges assigned (checked via OEHasPartialCharges),
    returns OELigandChargeType_CURRENT to use existing charges.
    Otherwise, returns the appropriate charge type based on user preference.

    Args:
        mol: Input molecule to check for existing charges
        charge_type: Desired charge type if no charges exist.
                     Either 'mmff94' (default) or 'am1bccelf10'

    Returns:
        OELigandChargeType enum value for use with SetLigandCharge
    """
    # Check if molecule already has partial charges
    if oechem.OEHasPartialCharges(mol):
        return oeszybki.OELigandChargeType_CURRENT

    # Return appropriate charge type based on user preference
    if charge_type.lower() == CHARGE_TYPE_AM1BCCELF10:
        return oeszybki.OELigandChargeType_AM1BCCELF10
    else:
        return oeszybki.OELigandChargeType_MMFF


def should_generate_elf_confs(charge_type: str) -> bool:
    """
    Determine if ELF conformer generation should be enabled.

    ELF conformer generation should only be enabled when using AM1BCCELF10 charges,
    as it is part of the AM1-BCC-ELF10 charge assignment protocol.

    Args:
        charge_type: The charge type being used ('mmff94' or 'am1bccelf10')

    Returns:
        True if ELF conformer generation should be enabled, False otherwise
    """
    return charge_type.lower() == CHARGE_TYPE_AM1BCCELF10


def molecule_has_charges(mol: oechem.OEMolBase) -> bool:
    """
    Check if a molecule already has partial charges assigned.

    Args:
        mol: Molecule to check

    Returns:
        True if molecule has partial charges, False otherwise
    """
    return oechem.OEHasPartialCharges(mol)


# =============================================================================
# Global Field Definitions for OEDB Storage
# =============================================================================

# Fields for receptor OEDB
du_field = oechem.OEField("Design Unit", oechem.Types.Chem.DesignUnit)
title_field = oechem.OEField("Title", oechem.Types.String)
ligand_field = oechem.OEField("Ligand Smiles", oechem.Types.String)
pocket_field = oechem.OEField("Site Residues", oechem.Types.StringVec)
iridium_field = oechem.OEField("Structure Quality", oechem.Types.String)

# Fields for docking results OEDB
mol_field = oechem.OEField("Molecule", oechem.Types.Chem.Mol)
method_field = oechem.OEField("Method", oechem.Types.String)


# =============================================================================
# Protein Preparation Functions
# =============================================================================


def split_components(
    mol: oechem.OEMol,
) -> Tuple[oechem.OEMol, oechem.OEMol, oechem.OEMol, oechem.OEMol]:
    """
    Split the components of a raw protein-ligand PDB/MMCIF file into
    ligand, protein, water, and other components.

    Args:
        mol: Input molecule containing protein-ligand complex

    Returns:
        Tuple of (ligand, protein, water, other) molecules
    """
    lig = oechem.OEMol()
    prot = oechem.OEMol()
    water = oechem.OEMol()
    other = oechem.OEMol()
    oechem.OESplitMolComplex(lig, prot, water, other, mol)
    return lig, prot, water, other


# def prep_structure(
#     mol: oechem.OEMol,
#     density_map: oegrid.OESkewGrid,
#     min_atoms: int,
#     max_atoms: int,
#     max_residues: int,
# ) -> List:
#     """
#     Prepare a protein structure using Spruce and create docking receptors.

#     A comprehensive protein preparation workflow is conducted using OEMakeDesignUnits.
#     The receptor grid is created with OEMakeReceptor including crystallographic waters.

#     Args:
#         mol: Input protein-ligand molecule
#         density_map: MTZ density map for structure validation
#         min_atoms: Minimum ligand atom count
#         max_atoms: Maximum ligand atom count
#         max_residues: Maximum ligand residues

#     Returns:
#         List of prepared OEDesignUnit objects
#     """
#     receptor_opts = oedocking.OEMakeReceptorOptions()
#     receptor_opts.SetTargetMask(oechem.OEDesignUnitComponents_TargetComplex)

#     metadata = oespruce.OEStructureMetadata()
#     du_split_options = oespruce.OEDesignUnitSplitOptions()
#     du_split_options.SetMinLigAtoms(min_atoms)
#     du_split_options.SetMaxLigAtoms(max_atoms)
#     du_split_options.SetMaxLigResidues(max_residues)

#     makedu_opts = oespruce.OEMakeDesignUnitOptions()
#     makedu_opts.SetSplitOptions(du_split_options)

#     design_units = oespruce.OEMakeDesignUnits(mol, density_map, metadata, makedu_opts)
#     dus = []
#     for design_unit in design_units:
#         oedocking.OEMakeReceptor(design_unit, receptor_opts)
#         dus.append(design_unit)
#     return dus


def prep_structure(
    mol: oechem.OEMol,
    density_map: oegrid.OESkewGrid,
    min_atoms: int,
    max_atoms: int,
    max_residues: int,
    build_sidechains: bool = False,
    cap_termini: bool = False,
    protonate: bool = False,
    handle_alt_locations: bool = False,
    flip_bias_scale: float = 1.0,
    verbose: bool = True,
) -> List:
    """
    Prepare a protein structure using Spruce and create docking receptors.

    A comprehensive protein preparation workflow is conducted:
    1. Handle alternate locations (keep highest occupancy)
    2. Build missing sidechains (optional)
    3. Cap broken termini (optional)
    4. Create design units with OEMakeDesignUnits
    5. Protonate with optimized hydrogen bond network (optional)
    6. Create receptor grids with OEMakeReceptor

    Args:
        mol: Input protein-ligand molecule
        density_map: MTZ density map for structure validation (can be empty OESkewGrid)
        min_atoms: Minimum ligand atom count
        max_atoms: Maximum ligand atom count
        max_residues: Maximum ligand residues
        build_sidechains: Whether to build missing sidechains (default: True)
        cap_termini: Whether to cap broken N/C termini (default: True)
        protonate: Whether to run protonation with flip optimization (default: True)
        handle_alt_locations: Whether to resolve alternate locations to highest occupancy (default: True)
        flip_bias_scale: Bias scale for flipping His/Asn/Gln sidechains during protonation (default: 1.0)
        verbose: Whether to print progress messages (default: False)

    Returns:
        List of prepared OEDesignUnit objects
    """
    # Work on a copy to avoid modifying the input
    mol_copy = oechem.OEMol(mol)

    # Step 1: Handle alternate locations - keep only highest occupancy conformation
    if handle_alt_locations:
        alf = oechem.OEAltLocationFactory(mol_copy)
        if alf.GetGroupCount() > 0:
            if verbose:
                print(
                    f"  Found {alf.GetGroupCount()} alternate location groups, keeping highest occupancy"
                )
            alf.MakePrimaryAltMol(mol_copy)

    # Step 2: Build missing sidechains
    if build_sidechains:
        sidechain_opts = oespruce.OESidechainBuilderOptions()
        sidechain_opts.SetDeleteClashingSolvent(True)
        sidechain_opts.SetMinimizeSidechains(True)
        sidechain_opts.SetRotamerLibrary(oechem.OERotamerLibrary_Dunbrack)
        if oespruce.OEBuildSidechains(mol_copy, sidechain_opts):
            if verbose:
                print("  Built missing sidechains")
        elif verbose:
            print("  No sidechains needed building or build failed")

    # Step 3: Cap broken termini
    if cap_termini:
        capped_n = oespruce.OECapNTermini(mol_copy)
        capped_c = oespruce.OECapCTermini(mol_copy)
        if verbose and (capped_n or capped_c):
            print(f"  Capped termini (N: {capped_n}, C: {capped_c})")

    # Step 4: Configure and create design units
    receptor_opts = oedocking.OEMakeReceptorOptions()
    receptor_opts.SetTargetMask(oechem.OEDesignUnitComponents_TargetComplex)

    metadata = oespruce.OEStructureMetadata()
    du_split_options = oespruce.OEDesignUnitSplitOptions()
    du_split_options.SetMinLigAtoms(min_atoms)
    du_split_options.SetMaxLigAtoms(max_atoms)
    du_split_options.SetMaxLigResidues(max_residues)

    makedu_opts = oespruce.OEMakeDesignUnitOptions()
    makedu_opts.SetSplitOptions(du_split_options)

    # Disable loop building by default (requires external database)
    makedu_opts.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetBuildTails(
        False
    )

    design_units = oespruce.OEMakeDesignUnits(
        mol_copy, density_map, metadata, makedu_opts
    )

    # Step 5: Protonate and create receptors
    dus = []
    for design_unit in design_units:
        # Protonate with flip optimization for His/Asn/Gln
        if protonate:
            place_h_opts = oechem.OEPlaceHydrogensOptions()
            place_h_opts.SetFlipBiasScale(flip_bias_scale)
            place_h_opts.SetWaterProcessing(
                oechem.OEPlaceHydrogensWaterProcessing_FullSearch
            )
            place_h_opts.SetStandardizeBondLen(True)
            place_h_opts.SetBadClashOverlapDistance(0.4)

            protonate_opts = oespruce.OEProtonateDesignUnitOptions()
            protonate_opts.SetPlaceHydrogensOptions(place_h_opts)

            oespruce.OEProtonateDesignUnit(design_unit, protonate_opts)
            if verbose:
                print("  Protonated design unit with flip optimization")

        # Create receptor grid
        oedocking.OEMakeReceptor(design_unit, receptor_opts)
        dus.append(design_unit)

    if verbose:
        print(f"  Created {len(dus)} design unit(s)")

    return dus


def set_record(du) -> oechem.OERecord:
    """
    Store the prepared OEDesignUnit on an OERecord with structure quality information.

    Extracts the Iridium score category (HT, MT, or NT) as an assessment of
    protein structure quality based on the overlap between coordinates and density map.

    Args:
        du: Prepared OEDesignUnit

    Returns:
        OERecord containing the design unit and metadata
    """
    structure_quality = du.GetStructureQuality()
    iridium_data = structure_quality.GetIridiumData()
    iridium_category = iridium_data.GetCategory()
    iridium_class = oechem.OEGetIridiumCategoryName(iridium_category)

    ligand = oechem.OEMol()
    du.GetLigand(ligand)
    ligand_smiles = oechem.OEMolToSmiles(ligand)
    title = du.GetTitle()
    site_residues = du.GetSiteResidues()

    record = oechem.OERecord()
    record.set_value(du_field, du)
    record.set_value(iridium_field, iridium_class)
    record.set_value(title_field, title)
    record.set_value(ligand_field, ligand_smiles)
    record.set_value(pocket_field, site_residues)
    return record


def extract_best_receptor(dus: List) -> Optional[oechem.OEDesignUnit]:
    """
    Extract the best receptor DU based on Iridium Score.

    Prioritizes HT (Highly Trustworthy) structures, then MT (Moderately Trustworthy).
    NT (Not Trustworthy) structures are not returned.

    Args:
        dus: List of OEDesignUnit objects

    Returns:
        Best OEDesignUnit based on Iridium score, or None if no trustworthy structure found
    """
    # First pass: look for HT structures
    for du in dus:
        structure_quality = du.GetStructureQuality()
        iridium_data = structure_quality.GetIridiumData()
        iridium_category = iridium_data.GetCategory()
        iridium_class = oechem.OEGetIridiumCategoryName(iridium_category)
        print(f"Iridium class: {iridium_class}")
        if iridium_class == "HT":
            return du

    # Second pass: look for MT structures
    for du in dus:
        structure_quality = du.GetStructureQuality()
        iridium_data = structure_quality.GetIridiumData()
        iridium_category = iridium_data.GetCategory()
        iridium_class = oechem.OEGetIridiumCategoryName(iridium_category)
        if iridium_class == "MT":
            return du

    print(
        "No structures were found to be moderately or highly trustworthy. "
        "Please revise inputs or remove structure"
    )
    return None


def load_receptor_from_oedb(oedb_file: str) -> oechem.OEDesignUnit:
    """
    Load receptor design unit from OEDB file.

    Args:
        oedb_file: Path to OEDB file containing receptor

    Returns:
        OEDesignUnit loaded from file

    Raises:
        ValueError: If receptor cannot be loaded from file
    """
    ifs = oechem.oeifstream(oedb_file)
    record = oechem.OERecord()

    du_field = oechem.OEField("Design Unit", oechem.Types.Chem.DesignUnit)
    for record in oechem.OEReadRecords(ifs):

        du = record.get_value(du_field)
        if du is not None:
            return du
        else:
            raise ValueError(f"Failed to read receptor from {oedb_file}")


# =============================================================================
# Conformer Generation Functions
# =============================================================================


def gen_confs(mol: oechem.OEMol) -> Optional[oechem.OEMol]:
    """
    Generate Omega 3D conformational ensemble using Pose mode.

    Pose Mode generates a max of 200 conformations if rotors <= 7 or 800 if rotors > 7.
    Also runs OEFlipper to enumerate unspecified stereocenters (up to 3).
    Applies OEGetReasonableProtomer for ionization/tautomer states at pH 7.4.

    Args:
        mol: Input molecule

    Returns:
        Molecule with generated conformers, or None if generation failed
    """
    flipper_opts = OEFlipperOptions()
    flipper_opts.SetMaxCenters(3)
    opts = OEOmegaOptions(oeomega.OEOmegaSampling_Pose)
    omega = OEOmega(opts)
    OEGetReasonableProtomer(mol)

    for isomer in OEFlipper(mol, flipper_opts):
        fmol = OEMol(isomer)
        ret_code = omega.Build(fmol)
        if ret_code == OEOmegaReturnCode_Success:
            return fmol
        else:
            print("Omega Failed because " + str(OEGetOmegaError(ret_code)))
            return None
    return None


# =============================================================================
# ROCS Overlay Functions
# =============================================================================


def rocs_overlay(query: oechem.OEMol, fitmol: oechem.OEMol) -> oechem.OEMol:
    """
    Perform a ROCS Overlay with default shape+color scoring.

    Returns the TanimotoCombo score which is used to decide between
    ShapeFit (TC >= 1.5) and HYBRID (TC < 1.5) docking methods.

    Args:
        query: Reference/query molecule
        fitmol: Molecule to be fitted/overlaid

    Returns:
        Fitted molecule with TanimotoCombo score stored as SD data
    """
    rotmat = oechem.OEFloatArray(9)
    transvec = oechem.OEFloatArray(3)
    prep = OEOverlapPrep()
    opts = OEOverlayOptions()
    overlay = OEOverlay(opts)

    prep.Prep(query)
    overlay.SetupRef(query)
    prep.Prep(fitmol)

    score = OEBestOverlayScore()
    overlay.BestOverlay(score, fitmol)
    TC = score.GetTanimotoCombo()
    score.GetRotMatrix(rotmat)
    score.GetTranslation(transvec)

    OESetSDData(fitmol, "TanimotoCombo", str(TC))
    return fitmol


# =============================================================================
# Docking Functions
# =============================================================================


def dock_ligand_hybrid(
    du: oechem.OEDesignUnit,
    confs: oechem.OEMol,
    num_poses: int = 50,
) -> Optional[oechem.OEMol]:
    """
    Dock Omega conformers into DU using HYBRID method.

    HYBRID uses both shape and pharmacophore features for docking.

    Scores are saved as SD data on each pose:
    - "Chemgauss4 Score": The Chemgauss4 docking score (lower is better)

    Args:
        du: Design unit containing the receptor
        confs: Molecule with conformers to dock
        num_poses: Number of poses to return (default: 50)

    Returns:
        Docked molecule with poses, or None if docking failed
    """
    OERemoveColorAtoms(confs)
    dock_opts = OEDockOptions()
    dock_opts.SetScoreMethod(OEDockMethod_Hybrid)
    dock = OEDock(dock_opts)
    dock.Initialize(du)

    docked_mol = OEMol()
    ret_code = dock.DockMultiConformerMolecule(docked_mol, confs, num_poses)

    if ret_code == OEDockingReturnCode_Success:
        # Annotate each pose with its docking score
        oedocking.OESetSDScore(docked_mol, dock, "Chemgauss4 Score")
        # Also add additional pose annotations
        dock.AnnotatePose(docked_mol)
        return docked_mol
    else:
        print(
            "Docking Failed because "
            + str(oedocking.OEDockingReturnCodeGetName(ret_code))
        )
        return None


def dock_ligand_shapefit(
    du: oechem.OEDesignUnit, confs: oechem.OEMol, num_poses: int = 1
) -> Optional[oechem.OEMol]:
    """
    Dock Omega conformers into DU with OEShapeFit.

    ShapeFit uses shape-based overlay for pose prediction.
    Scores are saved as SD data on each pose:
    - "Pose Probability": Probability that the predicted pose is within 2Å of the X-ray pose
    - "Shape Tanimoto": Shape similarity score (0-1, higher is better)
    - "Color Tanimoto": Color/pharmacophore similarity score (0-1, higher is better)

    Args:
        du: Design unit containing the receptor
        confs: Molecule with conformers to dock
        num_poses: Number of poses to return (default: 1)

    Returns:
        Multi-conformer molecule with docked poses, or None if no poses generated
    """
    OERemoveColorAtoms(confs)
    opts = OEShapeFitOptions()
    opts.SetFullConformationSearch(num_poses > 1)
    shapefit = OEShapeFit(opts)
    shapefit.SetupRef(du)

    results = shapefit.Fit(confs, num_poses)

    # Build multi-conformer molecule from poses (consistent with HYBRID output)
    docked_mol = None
    for res in results:
        pose = res.GetPose()
        # Save all available scores
        OESetSDData(pose, "Pose Probability", f"{res.GetScore():.4f}")

        if docked_mol is None:
            docked_mol = oechem.OEMol(pose)
        else:
            docked_mol.NewConf(pose)

    return docked_mol


# =============================================================================
# Energy Minimization Functions
# =============================================================================


def minimize_poses(
    du: oechem.OEDesignUnit,
    poses: oechem.OEMol,
    charge_type: str = CHARGE_TYPE_MMFF94,
    verbose: bool = False,
    filter_positive_energy: bool = True,
) -> Optional[oechem.OEMol]:
    """
    Energy minimize docking poses using fixed protein optimization.

    Works with both single-pose and multi-conformer molecules from any docking method.
    Uses AMBER ff14sb-SAGE force field. Charge assignment is determined by
    the charge_type parameter, or uses existing charges if present.
    Stores initial energy, final energy, and RMSD on each pose.

    Args:
        du: Design unit containing the protein
        poses: Molecule with docking poses (single or multi-conformer)
        charge_type: Charge type for ligand ('mmff94' or 'am1bccelf10'). Default: 'mmff94'
        verbose: Whether to print detailed minimization output. Default: False
        filter_positive_energy: Whether to filter out poses with positive final energy. Default: True

    Returns:
        Minimized poses (filtered if filter_positive_energy=True), or None if no poses remain
    """
    # Determine charge type based on molecule state
    ligand_charge_type = get_ligand_charge_type(poses, charge_type)

    protein_ligand_opts = OEProteinLigandOptOptions()
    protein_ligand_opts.SetSolventModel(oeszybki.OESolventModel_NoSolv)
    protein_ligand_opts.SetForceField("ff14sb_sage")
    protein_ligand_opts.SetGenerateElfConfs(should_generate_elf_confs(charge_type))
    protein_ligand_opts.SetLigandCharge(ligand_charge_type)

    optimizer = OEFixedProteinLigandOptimizer(protein_ligand_opts)
    optimizer.SetProtein(du, oechem.OEDesignUnitComponents_Protein)

    OEAddExplicitHydrogens(poses)

    # Track conformers to keep (negative energy) vs filter (positive energy)
    confs_to_keep = []  # List of (conf_idx, final_energy) tuples
    failed_count = 0
    positive_energy_count = 0
    last_error = None

    for i, pose in enumerate(poses.GetConfs()):
        results = OEProteinLigandOptResults()
        ret_code = optimizer.Optimize(results, pose)
        if ret_code == OESzybkiReturnCode_Success:
            initial = results.GetInitialEnergies()
            initial_energy = initial.GetTotalEnergy()
            final = results.GetFinalEnergies()
            final_energy = final.GetTotalEnergy()
            ligand_rmsd = results.GetLigandRMSD()
            OESetSDData(pose, "Initial Energy", str(initial_energy))
            OESetSDData(pose, "Final Energy", str(final_energy))
            OESetSDData(pose, "RMSD", str(ligand_rmsd))
            OESetSDData(pose, "Minimization", "Success")

            if verbose:
                print(
                    f"Minimization successful for pose {i}: Initial Energy={initial_energy:.2f}, "
                    f"Final Energy={final_energy:.2f}, RMSD={ligand_rmsd:.3f}"
                )

            # Track whether to keep this conformer
            if filter_positive_energy and final_energy >= 0:
                positive_energy_count += 1
                if verbose:
                    print(
                        f"  -> Filtering pose {i} (positive energy: {final_energy:.2f})"
                    )
            else:
                confs_to_keep.append(pose)
        else:
            # Mark failed pose so it can be filtered later if needed
            error_name = get_szybki_error_name(ret_code)
            OESetSDData(pose, "Minimization", f"Failed: {error_name}")
            failed_count += 1
            last_error = error_name
            if verbose:
                print(f"Minimization failed for pose {i}: {error_name}")
            continue

    if failed_count > 0:
        print(
            f"Minimization: {failed_count} conformers failed (last error: {last_error})"
        )

    if filter_positive_energy and positive_energy_count > 0:
        print(
            f"Minimization: Filtered {positive_energy_count} conformers with positive final energy"
        )

    if not confs_to_keep:
        if filter_positive_energy:
            print(
                "WARNING: No conformers with negative energy remain after minimization! Keeping lowest five energy conformers."
            )
            # Keep the five conformers with lowest final energy
            energy_conf_pairs = []
            for pose in poses.GetConfs():
                final_energy_str = OEGetSDData(pose, "Final Energy")
                try:
                    final_energy = float(final_energy_str)
                    energy_conf_pairs.append((final_energy, pose))
                except (TypeError, ValueError):
                    continue
            energy_conf_pairs.sort(key=lambda x: x[0])
            confs_to_keep = [pair[1] for pair in energy_conf_pairs[:5]]
            # Build result molecule with these conformers
            result_mol = oechem.OEMol(poses)
            result_mol.DeleteConfs()
            for conf in confs_to_keep:
                result_mol.NewConf(conf)
            return result_mol
        else:
            print(
                "WARNING: Minimization failed for ALL conformers! Returning unminimized poses."
            )
            return poses

    # Build result molecule with only the conformers to keep
    if filter_positive_energy:
        result_mol = oechem.OEMol(poses)
        result_mol.DeleteConfs()
        for conf in confs_to_keep:
            result_mol.NewConf(conf)
        return result_mol
    else:
        return poses


# Keep legacy aliases for backward compatibility
def minimize_poses_hybrid(
    du: oechem.OEDesignUnit,
    poses: oechem.OEMol,
    charge_type: str = CHARGE_TYPE_MMFF94,
) -> Optional[oechem.OEMol]:
    """Legacy alias for minimize_poses. Use minimize_poses instead."""
    return minimize_poses(du, poses, charge_type)


def minimize_pose_shapefit(
    du: oechem.OEDesignUnit,
    pose: oechem.OEMol,
    charge_type: str = CHARGE_TYPE_MMFF94,
) -> Optional[oechem.OEMol]:
    """Legacy alias for minimize_poses. Use minimize_poses instead."""
    return minimize_poses(du, pose, charge_type)


def minimize_poses_flex(
    du: oechem.OEDesignUnit,
    poses: oechem.OEMol,
    flex_range: float = 2.0,
    charge_type: str = CHARGE_TYPE_MMFF94,
    filter_positive_energy: bool = True,
) -> Tuple[Optional[oechem.OEMol], Optional[oechem.OEDesignUnit]]:
    """
    Energy minimize docking poses with flexible protein pocket.

    Works with both single-pose and multi-conformer molecules from any docking method.
    Uses AMBER ff14sb-SAGE force field. Charge assignment is determined by
    the charge_type parameter, or uses existing charges if present.
    Protein residues within flex_range of the ligand are treated as flexible.
    Stores initial energy, final energy, and RMSD on each pose.

    WARNING: Flexible protein optimization can be very slow. For large-scale
    workflows, consider using minimize_poses instead.

    Args:
        du: Design unit containing the protein
        poses: Molecule with docking poses (single or multi-conformer)
        flex_range: Distance cutoff (Angstroms) for flexible protein residues (default: 2.0)
        charge_type: Charge type for ligand ('mmff94' or 'am1bccelf10'). Default: 'mmff94'
        filter_positive_energy: Whether to filter out poses with positive final energy. Default: True

    Returns:
        Tuple of (minimized poses, updated design unit with optimized protein),
        or (None, None) if no poses remain after filtering
    """
    # Determine charge type based on molecule state
    ligand_charge_type = get_ligand_charge_type(poses, charge_type)

    protein_ligand_opts = OEProteinLigandOptOptions()
    protein_ligand_opts.SetSolventModel(oeszybki.OESolventModel_NoSolv)
    protein_ligand_opts.SetForceField("ff14sb_sage")
    protein_ligand_opts.SetGenerateElfConfs(should_generate_elf_confs(charge_type))
    protein_ligand_opts.SetLigandCharge(ligand_charge_type)

    # Configure flexible protein options
    flex_opts = OEProteinFlexOptions()
    flex_opts.SetFlexRange(flex_range)

    optimizer = OEFlexProteinLigandOptimizer(protein_ligand_opts)

    OEAddExplicitHydrogens(poses)

    # Track conformers to keep (negative energy) vs filter (positive energy)
    confs_to_keep = []  # List of conformer objects
    failed_count = 0
    positive_energy_count = 0
    last_successful_du = None
    last_error = None

    for pose in poses.GetConfs():
        # Create a copy of the design unit for this pose since flex optimization modifies it
        du_copy = oechem.OEDesignUnit(du)

        # Set the ligand in the design unit copy
        du_copy.SetLigand(oechem.OEMol(pose))

        results = OEProteinLigandOptResults()
        ret_code = optimizer.Optimize(
            results,
            du_copy,
            oechem.OEDesignUnitComponents_Protein,
            oechem.OEDesignUnitComponents_Ligand,
            flex_opts,
        )

        if ret_code == OESzybkiReturnCode_Success:
            initial = results.GetInitialEnergies()
            initial_energy = initial.GetTotalEnergy()
            final = results.GetFinalEnergies()
            final_energy = final.GetTotalEnergy()
            ligand_rmsd = results.GetLigandRMSD()

            # Get the optimized ligand from the design unit
            opt_ligand = oechem.OEMol()
            du_copy.GetLigand(opt_ligand)

            # Update pose coordinates from optimized ligand
            for src_atom, dst_atom in zip(opt_ligand.GetAtoms(), pose.GetAtoms()):
                coords = opt_ligand.GetCoords(src_atom)
                poses.SetCoords(dst_atom, coords)

            OESetSDData(pose, "Initial Energy", str(initial_energy))
            OESetSDData(pose, "Final Energy", str(final_energy))
            OESetSDData(pose, "RMSD", str(ligand_rmsd))
            OESetSDData(pose, "Minimization", "FlexProtein")

            # Track whether to keep this conformer
            if filter_positive_energy and final_energy >= 0:
                positive_energy_count += 1
            else:
                confs_to_keep.append(pose)
                # Keep the last successfully optimized design unit with negative energy
                last_successful_du = du_copy
        else:
            # Mark failed pose so it can be filtered later if needed
            error_name = get_szybki_error_name(ret_code)
            OESetSDData(pose, "Minimization", f"Failed: {error_name}")
            failed_count += 1
            last_error = error_name
            continue

    if failed_count > 0:
        print(
            f"Flex minimization: {failed_count} conformers failed (last error: {last_error})"
        )

    if filter_positive_energy and positive_energy_count > 0:
        print(
            f"Flex minimization: Filtered {positive_energy_count} conformers with positive final energy"
        )

    if not confs_to_keep:
        if filter_positive_energy:
            print(
                "WARNING: No conformers with negative energy remain after flex minimization!"
            )
            return None, None
        else:
            print(
                "Flex minimization: ALL conformers failed! Returning unminimized poses."
            )
            return poses, du

    # Build result molecule with only the conformers to keep
    if filter_positive_energy:
        result_mol = oechem.OEMol(poses)
        result_mol.DeleteConfs()
        for conf in confs_to_keep:
            result_mol.NewConf(conf)
        return result_mol, last_successful_du
    else:
        return poses, last_successful_du


# Legacy aliases for backward compatibility
def minimize_poses_hybrid_flex(
    du: oechem.OEDesignUnit,
    poses: oechem.OEMol,
    flex_range: float = 2.0,
    charge_type: str = CHARGE_TYPE_MMFF94,
) -> Tuple[Optional[oechem.OEMol], Optional[oechem.OEDesignUnit]]:
    """Legacy alias for minimize_poses_flex. Use minimize_poses_flex instead."""
    return minimize_poses_flex(du, poses, flex_range, charge_type)


def minimize_pose_shapefit_flex(
    du: oechem.OEDesignUnit,
    pose: oechem.OEMol,
    flex_range: float = 2.0,
    charge_type: str = CHARGE_TYPE_MMFF94,
) -> Tuple[Optional[oechem.OEMol], Optional[oechem.OEDesignUnit]]:
    """Legacy alias for minimize_poses_flex. Use minimize_poses_flex instead."""
    return minimize_poses_flex(du, pose, flex_range, charge_type)


# =============================================================================
# RMSD Calculation and Pose Selection Functions
# =============================================================================


def calc_rmsd_mcs(
    ref: oechem.OEMolBase,
    fit: oechem.OEMolBase,
) -> float:
    """
    Calculate RMSD using Maximum Common Substructure (MCS) matching.

    This is useful when molecules have different atom counts (e.g., different
    protonation states, tautomers, or minor structural differences).

    Args:
        ref: Reference molecule
        fit: Molecule to compare (will be aligned to ref)

    Returns:
        RMSD of matched atoms, or float('inf') if MCS fails
    """
    # Set up MCS search - use atom and bond expressions for chemistry matching
    atomexpr = oechem.OEExprOpts_DefaultAtoms
    bondexpr = oechem.OEExprOpts_DefaultBonds

    # Create MCS pattern from reference
    mcss = oechem.OEMCSSearch(ref, atomexpr, bondexpr, oechem.OEMCSType_Exhaustive)
    mcss.SetMCSFunc(oechem.OEMCSMaxAtomsCompleteCycles())

    # Find best MCS match
    best_rmsd = float("inf")
    for match in mcss.Match(fit, True):  # uniquify=True
        # Get matched atom pairs
        ref_coords = []
        fit_coords = []

        for mp in match.GetAtoms():
            ref_atom = mp.pattern
            fit_atom = mp.target

            ref_coords.append(ref.GetCoords(ref_atom))
            fit_coords.append(fit.GetCoords(fit_atom))

        if len(ref_coords) < 3:
            continue  # Need at least 3 atoms for meaningful RMSD

        # Calculate RMSD manually from matched coordinates
        ref_arr = np.array(ref_coords)
        fit_arr = np.array(fit_coords)

        # Simple RMSD without overlay (molecules should already be in same frame)
        diff = ref_arr - fit_arr
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

        if rmsd < best_rmsd:
            best_rmsd = rmsd

    return best_rmsd


def calc_rmsd(
    mol1: oechem.OEMolBase,
    mol2: oechem.OEMolBase,
    suppress_hydrogens: bool = True,
) -> float:
    """
    Calculate RMSD between two molecules.

    Handles symmetric molecules by using automorph RMSD. Falls back to
    MCS-based RMSD when molecules have different atom counts (e.g., different
    protonation states or tautomers).

    Args:
        mol1: First molecule (reference)
        mol2: Second molecule (pose)
        suppress_hydrogens: If True, remove hydrogens before RMSD calculation (default: True)

    Returns:
        RMSD value in Angstroms, or float('inf') if calculation fails
    """
    # Work on copies to avoid modifying originals
    ref = oechem.OEMol(mol1)
    fit = oechem.OEMol(mol2)

    # Suppress hydrogens if requested
    if suppress_hydrogens:
        OESuppressHydrogens(ref)
        OESuppressHydrogens(fit)

    # If same atom count, try standard RMSD methods first (faster)
    if ref.NumAtoms() == fit.NumAtoms():
        # Try direct RMSD first (same atom ordering)
        rmsd = oechem.OERMSD(ref, fit)
        if rmsd >= 0:
            return rmsd

        # Try automorph RMSD (handles symmetric molecules)
        rmsd = oechem.OERMSD(ref, fit, True)  # automorph=True
        if rmsd >= 0:
            return rmsd

        # Try with overlay=True which allows for translation/rotation
        rmsd = oechem.OERMSD(ref, fit, True, True)  # automorph=True, overlay=True
        if rmsd >= 0:
            return rmsd

    # Fall back to MCS-based RMSD (handles different atom counts)
    rmsd = calc_rmsd_mcs(ref, fit)
    return rmsd


def select_top_poses_by_rmsd(
    reference: oechem.OEMolBase,
    poses_list: List[oechem.OEMol],
    top_n: int,
) -> List[oechem.OEMol]:
    """
    Select top N poses based on RMSD to a reference ligand.

    Useful for validation against known binding modes or when you want
    poses most similar to a co-crystallized ligand.

    Args:
        reference: Reference molecule (e.g., co-crystallized query ligand)
        poses_list: List of multi-conformer molecules (docked poses)
        top_n: Number of top poses to select

    Returns:
        List of top poses sorted by RMSD to reference (lowest first)
    """
    all_poses_with_rmsd = []

    for mol_idx, mol in enumerate(poses_list):
        if mol is None:
            continue
        # Handle multi-conformer molecules
        for conf_idx, conf in enumerate(mol.GetConfs()):
            # calc_rmsd now handles hydrogen suppression internally
            rmsd = calc_rmsd(
                reference,
                conf,
                suppress_hydrogens=True,
            )
            OESetSDData(conf, "RMSD_to_Reference", f"{rmsd:.4f}")
            all_poses_with_rmsd.append((rmsd, conf))

            if rmsd == float("inf"):
                print(
                    f"  WARNING: RMSD calculation failed for mol {mol_idx}, conf {conf_idx}"
                )

    # Sort by RMSD (ascending - lower is better)
    all_poses_with_rmsd.sort(key=lambda x: x[0])

    valid_count = sum(1 for rmsd, _ in all_poses_with_rmsd if rmsd != float("inf"))
    print(
        f"RMSD calculation summary: {valid_count}/{len(all_poses_with_rmsd)} poses have valid RMSD"
    )

    # Select top N
    return [pose for _, pose in all_poses_with_rmsd[:top_n]]


def cluster_poses_by_rmsd(
    poses: oechem.OEMol,
    eps: float = 1.5,
    min_samples: int = 2,
    include_noise: bool = True,
) -> List[oechem.OEMol]:
    """
    Perform RMSD-based clustering using DBSCAN on docking poses.

    Groups similar poses into clusters and returns the centroid (most
    representative pose) from each cluster. Useful for reducing redundancy
    and getting structurally diverse poses.

    Args:
        poses: Multi-conformer molecule with docking poses
        eps: DBSCAN epsilon parameter - maximum RMSD to be in same cluster (default: 1.5 Å)
        min_samples: Minimum poses to form a cluster (default: 2)
        include_noise: If True, include poses labeled as noise (unclustered).
                       If False, only return cluster centroids (default: True)

    Returns:
        List of representative poses (cluster centroids + optionally noise poses)
    """
    ensemble = [oechem.OEMol(pose) for pose in poses.GetConfs()]
    number_poses = len(ensemble)

    if number_poses == 0:
        return []

    if number_poses == 1:
        return ensemble

    # Build symmetric RMSD distance matrix (only compute upper triangle)
    rmsd_matrix = npz((number_poses, number_poses))
    for i in range(number_poses):
        for j in range(i + 1, number_poses):
            rmsd = oechem.OERMSD(ensemble[i], ensemble[j], True, True, False)
            rmsd_matrix[i, j] = rmsd
            rmsd_matrix[j, i] = rmsd

    # Cluster using DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = db.fit_predict(rmsd_matrix)

    result_poses = []
    unique_labels = set(labels)

    for label in unique_labels:
        if label == -1:
            # Noise points (not in any cluster)
            if include_noise:
                noise_indices = npwhere(labels == label)[0]
                for idx in noise_indices:
                    OESetSDData(ensemble[idx], "RMSD_Cluster", "noise")
                    result_poses.append(ensemble[idx])
        else:
            # Find centroid of this cluster (pose with minimum total RMSD to others)
            indices = npwhere(labels == label)[0]
            submatrix = rmsd_matrix[npix(indices, indices)]
            centroid_idx = indices[npargmin(npsum(submatrix, axis=1))]
            OESetSDData(ensemble[centroid_idx], "RMSD_Cluster", f"centroid_{label}")
            result_poses.append(ensemble[centroid_idx])

    return result_poses


def select_diverse_poses_by_rmsd(
    poses_list: List[oechem.OEMol],
    top_n: int,
    rmsd_threshold: float = 1.5,
) -> List[oechem.OEMol]:
    """
    Select top N structurally diverse poses using greedy RMSD-based selection.

    This is an alternative to DBSCAN clustering that guarantees exactly top_n
    poses are returned (if available). Uses a greedy algorithm:
    1. Start with the first pose (usually best scored)
    2. Add next pose only if it's different enough (RMSD > threshold) from all selected
    3. Repeat until top_n poses selected or no more candidates

    Args:
        poses_list: List of multi-conformer molecules (docked poses)
        top_n: Number of diverse poses to select
        rmsd_threshold: Minimum RMSD between selected poses (default: 1.5 Å)

    Returns:
        List of diverse poses
    """
    # Flatten all poses into a single list
    all_poses = []
    for mol in poses_list:
        if mol is None:
            continue
        for conf in mol.GetConfs():
            all_poses.append(oechem.OEMol(conf))

    if not all_poses:
        return []

    # Greedy selection
    selected = [all_poses[0]]
    OESetSDData(selected[0], "Diversity_Rank", "1")

    for pose in all_poses[1:]:
        if len(selected) >= top_n:
            break

        # Check if this pose is different enough from all selected poses
        is_diverse = True
        for selected_pose in selected:
            rmsd = oechem.OERMSD(pose, selected_pose, True, True, False)
            if rmsd < rmsd_threshold:
                is_diverse = False
                break

        if is_diverse:
            OESetSDData(pose, "Diversity_Rank", str(len(selected) + 1))
            selected.append(pose)

    return selected


# =============================================================================
# Scoring Functions
# =============================================================================


def calc_zap_pbsa_binding_single_pose(
    receptor_du: oechem.OEDesignUnit,
    pose: oechem.OEMol,
    pose_idx: int,
    charge_type: str = CHARGE_TYPE_MMFF94,
) -> Tuple[oechem.OEMol, float, int]:
    """
    Calculate MM/PBSA binding energy for a single pose.

    This function is designed for parallel execution. Charge assignment
    is determined by checking if the molecule already has partial charges
    (via OEHasPartialCharges). If charges exist, they are used as-is.
    Otherwise, charges are assigned based on the charge_type parameter.

    Args:
        receptor_du: Design unit containing the protein
        pose: Single docking pose
        pose_idx: Index of the pose
        charge_type: Charge type for molecules ('mmff94' or 'am1bccelf10'). Default: 'mmff94'

    Returns:
        Tuple of (pose with SD data, binding_energy, pose_idx)
    """
    receptor = oechem.OEMol()
    receptor_du.GetProtein(receptor)
    oechem.OEAssignRadii(receptor, oechem.OERadiiType_Zap9)
    oechem.OEMMFFAtomTypes(receptor)
    # Assign charges to receptor if not already present
    if not oechem.OEHasPartialCharges(receptor):
        oequacpac.OEAssignCharges(receptor, oequacpac.OEMMFF94Charges())

    bind = oezap.OEBind()
    bind.SetProtein(receptor)
    bind.GetZap().SetInnerDielectric(10.0)
    bind.GetZap().SetSaltConcentration(0.1)
    bind.GetZap().SetOuterDielectric(80.0)

    pose_mol = oechem.OEMol(pose)
    oechem.OEAssignRadii(pose_mol, oechem.OERadiiType_Zap9)
    oechem.OEMMFFAtomTypes(pose_mol)
    # Assign charges to ligand if not already present
    if not oechem.OEHasPartialCharges(pose_mol):
        if charge_type.lower() == CHARGE_TYPE_AM1BCCELF10:
            oequacpac.OEAssignCharges(pose_mol, oequacpac.OEAM1BCCELF10Charges())
        else:
            oequacpac.OEAssignCharges(pose_mol, oequacpac.OEMMFF94Charges())

    results = oezap.OEBindResults()
    bind.Bind(pose_mol, results)
    binding_energy = results.GetBindingEnergy()

    OESetSDData(pose_mol, "Zap PBSA Binding Energy", str(binding_energy))
    OESetSDData(pose_mol, "Pose Idx", str(pose_idx))

    return pose_mol, binding_energy, pose_idx


def calc_zap_pbsa_binding_parallel(
    receptor_du: oechem.OEDesignUnit,
    poses_list: List[oechem.OEMol],
    n_jobs: int = 8,
    charge_type: str = CHARGE_TYPE_MMFF94,
) -> List[List[Tuple[oechem.OEMol, float]]]:
    """
    Calculate MM/PBSA binding energies in parallel for all poses.

    Charge assignment is determined by checking if molecules already have
    partial charges (via OEHasPartialCharges). If charges exist, they are
    used as-is. Otherwise, charges are assigned based on the charge_type parameter.

    Args:
        receptor_du: Design unit containing the protein
        poses_list: List of multi-conformer molecules with docking poses
        n_jobs: Number of parallel jobs
        charge_type: Charge type for molecules ('mmff94' or 'am1bccelf10'). Default: 'mmff94'

    Returns:
        List of lists, each containing (pose, energy) tuples for one input molecule
    """
    # Flatten all poses into a list of (mol_idx, pose_idx, pose) tuples
    all_poses = []
    for mol_idx, poses in enumerate(poses_list):
        for conf in poses.GetConfs():
            pose_idx = conf.GetIdx()
            all_poses.append((mol_idx, pose_idx, oechem.OEMol(conf)))

    print(f"  Calculating binding energies for {len(all_poses)} poses...")

    # Run parallel calculation
    parallel = Parallel(n_jobs=n_jobs, return_as="list", prefer="threads", verbose=10)
    results = parallel(
        delayed(calc_zap_pbsa_binding_single_pose)(
            receptor_du, pose, pose_idx, charge_type
        )
        for mol_idx, pose_idx, pose in all_poses
    )

    # Group results by original molecule index
    mol_results = {}
    for (mol_idx, orig_pose_idx, _), (scored_pose, energy, pose_idx) in zip(
        all_poses, results
    ):
        if mol_idx not in mol_results:
            mol_results[mol_idx] = []
        mol_results[mol_idx].append((scored_pose, energy))

    # Return as list of lists maintaining original order
    output_list = []
    for mol_idx in range(len(poses_list)):
        if mol_idx in mol_results:
            output_list.append(mol_results[mol_idx])
        else:
            output_list.append([])

    print(f"  Processed {len(output_list)} molecules with binding energies")
    return output_list


def select_top_poses_fast(
    poses_with_energies: List[Tuple[oechem.OEMol, float, int]],
    top_n: int = 5,
) -> List[oechem.OEMol]:
    """
    Fast selection of top N poses from pre-calculated energies.

    Args:
        poses_with_energies: List of (pose, energy, idx) tuples
        top_n: Number of top poses to select

    Returns:
        List of top N poses
    """
    sorted_poses = sorted(poses_with_energies, key=lambda x: x[1])
    return [pose for pose, energy, idx in sorted_poses[:top_n]]


def select_top_poses_from_list(
    poses_list: List[List[Tuple[oechem.OEMol, float]]],
    top_n: int = 5,
) -> List[oechem.OEMol]:
    """
    Select top N poses from each molecule based on binding energy.

    Args:
        poses_list: List of lists, each containing (pose, energy) tuples
        top_n: Number of top poses per molecule

    Returns:
        Flat list of top poses
    """
    all_top_poses = []

    for pose_energy_list in poses_list:
        # Sort by energy (lower is better)
        sorted_poses = sorted(pose_energy_list, key=lambda x: x[1])
        # Take top N
        top_poses = [pose for pose, energy in sorted_poses[:top_n]]
        all_top_poses.extend(top_poses)

    return all_top_poses


def calc_zap_pbsa_binding(
    du: oechem.OEDesignUnit,
    ligand: oechem.OEMol,
    multiple_poses: bool,
    charge_type: str = CHARGE_TYPE_MMFF94,
) -> oechem.OEMol:
    """
    Calculate Single Point MM/PBSA binding energies from minimized docking poses.

    Charge assignment is determined by checking if molecules already have
    partial charges (via OEHasPartialCharges). If charges exist, they are
    used as-is. Otherwise, charges are assigned based on the charge_type parameter.

    Args:
        du: Design unit containing the protein
        ligand: Docked ligand (single or multi-conformer)
        multiple_poses: If True, calculate for all conformers; if False, single pose
        charge_type: Charge type for molecules ('mmff94' or 'am1bccelf10'). Default: 'mmff94'

    Returns:
        Ligand with binding energy stored as SD data
    """
    receptor = oechem.OEMol()
    du.GetProtein(receptor)
    oechem.OEAssignRadii(receptor, oechem.OERadiiType_Zap9)
    oechem.OEMMFFAtomTypes(receptor)
    # Assign charges to receptor if not already present
    if not oechem.OEHasPartialCharges(receptor):
        oequacpac.OEAssignCharges(receptor, oequacpac.OEMMFF94Charges())

    bind = oezap.OEBind()
    bind.SetProtein(receptor)
    bind.GetZap().SetInnerDielectric(10.0)
    bind.GetZap().SetSaltConcentration(0.1)
    bind.GetZap().SetOuterDielectric(80.0)

    if multiple_poses:
        for pose in ligand.GetConfs():
            pose_idx = pose.GetIdx()
            oechem.OEAssignRadii(pose, oechem.OERadiiType_Zap9)
            oechem.OEMMFFAtomTypes(pose)
            # Assign charges to ligand if not already present
            if not oechem.OEHasPartialCharges(pose):
                if charge_type.lower() == CHARGE_TYPE_AM1BCCELF10:
                    oequacpac.OEAssignCharges(pose, oequacpac.OEAM1BCCELF10Charges())
                else:
                    oequacpac.OEAssignCharges(pose, oequacpac.OEMMFF94Charges())

            results = oezap.OEBindResults()
            bind.Bind(pose, results)
            binding_energy = results.GetBindingEnergy()
            OESetSDData(pose, "Zap PBSA Binding Energy", str(binding_energy))
            OESetSDData(pose, "Pose Idx", str(pose_idx))
    else:
        oechem.OEAssignRadii(ligand, oechem.OERadiiType_Zap9)
        oechem.OEMMFFAtomTypes(ligand)
        # Assign charges to ligand if not already present
        if not oechem.OEHasPartialCharges(ligand):
            if charge_type.lower() == CHARGE_TYPE_AM1BCCELF10:
                oequacpac.OEAssignCharges(ligand, oequacpac.OEAM1BCCELF10Charges())
            else:
                oequacpac.OEAssignCharges(ligand, oequacpac.OEMMFF94Charges())

        results = oezap.OEBindResults()
        bind.Bind(ligand, results)
        binding_energy = results.GetBindingEnergy()
        OESetSDData(ligand, "Zap PBSA Binding Energy", str(binding_energy))

    return ligand


def select_top_poses_mm_pbsa(poses: oechem.OEMol, top_n: int = 5) -> List[oechem.OEMol]:
    """
    Select top N best docking poses based on Zap MM/PBSA binding energy.

    Args:
        poses: Multi-conformer molecule with docking poses and binding energies
        top_n: Number of top poses to select (default: 5)

    Returns:
        List of top N poses sorted by binding energy
    """
    energies, idx = [], []
    for pose in poses.GetConfs():
        energy = float(oechem.OEGetSDData(pose, "Zap PBSA Binding Energy"))
        pose_idx = int(oechem.OEGetSDData(pose, "Pose Idx"))
        energies.append(energy)
        idx.append(pose_idx)

    df = pd.DataFrame({"Pose Idx": idx, "Zap PBSA Binding Energy": energies})
    sorted_df = df.sort_values(by="Zap PBSA Binding Energy", ascending=True)
    selected_indices = sorted_df.iloc[:top_n]

    top_poses = []
    for selected_idx in selected_indices["Pose Idx"]:
        selected_pose = poses.GetConf(oechem.OEHasConfIdx(selected_idx))
        top_poses.append(oechem.OEMol(selected_pose))

    return top_poses


# =============================================================================
# Record Saving Functions
# =============================================================================


def save_record(
    pose: oechem.OEMol, method: str, multiple_poses: bool
) -> oechem.OERecord:
    """
    Save a docking pose to an OERecord.

    Args:
        pose: Docked pose
        method: Docking method used ('HYBRID' or 'SHAPEFIT')
        multiple_poses: If True, extract first conformer; if False, use pose directly

    Returns:
        OERecord containing the pose and metadata
    """
    record = oechem.OERecord()
    title = pose.GetTitle()

    if not multiple_poses:
        record.set_value(mol_field, oechem.OEMol(pose))
    else:
        pose_conf = pose.GetConf(oechem.OEHasConfIdx(0))
        record.set_value(mol_field, oechem.OEMol(pose_conf))

    record.set_value(title_field, title)
    record.set_value(method_field, method)
    return record


# =============================================================================
# Others
# =============================================================================


def merge_ligand_with_pdb(
    protein_file: str,
    ligand_file: str,
    output_file: str,
    ligand_res_name: str = "LIG",
    ligand_chain_id: str = "L",
) -> str:
    """
    Merge a ligand file (SDF/MOL/PDB) with a protein structure file (PDB/CIF).

    Uses RDKit to read the ligand and biotite to read/write the protein structure.
    The ligand is added to the protein as a new residue with the specified name
    and chain ID.

    Args:
        protein_file: Path to input protein file (PDB or CIF)
        ligand_file: Path to input ligand file (SDF, MOL, or PDB)
        output_file: Path to output merged PDB file
        ligand_res_name: Residue name for the ligand (default: "LIG")
        ligand_chain_id: Chain ID for the ligand (default: "L")

    Returns:
        Path to the output merged PDB file

    Raises:
        FileNotFoundError: If input files don't exist
        ValueError: If ligand cannot be parsed or has no 3D coordinates
    """
    # Validate input files
    if not os.path.exists(protein_file):
        raise FileNotFoundError(f"Protein file not found: {protein_file}")
    if not os.path.exists(ligand_file):
        raise FileNotFoundError(f"Ligand file not found: {ligand_file}")

    # Load protein structure with biotite
    protein_path = Path(protein_file)
    if protein_path.suffix.lower() == ".cif":
        cif_file_obj = pdbx.CIFFile.read(str(protein_file))
        protein_array = pdbx.get_structure(
            cif_file_obj,
            model=1,
            include_bonds=True,
        )
    else:
        pdb_file_obj = pdb.PDBFile.read(str(protein_file))
        protein_array = pdb.get_structure(
            pdb_file_obj,
            model=1,
            extra_fields=["charge"],
            include_bonds=True,
        )

    # Load ligand with RDKit
    ligand_path = Path(ligand_file)
    suffix = ligand_path.suffix.lower()

    if suffix in [".sdf", ".mol"]:
        # Read SDF/MOL file
        supplier = Chem.SDMolSupplier(str(ligand_file), removeHs=False)
        rdkit_mol = next(supplier, None)
    elif suffix == ".pdb":
        rdkit_mol = Chem.MolFromPDBFile(str(ligand_file), removeHs=False)
    elif suffix == ".mol2":
        rdkit_mol = Chem.MolFromMol2File(str(ligand_file), removeHs=False)
    else:
        raise ValueError(f"Unsupported ligand file format: {suffix}")

    if rdkit_mol is None:
        raise ValueError(f"Failed to parse ligand file: {ligand_file}")

    # Get 3D coordinates from RDKit molecule
    conformer = rdkit_mol.GetConformer()
    if conformer is None:
        raise ValueError("Ligand has no 3D conformer")

    num_atoms = rdkit_mol.GetNumAtoms()
    if num_atoms == 0:
        raise ValueError("Ligand has no atoms")

    # Create biotite AtomArray for ligand
    ligand_array = struc.AtomArray(num_atoms)

    # Get the next residue ID (one more than max in protein)
    max_res_id = np.max(protein_array.res_id) if len(protein_array) > 0 else 0
    ligand_res_id = max_res_id + 1

    # Fill atom properties
    coords = np.array([conformer.GetAtomPosition(i) for i in range(num_atoms)])
    ligand_array.coord = coords

    for i, atom in enumerate(rdkit_mol.GetAtoms()):
        # Get element symbol
        element = atom.GetSymbol()
        ligand_array.element[i] = element

        # Get atom name - use PDB-style naming if available, otherwise generate
        pdb_info = atom.GetPDBResidueInfo()
        if pdb_info is not None:
            atom_name = pdb_info.GetName().strip()
        else:
            # Generate atom name: element + index (e.g., C1, C2, N1, O1)
            atom_name = f"{element}{i + 1}"
            # PDB atom names should be <= 4 chars, left-padded for short elements
            if len(atom_name) < 4:
                atom_name = f" {atom_name}" if len(element) == 1 else atom_name

        ligand_array.atom_name[i] = atom_name[:4]  # Truncate to 4 chars max
        ligand_array.res_name[i] = ligand_res_name[:3]  # 3 char max
        ligand_array.chain_id[i] = ligand_chain_id
        ligand_array.res_id[i] = ligand_res_id
        ligand_array.hetero[i] = True  # Ligand atoms are HETATM

    # Set B-factors and occupancies to default values
    ligand_array.set_annotation("b_factor", np.zeros(num_atoms))
    ligand_array.set_annotation("occupancy", np.ones(num_atoms))

    # Concatenate protein and ligand arrays
    complex_array = protein_array + ligand_array

    # Write merged structure to PDB file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdb_file_out = pdb.PDBFile()
    pdb.set_structure(pdb_file_out, complex_array)
    pdb_file_out.write(str(output_file))

    print(
        f"Merged {num_atoms} ligand atoms ({ligand_res_name}) "
        f"with protein ({len(protein_array)} atoms) -> {output_file}"
    )

    return output_file


def write_sdf_file(sdf_path, molecules, name="mol"):
    w = Chem.SDWriter(str(sdf_path))
    if len(molecules) == 1:
        if name:
            molecules[0].SetProp("_Name", name)
        if molecules[0] is not None:
            w.write(molecules[0])
        w.close()
    else:
        for i, m in enumerate(molecules):
            if name:
                m.SetProp("_Name", f"{name}_{i}")
            if m is not None:
                w.write(m)
        w.close()


def split_ligand_from_pdb(
    in_file: Optional[str] = None,
    pdb_id: Optional[str] = None,
    out_file: str = "protein_only.pdb",
    exclude_res: Optional[set] = ("LIG",),
    exclude_other: bool = False,
    save_ligand: bool = False,
    ligand_out_file: str = "ligand_only.sdf",
) -> None:
    """
    Split ligand from protein structure in PDB file and save protein-only structure.

    Args:
        in_file: Path to input PDB file with protein-ligand complex
        pdb_id: PDB ID to download structure from RCSB (if in_file not provided)
        out_file: Path to output file for protein-only structure (PDB or CIF)
        exclude_res: Set of residue names to exclude (default: {"LIG"})
    Returns:
        None
    """
    # Load protein into biotite atom array, split and save as new file
    assert in_file or pdb_id, "Either in_file or pdb_id must be provided"

    ## Get standard amino acid names first
    prot_seq = seq.ProteinSequence()
    aa_alphabet = prot_seq.get_alphabet()
    aa_names = [prot_seq.convert_letter_1to3(x) for x in aa_alphabet]

    # Common solvent/ion residue names to keep
    keep_residues = {
        "HOH",
        "WAT",
        "H2O",  # water
    }
    if not exclude_other:
        keep_residues.update(
            {
                "NA",
                "CL",
                "K",
                "MG",
                "CA",
                "ZN",
                "FE",
                "MN",
                "CU",
                "NI",
                "CO",  # ions
                "SO4",
                "PO4",
                "ACE",
                "NME",  # common caps/cofactors
                "SEP",
                "TPO",
                "PTR",  # phosphorylated residues
            }
        )

    # Download PDB file if pdb_id is provided
    if pdb_id and not in_file:
        print(f"Downloading PDB structure for ID: {pdb_id}")
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = rcsb.fetch(pdb_id, "cif", target_path=tmpdir)
            pdbx_file = pdbx.CIFFile.read(file_path)
            atom_array = pdbx.get_structure(pdbx_file, model=1, include_bonds=True)
    else:
        ## Load protein
        in_path = Path(in_file)
        if in_path.suffix.lower() == ".cif":
            cif_file_obj = pdbx.CIFFile.read(str(in_file))
            atom_array = pdbx.get_structure(
                cif_file_obj,
                model=1,
                include_bonds=True,
            )
        else:
            pdb_file_obj = pdb.PDBFile.read(str(in_file))
            extra = ["charge"]
            atom_array = pdb.get_structure(
                pdb_file_obj,
                model=1,
                extra_fields=extra,
                include_bonds=True,
            )

    res_names = np.unique(atom_array.res_name)

    # Get all non-standard amino acids (water ligand, ...)
    nonstd_res = [x for x in res_names if x not in aa_names]
    print(f"Non-standard residues found in the protein: {nonstd_res}")

    # Mask to keep protein and others, no ligand
    if exclude_res is None:
        # Auto-detect ligand residues if not specified
        exclude_res = set()
        for res in nonstd_res:
            if res not in keep_residues:
                exclude_res.add(res)
        print(f"Auto-detected ligand residues to exclude: {exclude_res}")

    mask_prot = np.ones(atom_array.res_name.shape, dtype=bool)
    for res in exclude_res:
        mask_prot &= atom_array.res_name != res
    prot_atom_array = atom_array[mask_prot]

    if not exclude_res:
        print(
            "Warning: No ligand residues identified to exclude. Output will be identical to input."
        )

    # Check if we have any atoms left
    if len(prot_atom_array) == 0:
        raise ValueError(
            f"No atoms remaining after excluding residues {exclude_res}. "
            "Check that the input file contains protein atoms."
        )

    print(f"Kept {len(prot_atom_array)} atoms after excluding ligand residues")

    # Check water molecules specifically
    water_mask = np.isin(prot_atom_array.res_name, ["HOH", "WAT", "H2O"])
    if water_mask.sum() > 0:
        print(f"Found {water_mask.sum()} water molecules in protein structure.")
        print("Water chain IDs:", np.unique(prot_atom_array.chain_id[water_mask]))
        print("Water res IDs:", np.unique(prot_atom_array.res_id[water_mask]))

        # The water likely has empty chain IDs - assign them a chain, otherwise this causes issues downstream
        empty_chain_mask = water_mask & (prot_atom_array.chain_id == "")
        if empty_chain_mask.sum() > 0:
            prot_atom_array.chain_id[empty_chain_mask] = "W"

    # Save as CIF file
    if out_file.endswith(".cif"):
        cif_file = pdbx.CIFFile()
        pdbx.set_structure(cif_file, prot_atom_array)
        cif_file.write(out_file)
    else:
        # Save as PDB file
        pdb_file = pdb.PDBFile()
        pdb.set_structure(pdb_file, prot_atom_array)
        pdb_file.write(out_file)

    if save_ligand:
        # Save ligand-only structure
        mask_ligand = np.zeros(atom_array.res_name.shape, dtype=bool)
        for res in exclude_res:
            mask_ligand |= atom_array.res_name == res
        ligand_atom_array = atom_array[mask_ligand]
        lig_mol = rdkit.to_mol(ligand_atom_array)
        write_sdf_file(ligand_out_file, [lig_mol])
        print(
            f"Saved ligand-only structure with {len(ligand_atom_array)} atoms to {ligand_out_file}"
        )


def split_other_from_pdb(
    in_file: str,
    out_file: str = "complex_clean.pdb",
    keep_ligand: bool = True,
) -> None:
    """
    Remove 'other' components (ions, cofactors, etc.) from PDB file.

    Keeps only protein, water, and optionally ligand. Removes common cofactors,
    ions, and other non-essential components that may interfere with structure
    preparation.

    Args:
        in_file: Path to input PDB/CIF file with protein-ligand complex
        out_file: Path to output file for cleaned structure (PDB or CIF)
        keep_ligand: Whether to keep ligand residues (default: True)
    Returns:
        None
    """
    # Get standard amino acid names
    prot_seq = seq.ProteinSequence()
    aa_alphabet = prot_seq.get_alphabet()
    aa_names = set(prot_seq.convert_letter_1to3(x) for x in aa_alphabet)

    # Water residue names to always keep
    water_residues = {"HOH", "WAT", "H2O"}

    # Common cofactors, ions, and other components to REMOVE
    # This is the inverse of split_ligand_from_pdb - we want to remove these
    other_residues_to_remove = {
        # Common ions
        "NA",
        "CL",
        "K",
        "MG",
        "CA",
        "ZN",
        "FE",
        "MN",
        "CU",
        "NI",
        "CO",
        "NA+",
        "CL-",
        "K+",
        "MG2",
        "CA2",
        "ZN2",
        "FE2",
        "FE3",
        "MN2",
        "CU2",
        "NI2",
        "CO2",
        # Common buffers and additives
        "SO4",
        "PO4",
        "ACT",
        "GOL",
        "EDO",
        "PEG",
        "DMS",
        "BME",
        "TRS",
        "CIT",
        "MES",
        "HEP",
        "EPE",
        "MPD",
        "PGE",
        "P6G",
        "1PE",
        "2PE",
        "XPE",
        "IMD",
        "IPA",
        "EOH",
        "ACE",
        "NME",
        "FOR",
        "NH4",
        "SCN",
        # Common cofactors
        "NAD",
        "NAP",
        "FAD",
        "FMN",
        "ADP",
        "ATP",
        "GDP",
        "GTP",
        "AMP",
        "GMP",
        "COA",
        "SAM",
        "SAH",
        "HEM",
        "HEC",
        "HEA",
        "HEB",
        "PLP",
        "TPP",
        "BTN",
        "B12",
        # Phosphorylated residues (keep these as they are modified amino acids)
        # "SEP", "TPO", "PTR",
        # Detergents
        "LDA",
        "SDS",
        "DXC",
        "BOG",
        "LMT",
        "OLC",
        "PLM",
        "MYR",
        # Cryoprotectants
        "GLY",
        "XYL",
        "SUC",
        "TRE",
        "MLI",
        # Other common crystallization additives
        "NO3",
        "BR",
        "I",
        "F",
        "IOD",
        "OXY",
    }

    # Load structure
    in_path = Path(in_file)
    if in_path.suffix.lower() == ".cif":
        cif_file_obj = pdbx.CIFFile.read(str(in_file))
        atom_array = pdbx.get_structure(
            cif_file_obj,
            model=1,
            include_bonds=True,
        )
    else:
        pdb_file_obj = pdb.PDBFile.read(str(in_file))
        extra = ["charge"]
        atom_array = pdb.get_structure(
            pdb_file_obj,
            model=1,
            extra_fields=extra,
            include_bonds=True,
        )

    res_names = np.unique(atom_array.res_name)

    # Get all non-standard residues
    nonstd_res = [x for x in res_names if x not in aa_names]
    print(f"Non-standard residues found: {nonstd_res}")

    # Identify what to remove vs keep
    residues_to_remove = set()
    ligand_residues = set()

    for res in nonstd_res:
        if res in water_residues:
            continue  # Always keep water
        elif res in other_residues_to_remove:
            residues_to_remove.add(res)
        else:
            # This is likely a ligand
            ligand_residues.add(res)

    print(f"Identified ligand residues: {ligand_residues}")
    print(f"'Other' residues to remove: {residues_to_remove}")

    # If not keeping ligand, add ligand residues to removal set
    if not keep_ligand:
        residues_to_remove.update(ligand_residues)
        print("Also removing ligand residues (keep_ligand=False)")

    # Create mask to keep everything except residues_to_remove
    mask_keep = np.ones(atom_array.res_name.shape, dtype=bool)
    for res in residues_to_remove:
        mask_keep &= atom_array.res_name != res

    clean_atom_array = atom_array[mask_keep]

    removed_count = len(atom_array) - len(clean_atom_array)
    print(f"Removed {removed_count} atoms from 'other' components")
    print(f"Kept {len(clean_atom_array)} atoms (protein + water + ligand)")

    # Check if we have any atoms left
    if len(clean_atom_array) == 0:
        raise ValueError(
            "No atoms remaining after removing 'other' components. "
            "Check that the input file contains protein/ligand atoms."
        )

    # Handle water chain IDs (assign chain 'W' if empty)
    water_mask = np.isin(clean_atom_array.res_name, list(water_residues))
    if water_mask.sum() > 0:
        empty_chain_mask = water_mask & (clean_atom_array.chain_id == "")
        if empty_chain_mask.sum() > 0:
            clean_atom_array.chain_id[empty_chain_mask] = "W"

    # Save output
    if out_file.endswith(".cif"):
        cif_file = pdbx.CIFFile()
        pdbx.set_structure(cif_file, clean_atom_array)
        cif_file.write(out_file)
    else:
        pdb_file = pdb.PDBFile()
        pdb.set_structure(pdb_file, clean_atom_array)
        pdb_file.write(out_file)

    print(f"Cleaned structure saved to {out_file}")

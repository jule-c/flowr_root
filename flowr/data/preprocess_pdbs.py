import argparse
import os
import subprocess
import tempfile
from glob import glob
from itertools import zip_longest
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import hydride
import numpy as np
import torch
from Bio.PDB.Polypeptide import is_aa
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from flowr.constants import AFFINITY_PROP_NAMES
from flowr.data.preprocess_pdb import transform_pdb
from flowr.util.molrepr import GeometricMol
from flowr.util.pocket import (
    PROLIF_INTERACTIONS,
    BindingInteractions,
    PocketComplex,
    PocketComplexBatch,
    ProteinPocket,
)
from posecheck.utils.biopython import (
    ids_scriptly_increasing,
    load_biopython_structure,
    remove_connect_lines,
    reorder_ids,
    save_biopython_structure,
)

"""
This script is used to preprocess data that comes as PDB (protein), SDF (ligand) and optionally txt files (residues in a given radius).
"""


def has_explicit_hydrogens(mol: Chem.rdchem.Mol) -> bool:
    """Check whether an RDKit molecule has explicit hydrogen atoms

    Args:
        mol (Chem.Mol): RDKit molecule object

    Returns:
        bool: True if the molecule has explicit hydrogen atoms, False otherwise
    """
    if mol is None:
        return False

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:  # Hydrogen has atomic number 1
            return True

    return False


def add_and_optimize_ligand_hs(
    ligand: [str, Chem.Mol],
    pocket_pdb_path: str,
    add_hs: bool = True,
    maxIters: int = 200,
) -> Chem.Mol:
    """
    Load a protein pocket (with hydrogens!) from a PDB file,
    add hydrogens to the ligand, then run a partial minimization
    where only the ligand's hydrogens are free to move.

    Args:
        ligand (Chem.Mol): The input RDKit molecule for the ligand.
        pocket_pdb_path (str): Path to the protein PDB file (which contains hydrogens).
        maxIters (int): Max iterations for the force-field minimization.

    Returns:
        Chem.Mol: The input ligand with updated (optimized) hydrogen coordinates.
    """

    protein = Chem.MolFromPDBFile(str(pocket_pdb_path), removeHs=False)
    if protein is None:
        protein = Chem.MolFromPDBFile(
            str(pocket_pdb_path), removeHs=False, proximityBonding=False
        )
        if protein is None:
            print(
                f"Could not read PDB file at: {str(pocket_pdb_path)}. Skipping hydrogen optimization."
            )
            return None

    if isinstance(ligand, str):
        # If ligand is a file path, read it
        ligand = Chem.SDMolSupplier(ligand, removeHs=False)[0]
        if ligand is None:
            print(
                f"Could not read ligand from {ligand}. Skipping hydrogen optimization."
            )
            return None
    ligand_with_H = Chem.AddHs(ligand, addCoords=True) if add_hs else Chem.Mol(ligand)

    # Combine ligand and pocket
    combined = Chem.CombineMols(protein, ligand_with_H)
    protein_num_atoms = protein.GetNumAtoms()
    ligand_num_atoms = ligand_with_H.GetNumAtoms()
    combined_mol = Chem.RWMol(combined)
    conf = Chem.Conformer(combined_mol.GetNumAtoms())
    prot_conf = protein.GetConformer()
    for i in range(protein_num_atoms):
        x, y, z = (
            prot_conf.GetAtomPosition(i).x,
            prot_conf.GetAtomPosition(i).y,
            prot_conf.GetAtomPosition(i).z,
        )
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(x, y, z))

    lig_conf = ligand_with_H.GetConformer()
    for i in range(ligand_num_atoms):
        x, y, z = (
            lig_conf.GetAtomPosition(i).x,
            lig_conf.GetAtomPosition(i).y,
            lig_conf.GetAtomPosition(i).z,
        )
        conf.SetAtomPosition(protein_num_atoms + i, Chem.rdGeometry.Point3D(x, y, z))

    combined_mol.AddConformer(conf)
    combined_mol = combined_mol.GetMol()

    # Set up the MMFF force field
    mp = AllChem.MMFFGetMoleculeProperties(combined_mol, mmffVariant="MMFF94s")
    if mp is None:
        print(
            "Could not get MMFF properties for combined molecule. Skipping hydrogen optimization."
        )
        return
    ff = AllChem.MMFFGetMoleculeForceField(combined_mol, mp)

    # Fix all atoms except for the ligand's hydrogens.
    for atom_idx in range(combined_mol.GetNumAtoms()):
        if atom_idx < protein_num_atoms:
            ff.AddFixedPoint(atom_idx)
        else:
            lig_atom = combined_mol.GetAtomWithIdx(atom_idx)
            if lig_atom.GetAtomicNum() != 1:
                ff.AddFixedPoint(atom_idx)

    # Minimize only the free atoms
    try:
        ff.Initialize()
        ff.Minimize(maxIters)
    except Exception as e:
        print(f"Minimization failed: {e}. Skipping hydrogen optimization.")
        return

    # Extract the updated coordinates for the ligand
    new_coords = combined_mol.GetConformer().GetPositions()
    for i in range(ligand_num_atoms):
        x, y, z = new_coords[protein_num_atoms + i]
        ligand_with_H.GetConformer().SetAtomPosition(
            i, Chem.rdGeometry.Point3D(x, y, z)
        )

    return ligand_with_H


def save_systems_(args, systems, split):
    batch = PocketComplexBatch(systems)
    save_dir = Path(args.save_path)
    save_dir.mkdir(exist_ok=True, parents=True)
    save_file = save_dir / f"{split}.smol"
    bytes_data = batch.to_bytes()
    save_file.write_bytes(bytes_data)
    print(f"Saved {len(systems)} systems to {save_file.resolve()}")


def canonicalize_atom_order(mol_conformer):
    """
    Renumber conformer atoms to match canonical SMILES atom ordering.
    Preserves 3D coordinates and explicit Hs.
    """
    # Generate SMILES with explicit Hs
    smiles = Chem.MolToSmiles(mol_conformer, allHsExplicit=True)
    mol_canonical = Chem.MolFromSmiles(smiles)

    # Get mapping
    match = mol_conformer.GetSubstructMatch(mol_canonical)

    if not match or len(match) != mol_canonical.GetNumAtoms():
        raise ValueError(
            f"Substructure match failed. Got {len(match)} of {mol_canonical.GetNumAtoms()} atoms"
        )

    mol_reordered = Chem.RenumberAtoms(mol_conformer, list(match))

    return mol_reordered


def process_pdb(
    pdb_file: str,
    txt_path: str = None,
    ligand: GeometricMol = None,
    add_bonds_to_protein: bool = True,
    add_hs_to_protein: bool = False,
    pocket_cutoff: float = 6.0,
    cut_pocket: bool = False,
    use_pdbfixer: bool = False,
):
    """
    Load a protein from a PDB file, optionally add hydrogens,
    and cut out a pocket around a given ligand.
    Args:
        pdb_file (str): Path to the PDB file.
        txt_path (str, optional): Path to a text file with residue IDs to keep.
        ligand (GeometricMol, optional): The ligand to cut out the pocket around.
        add_bonds_to_protein (bool): Whether to add bonds to the protein structure.
        add_hs_to_protein (bool): Whether to add hydrogens to the protein structure.
        pocket_cutoff (float): Cutoff distance for pocket extraction.
        cut_pocket (bool): Whether to cut out the pocket from the protein.
        use_pdbfixer (bool): Whether to use PDBFixer for fixing the PDB file.
    Returns:
        ProteinPocket: The extracted pocket as a ProteinPocket object.
    """

    if cut_pocket:
        assert ligand is not None, "Ligand must be provided to cut out pocket"

    fixed_pdb_file = pdb_file  # Default to original file
    if use_pdbfixer:
        from openmm.app import PDBFile
        from pdbfixer import PDBFixer

        # Create temporary file for the fixed PDB
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp_file:
            fixed_pdb_file = tmp_file.name

        fixer = PDBFixer(filename=pdb_file)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(True)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        if add_hs_to_protein:
            fixer.addMissingHydrogens(7.4)
        PDBFile.writeFile(fixer.topology, fixer.positions, open(fixed_pdb_file, "w"))

    name = Path(pdb_file).stem
    # Load protein from PDB (use fixed file if PDBFixer was used)
    if str(fixed_pdb_file).endswith(".cif"):
        pdb_file_obj = pdbx.CIFFile.read(str(fixed_pdb_file))
        read_fn = pdbx.get_structure
    elif str(fixed_pdb_file).endswith(".pdb"):
        pdb_file_obj = pdb.PDBFile.read(str(fixed_pdb_file))
        read_fn = pdb.get_structure
    else:
        print(
            f"Unsupported file format for {fixed_pdb_file}. Only PDB and CIF are supported."
        )
        if use_pdbfixer and fixed_pdb_file != pdb_file:
            if os.path.exists(fixed_pdb_file):
                os.remove(fixed_pdb_file)
        return

    # Sometimes reading charges fails so try to read first but fall back to without charge if not
    try:
        extra = ["charge"]
        structure = read_fn(
            pdb_file_obj,
            model=1,
            extra_fields=extra,
            include_bonds=add_bonds_to_protein,
        )
    except Exception:
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp_pdb:
                tmp_pdb_file = tmp_pdb.name
            structure = load_biopython_structure(str(fixed_pdb_file))
            if not ids_scriptly_increasing(structure):
                structure = reorder_ids(structure)
            save_biopython_structure(structure, tmp_pdb_file)
            pdb_file_obj = pdb.PDBFile.read(str(tmp_pdb_file))
            read_fn = pdb.get_structure
            extra = ["charge"]
            structure = read_fn(
                pdb_file_obj,
                model=1,
                extra_fields=extra,
                include_bonds=add_bonds_to_protein,
            )
            os.remove(tmp_pdb_file)
        except Exception as e:
            print(f"Error reading {fixed_pdb_file}: {e}")
            return

    # Add bonds if they couldn't be retrieved
    if add_bonds_to_protein:
        if (
            structure.bonds is None
            or len(structure.bonds.as_array()) == 0
            or sum(structure.bonds.as_array()[:, -1]) == 0
        ):
            bonds = struc.connect_via_residue_names(structure, inter_residue=True)
            if (
                bonds is None
                or len(bonds.as_array()) == 0
                or sum(bonds.as_array()[:, -1]) == 0
            ):
                try:
                    rdmol = Chem.MolFromPDBFile(
                        str(fixed_pdb_file), removeHs=False, proximityBonding=True
                    )
                except Exception:
                    try:
                        rdmol = Chem.MolFromPDBFile(
                            str(fixed_pdb_file), removeHs=False, proximityBonding=False
                        )
                    except Exception:
                        rdmol = None

                if rdmol is None:
                    print(f"Couldn't determine bond order for {name}. Skipping.")
                    if use_pdbfixer and fixed_pdb_file != pdb_file:
                        os.remove(fixed_pdb_file)
                    return

                from flowr.util.rdkit import retrieve_bonds_from_mol

                bonds = np.array(retrieve_bonds_from_mol(rdmol, infer_bonds=False))
                bonds = struc.BondList(rdmol.GetNumAtoms(), bonds)
            structure.bonds = bonds

    # Add hydrogens and optimize
    if add_hs_to_protein and not use_pdbfixer:
        structure, mask = hydride.add_hydrogen(structure)
        structure.coord = hydride.relax_hydrogen(structure)

    if txt_path is not None:
        with open(txt_path, "r") as f:
            ids = f.read().split()
            try:
                holo_residue_ids = [int(res.split(":")[-1]) for res in ids]
            except ValueError:
                try:
                    holo_residue_ids = [int(res.split(":")[0]) for res in ids]
                except ValueError:
                    try:
                        holo_residue_ids = [int(res.split(".")[0]) for res in ids]
                    except ValueError:
                        holo_residue_ids = [int(res.split(".")[-1]) for res in ids]

            # holo_chain_id = ids[0].split(":")[0]
        structure = structure[np.isin(structure.res_id, holo_residue_ids)]

    if cut_pocket:
        # Cut pocket
        ligand_coords = np.array(ligand._coords)
        distances = np.linalg.norm(
            structure.coord[:, None, :] - ligand_coords[None, :, :], axis=-1
        )
        atoms_in_pocket = (distances < pocket_cutoff).any(axis=1)
        chains_in_pocket = list(structure.chain_id[atoms_in_pocket])
        chains_in_pocket = [chain for chain in chains_in_pocket if len(chain) > 0]

        structure = structure[np.isin(structure.chain_id, chains_in_pocket)]

        # Create unique identifiers combining chain_id and res_id
        chain_res_pairs = set(zip(structure.chain_id, structure.res_id))
        res_filter_mask = np.zeros(len(structure), dtype=bool)

        for chain_id, res_id in chain_res_pairs:
            # Get residue atoms for this specific chain-residue combination
            res_mask = (structure.chain_id == chain_id) & (structure.res_id == res_id)
            res = structure[res_mask]

            if (
                is_aa(res.res_name[0], standard=True)
                and (
                    np.linalg.norm(
                        res.coord[:, None, :] - np.array(ligand._coords[None, :, :]),
                        axis=-1,
                    )
                ).min()
                < pocket_cutoff
            ):
                # Mark all atoms of this chain-residue combination for inclusion
                res_filter_mask |= res_mask

        # Filter structure using the boolean mask
        structure = structure[res_filter_mask]

    # Clean up temporary files if they were created
    if use_pdbfixer and fixed_pdb_file != pdb_file:
        if os.path.exists(fixed_pdb_file):
            os.remove(fixed_pdb_file)

    pocket = ProteinPocket.from_pocket_atoms(structure)
    return pocket


def load_protein_prolif(protein_path: str):
    import MDAnalysis as mda
    import prolif as plf

    """Load protein from PDB file using MDAnalysis
    and convert to plf.Molecule. Assumes hydrogens are present."""
    prot = mda.Universe(protein_path)
    prot = plf.Molecule.from_mda(prot, NoImplicit=False)
    return prot


def load_protein_from_pdb(pdb_path: str, add_hs: bool = False):
    """Load protein from PDB file, add hydrogens, and convert it to a prolif.Molecule.

    Args:
        pdb_path (str): The path to the PDB file.

    Returns:
        plf.Molecule: The loaded protein as a prolif.Molecule.
    """

    tmp_path = tempfile.mkstemp()[1] + ".pdb"
    tmp_protonated_path = tempfile.mkstemp()[1] + ".pdb"

    # Reorder residue IDs if necessary
    structure = load_biopython_structure(pdb_path)
    if not ids_scriptly_increasing(structure):
        structure = reorder_ids(structure)
    save_biopython_structure(structure, tmp_path)  # Save reordered structure

    if add_hs:
        # Run Hydride
        cmd = f"hydride -i {tmp_path} -o {tmp_protonated_path}"
        out = subprocess.run(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Check if Hydride failed
        if out.returncode != 0:
            print(out.stdout.decode())
            print(out.stderr.decode())
            raise Exception("Hydride failed")

        # - Remove CONECT lines from the PDB file - #
        # This is necessary because the CONECT lines are not handled correctly by MDAnalysis
        # and they are for some reason added by Hydride
        remove_connect_lines(tmp_protonated_path)
    else:
        tmp_protonated_path = tmp_path

    # Load the protein from the temporary PDB file
    prot = load_protein_prolif(tmp_protonated_path)
    os.remove(tmp_protonated_path)

    return prot


def convert_dative_bonds(mol):
    """Convert dative bonds to single bonds in an RDKit molecule.

    Args:
        mol: RDKit molecule object

    Returns:
        Modified RDKit molecule with dative bonds converted to single bonds
    """
    # Create a copy of the molecule to avoid modifying the original
    mol = Chem.Mol(mol)

    # Keep converting until no dative bonds remain
    max_iterations = 3  # Safety limit to prevent infinite loops
    iteration = 0

    while iteration < max_iterations:
        emol = Chem.EditableMol(mol)
        bonds_to_modify = []

        # Find all dative bonds
        for bond in mol.GetBonds():
            if (
                bond.GetBondType() == Chem.BondType.DATIVE
                or bond.GetBondType() == Chem.rdchem.BondType.DATIVE
            ):
                bonds_to_modify.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

        # If no dative bonds found, we're done
        if not bonds_to_modify:
            break

        # Remove dative bonds and add single bonds
        for begin_idx, end_idx in reversed(
            bonds_to_modify
        ):  # reverse to maintain indices
            emol.RemoveBond(begin_idx, end_idx)
            emol.AddBond(begin_idx, end_idx, Chem.BondType.SINGLE)

        mol = emol.GetMol()
        iteration += 1

    # Try to sanitize the modified molecule
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        try:
            # If sanitization fails, try to fix the molecule
            Chem.SanitizeMol(mol)
        except Exception as e:
            print(f"Warning: Sanitization failed: {e}")
            pass

    return mol


def safe_geometric_mol_from_rdkit(
    mol,
    ligand_info: dict = None,
    add_hs: bool = False,
    remove_hs: bool = False,
    sanitise: bool = True,
    repeated_sanitise: bool = False,
    kekulize: bool = False,
    min_atoms: int = 4,
    mol_source: str = "unknown",
):
    """Safely create GeometricMol from RDKit molecule, handling dative bonds.

    Args:
        mol: RDKit molecule object
        ligand_info: Dictionary containing potential_energy, atomic_forces, etc.
        add_hs: Whether to add hydrogens to the molecule
        kekulize: Whether to kekulize the molecule
        mol_source: Source description for error messages

    Returns:
        GeometricMol object or None if conversion fails
    """

    if mol is None:
        print(f"Received None molecule for {mol_source}")
        return None

    has_hydrogens = has_explicit_hydrogens(mol)
    if sanitise:
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            if repeated_sanitise:
                try:
                    mol = Chem.AddHs(mol, addCoords=True)
                    Chem.SanitizeMol(mol)
                    if not has_hydrogens:
                        mol = Chem.RemoveHs(mol)
                except Exception:
                    try:
                        mol = Chem.RemoveHs(mol)
                        mol = Chem.AddHs(mol, addCoords=True)
                        Chem.SanitizeMol(mol)
                        if not has_hydrogens:
                            mol = Chem.RemoveHs(mol)
                    except Exception as e2:
                        print(
                            f"Failed to sanitize molecule after multiple attempts: {e2}. "
                            "Please check the input molecule structure."
                        )
                        return None
            else:
                print(
                    f"Failed to sanitize molecule: {e}. "
                    "Please check the input molecule structure."
                )
                return None

    if add_hs:
        if has_hydrogens:
            print(
                f"FYI: Received molecule with both explicit hydrogens for {mol_source} and add_hs flag set to True."
                " Hydrogens will be added again. If this is not intended, please remove add_hs flag."
            )
        try:
            mol = Chem.AddHs(mol, addCoords=True)
            Chem.SanitizeMol(mol)
        except Exception as e:
            print(
                f"Failed to add hydrogens to molecule from {mol_source}: {e}. "
                "Please check the input molecule structure."
            )
            return None

    if kekulize:
        try:
            if remove_hs:
                Chem.Kekulize(mol, clearAromaticFlags=True)
            else:
                Chem.Kekulize(mol)
        except Chem.KekulizeException:
            print(
                f"Kekulization failed for {mol_source}. Attempting to sanitize before and try again."
            )
            try:
                Chem.SanitizeMol(mol)
                if remove_hs:
                    Chem.Kekulize(mol, clearAromaticFlags=True)
                else:
                    Chem.Kekulize(mol)
            except Exception as e:
                print(
                    f"Failed to sanitize/kekulize molecule: {e}. Please check the input molecule structure."
                )
                return None

    if remove_hs:
        try:
            mol = Chem.RemoveHs(mol)
            Chem.SanitizeMol(mol)
        except Exception as e:
            print(
                f"Failed to remove hydrogens from molecule: {e}. Please check the input molecule structure."
            )
            return None

    # Check for disconnected fragments
    fragments = Chem.GetMolFrags(mol)
    if len(fragments) > 1:
        print(
            f"Skipping {mol_source} due to multiple disconnected fragments ({len(fragments)} frags). SMILES: {Chem.MolToSmiles(mol)}"
        )
        return None

    # Check atom count
    num_atoms = mol.GetNumAtoms()
    if num_atoms < min_atoms:
        print(f"Skipping {mol_source} due to low atom count ({num_atoms}).")
        return None

    # Extract potential energy and forces from ligand_info if provided
    potential_energy = None
    forces = None

    if ligand_info is not None:
        if "potential_energy" in ligand_info:
            potential_energy = torch.tensor(ligand_info["potential_energy"])
        if "atomic_forces" in ligand_info:
            forces = torch.tensor(ligand_info["atomic_forces"])

    try:
        geom_mol = GeometricMol.from_rdkit(mol)
    except Exception as e:
        print(
            f"Failed to create GeometricMol from RDKit molecule for {mol_source}: {e}. "
            "Please check the input molecule structure."
        )
        return None
    try:
        geom_rdkit_mol = geom_mol.to_rdkit(sanitise=True)
        if geom_rdkit_mol is None:
            raise ValueError(
                f"Failed to convert GeometricMol to RDKit for {mol_source}. "
                "Please check the input molecule structure."
            )
    except Exception:
        try:
            mol = Chem.AddHs(
                mol, addCoords=True
            )  # make potential implicit hydrogens explicit (e.g. in pyrazoles)
            geom_mol = GeometricMol.from_rdkit(mol)
            geom_rdkit_mol = geom_mol.to_rdkit(sanitise=True)
            if geom_rdkit_mol is None:
                raise ValueError(
                    f"Failed to convert GeometricMol to RDKit for {mol_source}. "
                    "Please check the input molecule structure."
                )
            if not has_hydrogens and not add_hs:
                geom_mol = geom_mol.remove_hs()
        except Exception as e:
            print(f"Failed to create GeometricMol from RDKit: {e}")
            return None

    # Create a new GeometricMol with the additional properties
    if potential_energy is not None or forces is not None:
        geom_mol = geom_mol._copy_with(potential_energy=potential_energy, forces=forces)

    return geom_mol


def safe_load_mol_from_file(
    ligand_file_path: str = None,
    remove_hs: bool = False,
    ligand_idx: int = 0,
    canonicalize_conformer: bool = False,
):
    """Safely load an RDKit molecule from SDF or directly from a provided RDKit Mol object.

    Args:
        ligand_file_path (str): Path to the SDF or PDB file containing the ligand.

    Returns:
        Chem.Mol: The loaded RDKit molecule or None if loading fails.
    """
    if ligand_file_path.endswith(".sdf"):
        mol = Chem.SDMolSupplier(str(ligand_file_path), removeHs=remove_hs)[ligand_idx]
        if mol is None and ligand_idx == 0:
            mol = Chem.MolFromMolFile(str(ligand_file_path), removeHs=remove_hs)
        if mol is None and ligand_idx == 0:
            # Try with OpenBabel using temporary file
            with tempfile.NamedTemporaryFile(suffix=".mol", delete=False) as temp_file:
                temp_mol_path = temp_file.name

            try:
                cmd = f"obabel -isdf {ligand_file_path} -omol -O {temp_mol_path}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    mol = Chem.MolFromMolFile(temp_mol_path, removeHs=remove_hs)
                else:
                    print(
                        f"OpenBabel failed to convert {ligand_file_path}: {result.stderr}"
                    )
                    mol = None
            finally:
                # Always clean up the temporary file
                if os.path.exists(temp_mol_path):
                    os.remove(temp_mol_path)
    elif ligand_file_path.endswith(".pdb"):
        try:
            mol = Chem.MolFromPDBFile(str(ligand_file_path), removeHs=remove_hs)
        except Exception as e:
            print(f"Error reading {ligand_file_path}: {e}")
            mol = None
    elif ligand_file_path.endswith(".mol"):
        mol = Chem.MolFromMolFile(str(ligand_file_path), removeHs=remove_hs)
    else:
        raise ValueError("Ligand file must be in SDF or PDB format")

    if mol is not None and canonicalize_conformer:
        try:
            print(f"Canonicalizing atom order for {ligand_file_path}")
            mol = canonicalize_atom_order(mol)
        except Exception as e:
            print(f"Could not canonicalize conformer for {ligand_file_path}: {e}")

    return mol


def process_complex(
    pdb_id: str = None,
    pdb_path: str = None,
    ligand_id: str = None,
    ligand_mol: Chem.Mol = None,
    ligand_sdf_path: str = None,
    ligand_idx: int = 0,
    txt_path: str = None,
    chain_id: str = None,
    canonicalize_conformer: bool = False,
    add_hs: bool = False,
    add_hs_and_optimize: bool = False,
    add_rdkit_feats: bool = False,
    remove_hs: bool = False,
    kekulize: bool = False,
    use_pdbfixer: bool = False,
    add_bonds_to_protein: bool = True,
    add_hs_to_protein: bool = False,
    pocket_cutoff: float = 6.0,
    cut_pocket: bool = False,
    max_pocket_size: int = 1000,
    min_pocket_size: int = 10,
    compute_interactions: bool = False,
    pocket_type: str = "holo",
    affinity: dict = {},
    ligand_is_covalent: bool = False,
    ligand_info: dict = None,
    split: str = None,
):
    """Process a complex from PDB and SDF files.
    Args:
        pdb_id (str): The PDB ID of the protein.
        pdb_path (str): The path to the PDB file.
        ligand_id (str): The ligand identifier.
        ligand_sdf_path (str): The path to the SDF or PDB file.
        ligand_mol (rdkit.Chem.Mol): The RDKit molecule object for the ligand.
        txt_path (str): The path to the TXT file with residue IDs.
        remove_hs (bool): Whether to remove hydrogens from the ligand.
        kekulize (bool): Whether to kekulize the ligand.
        add_bonds_to_protein (bool): Whether to add bonds to the protein.
        add_hs_to_protein (bool): Whether to add hydrogens to the protein.
        pocket_cutoff (float): The cutoff distance for pocket extraction.
        cut_pocket (bool): Whether to cut out the pocket from the protein.
        max_pocket_size (int): The size threshold for the pocket.
        compute_interactions (bool): Whether to compute interactions.
        pocket_type (str): The type of pocket ("holo" or "apo").
        affinity (dict): The affinity values for the ligand.
        ligand_is_covalent (bool): Whether the ligand is covalent.
        split (str): The split name for saving the complex.
    Returns:
        PocketComplex: The processed pocket complex.
    """
    # Either add_hs or remove_hs, not both, or both False
    if (add_hs or add_hs_and_optimize) and remove_hs:
        raise ValueError("Cannot add and remove hydrogens at the same time")

    if pdb_path is None and pdb_id is None:
        raise ValueError("PDB path or ID must be provided")
    if pdb_path is None and pdb_id is not None:
        # Save downloaded files into the current working directory, delete afterwards
        # if no ligand_id is provided, the first ligand in the PDB file will be used
        save_path = Path.cwd()
        pdb_path, ligand_sdf_path = transform_pdb(save_path, pdb_id, ligand_id)

    if ligand_sdf_path is not None:
        mol = safe_load_mol_from_file(
            ligand_file_path=ligand_sdf_path,
            remove_hs=remove_hs,
            ligand_idx=ligand_idx,
            canonicalize_conformer=canonicalize_conformer,
        )
        if mol is None:
            print(f"Could not read ligand from {ligand_sdf_path}. Skipping.")
            return
        ligand = safe_geometric_mol_from_rdkit(
            mol,
            ligand_info,
            add_hs=add_hs or add_hs_and_optimize,
            kekulize=kekulize,
            mol_source=ligand_sdf_path,
        )
        if ligand is None:
            print("Could not process provided ligand molecule. Skipping.")
            return

    elif ligand_mol is not None:
        ligand = safe_geometric_mol_from_rdkit(
            ligand_mol,
            ligand_info,
            add_hs=add_hs or add_hs_and_optimize,
            kekulize=kekulize,
            mol_source="provided molecule",
        )
        if ligand is None:
            print("Could not process provided ligand molecule. Skipping.")
            return
    else:
        if cut_pocket:
            raise ValueError(
                "Either ligand_sdf_path or ligand_mol must be provided to process complex with pocket"
            )
        else:
            ligand = None

    ligand.full_mol = ligand.copy() if ligand is not None else None

    # Load protein from PDB
    pocket = process_pdb(
        pdb_path,
        ligand=ligand,
        txt_path=txt_path,
        add_bonds_to_protein=add_bonds_to_protein,
        add_hs_to_protein=add_hs_to_protein,
        pocket_cutoff=pocket_cutoff,
        cut_pocket=cut_pocket,
        use_pdbfixer=use_pdbfixer,
    )
    if pocket is None:
        print(f"Failed to process pocket from {pdb_path}. Skipping!")
        return

    if len(pocket) < min_pocket_size:
        print(f"Too small or empty pocket after processing {pdb_path}. Skipping!")
        return

    if max_pocket_size is not None:
        # Check if the pocket size is below the threshold
        hs_mask = pocket.atoms.element != "H"
        pocket_nohs = pocket.select_atoms(hs_mask)
        if len(pocket_nohs) > max_pocket_size:
            size_ratio = len(pocket_nohs) / max_pocket_size
            new_cutoff = pocket_cutoff * (0.9 / size_ratio)
            new_cutoff = max(new_cutoff, 6.0)

            print(
                f"Pocket size of {len(pocket_nohs)} is {size_ratio:.2f}x above threshold {max_pocket_size} for structure {pdb_path}. Reducing cutoff from {pocket_cutoff:.2f} to {new_cutoff:.2f} and trying again"
            )
            pocket = process_pdb(
                pdb_path,
                ligand=ligand,
                txt_path=txt_path,
                add_bonds_to_protein=add_bonds_to_protein,
                add_hs_to_protein=add_hs_to_protein,
                pocket_cutoff=new_cutoff,
                cut_pocket=cut_pocket,
                use_pdbfixer=use_pdbfixer,
            )
            hs_mask = pocket.atoms.element != "H"
            pocket_nohs = pocket.select_atoms(hs_mask)
            if len(pocket_nohs) > max_pocket_size:
                print(
                    f"Pocket size of {len(pocket_nohs)} is still above threshold {max_pocket_size} for structure {pdb_path}. Trying with cutoff 5.0!"
                )
                pocket = process_pdb(
                    pdb_path,
                    ligand=ligand,
                    txt_path=txt_path,
                    add_bonds_to_protein=add_bonds_to_protein,
                    add_hs_to_protein=add_hs_to_protein,
                    pocket_cutoff=5.0,
                    cut_pocket=cut_pocket,
                    use_pdbfixer=use_pdbfixer,
                )
                hs_mask = pocket.atoms.element != "H"
                pocket_nohs = pocket.select_atoms(hs_mask)
                if len(pocket_nohs) > max_pocket_size:
                    print(
                        f"Pocket size of {len(pocket_nohs)} is still above threshold {max_pocket_size} for structure {pdb_path}. Trying with cutoff 4.0!"
                    )
                    pocket = process_pdb(
                        pdb_path,
                        ligand=ligand,
                        txt_path=txt_path,
                        add_bonds_to_protein=add_bonds_to_protein,
                        add_hs_to_protein=add_hs_to_protein,
                        pocket_cutoff=4.0,
                        cut_pocket=cut_pocket,
                        use_pdbfixer=use_pdbfixer,
                    )
                    hs_mask = pocket.atoms.element != "H"
                    pocket_nohs = pocket.select_atoms(hs_mask)
                    if len(pocket_nohs) > max_pocket_size:
                        print(
                            f"Pocket size of {len(pocket_nohs)} is still above threshold {max_pocket_size} for structure {pdb_path}. Trying with cutoff 3.5!"
                        )
                        pocket = process_pdb(
                            pdb_path,
                            ligand=ligand,
                            txt_path=txt_path,
                            add_bonds_to_protein=add_bonds_to_protein,
                            add_hs_to_protein=add_hs_to_protein,
                            pocket_cutoff=3.5,
                            cut_pocket=cut_pocket,
                            use_pdbfixer=use_pdbfixer,
                        )
                        hs_mask = pocket.atoms.element != "H"
                        pocket_nohs = pocket.select_atoms(hs_mask)
                        if len(pocket_nohs) > max_pocket_size:
                            print(
                                f"Pocket size of {len(pocket_nohs)} is still above threshold {max_pocket_size} for structure {pdb_path}. Skipping!"
                            )
                            return

    # Save the full pocket structure as backup
    pocket.full_pocket = pocket.copy()

    # Optimize ligand hydrogens if requested
    if add_hs_and_optimize:
        tmp_pdb_path = tempfile.mktemp(suffix=".pdb")
        try:
            pocket.write_pdb(tmp_pdb_path, include_bonds=True)
            ligand_mol_with_hs = add_and_optimize_ligand_hs(
                ligand.to_rdkit(), tmp_pdb_path, add_hs=False
            )
            if ligand_mol_with_hs is not None:
                optimized_ligand = safe_geometric_mol_from_rdkit(
                    ligand_mol_with_hs,
                    ligand_info,
                    add_hs=False,  # Hydrogens are already added and optimized
                    kekulize=kekulize,
                    mol_source="optimized ligand",
                )
                if optimized_ligand is not None:
                    ligand = optimized_ligand
                else:
                    print(
                        "Failed to convert optimized ligand back to GeometricMol. Using original."
                    )
            else:
                print("Hydrogen optimization returned None. Using original ligand.")
        except Exception as e:
            print(f"Skipping ligand optimization due to error: {e}")
        finally:
            # Remove the temporary PDB file
            if os.path.exists(tmp_pdb_path):
                os.remove(tmp_pdb_path)

    # Get metadata
    if not affinity:
        for prop_name in mol.GetPropNames():
            if prop_name.lower() in AFFINITY_PROP_NAMES:
                affinity[prop_name.lower()] = float(mol.GetProp(prop_name))
    pdb_id = Path(pdb_path).stem
    metadata = {
        "system_id": pdb_id,
        "is_covalent": ligand_is_covalent,
        **affinity,
    }
    metadata["apo_type"] = None
    metadata["split"] = split

    if pocket_type == "holo":
        _complex = PocketComplex(
            holo=pocket,
            ligand=ligand,
            apo=None,
            metadata=metadata,
        )
    elif pocket_type == "apo":
        _complex = PocketComplex(
            holo=None,
            ligand=ligand,
            apo=pocket,
            metadata=metadata,
        )
    else:
        raise ValueError(f"Invalid pocket type: {pocket_type}")
    _complex.store_metrics_()

    if compute_interactions:
        try:
            interactions = BindingInteractions.from_system(
                _complex, interaction_types=PROLIF_INTERACTIONS
            )
            _complex.interactions = interactions.array
        except Exception as e:
            print(
                f"Error computing interactions for {metadata['system_id']}: {e}. Skipping complex!"
            )
            return
    return _complex


def pdb_to_sdf(pdb_path: str, sdf_path: str):
    """Convert a PDB file to an SDF file using Open Babel.

    Args:
        pdb_path (str): The path to the PDB file.
        sdf_path (str): The path to the output SDF file.
    """

    # Convert the PDB file to an SDF file using Open Babel
    cmd = f"obabel -ipdb {pdb_path} -osdf -O {sdf_path}"
    out = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # Check if Open Babel failed
    if out.returncode != 0:
        print(out.stdout.decode())
        print(out.stderr.decode())
        raise Exception("Open Babel failed")

    return sdf_path


def main(args):

    data = os.path.join(args.data_path, args.split)
    txt_files = glob(os.path.join(data, "*.txt"))
    pdb_files = [
        os.path.join(Path(file).parent, Path(file).stem.split("_")[0] + ".pdb")
        for file in txt_files
    ]
    if "pocket_ids" in txt_files[0]:
        sdf_files = [
            os.path.join(
                Path(file).parent,
                Path(file).name.replace(
                    "pocket_ids.txt", f"{Path(file).stem.split('_')[0]}.sdf"
                ),
            )
            for file in txt_files
        ]
    else:
        sdf_files = [
            os.path.join(Path(file).parent, Path(file).stem + ".sdf")
            for file in txt_files
        ]

    systems = []
    for txt_path, struct_path, sdf_path in tqdm(
        zip_longest(txt_files, pdb_files, sdf_files, fillvalue=None),
        total=len(pdb_files),
    ):

        # Load ligand from SDF
        # ligand_structure = biotite.structure.io.load_structure(sdf_path)
        # ligand_structure.res_name = ["LIG"] * len(ligand_structure)
        _complex = process_complex(
            struct_path,
            sdf_path=sdf_path,
            txt_path=txt_path,
            remove_hs=args.remove_hs,
            kekulize=args.kekulize,
            add_bonds_to_protein=args.add_bonds_to_protein,
            add_hs_to_protein=args.add_hs_to_protein,
            pocket_cutoff=args.pocket_cutoff,
            cut_pocket=args.cut_pocket,
            compute_interactions=args.compute_interactions,
            split=args.split,
        )

        systems.append(_complex)

    save_systems_(args, systems, args.split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Preprocess data that comes as PDB (protein), SDF (ligand) and TXT (residues in a given radius) files"
    )

    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--remove_hs", action="store_true")
    parser.add_argument("--add_bonds_to_protein", action="store_true")
    parser.add_argument("--add_hs_to_protein", action="store_true")
    parser.add_argument("--pocket_cutoff", type=float, default=6.0)
    parser.add_argument("--cut_pocket", action="store_true")
    parser.add_argument("--compute_interactions", action="store_true")
    parser.add_argument("--kekulize", action="store_true")

    args = parser.parse_args()
    main(args)

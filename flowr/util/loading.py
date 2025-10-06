import os
import subprocess
import tempfile
from functools import lru_cache
from typing import List, Union

import datamol as dm
import MDAnalysis as mda

# import openbabel as ob
import prolif as plf
from rdkit import Chem

from flowr.util.biopython import (
    ids_scriptly_increasing,
    load_biopython_structure,
    remove_connect_lines,
    reorder_ids,
    save_biopython_structure,
)
from flowr.util.chem import remove_radicals


def load_protein_prolif(protein_path: str):
    """Load protein from PDB file using MDAnalysis
    and convert to plf.Molecule. Assumes hydrogens are present."""
    prot = mda.Universe(protein_path)
    prot = plf.Molecule.from_mda(prot, NoImplicit=False)
    return prot


@lru_cache(maxsize=100)
def load_protein_from_pdb(pdb_path: str):
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

    # Run Hydrite
    cmd = f"hydride -i {tmp_path} -o {tmp_protonated_path}"
    out = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # Check if Hydrite failed
    if out.returncode != 0:
        print(out.stdout.decode())
        print(out.stderr.decode())
        raise Exception("Hydrite failed")

    # - Remove CONECT lines from the PDB file - #
    # This is necessary because the CONECT lines are not handled correctly by MDAnalysis
    # and they are for some reason added by Hydrite
    remove_connect_lines(tmp_protonated_path)

    # Load the protein from the temporary PDB file
    prot = load_protein_prolif(tmp_protonated_path)
    os.remove(tmp_protonated_path)

    return prot


# def get_pdbqt_mol(pdbqt_block: str) -> Chem.Mol:
#     """Convert pdbqt block to rdkit mol by converting with openbabel"""
#     # write pdbqt file
#     with open("test_pdbqt.pdbqt", "w") as f:
#         f.write(pdbqt_block)

#     # read pdbqt file from autodock
#     mol = ob.OBMol()
#     obConversion = ob.OBConversion()
#     obConversion.SetInAndOutFormats("pdbqt", "pdb")
#     obConversion.ReadFile(mol, "test_pdbqt.pdbqt")

#     # convert to RDKIT
#     mol = Chem.MolFromPDBBlock(obConversion.WriteString(mol))

#     # remove tmp file
#     os.remove("test_pdbqt.pdbqt")

#     return mol


# def read_pdbqt(pdbqt_file: str) -> Chem.Mol:
#     """Read pdbqt file from autodock and convert to rdkit mol.

#     Args:
#         pdbqt_file (str): path to pdbqt file

#     Returns:
#         Chem.Mol: rdkit mol
#     """
#     with open(pdbqt_file, "r") as f:
#         pdbqt_block = f.read()
#     return get_pdbqt_mol(pdbqt_block)


def load_sdf_prolif(sdf_path: str) -> plf.Molecule:
    """Load ligand from an SDF file and convert it to a prolif.Molecule.

    Args:
        sdf_path (str): Path to the SDF file.

    Returns:
        plf.Molecule: The loaded ligand as a prolif.Molecule.
    """
    return plf.sdf_supplier(sdf_path)


def load_mols_from_sdf(sdf_path: str, add_hs: bool = True) -> Union[plf.Molecule, List]:
    """Load ligand from an SDF file, add hydrogens, and convert it to a prolif.Molecule.

    Args:
        sdf_path (str): Path to the SDF file.
        add_hs (bool, optional): Whether to add hydrogens. Defaults to True.

    Returns:
        Union[plf.Molecule, List]: The loaded ligand as a prolif.Molecule, or an empty list if no molecule could be loaded.
    """
    tmp_path = sdf_path.split(".sdf")[0] + "_tmp.sdf"

    # Load molecules from the SDF file
    mols = dm.read_sdf(sdf_path)

    # Remove radicals from the molecules
    mols = [remove_radicals(mol) for mol in mols]

    # Add hydrogens to the molecules
    if add_hs:
        mols = [Chem.AddHs(m, addCoords=True) for m in mols if m is not None]

    if len(mols) == 0:
        return []

    # Write the molecules to a temporary file
    dm.to_sdf(mols, tmp_path)
    ligs = load_sdf_prolif(tmp_path)
    os.remove(tmp_path)

    # Turn into list
    ligs = list(ligs)

    return ligs


def load_mols_from_rdkit(
    mol_list: List[Chem.Mol], add_hs: bool = True
) -> Union[plf.Molecule, List]:
    """Load ligand from an RDKit molecule, add hydrogens, and convert it to a prolif.Molecule.

    Processes in the same way as src.utils.loading.load_mols_from_sdf but takes an RDKit molecule as input.

    Args:
        rdkit_mol (Chem.Mol): RDKit molecule.
        add_hs (bool, optional): Whether to add hydrogens. Defaults to True.

    Returns:
        Union[plf.Molecule, List]: The loaded ligand as a prolif.Molecule, or an empty list if no molecule could be loaded.
    """

    # Convert single molecule to list
    if isinstance(mol_list, Chem.Mol):
        print(
            "Converting single molecule to list, to be consistent with load_mols_from_sdf"
        )
        mol_list = [mol_list]

    # Add hydrogens to the molecules
    if add_hs:
        mol_list = [Chem.AddHs(m, addCoords=True) for m in mol_list if m is not None]

    # Check if any molecules were loaded
    if len(mol_list) == 0:
        print("No molecules loaded, returning empty list")
        return []

    return [plf.Molecule.from_rdkit(mol) for mol in mol_list]

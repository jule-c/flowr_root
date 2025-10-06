"""Functions to compute the interaction fingerprint of a molecule with a protein (pocket) in 3d space"""

import oddt
from oddt.interactions import hbonds
from oddt.fingerprints import InteractionFingerprint, tanimoto
from oddt.toolkits.ob import Molecule

from rdkit import Chem
import numpy as np
import tempfile
from typing import List, Union
from experiments.utils import write_sdf_file

RDKIT_MOL = Union[Chem.Mol, Chem.rdchem.Mol]
ODDT_MOL = Union[Molecule]

def convert_mol_to_oddt(mols: List[Chem.Mol]):
    """Converts a list of RDKit molecules to ODDT molecules"""
    temp_dir = tempfile.TemporaryDirectory()
    savedir = temp_dir.name + "/molecules.sdf"
    write_sdf_file(savedir, mols)
    oddt_mols = read_ligands(savedir)
    temp_dir.cleanup()
    return oddt_mols

def convert_rdkit_mol_to_oddt_mol(rdkit_mol: Chem.Mol) -> Molecule:
    """Converts an RDKit molecule to an ODDT molecule"""
    return convert_mol_to_oddt([rdkit_mol])[0]

def _check_mol_and_convert_to_oddt_mol(mol: Union[Molecule, Chem.Mol]) -> Molecule:
    """Checks if the molecule is an RDKit molecule and converts it to an ODDT molecule"""
    if isinstance(mol, RDKIT_MOL):
        return convert_rdkit_mol_to_oddt_mol(mol)
    elif isinstance(mol, ODDT_MOL):
        return mol
    else:
        raise ValueError('The molecule should be either an RDKit molecule or an ODDT molecule')

def read_protein(protein_file: str):
    """Reads a protein file and return a protein object"""
    protein = next(oddt.toolkit.readfile('pdb', protein_file))
    protein.protein = True
    return protein

def read_ligands(ligand_file: str):
    """Reads a ligand file and return a ligand object"""
    suppl = oddt.toolkit.readfile('sdf', ligand_file)
    ligand = [m for m in suppl]
    return ligand

def get_hbonds_residues(protein: Molecule, ligand: Union[Molecule, Chem.Mol]):
    """Returns the residues involved in hydrogen bonds with the ligand"""
    ligand = _check_mol_and_convert_to_oddt_mol(ligand)
    assert protein.protein, 'The protein object should have protein attribute set to True'
    assert not ligand.protein, 'The ligand object should have protein attribute set to False'
    protein_atoms, ligand_atoms, strict = hbonds(protein, ligand)
    formatted_atoms = [f'{resname}-{resnum}' for resname, resnum in
                   zip(protein_atoms['resname'], protein_atoms['resnum'])]
    return formatted_atoms

def get_interaction_fp(protein: Molecule, ligand: Union[Molecule, Chem.Mol]):
    """Returns the interaction fingerprint of a ligand with a protein"""
    # Get the interaction fingerprint
    ligand = _check_mol_and_convert_to_oddt_mol(ligand)
    assert protein.protein, 'The protein object should have protein attribute set to True'
    assert not ligand.protein, 'The ligand object should have protein attribute set to False'
    ifp = InteractionFingerprint(ligand=ligand, protein=protein)
    return ifp

def get_tanimoto_with_ref_ifp(ref_ifp: np.ndarray, 
                              ligands: List[Union[Molecule, Chem.Mol]],
                              protein: Molecule) -> List[float]:
    """Returns the Tanimoto similarity between the reference interaction fingerprint with IFP from ligands with protein"""
    all_tanimoto_sim = []
    for ligand in ligands:
        ligand = _check_mol_and_convert_to_oddt_mol(ligand)
        ifp = get_interaction_fp(protein, ligand)
        all_tanimoto_sim.append(tanimoto(ref_ifp, ifp))
    return all_tanimoto_sim
from copy import deepcopy

import numpy as np
import rdkit.Chem as Chem
from datamol.conformers._conformers import _get_ff
from rdkit import Chem
from rdkit.Chem import AllChem


def calculate_energy(
    mol: Chem.Mol, forcefield: str = "UFF", add_hs: bool = True
) -> float:
    """
    Evaluates the energy of a molecule using a force field.

    Args:
        mol: RDKit Mol object representing the molecule.
        forcefield: Force field to use for energy calculation (default: "UFF").
        add_hs: Whether to add hydrogens to the molecule (default: True).

    Returns:
        energy: Calculated energy of the molecule.
                Returns NaN if energy calculation fails.
    """
    mol = Chem.Mol(mol)  # Make a deep copy of the molecule

    if add_hs:
        mol = Chem.AddHs(mol, addCoords=True)

    try:
        ff = _get_ff(mol, forcefield=forcefield)
    except Exception:
        return np.nan

    try:
        energy = ff.CalcEnergy()
    except:
        return np.nan

    return energy


def relax_constrained(
    mol: Chem.Mol, forcefield: str = "UFF", add_hs: bool = True, maxDispl=0.1
) -> float:
    """
    Calculates the energy of a molecule using a force field.

    Args:
        mol: RDKit Mol object representing the molecule.
        forcefield: Force field to use for energy calculation (default: "UFF").
        add_hs: Whether to add hydrogens to the molecule (default: True).

    Returns:
        energy: Calculated energy of the molecule (rounded to 2 decimal places).
                Returns NaN if energy calculation fails.
    """
    mol = deepcopy(mol)  # Make a deep copy of the molecule
    #mol = Chem.Mol(mol)  # Make a deep copy of the molecule

    if add_hs:
        mol = Chem.AddHs(mol, addCoords=True)

    try:
        ff = _get_ff(mol, forcefield=forcefield)
    except Exception:
        return np.nan
    
    for i in range(mol.GetNumAtoms()):
        ff.UFFAddPositionConstraint(i, maxDispl=maxDispl, forceConstant=1.0e5)


    try:
        ff.Minimize()
        return mol
    except:
        None
        
def relax_global(mol: Chem.Mol) -> Chem.Mol:
    """Relax a molecule by adding hydrogens, embedding it, and optimizing it
    using the UFF force field.

    Args:
        mol (Chem.Mol): The molecule to relax.

    Returns:
        Chem.Mol: The relaxed molecule.
    """

    # if the molecule is None, return None
    if mol is None:
        return None

    # Incase ring info is not present
    Chem.GetSSSR(mol)  # SSSR: Smallest Set of Smallest Rings

    # make a copy of the molecule
    mol = deepcopy(mol)

    # add hydrogens
    mol = Chem.AddHs(mol, addCoords=True)

    # embed the molecule
    #AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    AllChem.EmbedMolecule(mol)

    # optimize the molecule
    try:
        AllChem.UFFOptimizeMolecule(mol)
    except:
        return None

    # return the molecule
    return mol

def relax_global_on_pose(mol: Chem.Mol) -> Chem.Mol:
    """Relax the given pose without position constraints by adding hydrogens and optimizing it
    using the UFF force field.

    Args:
        mol (Chem.Mol): The molecule to relax.

    Returns:
        Chem.Mol: The relaxed molecule.
    """

    # if the molecule is None, return None
    if mol is None:
        return None

    # Incase ring info is not present
    Chem.GetSSSR(mol)  # SSSR: Smallest Set of Smallest Rings

    # make a copy of the molecule
    mol = deepcopy(mol)

    # add hydrogens
    mol = Chem.AddHs(mol, addCoords=True)

    # optimize the molecule
    try:
        AllChem.UFFOptimizeMolecule(mol)
    except:
        return None

    return mol
        
def calculate_strain_energy(mol: Chem.Mol, maxDispl: float = 0.1, num_confs: int = 50) -> float:
    """Calculate the strain energy of a molecule.
    
    In order to evaluate the global strain energy of a molecule, rather than local imperfections
    in bonds distances and angles, we first perform a local relaxation of the molecule (by minimizing and allowing 
    a small displacement of the atoms) and then sample and minimize n conformers of the molecule.

    Args:
        mol (Chem.Mol): The molecule to calculate the strain energy for.
        maxDispl (float): The maximum displacement for position constraints during local relaxation. (Default: 0.1)
        num_confs (int): The number of conformers to generate for global relaxation.

    Returns:
        float: The calculated strain energy, or None if the calculation fails.
    """
    try:
        # relax molecule enforcing constraints on the atom positions
        locally_relaxed = relax_constrained(mol, maxDispl=maxDispl)
        # sample and minimize n conformers 
        global_relaxed = [relax_global(mol) for i in range(num_confs)]
        # alleviate insufficient sampling
        global_relaxed.append(relax_global_on_pose(mol))
            
        # calculate the energy of the locally relaxed molecule
        local_energy = calculate_energy(locally_relaxed)
        
        # calculate the energy of the globally relaxed molecules and take the minimum
        global_energy = min([calculate_energy(mol) for mol in global_relaxed if mol is not None])
        
        # calculate the strain energy
        strain_energy = local_energy - global_energy
        
        return strain_energy
    
    except Exception as e:
        print('Warning: Strain energy calculation failed')
        print(e)
        return None


if __name__ == "__main__":
    from posecheck.utils.constants import EXAMPLE_LIGAND_PATH, EXAMPLE_PDB_PATH
    from posecheck.utils.loading import load_mols_from_sdf, load_protein_from_pdb

    # Example molecules
    #prot = load_protein_from_pdb(EXAMPLE_PDB_PATH)
    lig = load_mols_from_sdf(EXAMPLE_LIGAND_PATH)[0]

    # Calculate strain
    strain = calculate_strain_energy(lig)

    print(f"Strain energy: {strain}")

    #assert round(strain, 2) == 19.11

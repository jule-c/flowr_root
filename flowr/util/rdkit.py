import hashlib
import pickle
import threading
from pathlib import Path
from typing import Dict, List, Optional, Union

import biotite.structure.io.pdb as io_pdb
import numpy as np
import rdkit
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from scipy.spatial.transform import Rotation

ArrT = np.ndarray


# *************************************************************************************************
# ************************************ Periodic Table class ***************************************
# *************************************************************************************************


class PeriodicTable:
    """Singleton class wrapper for the RDKit periodic table providing a neater interface"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self):
        self._table = Chem.GetPeriodicTable()

        # Just to be certain that vocab objects are thread safe
        self._pt_lock = threading.Lock()

    def atomic_from_symbol(self, symbol: str) -> int:
        with self._pt_lock:
            symbol = symbol.upper() if len(symbol) == 1 else symbol
            atomic = self._table.GetAtomicNumber(symbol)

        return atomic

    def symbol_from_atomic(self, atomic_num: int) -> str:
        with self._pt_lock:
            token = self._table.GetElementSymbol(atomic_num)

        return token

    def valence(self, atom: Union[str, int]) -> int:
        with self._pt_lock:
            valence = self._table.GetDefaultValence(atom)

        return valence


# *************************************************************************************************
# ************************************* Global Declarations ***************************************
# *************************************************************************************************


PT = PeriodicTable()

IDX_BOND_MAP = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC,
    # 5: Chem.BondType.DATIVE,
}
BOND_IDX_MAP = {bond: idx for idx, bond in IDX_BOND_MAP.items()}

IDX_CHARGE_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: -1, 5: -2, 6: -3}
CHARGE_IDX_MAP = {charge: idx for idx, charge in IDX_CHARGE_MAP.items()}


IDX_ADD_FEAT_MAP = {
    "is_aromatic": {False: 0, True: 1},
    "is_in_ring": {False: 0, True: 1},
    "hybridization": {
        0: rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED,
        1: rdkit.Chem.rdchem.HybridizationType.S,
        2: rdkit.Chem.rdchem.HybridizationType.SP,
        3: rdkit.Chem.rdchem.HybridizationType.SP2,
        4: rdkit.Chem.rdchem.HybridizationType.SP3,
        5: rdkit.Chem.rdchem.HybridizationType.SP2D,
        6: rdkit.Chem.rdchem.HybridizationType.SP3D,
        7: rdkit.Chem.rdchem.HybridizationType.SP3D2,
        8: rdkit.Chem.rdchem.HybridizationType.OTHER,
    },
}
ADD_FEAT_IDX_MAP = {
    "is_aromatic": {v: k for k, v in IDX_ADD_FEAT_MAP["is_aromatic"].items()},
    "is_in_ring": {v: k for k, v in IDX_ADD_FEAT_MAP["is_in_ring"].items()},
    "hybridization": {v: k for k, v in IDX_ADD_FEAT_MAP["hybridization"].items()},
}
# *************************************************************************************************

StructuralAlertSmarts = [
    "*1[O,S,N]*1",
    "[S,C](=[O,S])[F,Br,Cl,I]",
    "[CX4][Cl,Br,I]",
    "[#6]S(=O)(=O)O[#6]",
    "[$([CH]),$(CC)]#CC(=O)[#6]",
    "[$([CH]),$(CC)]#CC(=O)O[#6]",
    "n[OH]",
    "[$([CH]),$(CC)]#CS(=O)(=O)[#6]",
    "C=C(C=O)C=O",
    "n1c([F,Cl,Br,I])cccc1",
    "[CH1](=O)",
    "[#8][#8]",
    "[C;!R]=[N;!R]",
    "[N!R]=[N!R]",
    "[#6](=O)[#6](=O)",
    "[#16][#16]",
    "[#7][NH2]",
    "C(=O)N[NH2]",
    "[#6]=S",
    "[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]=[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]",
    "C1(=[O,N])C=CC(=[O,N])C=C1",
    "C1(=[O,N])C(=[O,N])C=CC=C1",
    "a21aa3a(aa1aaaa2)aaaa3",
    "a31a(a2a(aa1)aaaa2)aaaa3",
    "a1aa2a3a(a1)A=AA=A3=AA=A2",
    "c1cc([NH2])ccc1",
    "[Hg,Fe,As,Sb,Zn,Se,se,Te,B,Si,Na,Ca,Ge,Ag,Mg,K,Ba,Sr,Be,Ti,Mo,Mn,Ru,Pd,Ni,Cu,Au,Cd,"
    + "Al,Ga,Sn,Rh,Tl,Bi,Nb,Li,Pb,Hf,Ho]",
    "I",
    "OS(=O)(=O)[O-]",
    "[N+](=O)[O-]",
    "C(=O)N[OH]",
    "C1NC(=O)NC(=O)1",
    "[SH]",
    "[S-]",
    "c1ccc([Cl,Br,I,F])c([Cl,Br,I,F])c1[Cl,Br,I,F]",
    "c1cc([Cl,Br,I,F])cc([Cl,Br,I,F])c1[Cl,Br,I,F]",
    "[CR1]1[CR1][CR1][CR1][CR1][CR1][CR1]1",
    "[CR1]1[CR1][CR1]cc[CR1][CR1]1",
    "[CR2]1[CR2][CR2][CR2][CR2][CR2][CR2][CR2]1",
    "[CR2]1[CR2][CR2]cc[CR2][CR2][CR2]1",
    "[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1",
    "[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1",
    "C#C",
    "[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]",
    "[$([N+R]),$([n+R]),$([N+]=C)][O-]",
    "[#6]=N[OH]",
    "[#6]=NOC=O",
    "[#6](=O)[CX4,CR0X3,O][#6](=O)",
    "c1ccc2c(c1)ccc(=O)o2",
    "[O+,o+,S+,s+]",
    "N=C=O",
    "[NX3,NX4][F,Cl,Br,I]",
    "c1ccccc1OC(=O)[#6]",
    "[CR0]=[CR0][CR0]=[CR0]",
    "[C+,c+,C-,c-]",
    "N=[N+]=[N-]",
    "C12C(NC(N1)=O)CSC2",
    "c1c([OH])c([OH,NH2,NH])ccc1",
    "P",
    "[N,O,S]C#N",
    "C=C=O",
    "[Si][F,Cl,Br,I]",
    "[SX2]O",
    "[SiR0,CR0](c1ccccc1)(c2ccccc2)(c3ccccc3)",
    "O1CCCCC1OC2CCC3CCCCC3C2",
    "N=[CR0][N,n,O,S]",
    "[cR2]1[cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2][cR2]1[cR2]2[cR2][cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2]2",
    "C=[C!r]C#N",
    "[cR2]1[cR2]c([N+0X3R0,nX3R0])c([N+0X3R0,nX3R0])[cR2][cR2]1",
    "[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2]c([N+0X3R0,nX3R0])[cR2]1",
    "[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2][cR2]c1([N+0X3R0,nX3R0])",
    "[OH]c1ccc([OH,NH2,NH])cc1",
    "c1ccccc1OC(=O)O",
    "[SX2H0][N]",
    "c12ccccc1(SC(S)=N2)",
    "c12ccccc1(SC(=S)N2)",
    "c1nnnn1C=O",
    "s1c(S)nnc1NC=O",
    "S1C=CSC1=S",
    "C(=O)Onnn",
    "OS(=O)(=O)C(F)(F)F",
    "N#CC[OH]",
    "N#CC(=O)",
    "S(=O)(=O)C#N",
    "N[CH2]C#N",
    "C1(=O)NCC1",
    "S(=O)(=O)[O-,OH]",
    "NC[F,Cl,Br,I]",
    "C=[C!r]O",
    "[NX2+0]=[O+0]",
    "[OR0,NR0][OR0,NR0]",
    "C(=O)O[C,H1].C(=O)O[C,H1].C(=O)O[C,H1]",
    "[CX2R0][NX3R0]",
    "c1ccccc1[C;!R]=[C;!R]c2ccccc2",
    "[NX3R0,NX4R0,OR0,SX2R0][CX4][NX3R0,NX4R0,OR0,SX2R0]",
    "[s,S,c,C,n,N,o,O]~[n+,N+](~[s,S,c,C,n,N,o,O])(~[s,S,c,C,n,N,o,O])~[s,S,c,C,n,N,o,O]",
    "[s,S,c,C,n,N,o,O]~[nX3+,NX3+](~[s,S,c,C,n,N])~[s,S,c,C,n,N]",
    "[*]=[N+]=[*]",
    "[SX3](=O)[O-,OH]",
    "N#N",
    "F.F.F.F",
    "[R0;D2][R0;D2][R0;D2][R0;D2]",
    "[cR,CR]~C(=O)NC(=O)~[cR,CR]",
    "C=!@CC=[O,S]",
    "[#6,#8,#16][#6](=O)O[#6]",
    "c[C;R0](=[O,S])[#6]",
    "c[SX2][C;!R]",
    "C=C=C",
    "c1nc([F,Cl,Br,I,S])ncc1",
    "c1ncnc([F,Cl,Br,I,S])c1",
    "c1nc(c2c(n1)nc(n2)[F,Cl,Br,I])",
    "[#6]S(=O)(=O)c1ccc(cc1)F",
    "[15N]",
    "[13C]",
    "[18O]",
    "[34S]",
]
StructuralAlerts = [Chem.MolFromSmarts(smarts) for smarts in StructuralAlertSmarts]


CONT_PROPERTIES_RDKIT = {
    "MolWt": Descriptors.MolWt,
    "LogP": Descriptors.MolLogP,
    "SlogP_VSA1": Descriptors.SlogP_VSA1,
    "SlogP_VSA2": Descriptors.SlogP_VSA2,
    "SlogP_VSA3": Descriptors.SlogP_VSA3,
    "SlogP_VSA4": Descriptors.SlogP_VSA4,
    "SlogP_VSA5": Descriptors.SlogP_VSA5,
    "SlogP_VSA6": Descriptors.SlogP_VSA6,
    "SlogP_VSA7": Descriptors.SlogP_VSA7,
    "SlogP_VSA8": Descriptors.SlogP_VSA8,
    "SlogP_VSA9": Descriptors.SlogP_VSA9,
    "MolMR": Descriptors.MolMR,
    "FractionCSP3": Descriptors.FractionCSP3,
    "TPSA": Descriptors.TPSA,
}

DISC_PROPERTIES_RDKIT = {
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHeteroatoms": Descriptors.NumHeteroatoms,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "NumHeavyAtoms": Descriptors.HeavyAtomCount,
    "NumAliphaticCarbocycles": Descriptors.NumAliphaticCarbocycles,
    "NumAliphaticHeterocycles": Descriptors.NumAliphaticHeterocycles,
    "NumAliphaticRings": Descriptors.NumAliphaticRings,
    "NumAromaticCarbocycles": Descriptors.NumAromaticCarbocycles,
    "NumAromaticHeterocycles": Descriptors.NumAromaticHeterocycles,
    "NumSaturatedCarbocycles": Descriptors.NumSaturatedCarbocycles,
    "NumSaturatedHeterocycles": Descriptors.NumSaturatedHeterocycles,
    "NumAromaticRings": Descriptors.NumAromaticRings,
    "RingCount": Descriptors.RingCount,
    "NumChiralCenters": lambda mol: len(rdkit.Chem.FindMolChiralCenters(mol)),
    "NumAlerts": lambda mol: sum(
        1 for alert in StructuralAlerts if mol.HasSubstructMatch(alert)
    ),
}


# *************************************************************************************************
# *************************************** Util Functions ******************************************
# *************************************************************************************************


def _check_shape_len(arr, allowed, name="object"):
    num_dims = len(arr.shape)
    allowed = [allowed] if isinstance(allowed, int) else allowed
    if num_dims not in allowed:
        raise RuntimeError(
            f"Number of dimensions of {name} must be in {str(allowed)}, got {num_dims}"
        )


def _check_dim_shape(arr, dim, allowed, name="object"):
    shape = arr.shape[dim]
    allowed = [allowed] if isinstance(allowed, int) else allowed
    if shape not in allowed:
        raise RuntimeError(
            f"Shape of {name} for dim {dim} must be in {allowed}, got {shape}"
        )


# *************************************************************************************************
# ************************************* External Functions ****************************************
# *************************************************************************************************


def retrieve_rdkit_cont_feats_from_mol(mol: Chem.rdchem.Mol):
    """Retrieve continuous properties from a molecule
    Args:
        mol (Chem.rdchem.Mol): RDKit molecule object
    Returns:
        list: List containing continuous properties
    """
    return [func(mol) for func in CONT_PROPERTIES_RDKIT.values()]


def retrieve_rdkit_disc_feats_from_mol(mol: Chem.rdchem.Mol):
    """Retrieve discrete properties from a molecule
    Args:
        mol (Chem.rdchem.Mol): RDKit molecule object
    Returns:
        list: List containing discrete properties
    """
    return [func(mol) for func in DISC_PROPERTIES_RDKIT.values()]


def retrieve_hybridization_from_mol(mol: Chem.rdchem.Mol):
    hybridization = [
        ADD_FEAT_IDX_MAP["hybridization"][atom.GetHybridization()]
        for atom in mol.GetAtoms()
    ]
    return hybridization


def retrieve_is_aromatic_from_mol(mol: Chem.rdchem.Mol):
    is_aromatic = [
        ADD_FEAT_IDX_MAP["is_aromatic"].get(atom.GetIsAromatic())
        for atom in mol.GetAtoms()
    ]
    return is_aromatic


def retrieve_is_in_ring_from_mol(mol: Chem.rdchem.Mol):
    is_in_ring = [
        ADD_FEAT_IDX_MAP["is_in_ring"].get(atom.IsInRing()) for atom in mol.GetAtoms()
    ]
    return is_in_ring


def retrieve_donor_acceptor_from_mol(mol: Chem.rdchem.Mol):
    import os

    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures

    fdefName = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

    feats = factory.GetFeaturesForMol(mol)
    donor_ids = []
    acceptor_ids = []
    for f in feats:
        if f.GetFamily().lower() == "donor":
            donor_ids.append(f.GetAtomIds())
        elif f.GetFamily().lower() == "acceptor":
            acceptor_ids.append(f.GetAtomIds())

    if len(donor_ids) > 0:
        donor_ids = np.concatenate(donor_ids)
    else:
        donor_ids = np.array([])

    if len(acceptor_ids) > 0:
        acceptor_ids = np.concatenate(acceptor_ids)
    else:
        acceptor_ids = np.array([])
    is_acceptor = np.zeros(mol.GetNumAtoms(), dtype=np.uint8)
    is_donor = np.zeros(mol.GetNumAtoms(), dtype=np.uint8)
    if len(donor_ids) > 0:
        is_donor[donor_ids] = 1
    if len(acceptor_ids) > 0:
        is_acceptor[acceptor_ids] = 1

    return (is_donor, is_acceptor)


def retrieve_bonds_from_mol(mol: Chem.rdchem.Mol, infer_bonds: bool = False):
    if infer_bonds:
        mol = _infer_bonds(mol)
    bonds = []
    for bond in mol.GetBonds():
        bond_start = bond.GetBeginAtomIdx()
        bond_end = bond.GetEndAtomIdx()

        # TODO perhaps print a warning but just don't add the bond?
        bond_type = BOND_IDX_MAP.get(bond.GetBondType())
        if bond_type is None:
            raise NotImplementedError(f"Unsupported bond type {bond.GetBondType()}")

        bonds.append([bond_start, bond_end, bond_type])
    return bonds


def mol_is_valid(
    mol: Chem.rdchem.Mol,
    with_hs: bool = True,
    connected: bool = True,
    add_hs=False,
) -> bool:
    """Whether the mol can be sanitised and, optionally, whether it's fully connected

    Args:
        mol (Chem.Mol): RDKit molecule to check
        with_hs (bool): Whether to check validity including hydrogens (if they are in the input mol), default True
        connected (bool): Whether to also assert that the mol must not have disconnected atoms, default True

    Returns:
        bool: Whether the mol is valid
    """

    if mol is None:
        return False

    mol_copy = Chem.Mol(mol)
    if not with_hs:
        mol_copy = Chem.RemoveAllHs(mol_copy)

    if add_hs:
        mol_copy = Chem.AddHs(mol_copy)

    try:
        AllChem.SanitizeMol(mol_copy)
    except Exception:
        return False

    n_frags = len(AllChem.GetMolFrags(mol_copy))
    if connected and n_frags != 1:
        return False

    return True


def remove_radicals(mol: Chem.Mol, sanitize: bool = True) -> Chem.Mol:
    """Remove free radicals from a molecule."""

    # Iterate over all atoms in the molecule
    for atom in mol.GetAtoms():
        # Check if the current atom has any radical electrons
        num_radicals = atom.GetNumRadicalElectrons()
        if num_radicals > 0:
            # Saturate the atom with hydrogen atoms
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + num_radicals)
            atom.SetNumRadicalElectrons(0)

    if sanitize:
        AllChem.SanitizeMol(mol)

    return mol


def has_radicals(mol: Chem.Mol) -> bool:
    """Check if a molecule has any free radicals."""

    # Iterate over all atoms in the molecule
    for atom in mol.GetAtoms():
        # Check if the current atom has any radical electrons
        num_radicals = atom.GetNumRadicalElectrons()
        if num_radicals > 0:
            return True

    return False


def calc_energy(mol: Chem.rdchem.Mol, per_atom: bool = False) -> float:
    """Calculate the energy for an RDKit molecule using the MMFF forcefield

    The energy is only calculated for the first (0th index) conformer within the molecule. The molecule is copied so
    the original is not modified.

    Args:
        mol (Chem.Mol): RDKit molecule
        per_atom (bool): Whether to normalise by number of atoms in mol, default False

    Returns:
        float: Energy of the molecule or None if the energy could not be calculated
    """

    mol_copy = Chem.Mol(mol)

    try:
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol_copy, mmffVariant="MMFF94")
        ff = AllChem.MMFFGetMoleculeForceField(mol_copy, mmff_props, confId=0)
        energy = ff.CalcEnergy()
        energy = energy / mol.GetNumAtoms() if per_atom else energy
    except:
        energy = None

    return energy


def optimise_mol(mol: Chem.rdchem.Mol, max_iters: int = 1000) -> Chem.rdchem.Mol:
    """Optimise the conformation of an RDKit molecule

    Only the first (0th index) conformer within the molecule is optimised. The molecule is copied so the original
    is not modified.

    Args:
        mol (Chem.Mol): RDKit molecule
        max_iters (int): Max iterations for the conformer optimisation algorithm

    Returns:
        Chem.Mol: Optimised molecule or None if the molecule could not be optimised within the given number of
                iterations
    """

    mol_copy = Chem.Mol(mol)
    try:
        exitcode = AllChem.MMFFOptimizeMolecule(mol_copy, maxIters=max_iters)
    except:
        exitcode = -1

    if exitcode == 0:
        return mol_copy

    return None


def conf_distance(
    mol1: Chem.rdchem.Mol, mol2: Chem.rdchem.Mol, fix_order: bool = True
) -> float:
    """Approximately align two molecules and then calculate RMSD between them

    Alignment and distance is calculated only between the default conformers of each molecule.

    Args:
        mol1 (Chem.Mol): First molecule to align
        mol2 (Chem.Mol): Second molecule to align
        fix_order (bool): Whether to fix the atom order of the molecules

    Returns:
        float: RMSD between molecules after approximate alignment if fix_order is set to true
    """

    assert len(mol1.GetAtoms()) == len(mol2.GetAtoms())
    coords1 = np.array(mol1.GetConformer().GetPositions())
    coords2 = np.array(mol2.GetConformer().GetPositions())
    if fix_order:
        # Firstly, centre both molecules
        coords1 = coords1 - (coords1.sum(axis=0) / coords1.shape[0])
        coords2 = coords2 - (coords2.sum(axis=0) / coords2.shape[0])

        # Find the best rotation alignment between the centred mols
        rotation, _ = Rotation.align_vectors(coords1, coords2)
        aligned_coords2 = rotation.apply(coords2)

        sqrd_dists = (coords1 - aligned_coords2) ** 2
        rmsd = np.sqrt(sqrd_dists.sum(axis=1).mean())
    else:
        rmsd = np.sqrt(((coords1 - coords2) ** 2).sum(axis=1).mean())

    return rmsd


# TODO could allow more args
def smiles_from_mol(
    mol: Chem.rdchem.Mol,
    canonical: bool = True,
    include_stereocenters: bool = True,
    remove_hs: bool = False,
    explicit_hs: bool = False,
) -> Union[str, None]:
    """Create a SMILES string from a molecule

    Args:
        mol (Chem.Mol): RDKit molecule object
        canonical (bool): Whether to create a canonical SMILES, default True
        explicit_hs (bool): Whether to embed hydrogens in the mol before creating a SMILES, default False. If True
                this will create a new mol with all hydrogens embedded. Note that the SMILES created by doing this
                is not necessarily the same as creating a SMILES showing implicit hydrogens.

    Returns:
        str: SMILES string which could be None if the SMILES generation failed
    """

    if mol is None:
        return None

    if explicit_hs:
        mol = Chem.AddHs(mol)

    if remove_hs:
        mol = Chem.RemoveHs(mol)

    try:
        smiles = Chem.MolToSmiles(
            mol, canonical=True, isomericSmiles=include_stereocenters
        )
    except Exception as e:
        print(f"Error generating SMILES: {e}")
        smiles = None

    return smiles


def mol_from_smiles(
    smiles: str, explicit_hs: bool = False
) -> Union[Chem.rdchem.Mol, None]:
    """Create a RDKit molecule from a SMILES string

    Args:
        smiles (str): SMILES string
        explicit_hs (bool): Whether to embed explicit hydrogens into the mol

    Returns:
        Chem.Mol: RDKit molecule object or None if one cannot be created from the SMILES
    """

    if smiles is None:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol) if explicit_hs else mol
    except Exception:
        mol = None

    return mol


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


def mol_from_atoms(
    coords: ArrT,
    tokens: list[str],
    bonds: Optional[ArrT] = None,
    charges: Optional[ArrT] = None,
    hybridization: Optional[ArrT] = None,
    aromaticity: Optional[ArrT] = None,
    sanitise: bool = True,
    repeated_sanitise: bool = False,
    add_hs: bool = False,
    remove_hs: bool = False,
    kekulize: bool = False,
    fix_aromaticity: bool = False,
):
    """Create RDKit mol from atom coords and atom tokens (and optionally bonds)

    If any of the atom tokens are not valid atoms (do not exist on the periodic table), None will be returned.

    If bonds are not provided this function will create a partial molecule using the atomics and coordinates and then
    infer the bonds based on the coordinates using OpenBabel. Otherwise the bonds are added to the molecule as they
    are given in the bond array.

    If bonds are provided they must not contain any duplicates.

    If charges are not provided they are assumed to be 0 for all atoms.

    Args:
        coords (np.ndarray): Coordinate tensor, shape [n_atoms, 3]
        atomics (list[str]): Atomic numbers, length must be n_atoms
        bonds (np.ndarray, optional): Bond indices and types, shape [n_bonds, 3]
        charges (np.ndarray, optional): Charge for each atom, shape [n_atoms]
        hybridization (List, optional): Hybridization types for each atom, shape [n_atoms]
        aromaticity (List, optional): Whether each atom is aromatic, shape [n_atoms]
        sanitise_repeated (bool): Whether to repeatedly sanitise the molecule until it is valid, default False
        sanitise (bool): Whether to apply RDKit sanitization to the molecule, default True
        add_hs (bool): Whether to add explicit hydrogens to the molecule, default False
        remove_hs (bool): Whether to remove explicit hydrogens from the molecule, default False
        kekulize (bool): Whether to kekulize the molecule, default False
        fix_aromaticity (bool): Whether to fix aromaticity of the molecule, default False

    Returns:
        Chem.rdchem.Mol: RDKit molecule or None if one cannot be created
    """

    _check_shape_len(coords, 2, "coords")
    _check_dim_shape(coords, 1, 3, "coords")

    if coords.shape[0] != len(tokens):
        raise ValueError(
            "coords and atomics tensor must have the same number of atoms."
        )

    if bonds is not None:
        _check_shape_len(bonds, 2, "bonds")
        _check_dim_shape(bonds, 1, 3, "bonds")

    if charges is not None:
        _check_shape_len(charges, 1, "charges")
        _check_dim_shape(charges, 0, len(tokens), "charges")

    try:
        atomics = [PT.atomic_from_symbol(token) for token in tokens]
    except Exception as e:
        # print(f"Error: {e}")
        return None

    charges = charges.tolist() if charges is not None else [0] * len(tokens)

    # Add atom types and charges
    try:
        mol = Chem.EditableMol(Chem.Mol())
        for idx, atomic in enumerate(atomics):
            atom = Chem.Atom(atomic)
            atom.SetFormalCharge(charges[idx])
            if hybridization:
                atom.SetHybridization(hybridization[idx])
            if aromaticity:
                atom.SetIsAromatic(aromaticity[idx])
            mol.AddAtom(atom)
    except Exception as e:
        # print("Error adding atoms and charges: {e}")
        return None

    # Add 3D coords
    conf = Chem.Conformer(coords.shape[0])
    for idx, coord in enumerate(coords.tolist()):
        conf.SetAtomPosition(idx, coord)

    mol = mol.GetMol()
    mol.AddConformer(conf)

    if bonds is None:
        return _infer_bonds(mol)

    # Add bonds if they have been provided
    mol = Chem.EditableMol(mol)
    for bond in bonds.astype(np.int32).tolist():
        start, end, b_type = bond

        if b_type not in IDX_BOND_MAP:
            # print(f"Invalid bond type {b_type}")
            return None

        # Don't add self connections
        if start != end:
            b_type = IDX_BOND_MAP[b_type]
            if fix_aromaticity and b_type == 4:  # aromatic bond
                b_type = 1  # single bond
            mol.AddBond(start, end, b_type)

    try:
        mol = mol.GetMol()
        for atom in mol.GetAtoms():
            atom.UpdatePropertyCache(strict=False)
    except Exception:
        # print("Error building the molecule")
        return None

    if fix_aromaticity:
        try:
            for atom in mol.GetAtoms():
                atom.SetIsAromatic(False)
            for bond in mol.GetBonds():
                bond.SetIsAromatic(False)
            Chem.SanitizeMol(mol)
        except Exception as e:
            print(f"Failed to fix aromaticity during to_rdkit conversion: {e}")
            return None

    if repeated_sanitise:
        mol = _repeated_sanitization(mol)
        if mol is None:
            # print("Failed to sanitise the molecule after multiple attempts")
            return None

    if add_hs:
        try:
            mol = Chem.AddHs(mol, addCoords=True)
        except Exception:
            # print("Error adding hydrogens to the molecule")
            return None

    if remove_hs:
        try:
            mol = Chem.RemoveHs(mol)
        except Exception:
            # print("Error removing hydrogens from the molecule")
            return None

    if kekulize:
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            try:
                mol = Chem.RemoveHs(mol)
                Chem.Kekulize(mol, clearAromaticFlags=True)
            except Exception:
                # print("Error kekulizing the molecule")
                return None

    if sanitise and not fix_aromaticity:
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            print(
                f"Failed to sanitize molecule: {e}. "
                "Returning None and potentially correcting downstream."
            )
            return None

    return mol


def _repeated_sanitization(
    mol: Chem.rdchem.Mol, max_attempts: int = 5
) -> Chem.rdchem.Mol:
    """Repeatedly sanitise a molecule to ensure it is valid"""
    for attempt in range(max_attempts):
        try:
            Chem.SanitizeMol(mol)
            return mol
        except Exception:
            continue
    return None


def _infer_bonds(mol: Chem.rdchem.Mol):
    from openbabel import pybel

    coords = mol.GetConformer().GetPositions().tolist()
    coord_strs = ["\t".join([f"{c:.6f}" for c in cs]) for cs in coords]
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

    xyz_str_header = f"{str(mol.GetNumAtoms())}\n\n"
    xyz_strs = [
        f"{str(atom)}\t{coord_str}" for coord_str, atom in zip(coord_strs, atom_symbols)
    ]
    xyz_str = xyz_str_header + "\n".join(xyz_strs)

    try:
        pybel_mol = pybel.readstring("xyz", xyz_str)
    except Exception:
        pybel_mol = None

    if pybel_mol is None:
        return None

    mol_str = pybel_mol.write("mol")
    mol = Chem.MolFromMolBlock(mol_str, removeHs=False, sanitize=True)
    return mol


def generate_conformer(mol: Chem.rdchem.Mol, explicit_hs=True) -> Chem.rdchem.Mol:
    """Generate a conformer for an RDKit molecule

    Args:
        mol (Chem.Mol): RDKit molecule

    Returns:
        Chem.Mol: Molecule with generated conformer
    """
    # Copy the molecule to avoid modifying the original
    mol_copy = Chem.Mol(mol)
    mol_copy = Chem.AddHs(mol_copy) if explicit_hs else mol_copy

    # Generate the conformer
    AllChem.EmbedMolecule(mol_copy)

    return mol_copy


def write_sdf_file(sdf_path, molecules, name="mol"):
    w = Chem.SDWriter(str(sdf_path))
    for i, m in enumerate(molecules):
        if name:
            m.SetProp("_Name", f"{name}_{i}")
        if m is not None:
            w.write(m)
    w.close()


def canonicalize(smiles: str, include_stereocenters=True, remove_hs=False):
    mol = Chem.MolFromSmiles(smiles)
    if remove_hs:
        mol = Chem.RemoveHs(mol)

    if mol is not None:
        return Chem.MolToSmiles(
            mol, canonical=True, isomericSmiles=include_stereocenters
        )
    else:
        return None


def canonicalize_list(
    smiles_list,
    include_stereocenters=True,
    remove_hs=False,
):
    canonicalized_smiles = [
        canonicalize(smiles, include_stereocenters, remove_hs=remove_hs)
        for smiles in smiles_list
    ]
    # Remove None elements
    canonicalized_smiles = [s for s in canonicalized_smiles]

    return remove_duplicates(canonicalized_smiles)


def remove_duplicates(list_with_duplicates):
    unique_set = set()
    unique_list = []
    ids = []
    for i, element in enumerate(list_with_duplicates):
        if element not in unique_set and element is not None:
            unique_set.add(element)
            unique_list.append(element)
        else:
            ids.append(i)

    return unique_list, ids


def canonicalize_mol_list(
    mols: list[Chem.rdchem.Mol],
    ref_smiles: list[str],
    include_stereocenters=True,
    remove_hs=False,
):
    # Convert ref_smiles to set for O(1) lookup instead of O(n)
    ref_smiles_set = set(ref_smiles)

    # Use list comprehension with set for tracking uniqueness
    seen_smiles = set()
    unique_mols = []

    for mol in mols:
        smiles = smiles_from_mol(
            mol,
            canonical=True,
            include_stereocenters=include_stereocenters,
            remove_hs=remove_hs,
        )

        # Check all conditions in one go
        if (
            smiles is not None
            and smiles not in seen_smiles
            and smiles not in ref_smiles_set
        ):
            seen_smiles.add(smiles)
            unique_mols.append(mol)

    return unique_mols


def sanitize_list(
    mols: list[Chem.rdchem.Mol],
    ref_mols: Optional[list[Chem.rdchem.Mol]] = None,
    ref_mols_with_hs: Optional[list[Chem.rdchem.Mol]] = None,
    pdbs: Optional[list[str]] = None,
    pdbs_with_hs: Optional[list[str]] = None,
    sanitize: bool = False,
    filter_uniqueness: bool = False,
    filter_pdb: bool = False,
):

    rdkit_valid = [mol_is_valid(mol, connected=True) for mol in mols]
    valid_mols = [mol for mol, valid in zip(mols, rdkit_valid) if valid]
    if sanitize:
        for mol in valid_mols:
            AllChem.SanitizeMol(mol)
    if ref_mols is not None:
        valid_ref_mols = [mol for mol, valid in zip(ref_mols, rdkit_valid) if valid]
    if ref_mols_with_hs is not None:
        valid_ref_mols_with_hs = [
            mol for mol, valid in zip(ref_mols_with_hs, rdkit_valid) if valid
        ]
    if pdbs is not None:
        valid_pdbs = [pdb for pdb, valid in zip(pdbs, rdkit_valid) if valid]
    if pdbs_with_hs is not None:
        valid_pdbs_with_hs = [
            pdb for pdb, valid in zip(pdbs_with_hs, rdkit_valid) if valid
        ]

    if filter_uniqueness:
        valid_smiles = [Chem.MolToSmiles(mol) for mol in valid_mols]
        unique_valid_smiles, duplicate_ids = canonicalize_list(valid_smiles)
        valid_mols = [mol for i, mol in enumerate(valid_mols) if i not in duplicate_ids]
        if ref_mols is not None:
            valid_ref_mols = [
                mol for i, mol in enumerate(valid_ref_mols) if i not in duplicate_ids
            ]
        if ref_mols_with_hs is not None:
            valid_ref_mols_with_hs = [
                mol
                for i, mol in enumerate(valid_ref_mols_with_hs)
                if i not in duplicate_ids
            ]
        if pdbs is not None:
            valid_pdbs = [
                pdb for i, pdb in enumerate(valid_pdbs) if i not in duplicate_ids
            ]
        if pdbs_with_hs is not None:
            valid_pdbs_with_hs = [
                pdb
                for i, pdb in enumerate(valid_pdbs_with_hs)
                if i not in duplicate_ids
            ]

    if len(valid_mols) == 0:
        out = (
            ([], pdbs)
            if pdbs is not None
            else ([], pdbs_with_hs) if pdbs_with_hs is not None else []
        )
        return out

    if filter_pdb:
        read_fn = io_pdb.get_structure
        if pdbs is not None or pdbs_with_hs is not None:
            if pdbs is not None:
                assert pdbs_with_hs is None, "Cannot filter both pdbs and pdbs_with_hs"
                pdb_valid = [
                    (
                        read_fn(
                            io_pdb.PDBFile.read(str(pdb)), model=1, include_bonds=True
                        )
                        if pdb is not None
                        else None
                    )
                    for pdb in valid_pdbs
                ]
            elif pdbs_with_hs is not None:
                assert pdbs is None, "Cannot filter both pdbs and pdbs_with_hs"
                pdb_valid = [
                    (
                        read_fn(
                            io_pdb.PDBFile.read(str(pdb)), model=1, include_bonds=True
                        )
                        if pdb is not None
                        else None
                    )
                    for pdb in valid_pdbs_with_hs
                ]
            valid_pdbs = [pdb for pdb, valid in zip(valid_pdbs, pdb_valid) if valid]
            valid_mols = [mol for mol, valid in zip(valid_mols, pdb_valid) if valid]
            if ref_mols is not None:
                valid_ref_mols = [
                    mol for mol, valid in zip(valid_ref_mols, pdb_valid) if valid
                ]
            if ref_mols_with_hs is not None:
                valid_ref_mols_with_hs = [
                    mol
                    for mol, valid in zip(valid_ref_mols_with_hs, pdb_valid)
                    if valid
                ]
        else:
            raise ValueError("No PDB files provided to filter")

    if len(valid_mols) == 0:
        return []

    result = [valid_mols]
    if ref_mols is not None:
        result.append(valid_ref_mols)
    if ref_mols_with_hs is not None:
        result.append(valid_ref_mols_with_hs)
    if pdbs is not None:
        result.append(valid_pdbs)
    if pdbs_with_hs is not None:
        result.append(valid_pdbs_with_hs)
    if len(result) == 1:
        return result[0]
    return tuple(result)


class ConformerGenerator:
    """
    Generates and caches RDKit conformers for molecular graphs.
    Uses GeometricMol.to_rdkit() for reliable molecule conversion.
    Stores multiple conformers per molecule and randomly samples from them.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_conformers: int = 5,
        max_iters: int = 200,
        enable_caching: bool = True,
        vocab: Optional[dict] = None,
    ):
        self.max_conformers = max_conformers
        self.max_iters = max_iters
        self.enable_caching = enable_caching
        self.vocab = vocab

        # Setup cache directory
        if cache_dir is None:
            cache_dir = Path.home() / ".flowr_cache" / "conformers"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache for faster access - now stores lists of conformers
        self._memory_cache: Dict[str, List[torch.Tensor]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_molecule_hash(self, to_mol) -> str:
        """Generate a hash for caching based on molecular graph"""
        try:
            # Use GeometricMol's built-in RDKit conversion
            rdkit_mol = to_mol.to_rdkit(vocab=self.vocab, sanitise=True)
            if rdkit_mol is None:
                return None

            # Create hash from canonical SMILES
            canonical_smiles = Chem.MolToSmiles(rdkit_mol)
            return hashlib.md5(canonical_smiles.encode()).hexdigest()
        except Exception as e:
            print(f"Error generating molecule hash: {e}")
            return None

    def _load_from_cache(self, mol_hash: str) -> Optional[List[torch.Tensor]]:
        """Load all conformers from cache"""
        if not self.enable_caching or mol_hash is None:
            return None

        # Check memory cache first
        if mol_hash in self._memory_cache:
            self._cache_hits += 1
            # Return deep copies to avoid modification of cached data
            return [conf.clone() for conf in self._memory_cache[mol_hash]]

        # Check disk cache
        cache_file = self.cache_dir / f"{mol_hash}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    conformers_list = pickle.load(f)
                    conformers_tensors = [
                        torch.tensor(coords, dtype=torch.float32)
                        for coords in conformers_list
                    ]

                # Store in memory cache
                self._memory_cache[mol_hash] = [
                    conf.clone() for conf in conformers_tensors
                ]
                self._cache_hits += 1
                return conformers_tensors
            except Exception as e:
                print(f"Error loading cached conformers: {e}")

        self._cache_misses += 1
        return None

    def _save_to_cache(self, mol_hash: str, conformers: List[torch.Tensor]):
        """Save all conformers to cache"""
        if not self.enable_caching or mol_hash is None or not conformers:
            return

        # Save to memory cache
        self._memory_cache[mol_hash] = [conf.clone() for conf in conformers]

        # Save to disk cache - convert to numpy for serialization
        cache_file = self.cache_dir / f"{mol_hash}.pkl"
        try:
            conformers_numpy = [conf.numpy() for conf in conformers]
            with open(cache_file, "wb") as f:
                pickle.dump(conformers_numpy, f)
        except Exception as e:
            print(f"Error saving conformers to cache: {e}")

    def generate_conformer_from_graph(self, to_mol) -> Optional[torch.Tensor]:
        """Generate/retrieve RDKit conformer from GeometricMol and randomly sample from available conformers"""
        mol_hash = self._get_molecule_hash(to_mol)

        # Try to load all conformers from cache first
        cached_conformers = self._load_from_cache(mol_hash)
        if cached_conformers is not None and len(cached_conformers) > 0:
            # Randomly sample from cached conformers
            selected_conformer = cached_conformers[
                np.random.randint(len(cached_conformers))
            ]
            return selected_conformer

        # Generate new conformers using GeometricMol's to_rdkit method
        new_conformers = self._generate_all_conformers_from_mol(to_mol)

        # Cache all generated conformers
        if new_conformers is not None and len(new_conformers) > 0:
            self._save_to_cache(mol_hash, new_conformers)
            # Randomly sample from newly generated conformers
            selected_conformer = new_conformers[np.random.randint(len(new_conformers))]
            return selected_conformer

        return None

    def _generate_all_conformers_from_mol(self, to_mol) -> Optional[List[torch.Tensor]]:
        """Generate all conformers from GeometricMol using its to_rdkit() method"""
        try:
            # Use GeometricMol's built-in RDKit conversion
            rdkit_mol = to_mol.to_rdkit(vocab=self.vocab, sanitise=True)
            if rdkit_mol is None:
                return None

            rdkit_mol = Chem.AddHs(rdkit_mol)

            # Generate multiple conformers with random seeds
            conformer_ids = AllChem.EmbedMultipleConfs(
                rdkit_mol,
                numConfs=self.max_conformers,
                randomSeed=np.random.randint(0, 100000),
                clearConfs=True,
                useRandomCoords=True,
            )

            if len(conformer_ids) == 0:
                # Fallback to single conformer with different random seeds
                conformers = []
                for attempt in range(min(self.max_conformers, 10)):
                    temp_mol = Chem.Mol(rdkit_mol)
                    # Use different random seed for each attempt
                    result = AllChem.EmbedMolecule(
                        temp_mol, randomSeed=np.random.randint(0, 100000)
                    )
                    if result == 0:
                        coords = self._extract_coordinates(temp_mol, to_mol.seq_length)
                        if coords is not None:
                            conformers.append(coords)

                return conformers if conformers else None

            # Extract all conformer coordinates
            conformers = []
            for conf_id in conformer_ids:
                coords = self._extract_coordinates(
                    rdkit_mol, to_mol.seq_length, conf_id
                )
                if coords is not None:
                    conformers.append(coords)

            return conformers if conformers else None

        except Exception as e:
            print(f"Error generating conformers: {e}")
            return None

    def _extract_coordinates(
        self, mol: Chem.Mol, expected_atoms: int, conf_id: int = 0
    ) -> Optional[torch.Tensor]:
        """Extract coordinates from conformer, handling hydrogen mismatch"""
        try:
            conf = mol.GetConformer(conf_id)
            coords = conf.GetPositions()

            # If we have too many atoms (likely due to added hydrogens), remove them
            if coords.shape[0] != expected_atoms and coords.shape[0] > expected_atoms:
                mol_no_h = Chem.RemoveHs(mol)
                if mol_no_h.GetNumAtoms() == expected_atoms:
                    conf_no_h = mol_no_h.GetConformer(conf_id)
                    coords = conf_no_h.GetPositions()

            # Only return if we have the expected number of atoms
            if coords.shape[0] == expected_atoms:
                return torch.tensor(coords, dtype=torch.float32)

            return None

        except Exception as e:
            print(f"Error extracting coordinates: {e}")
            return None

    def get_all_conformers(self, to_mol) -> Optional[List[torch.Tensor]]:
        """Get all available conformers for a molecule (useful for analysis)"""
        mol_hash = self._get_molecule_hash(to_mol)
        cached_conformers = self._load_from_cache(mol_hash)

        if cached_conformers is not None:
            return cached_conformers

        # Generate if not cached
        new_conformers = self._generate_all_conformers_from_mol(to_mol)
        if new_conformers is not None:
            self._save_to_cache(mol_hash, new_conformers)

        return new_conformers

    def clear_cache(self):
        """Clear both memory and disk cache"""
        self._memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()

    def get_cache_stats(self) -> dict:
        """Get cache performance statistics"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

        total_conformers = sum(
            len(conformers) for conformers in self._memory_cache.values()
        )

        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "memory_cache_size": len(self._memory_cache),
            "total_cached_conformers": total_conformers,
            "disk_cache_size": len(list(self.cache_dir.glob("*.pkl"))),
        }

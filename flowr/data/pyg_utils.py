import os
import pickle
import threading
from collections import defaultdict
from glob import glob
from typing import Optional, Union

import numpy as np
import rdkit
import rmsd
import torch
from Bio.PDB.Polypeptide import is_aa
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import AllChem, ChemicalFeatures
from torch_geometric.data import Data
from torch_geometric.utils import sort_edge_index, subgraph

from flowr.util.rdkit import _infer_bonds

fdefName = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

RDLogger.DisableLog("rdApp.*")

x_map = {
    "is_aromatic": [False, True],
    "is_in_ring": [False, True],
    "hybridization": [
        rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED,
        rdkit.Chem.rdchem.HybridizationType.S,
        rdkit.Chem.rdchem.HybridizationType.SP,
        rdkit.Chem.rdchem.HybridizationType.SP2,
        rdkit.Chem.rdchem.HybridizationType.SP3,
        rdkit.Chem.rdchem.HybridizationType.SP2D,
        rdkit.Chem.rdchem.HybridizationType.SP3D,
        rdkit.Chem.rdchem.HybridizationType.SP3D2,
        rdkit.Chem.rdchem.HybridizationType.OTHER,
    ],
    "is_h_donor": [False, True],
    "is_h_acceptor": [False, True],
}


def get_fc_edge_index_with_offset(n, offset: int = 0, device="cpu"):
    row = torch.arange(n, dtype=torch.long)
    col = torch.arange(n, dtype=torch.long)
    row = row.view(-1, 1).repeat(1, n).view(-1)
    col = col.repeat(n)
    fc_edge_index = torch.stack([col, row], dim=0)
    mask = fc_edge_index[0] != fc_edge_index[1]
    fc_edge_index = fc_edge_index[:, mask]
    fc_edge_index += offset
    return fc_edge_index.to(device)


def mol_to_torch_geometric(
    mol,
    atom_encoder,
    smiles=None,
    remove_hydrogens: bool = False,
    cog_proj: bool = True,
    add_ad=True,
    add_pocket=False,
    **kwargs,
):

    if remove_hydrogens:
        # mol = Chem.RemoveAllHs(mol)
        mol = Chem.RemoveHs(
            mol
        )  # only remove (explicit) hydrogens attached to molecular graph
        Chem.Kekulize(mol, clearAromaticFlags=True)

    if smiles is None:
        smiles = Chem.MolToSmiles(mol)
    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
    edge_index = adj.nonzero().contiguous().T
    bond_types = adj[edge_index[0], edge_index[1]]
    bond_types[bond_types == 1.5] = 4
    if remove_hydrogens:
        assert max(bond_types) != 4
    edge_attr = bond_types.long()

    pos = torch.tensor(mol.GetConformers()[0].GetPositions()).float()
    if cog_proj:
        pos = pos - torch.mean(pos, dim=0, keepdim=True)
    atom_types = []
    all_charges = []
    is_aromatic = []
    is_in_ring = []
    sp_hybridization = []

    for atom in mol.GetAtoms():
        atom_types.append(atom_encoder[atom.GetSymbol()])
        all_charges.append(
            atom.GetFormalCharge()
        )  # TODO: check if implicit Hs should be kept
        is_aromatic.append(x_map["is_aromatic"].index(atom.GetIsAromatic()))
        is_in_ring.append(x_map["is_in_ring"].index(atom.IsInRing()))
        sp_hybridization.append(x_map["hybridization"].index(atom.GetHybridization()))

    atom_types = torch.Tensor(atom_types).long()
    all_charges = torch.Tensor(all_charges).long()

    is_aromatic = torch.Tensor(is_aromatic).long()
    is_in_ring = torch.Tensor(is_in_ring).long()
    hybridization = torch.Tensor(sp_hybridization).long()
    if add_ad:
        # hydrogen bond acceptor and donor
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

        is_donor = torch.from_numpy(is_donor).long()
        is_acceptor = torch.from_numpy(is_acceptor).long()
    else:
        is_donor = is_acceptor = None

    additional = {}
    if "wbo" in kwargs:
        wbo = torch.Tensor(kwargs["wbo"])[edge_index[0], edge_index[1]].float()
        additional["wbo"] = wbo
    if "mulliken" in kwargs:
        mulliken = torch.Tensor(kwargs["mulliken"]).float()
        additional["mulliken"] = mulliken
    if "grad" in kwargs:
        grad = torch.Tensor(kwargs["grad"]).float()
        additional["grad"] = grad

    data = Data(
        x=atom_types,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        charges=all_charges,
        smiles=smiles,
        is_aromatic=is_aromatic,
        is_in_ring=is_in_ring,
        is_h_donor=is_donor,
        is_h_acceptor=is_acceptor,
        hybridization=hybridization,
        mol=mol,
        **additional,
    )

    return data


# in case the rdkit.molecule has explicit hydrogens, the number of attached hydrogens to heavy atoms are not saved
def remove_hydrogens(data: Data):
    to_keep = data.x > 0
    new_edge_index, new_edge_attr = subgraph(
        to_keep,
        data.edge_index,
        data.edge_attr,
        relabel_nodes=True,
        num_nodes=len(to_keep),
    )
    new_pos = data.pos[to_keep] - torch.mean(data.pos[to_keep], dim=0)

    newdata = Data(
        x=data.x[to_keep] - 1,  # Shift onehot encoding to match atom decoder
        pos=new_pos,
        charges=data.charges[to_keep],
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
        mol=data.mol,
    )

    if hasattr(data, "is_aromatic"):
        newdata["is_aromatic"] = data.get("is_aromatic")[to_keep]
    if hasattr(data, "is_in_ring"):
        newdata["is_in_ring"] = data.get("is_in_ring")[to_keep]
    if hasattr(data, "hybridization"):
        newdata["hybridization"] = data.get("hybridization")[to_keep]

    return newdata


def save_pickle(array, path, exist_ok=True):
    if exist_ok:
        with open(path, "wb") as f:
            pickle.dump(array, f)
    else:
        if not os.path.exists(path):
            with open(path, "wb") as f:
                pickle.dump(array, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def write_xyz_file(coords, atom_types, filename):
    out = f"{len(coords)}\n\n"
    assert len(coords) == len(atom_types)
    for i in range(len(coords)):
        out += f"{atom_types[i]} {coords[i, 0]:.3f} {coords[i, 1]:.3f} {coords[i, 2]:.3f}\n"
    with open(filename, "w") as f:
        f.write(out)


def write_xyz_file_from_batch(
    pos,
    atoms,
    batch,
    atom_decoder=None,
    pos_pocket=None,
    atoms_pocket=None,
    batch_pocket=None,
    pocket_name=None,
    joint_traj=False,
    path="/scratch1/e3moldiffusion/logs/crossdocked",
    i=0,
):
    if not os.path.exists(path):
        os.makedirs(path)

    atomsxmol = batch.bincount()
    num_atoms_prev = 0
    for k, num_atoms in enumerate(atomsxmol):
        if pocket_name is not None:
            pdb = pocket_name[k].split("_")[0].split(".pdb")[0]
            save_dir = os.path.join(path, f"{pdb}")
        else:
            save_dir = os.path.join(path, f"graph_{k}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        ats = torch.argmax(atoms[num_atoms_prev : num_atoms_prev + num_atoms], dim=1)
        types = [atom_decoder[int(a)] for a in ats]
        positions = pos[num_atoms_prev : num_atoms_prev + num_atoms]
        write_xyz_file(positions, types, os.path.join(save_dir, f"mol_{i}.xyz"))

        num_atoms_prev += num_atoms

    if joint_traj:
        atomsxmol = batch.bincount()
        atomsxmol_pocket = batch_pocket.bincount()
        num_atoms_prev = 0
        num_atoms_prev_pocket = 0
        for k, (num_atoms, num_atoms_pocket) in enumerate(
            zip(atomsxmol, atomsxmol_pocket)
        ):
            save_dir = os.path.join(path, f"graph_{k}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            ats = torch.argmax(
                atoms[num_atoms_prev : num_atoms_prev + num_atoms], dim=1
            )
            ats_pocket = torch.argmax(
                atoms_pocket[
                    num_atoms_prev_pocket : num_atoms_prev_pocket + num_atoms_pocket
                ],
                dim=1,
            )
            types = [atom_decoder[int(a)] for a in ats]
            types_pocket = [
                "B" for _ in range(len(ats_pocket))
            ]  # [atom_decoder[int(a)] for a in ats_pocket]
            positions = pos[num_atoms_prev : num_atoms_prev + num_atoms]
            positions_pocket = pos_pocket[
                num_atoms_prev_pocket : num_atoms_prev_pocket + num_atoms_pocket
            ]

            types_joint = types + types_pocket
            positions_joint = torch.cat([positions, positions_pocket], dim=0)

            write_xyz_file(
                positions_joint,
                types_joint,
                os.path.join(save_dir, f"lig_pocket_{i}.xyz"),
            )

            num_atoms_prev += num_atoms
            num_atoms_prev_pocket += num_atoms_pocket


def get_key(fp):
    filename = os.path.splitext(os.path.basename(fp))[0]
    int_part = filename.split("_")[-1]
    return int(int_part)


def write_trajectory_as_xyz(
    molecules,
    path,
    strict=True,
    joint_traj=False,
):
    try:
        os.makedirs(path)
    except OSError:
        pass

    for i, mol in enumerate(molecules):
        rdkit_mol = mol.rdkit_mol
        valid = (
            rdkit_mol is not None
            and mol.compute_validity(rdkit_mol, strict=strict) is not None
        )
        if valid:
            if joint_traj:
                files = sorted(
                    glob(os.path.join(path, f"graph_{i}/lig_pocket_*.xyz")), key=get_key
                )
            else:
                files = sorted(
                    glob(os.path.join(path, f"graph_{i}/mol_*.xyz")), key=get_key
                )
            traj_path = os.path.join(path, f"trajectory_{i}.xyz")
            molecules[i].trajectory = traj_path
            for j, file in enumerate(files):
                with open(file, "r") as f:
                    lines = f.readlines()

                with open(traj_path, "a") as file:
                    for line in lines:
                        file.write(line)
                    if (
                        j == len(files) - 1
                    ):  ####write the last timestep 10x for better visibility
                        for _ in range(10):
                            for line in lines:
                                file.write(line)
        else:
            molecules[i].trajectory = None


def calc_rmsd(mol1, mol2):
    U = rmsd.kabsch(mol1, mol2)
    mol1 = np.dot(mol1, U)
    return rmsd.rmsd(mol1, mol2)


def create_bond_graph(data, atom_encoder):
    mol = data.mol
    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
    edge_index = adj.nonzero().contiguous().T
    bond_types = adj[edge_index[0], edge_index[1]]
    bond_types[bond_types == 1.5] = 4
    edge_attr = bond_types.long()

    pos = torch.tensor(mol.GetConformers()[0].GetPositions()).float()
    assert calc_rmsd(pos.numpy(), data.pos.numpy()) < 1.0e-3

    atom_types = []
    all_charges = []
    for atom in mol.GetAtoms():
        atom_types.append(atom_encoder[atom.GetSymbol()])
        all_charges.append(atom.GetFormalCharge())
    atom_types = torch.Tensor(atom_types).long()
    assert (atom_types == data.x).all()

    all_charges = torch.Tensor(all_charges).long()
    data.charges = all_charges

    # edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    # E = to_dense_adj(
    #     edge_index=edge_index,
    #     batch=torch.zeros_like(atom_types),
    #     edge_attr=edge_attr,
    #     max_num_nodes=len(atom_types),
    # )
    # diag_mask = ~torch.eye(5, dtype=torch.bool)
    # E = F.one_hot(E, num_classes=5).float() * diag_mask
    data.bond_index = edge_index
    data.bond_attr = edge_attr

    data = fully_connected_edge_idx(data=data, without_self_loop=True)

    return data


def fully_connected_edge_idx(data: Data, without_self_loop: bool = True):
    N = data.pos.size(0)
    row = torch.arange(N, dtype=torch.long)
    col = torch.arange(N, dtype=torch.long)
    row = row.view(-1, 1).repeat(1, N).view(-1)
    col = col.repeat(N)
    fc_edge_index = torch.stack([row, col], dim=0)
    if without_self_loop:
        mask = fc_edge_index[0] != fc_edge_index[1]
        fc_edge_index = fc_edge_index[:, mask]

    fc_edge_index = sort_edge_index(fc_edge_index, sort_by_row=False, num_nodes=N)
    data.fc_edge_index = fc_edge_index

    return data


def atom_type_config(dataset: str = "qm9"):
    if dataset == "qm9":
        mapping = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
    elif dataset == "aqm":
        mapping = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4, "P": 5, "S": 6, "Cl": 7}
    elif dataset == "fullerene":
        mapping = {"H": 0, "C": 1, "N": 2, "Cl": 3}
    elif dataset == "drugs":
        mapping = {
            "H": 0,
            "B": 1,
            "C": 2,
            "N": 3,
            "O": 4,
            "F": 5,
            "Al": 6,
            "Si": 7,
            "P": 8,
            "S": 9,
            "Cl": 10,
            "As": 11,
            "Br": 12,
            "I": 13,
            "Hg": 14,
            "Bi": 15,
        }
    else:
        raise ValueError("Dataset not found!")
    return mapping


class Statistics:
    def __init__(
        self,
        num_nodes,
        atom_types,
        bond_types,
        charge_types,
        valencies,
        bond_lengths,
        bond_angles,
        dihedrals=None,
        is_in_ring=None,
        is_aromatic=None,
        hybridization=None,
        force_norms=None,
        is_h_donor=None,
        is_h_acceptor=None,
    ):
        self.num_nodes = num_nodes
        # print("NUM NODES IN STATISTICS", num_nodes)
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.charge_types = charge_types
        self.valencies = valencies
        self.bond_lengths = bond_lengths
        self.bond_angles = bond_angles
        self.dihedrals = dihedrals
        self.is_in_ring = is_in_ring
        self.is_aromatic = is_aromatic
        self.hybridization = hybridization
        self.force_norms = force_norms
        self.is_h_donor = is_h_donor
        self.is_h_acceptor = is_h_acceptor


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
    except:
        return False

    n_frags = len(AllChem.GetMolFrags(mol_copy))
    if connected and n_frags != 1:
        return False

    return True


IDX_BOND_MAP = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC,
}
BOND_IDX_MAP = {bond: idx for idx, bond in IDX_BOND_MAP.items()}

IDX_CHARGE_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: -1, 5: -2, 6: -3}
CHARGE_IDX_MAP = {charge: idx for idx, charge in IDX_CHARGE_MAP.items()}


ArrT = np.ndarray


def mol_from_atoms(
    coords: ArrT,
    tokens: list[str],
    bonds: Optional[ArrT] = None,
    charges: Optional[ArrT] = None,
    sanitise=True,
    remove_hs=False,
    kekulize=False,
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
        sanitise (bool): Whether to apply RDKit sanitization to the molecule, default True

    Returns:
        Chem.rdchem.Mol: RDKit molecule or None if one cannot be created
    """

    PT = PeriodicTable()
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
    except Exception:
        # print(f"Error: {e}")
        return None

    charges = charges.tolist() if charges is not None else [0] * len(tokens)

    # Add atom types and charges
    mol = Chem.EditableMol(Chem.Mol())
    for idx, atomic in enumerate(atomics):
        atom = Chem.Atom(atomic)
        atom.SetFormalCharge(charges[idx])
        mol.AddAtom(atom)

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
            mol.AddBond(start, end, b_type)

    try:
        mol = mol.GetMol()
        for atom in mol.GetAtoms():
            atom.UpdatePropertyCache(strict=False)
    except Exception:
        # print("Error building the molecule")
        return None

    if remove_hs:
        mol = Chem.RemoveHs(mol)

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

    if sanitise:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            # print("Error sanitising the molecule")
            return None

    return mol


def _remove_hs_from_dict(mol: dict):
    PT = PeriodicTable()
    device = mol["device"]

    coords = mol["coords"].numpy()
    atomics = mol["atomics"].numpy()
    bond_types = mol["bond_types"]
    bond_indices = mol["bond_indices"]
    bonds = torch.cat((bond_indices, bond_types.unsqueeze(1)), dim=-1).numpy()
    charges = mol["charges"].numpy()
    atomics = atomics.tolist()
    tokens = [PT.symbol_from_atomic(a) for a in atomics]
    mol = mol_from_atoms(coords, tokens, bonds, charges, sanitise=False, kekulize=False)
    mol_data = mol_to_torch(mol, remove_hs=True)
    return {
        "coords": mol_data["coords"],
        "atomics": mol_data["atomics"],
        "bond_indices": mol_data["bond_indices"],
        "bond_types": mol_data["bond_types"],
        "charges": mol_data["charges"],
        "id": mol_data["smiles"],
        "device": device,
    }


def torch_to_mol(mol: dict):
    PT = PeriodicTable()
    coords = mol["coords"].numpy()
    atomics = mol["atomics"].numpy()
    bond_types = mol["bond_types"]
    bond_indices = mol["bond_indices"]
    bonds = torch.cat((bond_indices, bond_types.unsqueeze(1)), dim=-1).numpy()
    charges = mol["charges"].numpy()
    atomics = atomics.tolist()
    tokens = [PT.symbol_from_atomic(a) for a in atomics]
    mol = mol_from_atoms(coords, tokens, bonds, charges, sanitise=False, kekulize=False)
    return mol


def mol_to_torch(
    mol,
    smiles=None,
    remove_hs: bool = False,
):
    if remove_hs:
        # mol = Chem.RemoveAllHs(mol)
        mol = Chem.RemoveHs(
            mol
        )  # only remove (explicit) hydrogens attached to molecular graph
        Chem.Kekulize(mol, clearAromaticFlags=True)

    if smiles is None:
        smiles = Chem.MolToSmiles(mol)
    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
    bond_indices = adj.nonzero().contiguous().T
    bond_indices = bond_indices[:, bond_indices[0] < bond_indices[1]]
    bond_types = adj[bond_indices[0], bond_indices[1]]
    bond_types[bond_types == 1.5] = 4
    if remove_hs:
        assert max(bond_types) < 4
    bond_types = bond_types.long()
    bond_indices = bond_indices.T

    pos = torch.tensor(mol.GetConformers()[0].GetPositions()).float()
    atom_types = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()]).long()
    all_charges = torch.tensor([a.GetFormalCharge() for a in mol.GetAtoms()]).long()

    return pos, atom_types, bond_indices, bond_types, all_charges, smiles


def _check_mol_valid(mol: dict):
    PT = PeriodicTable()
    coords = mol["coords"].numpy()
    atomics = mol["atomics"].numpy()
    bond_types = mol["bond_types"]
    bond_indices = mol["bond_indices"]
    bonds = torch.cat((bond_indices, bond_types.unsqueeze(1)), dim=-1).numpy()
    charges = mol["charges"]
    atomics = atomics.tolist()
    tokens = [PT.symbol_from_atomic(a) for a in atomics]
    mol = mol_from_atoms(coords, tokens, bonds, charges, sanitise=False, kekulize=True)
    return mol_is_valid(mol, connected=True)


def from_bytes(
    data: bytes,
    atom_encoder: dict,
    remove_hs_lig: bool = False,
    remove_hs_pocket: bool = True,
    cutoff: float = None,
):
    obj = pickle.loads(data)

    lig_obj = pickle.loads(obj["ligand"])
    if remove_hs_lig:
        # lig_obj = BuildMolecule(lig_obj, remove_hs=True, kekulize=True).molecule
        lig_obj = _remove_hs_from_dict(lig_obj)
    # if not _check_mol_valid(lig_obj):
    #     return
    mol = torch_to_mol(lig_obj)
    lig_coords = torch.tensor(mol.GetConformers()[0].GetPositions()).float()
    lig_atoms = torch.tensor([atom_encoder[a.GetSymbol()] for a in mol.GetAtoms()])

    pocket_atoms = defaultdict()
    pocket_bonds = defaultdict()
    pocket_atoms["apo"] = None
    pocket_bonds["apo"] = None
    pocket_obj = pickle.loads(obj["holo"])
    if remove_hs_pocket:
        atoms = pocket_obj["atoms"]
        bonds = pocket_obj["bonds"]
        protein_atom_mask = atoms.element != "H"
        atom_struc = atoms.copy()
        atom_struc.bonds = bonds.copy()
        structure = atom_struc[protein_atom_mask]
        bonds = structure.bonds

    if cutoff is not None:
        # Cut pocket
        res_ids = set(structure.res_id)
        res_id_filter = []
        for res_id in res_ids:
            res = structure[structure.res_id == res_id]
            if (
                is_aa(res.res_name[0], standard=True)
                and (
                    np.linalg.norm(
                        res.coord[:, None, :] - lig_coords.numpy()[None, :, :], axis=-1
                    )
                ).min()
                < cutoff
            ):
                res_id_filter.append(res_id)
        structure = structure[np.isin(structure.res_id, res_id_filter)]
        bonds = structure.bonds

    pocket_atoms["holo"] = structure
    pocket_bonds["holo"] = bonds

    holo_coords = torch.tensor(pocket_atoms["holo"].coord)
    holo_atoms = torch.tensor([atom_encoder[a] for a in pocket_atoms["holo"].element])

    return mol, lig_coords, lig_atoms, holo_coords, holo_atoms

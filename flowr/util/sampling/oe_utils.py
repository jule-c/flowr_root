import itertools

# import torch_scatter
import math
import random
from copy import deepcopy

import networkx as nx
import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
import rdkit.Chem.rdMolTransforms
import torch
import torch_geometric
from rdkit.Chem import rdMolTransforms
from rdkit.Geometry import Point3D
from tqdm import tqdm


def logger(text):
    print(text)


atomTypes = ["H", "C", "B", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"]
formalCharge = [-1, -2, 1, 2, 0]
hybridization = [
    rdkit.Chem.rdchem.HybridizationType.S,
    rdkit.Chem.rdchem.HybridizationType.SP,
    rdkit.Chem.rdchem.HybridizationType.SP2,
    rdkit.Chem.rdchem.HybridizationType.SP3,
    rdkit.Chem.rdchem.HybridizationType.SP3D,
    rdkit.Chem.rdchem.HybridizationType.SP3D2,
    rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED,
]
num_single_bonds = [0, 1, 2, 3, 4, 5, 6]
num_double_bonds = [0, 1, 2, 3, 4]
num_triple_bonds = [0, 1, 2]
num_aromatic_bonds = [0, 1, 2, 3, 4]
bondTypes = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]


class AtomFragmentLibrary(torch_geometric.data.Dataset):
    def __init__(self, db):
        super(AtomFragmentLibrary, self).__init__()
        self.db = db

    def __len__(self):
        return len(self.db)

    def process_fragment(self, key):
        mol = self.db.iloc[key].mol
        xyz = torch.tensor(mol.GetConformer().GetPositions())
        center_of_mass = torch.sum(xyz, dim=0) / len(xyz)
        positions = xyz - center_of_mass

        adj = rdkit.Chem.GetAdjacencyMatrix(mol)
        edge_index = adjacency_to_undirected_edge_index(adj)

        # Edge Features --> rdkit ordering of edges
        bonds = []
        for b in range(int(edge_index.shape[1] / 2)):
            bond_index = edge_index[:, ::2][:, b]
            bond = mol.GetBondBetweenAtoms(int(bond_index[0]), int(bond_index[1]))
            bonds.append(bond)
        edge_features = getEdgeFeatures(bonds)

        # Node Features --> rdkit ordering of atoms
        atoms = rdkit.Chem.rdchem.Mol.GetAtoms(mol)
        atom_symbols = [atom.GetSymbol() for atom in atoms]
        node_features = getNodeFeatures(atoms)

        x = torch.as_tensor(node_features)
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        edge_attr = torch.as_tensor(edge_features)
        pos = positions

        data = torch_geometric.data.Data(
            x=x.float(),
            edge_index=edge_index,
            edge_attr=edge_attr.float(),
            pos=pos.float(),
        )

        return data

    def process_atom(self, key):
        x = torch.tensor(self.db.iloc[key].atom_features).unsqueeze(0)
        pos = torch.tensor(np.zeros((1, 3)))  # centered at origin

        data = torch_geometric.data.Data(
            x=x.float(),
            edge_index=torch.tensor([[], []], dtype=torch.long),
            edge_attr=torch.tensor([]),
            pos=pos.float(),
        )

        return data

    def __getitem__(self, key):
        if self.db.iloc[key].is_fragment == 1:
            data = self.process_fragment(key)
        else:
            data = self.process_atom(key)

        return data


def filter_mol(mol, unique_atoms, fragment_library_smiles):
    passed = True

    ring_fragments = get_ring_fragments(mol)
    for frag in ring_fragments:
        smiles = get_fragment_smiles(mol, frag)
        if smiles not in fragment_library_smiles:
            passed = False
            break

    atoms = rdkit.Chem.rdchem.Mol.GetAtoms(mol)
    node_features = getNodeFeatures(atoms)
    for f, feat in enumerate(node_features):
        if list(feat) not in unique_atoms.tolist():
            passed = False
            break

    return passed


def get_rings(mol):
    return mol.GetRingInfo().AtomRings()


def get_acyclic_single_bonds(mol):
    AcyclicBonds = rdkit.Chem.MolFromSmarts("[*]!@[*]")
    SingleBonds = rdkit.Chem.MolFromSmarts("[*]-[*]")
    acyclicBonds = mol.GetSubstructMatches(AcyclicBonds)
    singleBonds = mol.GetSubstructMatches(SingleBonds)

    acyclicBonds = [tuple(sorted(b)) for b in acyclicBonds]
    singleBonds = [tuple(sorted(b)) for b in singleBonds]

    select_bonds = set(acyclicBonds).intersection(set(singleBonds))
    return select_bonds


def get_multiple_bonds_to_ring(mol):
    BondToRing = rdkit.Chem.MolFromSmarts("[r]!@[*]")
    bonds_to_rings = mol.GetSubstructMatches(BondToRing)
    NonSingleBond = rdkit.Chem.MolFromSmarts("[*]!-[*]")
    non_single_bonds = mol.GetSubstructMatches(NonSingleBond)

    bonds_to_rings = [tuple(sorted(b)) for b in bonds_to_rings]
    non_single_bonds = [tuple(sorted(b)) for b in non_single_bonds]

    return tuple(set(bonds_to_rings).intersection(set(non_single_bonds)))


def get_rigid_ring_linkers(mol):
    RingLinker = rdkit.Chem.MolFromSmarts("[r]!@[r]")
    ring_linkers = mol.GetSubstructMatches(RingLinker)

    NonSingleBond = rdkit.Chem.MolFromSmarts("[*]!-[*]")
    non_single_bonds = mol.GetSubstructMatches(NonSingleBond)

    ring_linkers = [tuple(sorted(b)) for b in ring_linkers]
    non_single_bonds = [tuple(sorted(b)) for b in non_single_bonds]

    return tuple(set(ring_linkers).intersection(set(non_single_bonds)))


def get_ring_fragments(mol):
    rings = get_rings(mol)

    rings = [set(r) for r in rings]

    # combining rigid ring structures connected by rigid (non-single) bond (they will be combined in the next step)
    rigid_ring_linkers = get_rigid_ring_linkers(mol)
    new_rings = []
    for ring in rings:
        new_ring = ring
        for bond in rigid_ring_linkers:
            if (bond[0] in ring) or (bond[1] in ring):
                new_ring = new_ring.union(set(bond))
        new_rings.append(new_ring)
    rings = new_rings

    # joining ring structures
    N_rings = len(rings)
    done = False

    joined_rings = []
    for i in range(0, len(rings)):
        joined_ring_i = set(rings[i])
        done = False
        while not done:
            for j in range(0, len(rings)):  # i+1
                ring_j = set(rings[j])
                if (len(joined_ring_i.intersection(ring_j)) > 0) & (
                    joined_ring_i.union(ring_j) != joined_ring_i
                ):
                    joined_ring_i = joined_ring_i.union(ring_j)
                    done = False
                    break
            else:
                done = True

        if joined_ring_i not in joined_rings:
            joined_rings.append(joined_ring_i)

    rings = joined_rings

    # adding in rigid (non-single) bonds to these ring structures
    multiple_bonds_to_rings = get_multiple_bonds_to_ring(mol)
    new_rings = []
    for ring in rings:
        new_ring = ring
        for bond in multiple_bonds_to_rings:
            if (bond[0] in ring) or (bond[1] in ring):
                new_ring = new_ring.union(set(bond))
        new_rings.append(new_ring)
    rings = new_rings

    return rings


def get_fragment_smarts(mol, ring_fragment):
    return rdkit.Chem.rdmolfiles.MolFragmentToSmarts(
        mol, ring_fragment, isomericSmarts=False
    )


def get_graph_edit_distance(mol1, mol2):
    G1 = get_substructure_graph(mol1, list(range(0, mol1.GetNumAtoms())))
    G2 = get_substructure_graph(mol2, list(range(0, mol2.GetNumAtoms())))

    nm = nx.algorithms.isomorphism.generic_node_match(
        ["atom_features"], [None], [np.allclose]
    )
    em = nx.algorithms.isomorphism.numerical_edge_match("bond_type", 1.0)

    distance = nx.graph_edit_distance(G1, G2, node_match=nm, edge_match=em)
    return distance


def get_reindexing_map_for_matching(mol, fragment_indices, partial_mol):
    G1 = get_substructure_graph_for_matching(mol, fragment_indices)
    G2 = get_substructure_graph_for_matching(
        partial_mol, list(range(0, partial_mol.GetNumAtoms()))
    )

    nm = nx.algorithms.isomorphism.generic_node_match(
        ["atom_features"], [None], [np.allclose]
    )
    em = nx.algorithms.isomorphism.numerical_edge_match("bond_type", 1.0)

    # getting map from old indices to new indices
    GM = nx.algorithms.isomorphism.GraphMatcher(G1, G2, node_match=nm, edge_match=em)
    assert GM.is_isomorphic()  # THIS NEEDS TO BE CALLED FOR GM.mapping to be initiated
    idx_map = GM.mapping

    return idx_map


def get_attachment_index_of_fragment(
    mol, fragment_indices, fragment_mol, idx, canonical=False
):
    # returns the index of the attachment point in the new fragment

    # mol is the original mol
    # fragment_indices is the list of node indices of the fragment in the original molecule
    # fragment_mol is the 3D mol object of the fragment from the fragment library
    # idx is the attachment point index in the new fragment, but in the ORIGINAL molecule indexing ordering
    # canonical is ignored

    idx_map = get_reindexing_map(mol, fragment_indices, fragment_mol)

    return idx_map[idx]


def get_multi_hot_attachment_points(fragment_mol, idx):
    # idx is the attachment point in the fragment
    # returns multi-hot array indicating which atoms in fragment_mol are graph-equivalent to the atom with index idx
    equivalent_atoms = list(
        rdkit.Chem.CanonicalRankAtoms(fragment_mol, breakTies=False)
    )
    multi_hot = np.array(np.array(equivalent_atoms) == equivalent_atoms[idx], dtype=int)
    return multi_hot


def get_equivalency_mask(fragment_mol):
    # returns a mask over atoms in fragment_mol where all 1s indicate the canonical atom amongst its graph-equivalent symmetry class
    equivalent_atoms = list(
        rdkit.Chem.CanonicalRankAtoms(fragment_mol, breakTies=False)
    )
    mask = np.zeros(len(equivalent_atoms))
    seen = set()
    for i in range(len(equivalent_atoms)):
        if equivalent_atoms[i] not in seen:
            mask[i] = 1
            seen.add(equivalent_atoms[i])
    return mask


def generate_conformer(smiles, addHs=False):
    mol = rdkit.Chem.MolFromSmiles(smiles)
    mol = rdkit.Chem.AddHs(mol)
    rdkit.Chem.AllChem.EmbedMolecule(mol, randomSeed=0xF00D)

    if not addHs:
        mol = rdkit.Chem.RemoveHs(mol)
    return mol


def get_atoms_in_fragment(atom_idx, ring_fragments):
    ring_idx = None
    for r, ring in enumerate(ring_fragments):
        if atom_idx in ring:
            ring_idx = r
            break
    else:
        return [atom_idx]

    return list(ring_fragments[ring_idx])


def get_source_atom(mol, focal, completed_atoms):
    mol_bonds = [
        sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        for bond in mol.GetBonds()
    ]
    source = -1
    source_of_source = -1
    for c in completed_atoms:
        if sorted([c, focal]) in mol_bonds:
            source = c
            for s in completed_atoms:
                if sorted([s, source]) in mol_bonds:
                    source_of_source = s
                    break
            break
    return source, source_of_source


def update_completed_atoms(completed_atoms, atoms=[]):
    completed_atoms = list(set(completed_atoms + atoms))
    return completed_atoms


def get_bonded_connections(mol, atoms=[], completed_atoms=[]):
    mol_bonds = [
        sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        for bond in mol.GetBonds()
    ]
    bonds = []
    for idx in atoms:
        for b in mol_bonds:
            if idx in b:
                if (int((set(b) - set([idx])).pop())) not in atoms:
                    bonds.append([idx, int((set(b) - set([idx])).pop())])

    # remove any completed atoms from list of new bonds
    if len(completed_atoms) > 0:
        bonds = [b for b in bonds if b[1] not in completed_atoms]

    return bonds, [b[1] for b in bonds]


def get_dihedral_indices(mol, source, focal):
    _, source_bonds = get_bonded_connections(
        mol, atoms=[source], completed_atoms=[focal]
    )
    _, focal_bonds = get_bonded_connections(
        mol, atoms=[focal], completed_atoms=[source]
    )
    if (len(source_bonds) == 0) or (len(focal_bonds) == 0):
        return [-1, source, focal, -1]
    return [source_bonds[0], source, focal, focal_bonds[0]]


def update_queue(queue):
    new_queue = queue[1:]
    return new_queue


def is_bond_rotatable(mol, source, focal_atom):
    return mol.GetBondBetweenAtoms(source, focal_atom).GetBondTypeAsDouble() == 1.0


def translate_node_features(feat):
    n = 0
    atom_type = atomTypes[np.argmax(feat[0 : len(atomTypes) + 1])]
    n += len(atomTypes) + 1
    formal_charge = formalCharge[np.argmax(feat[n : n + len(formalCharge) + 1])]
    n += len(formalCharge) + 1
    # hybrid = hybridization[np.argmax(feat[n: n + len(hybridization) + 1])]
    # n += len(hybridization) + 1
    aromatic = feat[n]
    n += 1
    mass = feat[n]
    n += 1
    n_single = num_single_bonds[np.argmax(feat[n : n + len(num_single_bonds) + 1])]
    n += len(num_single_bonds) + 1
    n_double = num_double_bonds[np.argmax(feat[n : n + len(num_double_bonds) + 1])]
    n += len(num_double_bonds) + 1
    n_triple = num_triple_bonds[np.argmax(feat[n : n + len(num_triple_bonds) + 1])]
    n += len(num_triple_bonds) + 1
    n_aromatic = num_aromatic_bonds[
        np.argmax(feat[n : n + len(num_aromatic_bonds) + 1])
    ]
    n += len(num_aromatic_bonds) + 1

    return (
        atom_type,
        formal_charge,
        aromatic,
        mass * 100,
        n_single,
        n_double,
        n_triple,
        n_aromatic,
    )


def one_hot_embedding(value, options):
    embedding = [0] * (len(options) + 1)
    index = options.index(value) if value in options else -1
    embedding[index] = 1
    return embedding


def adjacency_to_undirected_edge_index(adj):
    adj = np.triu(
        np.array(adj, dtype=int)
    )  # keeping just upper triangular entries from sym matrix
    array_adj = np.array(
        np.nonzero(adj), dtype=int
    )  # indices of non-zero values in adj matrix
    edge_index = np.zeros(
        (2, 2 * array_adj.shape[1]), dtype=int
    )  # placeholder for undirected edge list
    edge_index[:, ::2] = array_adj
    edge_index[:, 1::2] = np.flipud(array_adj)
    return edge_index


def getNodeFeatures(list_rdkit_atoms):
    F_v = len(atomTypes) + 1
    F_v += len(formalCharge) + 1
    F_v += 1 + 1

    F_v += 8
    F_v += 6
    F_v += 4
    F_v += 6

    node_features = np.zeros((len(list_rdkit_atoms), F_v))
    for node_index, node in enumerate(list_rdkit_atoms):
        features = one_hot_embedding(
            node.GetSymbol(), atomTypes
        )  # atom symbol, dim=12 + 1
        features += one_hot_embedding(
            node.GetFormalCharge(), formalCharge
        )  # formal charge, dim=5+1
        features += [
            int(node.GetIsAromatic())
        ]  # whether atom is part of aromatic system, dim = 1
        features += [node.GetMass() * 0.01]  # atomic mass / 100, dim=1

        atom_bonds = np.array([b.GetBondTypeAsDouble() for b in node.GetBonds()])
        N_single = int(
            sum(atom_bonds == 1.0) + node.GetNumImplicitHs() + node.GetNumExplicitHs()
        )
        N_double = int(sum(atom_bonds == 2.0))
        N_triple = int(sum(atom_bonds == 3.0))
        N_aromatic = int(sum(atom_bonds == 1.5))

        features += one_hot_embedding(N_single, num_single_bonds)
        features += one_hot_embedding(N_double, num_double_bonds)
        features += one_hot_embedding(N_triple, num_triple_bonds)
        features += one_hot_embedding(N_aromatic, num_aromatic_bonds)

        node_features[node_index, :] = features

    return np.array(node_features, dtype=np.float32)


def getEdgeFeatures(list_rdkit_bonds):
    F_e = len(bondTypes) + 1  # + 1 + (6+1)

    edge_features = np.zeros((len(list_rdkit_bonds) * 2, F_e))
    for edge_index, edge in enumerate(list_rdkit_bonds):
        features = one_hot_embedding(str(edge.GetBondType()), bondTypes)  # dim=4+1

        # Encode both directed edges to get undirected edge
        edge_features[2 * edge_index : 2 * edge_index + 2, :] = features

    return np.array(edge_features, dtype=np.float32)


def retrieve_atom_ID(atom_features, atom_lookup):
    atom_ID = [
        i
        for i in range(atom_lookup.shape[0])
        if np.array_equal(atom_features, atom_lookup[i])
    ][0]
    return atom_ID


def retrieve_bond_ID(bond_prop, bond_lookup_table):
    try:
        bond_ID = int(
            bond_lookup_table[
                (bond_lookup_table[0] == bond_prop[0])
                & (bond_lookup_table[1] == bond_prop[1])
                & (bond_lookup_table[2] == bond_prop[2])
            ].index[0]
        )
    except:
        return None

    return bond_ID


def rigid_transform_3D(A, B):
    # Input: expects 3xN matrix of points
    # Returns R,t
    # R = 3x3 rotation matrix
    # t = 3x1 column vector

    # Outputs R and t TO BE APPLIED TO A to get B

    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    B2 = (R @ A) + t

    return R, t, np.linalg.norm(B2 - B)


def get_xyz(p1, p2, p3, d, theta, psi):
    theta = np.pi - theta
    D2 = np.array(
        [
            d * np.cos(theta),
            d * np.cos(psi) * np.sin(theta),
            d * np.sin(psi) * np.sin(theta),
        ]
    )
    bc = (p3 - p2) / np.linalg.norm(p3 - p2)
    AB = p2 - p1
    n_hat = np.cross(AB, bc) / np.linalg.norm(np.cross(AB, bc))
    M = np.array(
        [bc, np.cross(n_hat, bc) / np.linalg.norm(np.cross(n_hat, bc)), n_hat]
    ).T

    q = M @ (D2.T) + p3
    return q


def VAB_2nd_order_batched(
    centers_1, centers_2, alphas_1, alphas_2, prefactors_1, prefactors_2
):
    R2 = ((torch.cdist(centers_1, centers_2) ** 2.0).T).permute(2, 0, 1)
    prefactor1_prod_prefactor2 = prefactors_1.unsqueeze(1) * prefactors_2.unsqueeze(2)
    alpha1_prod_alpha2 = alphas_1.unsqueeze(1) * alphas_2.unsqueeze(2)
    alpha1_sum_alpha2 = alphas_1.unsqueeze(1) + alphas_2.unsqueeze(2)
    VAB_2nd_order = torch.sum(
        torch.sum(
            np.pi ** (1.5)
            * prefactor1_prod_prefactor2
            * torch.exp(-(alpha1_prod_alpha2 / alpha1_sum_alpha2) * R2)
            / (alpha1_sum_alpha2 ** (1.5)),
            dim=2,
        ),
        dim=1,
    )
    return VAB_2nd_order


def shape_tanimoto_batched(
    centers_1, centers_2, alphas_1, alphas_2, prefactors_1, prefactors_2
):
    VAA = VAB_2nd_order_batched(
        centers_1, centers_1, alphas_1, alphas_1, prefactors_1, prefactors_1
    )
    VBB = VAB_2nd_order_batched(
        centers_2, centers_2, alphas_2, alphas_2, prefactors_2, prefactors_2
    )
    VAB = VAB_2nd_order_batched(
        centers_1, centers_2, alphas_1, alphas_2, prefactors_1, prefactors_2
    )
    return VAB / (VAA + VBB - VAB)


def VAB_2nd_order(centers_1, centers_2, alphas_1, alphas_2, prefactors_1, prefactors_2):
    R2 = (torch.cdist(centers_1, centers_2) ** 2.0).T
    prefactor1_prod_prefactor2 = prefactors_1 * prefactors_2.unsqueeze(1)
    alpha1_prod_alpha2 = alphas_1 * alphas_2.unsqueeze(1)
    alpha1_sum_alpha2 = alphas_1 + alphas_2.unsqueeze(1)

    VAB_2nd_order = torch.sum(
        np.pi ** (1.5)
        * prefactor1_prod_prefactor2
        * torch.exp(-(alpha1_prod_alpha2 / alpha1_sum_alpha2) * R2)
        / (alpha1_sum_alpha2 ** (1.5))
    )
    return VAB_2nd_order


def shape_tanimoto(
    centers_1, centers_2, alphas_1, alphas_2, prefactors_1, prefactors_2
):
    VAA = VAB_2nd_order(
        centers_1, centers_1, alphas_1, alphas_1, prefactors_1, prefactors_1
    )
    VBB = VAB_2nd_order(
        centers_2, centers_2, alphas_2, alphas_2, prefactors_2, prefactors_2
    )
    VAB = VAB_2nd_order(
        centers_1, centers_2, alphas_1, alphas_2, prefactors_1, prefactors_2
    )
    return VAB / (VAA + VBB - VAB)


def get_ROCS(centers_1, centers_2, prefactor=0.8, alpha=0.81):
    # centers_1 = torch.tensor(centers_1)
    # centers_2 = torch.tensor(centers_2)
    prefactors_1 = torch.ones(centers_1.shape[0]) * prefactor
    prefactors_2 = torch.ones(centers_2.shape[0]) * prefactor
    alphas_1 = torch.ones(prefactors_1.shape[0]) * alpha
    alphas_2 = torch.ones(prefactors_2.shape[0]) * alpha

    tanimoto = shape_tanimoto(
        centers_1, centers_2, alphas_1, alphas_2, prefactors_1, prefactors_2
    )
    return tanimoto


def get_ROCS_mols(mol_1, mol_2, prefactor=0.8, alpha=0.81):
    pos_1 = torch.from_numpy(mol_1.GetConformer().GetPositions())
    pos_2 = torch.from_numpy(mol_2.GetConformer().GetPositions())
    return get_ROCS(pos_1, pos_2, prefactor=prefactor, alpha=alpha)


def get_substructure_graph(mol, atom_indices, node_features=None):
    G = nx.Graph()
    bonds = list(mol.GetBonds())
    bond_indices = [sorted([b.GetBeginAtomIdx(), b.GetEndAtomIdx()]) for b in bonds]

    for atom_idx in atom_indices:
        atom = mol.GetAtomWithIdx(int(atom_idx))
        if node_features is None:
            atom_features = getNodeFeatures([atom])[0]
        else:
            atom_features = node_features[int(atom_idx)]
        G.add_node(int(atom_idx), atom_features=atom_features)

    for i in atom_indices:
        for j in atom_indices:
            if sorted([int(i), int(j)]) in bond_indices:
                G.add_edge(
                    int(i),
                    int(j),
                    bond_type=mol.GetBondBetweenAtoms(
                        int(i), int(j)
                    ).GetBondTypeAsDouble(),
                )
    return G


def get_reindexing_map(mol, fragment_indices, partial_mol):
    G1 = get_substructure_graph(mol, fragment_indices)
    G2 = get_substructure_graph(partial_mol, list(range(0, partial_mol.GetNumAtoms())))

    nm = nx.algorithms.isomorphism.generic_node_match(
        ["atom_features"], [None], [np.allclose]
    )
    em = nx.algorithms.isomorphism.numerical_edge_match("bond_type", 1.0)

    # getting map from old indices to new indices
    GM = nx.algorithms.isomorphism.GraphMatcher(G1, G2, node_match=nm, edge_match=em)
    assert GM.is_isomorphic()  # THIS NEEDS TO BE CALLED FOR GM.mapping to be initiated
    idx_map = GM.mapping

    return idx_map


def get_atom_fragment_ID_index(
    mol,
    atom_idx,
    node_features,
    ring_fragments,
    atom_fragment_library,
    fragment_library_atom_features,
):
    ring_fragment = [list(r) for r in ring_fragments if atom_idx in r]
    if len(ring_fragment) > 0:
        assert len(ring_fragment) == 1
        frag_ID_smiles = get_fragment_smiles(mol, ring_fragment[0])
        atom_fragment_ID_index = atom_fragment_library.index[
            atom_fragment_library["smiles"] == frag_ID_smiles
        ].tolist()[0]
    else:
        atom_features = node_features[atom_idx]
        atom_fragment_ID_index = np.where(
            np.all(fragment_library_atom_features == atom_features, axis=1)
        )[0][0]
    return atom_fragment_ID_index


def get_ground_truth_generation_sequence(
    frame, atom_fragment_library, fragment_library_atom_features
):
    sequence = []

    mol = deepcopy(frame.iloc[0].rdkit_mol_cistrans_stereo)
    ring_fragments = get_ring_fragments(mol)
    node_features = getNodeFeatures(mol.GetAtoms())

    for i in range(len(frame)):
        seq = [None, None, None, None, None]
        focal_indices = frame.iloc[i].focal_indices_sorted
        focal_attachment_index = frame.iloc[i].focal_attachment_index
        focal_root_node_index = frame.iloc[i].focal_root_node

        partial_graph_indices = frame.iloc[i].partial_graph_indices
        canonical_ordering = list(
            rdkit.Chem.rdmolfiles.CanonicalRankAtomsInFragment(
                mol=mol,
                atomsToUse=[int(p) for p in partial_graph_indices],
                bondsToUse=[
                    b.GetIdx()
                    for b in mol.GetBonds()
                    if (
                        (b.GetBeginAtomIdx() in partial_graph_indices)
                        & (b.GetEndAtomIdx() in partial_graph_indices)
                    )
                ],
                breakTies=False,
                includeChirality=False,
                includeIsotopes=True,
            )
        )

        canonical_partial_graph_indices = [
            canonical_ordering[j] for j in partial_graph_indices
        ]
        canonical_focal_root_node_index = canonical_ordering[focal_root_node_index]

        if (
            len(
                [
                    j
                    for j in canonical_partial_graph_indices
                    if j == canonical_focal_root_node_index
                ]
            )
            > 1
        ) & (len(focal_indices) > 1):
            raise Exception  # Exception('Sequence determination failed. Graph symmetry detected for focal root node (queue node) -- cannot unambiguously determine focal attachment point.')

        next_atom_idx = frame.iloc[i].next_atom_index
        if next_atom_idx == -1:
            next_atom_fragment_ID_index = -1
            seq[0] = next_atom_fragment_ID_index
            sequence.append(seq)
            continue
        else:
            next_atom_fragment_ID_index = get_atom_fragment_ID_index(
                mol,
                next_atom_idx,
                node_features,
                ring_fragments,
                atom_fragment_library,
                fragment_library_atom_features,
            )
            seq[0] = next_atom_fragment_ID_index

        seq[1] = [
            get_substructure_graph(mol, partial_graph_indices),
            focal_attachment_index,
        ]

        if atom_fragment_library.iloc[next_atom_fragment_ID_index].is_fragment == 0:
            seq[2] = 0
        else:
            next_atom_fragment_mol = atom_fragment_library.iloc[
                next_atom_fragment_ID_index
            ].mol
            next_atom_fragment_indices = frame.iloc[i].next_atom_fragment_indices_sorted
            next_atom_fragment_attachment_index = get_attachment_index_of_fragment(
                mol, next_atom_fragment_indices, next_atom_fragment_mol, next_atom_idx
            )

            next_fragment_attachment_index_rel_next = (
                next_atom_fragment_attachment_index
            )
            seq[2] = next_fragment_attachment_index_rel_next

        bond_type = str(
            mol.GetBondBetweenAtoms(
                int(focal_attachment_index), int(next_atom_idx)
            ).GetBondType()
        )
        bond_type_idx = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"].index(bond_type)
        seq[3] = bond_type_idx

        sequence.append(seq)

    return sequence


def add_to_queue_BFS(queue, indices, canonical=False, mol=None, subgraph_indices=None):
    if canonical:
        if subgraph_indices is None:
            canon_ranks = list(
                rdkit.Chem.rdmolfiles.CanonicalRankAtomsInFragment(
                    mol,
                    atomsToUse=list(range(0, len(mol.GetAtoms()))),
                    bondsToUse=[
                        b.GetIdx()
                        for b in mol.GetBonds()
                        if (
                            (b.GetBeginAtomIdx() in list(range(0, len(mol.GetAtoms()))))
                            & (b.GetEndAtomIdx() in list(range(0, len(mol.GetAtoms()))))
                        )
                    ],
                    breakTies=True,
                    includeChirality=False,
                    includeIsotopes=True,
                )
            )
        else:
            subgraph_indices = [int(s) for s in subgraph_indices]
            canon_ranks = list(
                rdkit.Chem.rdmolfiles.CanonicalRankAtomsInFragment(
                    mol,
                    atomsToUse=subgraph_indices,
                    bondsToUse=[
                        b.GetIdx()
                        for b in mol.GetBonds()
                        if (
                            (b.GetBeginAtomIdx() in subgraph_indices)
                            & (b.GetEndAtomIdx() in subgraph_indices)
                        )
                    ],
                    breakTies=True,
                    includeChirality=False,
                    includeIsotopes=True,
                )
            )

        ranks = [canon_ranks[i] for i in indices]
        ranks_index = sorted(range(len(ranks)), key=lambda k: ranks[k])
        indices = [indices[r] for r in ranks_index]

    else:
        random.shuffle(indices)

    new_queue = queue + indices
    return new_queue


def get_partial_subgraphs_BFS(mol, seed_idx, canonical=False):
    ring_fragments = get_ring_fragments(mol)
    node_features = getNodeFeatures(mol.GetAtoms())

    queue = []  # the queue should only contain atoms that have unknown attachments
    completed_focal_atoms = []
    positioned_atoms = []
    placed_atoms = []

    list_completed_focal_atoms = []
    list_dihedrals = []
    list_positioned_atoms = []

    list_2D_partial_graphs = []
    list_2D_focal_atom_fragment = []
    list_2D_focal_attachment = []
    list_2D_next_atom_fragment = []
    list_2D_next_atom_fragment_indices = []
    list_2D_focal_root_node = []  # queue node

    # initial seeding procedure
    source = seed_idx
    source_fragment = get_atoms_in_fragment(source, ring_fragments)
    source_bonds, source_coupled_atoms = get_bonded_connections(mol, source_fragment)

    subgraph_indices = list(
        set(positioned_atoms + source_fragment + source_coupled_atoms)
    )
    subgraph_indices = list(
        set(
            [
                item
                for sublist in [
                    get_atoms_in_fragment(s, ring_fragments) for s in subgraph_indices
                ]
                for item in sublist
            ]
        )
    )
    queue = add_to_queue_BFS(
        queue,
        source_coupled_atoms,
        canonical=canonical,
        mol=mol,
        subgraph_indices=subgraph_indices,
    )
    completed_focal_atoms = update_completed_atoms(
        completed_focal_atoms, source_fragment
    )

    positioned_atoms = list(
        set(positioned_atoms + source_fragment + source_coupled_atoms)
    )

    list_completed_focal_atoms.append(completed_focal_atoms)
    list_positioned_atoms.append(positioned_atoms)
    list_dihedrals.append(get_dihedral_indices(mol, source, queue[0]))

    placed_atoms = list(set(placed_atoms + source_fragment))
    for s in random.sample(
        source_coupled_atoms, len(source_coupled_atoms)
    ):  # random ordering of atoms to generate and connect to the focus
        list_2D_partial_graphs.append(placed_atoms)
        list_2D_focal_atom_fragment.append(source_fragment)
        list_2D_next_atom_fragment.append(s)
        list_2D_next_atom_fragment_indices.append(
            get_atoms_in_fragment(s, ring_fragments)
        )
        list_2D_focal_root_node.append(source)

        focal_attachment_point = [
            source_bonds[i][0]
            for i in range(len(source_bonds))
            if source_bonds[i][1] == s
        ]
        assert len(focal_attachment_point) == 1
        list_2D_focal_attachment.append(focal_attachment_point[0])

        placed_atoms = list(set(placed_atoms + [s]))

    # STOP TOKEN
    list_2D_partial_graphs.append(placed_atoms)
    list_2D_focal_atom_fragment.append(source_fragment)
    list_2D_focal_attachment.append(
        -1
    )  # there is no attachment point in the focal fragment for a stop token
    list_2D_next_atom_fragment.append(-1)
    list_2D_next_atom_fragment_indices.append(-1)
    list_2D_focal_root_node.append(source)

    # choosing the first focal atom, which is bonded to the seed
    focal_atom = queue[0]
    source, source_of_source = get_source_atom(mol, focal_atom, completed_focal_atoms)
    focal_fragment = get_atoms_in_fragment(focal_atom, ring_fragments)
    focal_bonds, focal_coupled_atoms = get_bonded_connections(
        mol, focal_fragment, completed_atoms=completed_focal_atoms
    )

    subgraph_indices = list(
        set(positioned_atoms + focal_fragment + focal_coupled_atoms)
    )
    subgraph_indices = list(
        set(
            [
                item
                for sublist in [
                    get_atoms_in_fragment(s, ring_fragments) for s in subgraph_indices
                ]
                for item in sublist
            ]
        )
    )
    queue = add_to_queue_BFS(
        queue,
        focal_coupled_atoms,
        canonical=canonical,
        mol=mol,
        subgraph_indices=subgraph_indices,
    )
    completed_focal_atoms = update_completed_atoms(
        completed_focal_atoms, focal_fragment
    )
    queue = update_queue(queue)

    positioned_atoms = list(
        set(positioned_atoms + focal_fragment + focal_coupled_atoms)
    )

    if positioned_atoms not in list_positioned_atoms:
        placed_atoms = list(set(placed_atoms + focal_fragment))
        for s in random.sample(focal_coupled_atoms, len(focal_coupled_atoms)):
            if s not in list_2D_partial_graphs[-1]:
                list_2D_partial_graphs.append(placed_atoms)
                list_2D_focal_atom_fragment.append(focal_fragment)
                list_2D_next_atom_fragment.append(s)
                list_2D_next_atom_fragment_indices.append(
                    get_atoms_in_fragment(s, ring_fragments)
                )
                list_2D_focal_root_node.append(focal_atom)

                focal_attachment_point = [
                    focal_bonds[i][0]
                    for i in range(len(focal_bonds))
                    if focal_bonds[i][1] == s
                ]
                assert len(focal_attachment_point) == 1
                list_2D_focal_attachment.append(focal_attachment_point[0])

                placed_atoms = list(set(placed_atoms + [s]))

        # STOP TOKEN
        list_2D_partial_graphs.append(placed_atoms)
        list_2D_focal_atom_fragment.append(focal_fragment)
        list_2D_focal_attachment.append(-1)
        list_2D_next_atom_fragment.append(-1)
        list_2D_next_atom_fragment_indices.append(-1)
        list_2D_focal_root_node.append(focal_atom)

        list_positioned_atoms.append(positioned_atoms)
        list_completed_focal_atoms.append(completed_focal_atoms)
        list_dihedrals.append(get_dihedral_indices(mol, source, focal_atom))

    else:  # ADD A STOP TOKEN
        placed_atoms = list(set(placed_atoms + focal_fragment))
        list_2D_partial_graphs.append(placed_atoms)
        list_2D_focal_atom_fragment.append(focal_fragment)
        list_2D_focal_attachment.append(-1)
        list_2D_next_atom_fragment.append(-1)
        list_2D_next_atom_fragment_indices.append(-1)
        list_2D_focal_root_node.append(focal_atom)

    while len(queue) > 0:
        focal_atom = queue[0]
        source, source_of_source = get_source_atom(
            mol, focal_atom, completed_focal_atoms
        )
        focal_fragment = get_atoms_in_fragment(focal_atom, ring_fragments)
        focal_bonds, focal_coupled_atoms = get_bonded_connections(
            mol, focal_fragment, completed_atoms=completed_focal_atoms
        )
        subgraph_indices = list(
            set(positioned_atoms + focal_fragment + focal_coupled_atoms)
        )
        subgraph_indices = list(
            set(
                [
                    item
                    for sublist in [
                        get_atoms_in_fragment(s, ring_fragments)
                        for s in subgraph_indices
                    ]
                    for item in sublist
                ]
            )
        )
        queue = add_to_queue_BFS(
            queue,
            focal_coupled_atoms,
            canonical=canonical,
            mol=mol,
            subgraph_indices=subgraph_indices,
        )
        completed_focal_atoms = update_completed_atoms(
            completed_focal_atoms, focal_fragment
        )
        queue = update_queue(queue)

        is_rotatable = is_bond_rotatable(mol, source, focal_atom)

        old_positioned_atoms = positioned_atoms
        positioned_atoms = list(
            set(positioned_atoms + focal_fragment + focal_coupled_atoms)
        )

        if (
            positioned_atoms not in list_positioned_atoms
        ):  # we only care about steps where new atoms are added
            placed_atoms = list(set(placed_atoms + focal_fragment))
            for s in random.sample(focal_coupled_atoms, len(focal_coupled_atoms)):
                if s not in list_2D_partial_graphs[-1]:
                    list_2D_partial_graphs.append(placed_atoms)
                    list_2D_focal_atom_fragment.append(focal_fragment)
                    list_2D_next_atom_fragment.append(s)
                    list_2D_next_atom_fragment_indices.append(
                        get_atoms_in_fragment(s, ring_fragments)
                    )
                    list_2D_focal_root_node.append(focal_atom)

                    focal_attachment_point = [
                        focal_bonds[i][0]
                        for i in range(len(focal_bonds))
                        if focal_bonds[i][1] == s
                    ]
                    assert len(focal_attachment_point) == 1
                    list_2D_focal_attachment.append(focal_attachment_point[0])

                    placed_atoms = list(set(placed_atoms + [s]))

            # STOP TOKEN
            list_2D_partial_graphs.append(placed_atoms)
            list_2D_focal_atom_fragment.append(focal_fragment)
            list_2D_focal_attachment.append(-1)
            list_2D_next_atom_fragment.append(-1)
            list_2D_next_atom_fragment_indices.append(-1)
            list_2D_focal_root_node.append(focal_atom)

            list_positioned_atoms.append(positioned_atoms)
            list_completed_focal_atoms.append(completed_focal_atoms)
            list_dihedrals.append(get_dihedral_indices(mol, source, focal_atom))
            # i += 1

        else:  # ADD A STOP TOKEN
            placed_atoms = list(set(placed_atoms + focal_fragment))
            list_2D_partial_graphs.append(placed_atoms)
            list_2D_focal_atom_fragment.append(focal_fragment)
            list_2D_focal_attachment.append(-1)
            list_2D_next_atom_fragment.append(-1)
            list_2D_next_atom_fragment_indices.append(-1)
            list_2D_focal_root_node.append(focal_atom)

    # re-ordering indices
    ordered_list_positioned_atoms = [list_positioned_atoms[0]]
    for l in range(1, len(list_positioned_atoms)):
        new_atoms = set(list_positioned_atoms[l]) - set(list_positioned_atoms[l - 1])
        same_atoms = set(list_positioned_atoms[l]).intersection(
            set(list_positioned_atoms[l - 1])
        )
        ordered_list_positioned_atoms.append(list(same_atoms) + list(new_atoms))

    # SORTING ALL
    list_2D_partial_graphs = [
        sorted(l) if type(l) == list else l for l in list_2D_partial_graphs
    ]
    list_2D_focal_atom_fragment = [
        sorted(l) if type(l) == list else l for l in list_2D_focal_atom_fragment
    ]
    list_2D_focal_attachment = [
        sorted(l) if type(l) == list else l for l in list_2D_focal_attachment
    ]
    list_2D_next_atom_fragment = [
        sorted(l) if type(l) == list else l for l in list_2D_next_atom_fragment
    ]
    list_2D_next_atom_fragment_indices = [
        sorted(l) if type(l) == list else l for l in list_2D_next_atom_fragment_indices
    ]

    return (
        (
            list_2D_partial_graphs,
            list_2D_focal_atom_fragment,
            list_2D_focal_attachment,
            list_2D_next_atom_fragment,
            list_2D_next_atom_fragment_indices,
            list_2D_focal_root_node,
        ),
        ordered_list_positioned_atoms[0:-1],
        ordered_list_positioned_atoms[1:],
        list_dihedrals[1:],
    )


def get_point_cloud(centers, N, per_node=True, var=1.0 / (12.0 * 1.7)):
    if per_node:
        N_points_per_atom = N
    else:
        N_total = N
        N_points_per_atom = int(math.ceil(N_total / (centers.shape[0] - 1)))

    volume = []
    for center in centers:
        points = sample_atom_volume(center.numpy(), N_points_per_atom, var=var)
        # probs = GMM(points, centers, p = 0.8, alpha = 0.81)
        volume_points = points
        volume.append(points)

    cloud_batch_index_all = torch.LongTensor(
        torch.cat(
            [
                torch.ones(N_points_per_atom, dtype=torch.long) * i
                for i in range(centers.shape[0])
            ]
        )
    )
    cloud_all = torch.tensor(np.concatenate(volume))

    if per_node == False:
        subsamples = torch.LongTensor(
            np.sort(
                np.random.choice(
                    np.arange(0, cloud_all.shape[0]), N_total, replace=False
                )
            )
        )
        cloud = cloud_all[subsamples]
        cloud_batch_index = cloud_batch_index_all[subsamples]
    else:
        cloud = cloud_all
        cloud_batch_index = cloud_batch_index_all

    return cloud, cloud_batch_index


def sample_atom_volume(center, N, var=1.0 / (12.0 * 1.7)):
    x = np.random.multivariate_normal(
        center,
        [[var, 0, 0], [0, var, 0], [0, 0, var]],
        size=N,
        check_valid="warn",
        tol=1e-8,
    )
    return x


def process_partial_mol_rotated(
    partial_mol,
    positioned_atoms_indices,
    focal_indices,
    atom_fragment_library,
    atom_to_library_ID_map,
    list_rotated_positions,
    N_points,
    pointCloudVar=1.0 / (4.0 * 1.7),
):
    mol = deepcopy(partial_mol)

    indices_partial_after = positioned_atoms_indices
    query_indices = focal_indices
    query_indices_ref_partial = [
        indices_partial_after.index(idx) for idx in query_indices
    ]

    # FULL GRAPH DATA
    # Edge Index
    adj = rdkit.Chem.GetAdjacencyMatrix(mol)
    edge_index = adjacency_to_undirected_edge_index(adj)

    # Edge Features --> rdkit ordering of edges
    bonds = []
    for b in range(int(edge_index.shape[1] / 2)):
        bond_index = edge_index[:, ::2][:, b]
        bond = mol.GetBondBetweenAtoms(int(bond_index[0]), int(bond_index[1]))
        bonds.append(bond)
    edge_features = getEdgeFeatures(bonds)

    # Node Features --> rdkit ordering of atoms
    atoms = rdkit.Chem.rdchem.Mol.GetAtoms(mol)
    atom_symbols = [atom.GetSymbol() for atom in atoms]
    node_features = getNodeFeatures(atoms)

    for k in atom_to_library_ID_map:
        node_features[k] = atom_fragment_library.iloc[
            atom_to_library_ID_map[k]
        ].atom_features

    # Fragment library node associations
    fragment_library_atom_features = np.concatenate(
        atom_fragment_library["atom_features"], axis=0
    ).reshape((len(atom_fragment_library), -1))
    if type(atom_fragment_library) != type(None):
        ring_fragments = get_ring_fragments(mol)
        atom_fragment_associations = np.zeros(len(atoms), dtype=int)
        for i, atom in enumerate(atoms):
            ring_fragment = [list(r) for r in ring_fragments if i in r]
            if len(ring_fragment) > 0:
                assert len(ring_fragment) == 1
                frag_ID_smiles = get_fragment_smiles(mol, ring_fragment[0])
                atom_fragment_ID_index = atom_fragment_library.index[
                    atom_fragment_library["smiles"] == frag_ID_smiles
                ].tolist()[0]
            else:
                atom_features = node_features[i]
                atom_fragment_ID_index = np.where(
                    np.all(fragment_library_atom_features == atom_features, axis=1)
                )[0][0]
            atom_fragment_associations[i] = atom_fragment_ID_index
        atom_fragment_associations = torch.tensor(
            atom_fragment_associations, dtype=torch.long
        )
    else:
        atom_fragment_associations = torch.tensor(
            np.zeros(len(atoms)), dtype=torch.long
        )

    subgraph_node_index = torch.as_tensor(indices_partial_after, dtype=torch.long)

    n_idx = torch.zeros(node_features.shape[0], dtype=torch.long)
    n_idx[subgraph_node_index] = torch.arange(subgraph_node_index.size(0))

    # SUBGRAPH INDEX DATA
    subgraph_edge_index, subgraph_edge_attr = torch_geometric.utils.subgraph(
        indices_partial_after,
        torch.as_tensor(edge_index, dtype=torch.long),
        torch.as_tensor(edge_features),
        relabel_nodes=True,
    )

    rotated_partial_positions = torch.cat(
        list_rotated_positions, dim=0
    )  # torch.cat(rotated_partial_positions_list, dim = 0)

    N = len(list_rotated_positions)

    x = torch.as_tensor(node_features)

    x_subgraph = x[subgraph_node_index]
    edge_index_subgraph = torch.as_tensor(subgraph_edge_index, dtype=torch.long)
    edge_attr_subgraph = torch.as_tensor(subgraph_edge_attr)

    atom_fragment_associations_subgraph = atom_fragment_associations[
        subgraph_node_index
    ]
    query_index_subgraph = n_idx[torch.as_tensor(query_indices, dtype=torch.long)]

    subgraph_size = torch.as_tensor(np.array([len(indices_partial_after)]))
    query_size = torch.as_tensor(np.array([len(query_indices)]))

    add_to_subgraph_node_index = (
        torch.cat(
            [
                torch.ones(subgraph_node_index.shape, dtype=torch.long) * i
                for i in range(N)
            ],
            dim=0,
        )
        * x.shape[0]
    )

    add_to_edge_index_subgraph = (
        torch.cat(
            [
                torch.ones(edge_index_subgraph.shape, dtype=torch.long) * i
                for i in range(N)
            ],
            dim=1,
        )
        * x_subgraph.shape[0]
    )
    add_to_query_index_subgraph = (
        torch.cat(
            [
                torch.ones(query_index_subgraph.shape, dtype=torch.long) * i
                for i in range(N)
            ],
            dim=0,
        )
        * x_subgraph.shape[0]
    )

    x_subgraph_repeat = x_subgraph.repeat((N, 1))
    edge_index_subgraph_repeat = (
        edge_index_subgraph.repeat((1, N)) + add_to_edge_index_subgraph
    )
    subgraph_node_index_repeat = (
        subgraph_node_index.repeat((N)) + add_to_subgraph_node_index
    )
    edge_attr_subgraph_repeat = edge_attr_subgraph.repeat((N, 1))
    query_index_subgraph_repeat = (
        query_index_subgraph.repeat((N)) + add_to_query_index_subgraph
    )
    atom_fragment_associations_subgraph_repeat = (
        atom_fragment_associations_subgraph.repeat(N)
    )

    pos_subgraph_repeat = rotated_partial_positions[subgraph_node_index_repeat]

    subgraph_size_repeat = subgraph_size.repeat((N))
    query_size_repeat = query_size.repeat((N))

    # are indices getting mixed up here?
    list_rotated_partial_positions = list(pos_subgraph_repeat.reshape((N, -1, 3)))
    cloud_subgraph_temp_list = [
        get_point_cloud(p, N_points, per_node=True, var=pointCloudVar)
        for p in list_rotated_partial_positions
    ]
    cloud_subgraph_list, cloud_batch_indices_subgraph_list = [
        t[0] for t in cloud_subgraph_temp_list
    ], [t[1] for t in cloud_subgraph_temp_list]

    cloud_subgraph = torch.cat(cloud_subgraph_list, dim=0)
    cloud_batch_indices_subgraph = torch.cat(cloud_batch_indices_subgraph_list, dim=0)

    data = torch_geometric.data.Data(
        x_subgraph=x_subgraph_repeat.float(),
        edge_index_subgraph=edge_index_subgraph_repeat,
        subgraph_node_index=subgraph_node_index_repeat,
        edge_attr_subgraph=edge_attr_subgraph_repeat.float(),
        pos_subgraph=pos_subgraph_repeat.float(),
        cloud_subgraph=cloud_subgraph.float(),
        cloud_indices_subgraph=cloud_batch_indices_subgraph,
        atom_fragment_associations_subgraph=atom_fragment_associations_subgraph_repeat,
        query_index_subgraph=query_index_subgraph_repeat,
        subgraph_size=subgraph_size_repeat,
        query_size=query_size_repeat,
        new_batch_subgraph=torch.cat(
            [torch.ones(x_subgraph.shape[0], dtype=torch.long) * i for i in range(N)],
            dim=0,
        ),
    )

    return data


def process_mol(mol, atom_fragment_library, N_points, pointCloudVar=1.0 / (4.0 * 1.7)):
    fragment_library_atom_features = np.concatenate(
        atom_fragment_library["atom_features"], axis=0
    ).reshape((len(atom_fragment_library), -1))

    bondTypes = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

    adj = rdkit.Chem.GetAdjacencyMatrix(mol)
    edge_index = adjacency_to_undirected_edge_index(adj)

    bonds = []
    for b in range(int(edge_index.shape[1] / 2)):
        bond_index = edge_index[:, ::2][:, b]
        bond = mol.GetBondBetweenAtoms(int(bond_index[0]), int(bond_index[1]))
        bonds.append(bond)
    edge_features = getEdgeFeatures(bonds)

    atoms = rdkit.Chem.rdchem.Mol.GetAtoms(mol)
    atom_symbols = [atom.GetSymbol() for atom in atoms]
    node_features = getNodeFeatures(atoms)

    ring_fragments = get_ring_fragments(mol)
    atom_fragment_associations = np.zeros(len(atoms), dtype=int)
    for i, atom in enumerate(atoms):
        ring_fragment = [list(r) for r in ring_fragments if i in r]
        if len(ring_fragment) > 0:
            assert len(ring_fragment) == 1
            frag_ID_smiles = get_fragment_smiles(mol, ring_fragment[0])
            atom_fragment_ID_index = atom_fragment_library.index[
                atom_fragment_library["smiles"] == frag_ID_smiles
            ].tolist()[0]
        else:
            atom_features = node_features[i]
            atom_fragment_ID_index = np.where(
                np.all(fragment_library_atom_features == atom_features, axis=1)
            )[0][0]
        atom_fragment_associations[i] = atom_fragment_ID_index

    positions = torch.tensor(mol.GetConformer().GetPositions())
    center_of_mass = torch.sum(positions, dim=0) / len(positions)
    positions = positions - center_of_mass

    cloud, cloud_batch_indices = get_point_cloud(
        positions, N_points, per_node=True, var=pointCloudVar
    )

    data = torch_geometric.data.Data(
        x=torch.as_tensor(node_features).float(),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.as_tensor(edge_features).float(),
        pos=positions.float(),
        cloud=cloud.float(),
        cloud_indices=cloud_batch_indices,
        cloud_index=cloud_batch_indices,
        atom_fragment_associations=torch.tensor(
            atom_fragment_associations, dtype=torch.long
        ),
    )

    return data


def process_partial_mol(
    mol,
    positioned_atoms_indices,
    focal_indices,
    atom_fragment_library,
    N_points,
    atom_to_library_ID_map={},
    pointCloudVar=1.0 / (4.0 * 1.7),
):
    fragment_library_atom_features = np.concatenate(
        atom_fragment_library["atom_features"], axis=0
    ).reshape((len(atom_fragment_library), -1))

    bondTypes = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
    adj = rdkit.Chem.GetAdjacencyMatrix(mol)
    edge_index = adjacency_to_undirected_edge_index(adj)

    bonds = []
    for b in range(int(edge_index.shape[1] / 2)):
        bond_index = edge_index[:, ::2][:, b]
        bond = mol.GetBondBetweenAtoms(int(bond_index[0]), int(bond_index[1]))
        bonds.append(bond)
    edge_features = getEdgeFeatures(bonds)

    atoms = rdkit.Chem.rdchem.Mol.GetAtoms(mol)
    atom_symbols = [atom.GetSymbol() for atom in atoms]
    node_features = getNodeFeatures(atoms)

    for k in atom_to_library_ID_map:
        node_features[k] = atom_fragment_library.iloc[
            atom_to_library_ID_map[k]
        ].atom_features

    ring_fragments = get_ring_fragments(mol)
    atom_fragment_associations = np.zeros(len(atoms), dtype=int)
    for i, atom in enumerate(atoms):
        ring_fragment = [list(r) for r in ring_fragments if i in r]
        if len(ring_fragment) > 0:
            assert len(ring_fragment) == 1
            frag_ID_smiles = get_fragment_smiles(mol, ring_fragment[0])
            assert frag_ID_smiles != None
            atom_fragment_ID_index = atom_fragment_library.index[
                atom_fragment_library["smiles"] == frag_ID_smiles
            ].tolist()[0]
        else:
            atom_features = node_features[i]
            if len(atom_features.shape) == 1:
                atom_features = np.expand_dims(atom_features, 0)
            atom_fragment_ID_index = np.where(
                np.all(fragment_library_atom_features == atom_features, axis=1)
            )[0][0]
        atom_fragment_associations[i] = atom_fragment_ID_index

    # indices of focal atoms reference w.r.t. the positioned atoms
    focal_indices_ref_partial = [
        positioned_atoms_indices.index(f) for f in focal_indices
    ]

    positions = torch.tensor(mol.GetConformer().GetPositions())

    subgraph_node_index = torch.as_tensor(positioned_atoms_indices, dtype=torch.long)

    n_idx = torch.zeros(positions.shape[0], dtype=torch.long)
    n_idx[subgraph_node_index] = torch.arange(subgraph_node_index.size(0))

    subgraph_edge_index, subgraph_edge_attr = torch_geometric.utils.subgraph(
        positioned_atoms_indices,
        torch.as_tensor(edge_index, dtype=torch.long),
        torch.as_tensor(edge_features),
        relabel_nodes=True,
    )

    positions_subgraph = positions[subgraph_node_index]
    atom_fragment_associations_subgraph = atom_fragment_associations[
        subgraph_node_index
    ]

    subgraph_size = torch.as_tensor(np.array([len(positioned_atoms_indices)]))
    focal_size = torch.as_tensor(np.array([len(focal_indices)]))
    focal_index = torch.as_tensor(focal_indices, dtype=torch.long)

    focal_index_subgraph = n_idx[focal_index]

    cloud_subgraph, cloud_batch_indices_subgraph = get_point_cloud(
        positions_subgraph, N_points, per_node=True, var=pointCloudVar
    )

    data = torch_geometric.data.Data(
        x=torch.as_tensor(node_features).float(),
        edge_index_subgraph=subgraph_edge_index,
        x_subgraph=torch.as_tensor(node_features)[subgraph_node_index].float(),
        edge_attr_subgraph=subgraph_edge_attr.float(),
        pos_subgraph=positions_subgraph.float(),
        cloud_subgraph=cloud_subgraph.float(),
        cloud_index_subgraph=cloud_batch_indices_subgraph,
        cloud_indices_subgraph=cloud_batch_indices_subgraph,
        atom_fragment_associations_subgraph=torch.tensor(
            atom_fragment_associations_subgraph, dtype=torch.long
        ),
        subgraph_size=subgraph_size,
        focal_size=focal_size,
        focal_index=focal_index,
        focal_index_subgraph=focal_index_subgraph,
    )

    return data


def focal_attachment_valency_mask(mol, focal_indices, node_features):
    if type(focal_indices) == int:
        focal_indices = [focal_indices]
    else:
        focal_indices = list(focal_indices)
    occupied_focal_bonds = [
        [b.GetBondTypeAsDouble() for b in mol.GetAtomWithIdx(i).GetBonds()]
        for i in focal_indices
    ]
    occupied_single_bonds = [
        sum(np.array(bond_types) == 1.0) for bond_types in occupied_focal_bonds
    ]
    occupied_double_bonds = [
        sum(np.array(bond_types) == 2.0) for bond_types in occupied_focal_bonds
    ]
    occupied_triple_bonds = [
        sum(np.array(bond_types) == 3.0) for bond_types in occupied_focal_bonds
    ]
    occupied_bonds = np.array(
        [
            np.array(
                [
                    occupied_single_bonds[i],
                    occupied_double_bonds[i],
                    occupied_triple_bonds[i],
                ]
            )
            for i in range(len(focal_indices))
        ]
    )

    allowed_bond_types = np.array(
        [
            np.array(translate_node_features(i)[4:7])
            for i in node_features[focal_indices]
        ]
    )
    available_bond_types_per_focal_atom = allowed_bond_types - occupied_bonds

    available_bond_types = (
        np.sum(available_bond_types_per_focal_atom, axis=0) > 0.0
    )  # available bond types for the entire fragment, if focal is a fragment
    has_available_bonds = (
        np.sum(available_bond_types_per_focal_atom, axis=1) > 0.0
    )  # tells us if each focal atom in the atom/fragment has available valence for bonding

    if (
        sum(available_bond_types[1:]) > 0.0
    ):  # if there are unsatisfied double/triple bonds, we can't stop
        stop = 0
    elif (
        sum(available_bond_types) == 0.0
    ):  # if there are no more bonds available, we have to stop
        stop = 1
    else:
        stop = None

    return stop, has_available_bonds, available_bond_types_per_focal_atom


def next_ID_valency_mask(focal_bond_types, AtomFragment_database):
    next_ID_mask = (
        np.array(AtomFragment_database[["N_single", "N_double", "N_triple"]])[
            :, focal_bond_types
        ].sum(axis=1)
        > 0.0
    )
    no_aromatic_bonds = list(AtomFragment_database.N_aromatic == 0)
    next_ID_mask = next_ID_mask & no_aromatic_bonds
    return next_ID_mask


def next_attachment_valency_mask(next_ID, AtomFragment_database):
    if AtomFragment_database.iloc[next_ID].is_fragment == 0:
        return np.array([True])

    next_mol_atoms = AtomFragment_database.iloc[next_ID].mol.GetAtoms()
    next_attachment_mask = np.array(
        [atom.GetTotalNumHs() > 0.0 for atom in next_mol_atoms]
    )

    return next_attachment_mask


def bond_type_valency_mask(focal_bond_types, next_ID, AtomFragment_database):
    if AtomFragment_database.iloc[next_ID].is_fragment == 1:
        return np.array(
            [True, False, False, False]
        )  # only generate single bonds to fragments

    next_bond_types = (
        np.array(
            AtomFragment_database.iloc[next_ID][["N_single", "N_double", "N_triple"]]
        )
        > 0.0
    )

    bond_type_mask = focal_bond_types & next_bond_types
    return np.array(
        list(bond_type_mask) + [False]
    )  # we never generate aromatic bonds (all internal to fragments)


def update_2D_mol_BFS_switched(
    model,
    Z_inv,
    partial_mol,
    queue,
    positioned_atoms_indices,
    AtomFragment_database,
    fragment_batch,
    fragment_library_node_features,
    fragment_library_features,
    atom_to_library_ID_map,
    unique_atoms,
    bond_lookup,
    N_points,
    mask_first_stop=False,
    ground_truth_sequence=[],
    seq=0,
    canonical=False,
    pointCloudVar=1.0 / (4.0 * 1.7),
):
    all_fragments = get_ring_fragments(partial_mol)

    focal_atom = queue[0]
    positioned_source, positioned_source_of_source = get_source_atom(
        partial_mol, focal_atom, positioned_atoms_indices
    )

    focal_indices = [list(f) for f in all_fragments if focal_atom in f]
    focal_indices.append([focal_atom])
    focal_indices = [item for sublist in focal_indices for item in sublist]
    focal_indices = list(set(focal_indices).union(set([focal_atom])))

    old_positioned_atom_indices = list(
        set(list(set(positioned_atoms_indices) - set(focal_indices)) + [focal_atom])
    )
    positioned_atoms_indices = list(
        set(positioned_atoms_indices).union(set(focal_indices))
    )

    updated_queue = queue

    STOP = False
    all_STOP = True
    mask_stop = mask_first_stop

    add_to_queue = []
    while not STOP:
        if (seq < len(ground_truth_sequence)) and (partial_mol.GetNumAtoms() == 1):
            next_atom_fragment_ID = ground_truth_sequence[seq][0]
            focal_attachment_index = 0
            next_fragment_attachment_index_rel_next = ground_truth_sequence[seq][2]
            bond_type_idx = ground_truth_sequence[seq][3]
            node_features = np.expand_dims(
                AtomFragment_database.iloc[atom_to_library_ID_map[0]].atom_features, 0
            )

        else:
            data = process_partial_mol(
                partial_mol,
                positioned_atoms_indices,
                focal_indices,
                AtomFragment_database,
                N_points,
                atom_to_library_ID_map,
                pointCloudVar=pointCloudVar,
            )

            batch_index = torch.zeros(
                data.x.shape[0], dtype=torch.long
            )  # only 1 molecule per run
            batch_size = 1

            focal_index_rel_partial = data.focal_index_subgraph
            focal_batch_index = batch_index[data.focal_index_subgraph]
            focal_reindexing_map = {
                int(j): int(i)
                for i, j in zip(
                    torch.arange(torch.unique(focal_batch_index).shape[0]),
                    torch.unique(focal_batch_index),
                )
            }
            focal_batch_index_reindexed = torch.tensor(
                [focal_reindexing_map[int(i)] for i in focal_batch_index],
                dtype=torch.long,
            )

            # use ground truth sequence
            if (len(ground_truth_sequence) > seq) and (
                ground_truth_sequence[seq][0] != None
            ):
                next_atom_fragment_ID = ground_truth_sequence[seq][0]
                stop_decision = next_atom_fragment_ID == -1

            # encode partial molecule and predict stop token
            elif model is not None:
                x_inv_subgraph, Z_inv_subgraph, Z_inv_select = model.Encoder.encode(
                    x=torch.cat(
                        (
                            data.x_subgraph,
                            fragment_library_features[
                                data.atom_fragment_associations_subgraph
                            ],
                        ),
                        dim=1,
                    ),
                    edge_index=data.edge_index_subgraph,
                    edge_attr=data.edge_attr_subgraph,
                    batch_size=1,
                    select_indices=focal_index_rel_partial,
                    select_indices_batch=focal_batch_index,
                    shared_encoders=model.shared_encoders,
                )
                graph_subgraph_focal_features_concat = model.Encoder.mix_codes(
                    batch_size, Z_inv, Z_inv_subgraph, Z_inv_select
                )
                h_partial = x_inv_subgraph
                h_focal = h_partial[focal_index_rel_partial]

                (
                    stop_logit_mask,
                    focal_attachment_mask,
                    available_bond_types_per_focal_atom,
                ) = focal_attachment_valency_mask(
                    mol=partial_mol,
                    focal_indices=focal_indices,
                    node_features=data.x.detach().numpy(),
                )

                stop_logits = model.Decoder.decode_stop(
                    graph_subgraph_focal_features_concat
                )

                if stop_logit_mask is not None:
                    if stop_logit_mask == 1:  # force stop
                        stop_decision = True
                    elif stop_logit_mask == 0:  # force continuation
                        stop_decision = False
                else:
                    stop_decision = (
                        torch.sigmoid(stop_logits) < 0.5
                    )  # if True, STOP. Else, continue with generation.

            if mask_stop:
                stop_decision = False

            # update STOP
            if stop_decision:
                seq += 1
                STOP = True
                updated_partial_mol = partial_mol

                if len(focal_indices) > 1:
                    all_STOP = False

                break
            all_STOP = False

            if (len(ground_truth_sequence) > seq) and (
                ground_truth_sequence[seq][1] != None
            ):
                if len(focal_index_rel_partial) > 1:
                    (
                        G_subgraph,
                        focal_attachment_index_ground_truth,
                    ) = ground_truth_sequence[seq][1]

                    G_partial = get_substructure_graph(
                        partial_mol,
                        positioned_atoms_indices,
                        node_features=data.x.numpy(),
                    )

                    nm = nx.algorithms.isomorphism.generic_node_match(
                        ["atom_features"], [None], [np.allclose]
                    )
                    em = nx.algorithms.isomorphism.numerical_edge_match(
                        "bond_type", 1.0
                    )
                    GM = nx.algorithms.isomorphism.GraphMatcher(
                        G_subgraph, G_partial, node_match=nm, edge_match=em
                    )

                    assert (
                        GM.is_isomorphic()
                    )  # THIS NEEDS TO BE CALLED FOR GM.mapping to appear
                    focal_attachment_index = GM.mapping[
                        focal_attachment_index_ground_truth
                    ]
                    focal_attachment_index_rel_partial = torch.tensor(
                        focal_attachment_index
                    )

                else:
                    focal_attachment_index_rel_focal = 0
                    focal_attachment_index_rel_partial = focal_index_rel_partial[
                        focal_attachment_index_rel_focal
                    ]
                    focal_attachment_index = data.focal_index[
                        focal_attachment_index_rel_focal
                    ]

            elif model is not None:  # predict focal attachment point
                fragment_attachment_scores_softmax = model.Decoder.decode_focal_attachment_point(
                    graph_subgraph_focal_features_concat=graph_subgraph_focal_features_concat,
                    h_focal=h_focal,
                    focal_batch_index_reindexed=focal_batch_index_reindexed,
                )

                fragment_attachment_scores_softmax[~focal_attachment_mask] = 0.0
                focal_attachment_index_rel_focal = int(
                    torch.argmax(fragment_attachment_scores_softmax)
                )

                focal_attachment_index_rel_partial = focal_index_rel_partial[
                    focal_attachment_index_rel_focal
                ]
                focal_attachment_index = data.focal_index[
                    focal_attachment_index_rel_focal
                ]

                focal_bond_types = (
                    available_bond_types_per_focal_atom[
                        focal_attachment_index_rel_focal
                    ]
                    > 0.0
                )

            # predict next atom/fragment type
            if (len(ground_truth_sequence) > seq) and (
                ground_truth_sequence[seq][0] != None
            ):
                next_atom_fragment_ID = ground_truth_sequence[seq][0]
            elif model is not None:
                focal_atom_features = h_partial[
                    focal_attachment_index_rel_partial.unsqueeze(0)
                ]
                graph_subgraph_focal_focalAtom_features_concat = torch.cat(
                    (graph_subgraph_focal_features_concat, focal_atom_features), dim=1
                )

                next_atom_fragment_logits = model.Decoder.decode_next_atom_fragment(
                    graph_subgraph_focal_focalAtom_features_concat,
                    fragment_library_features,
                ).squeeze()
                next_ID_mask = next_ID_valency_mask(
                    focal_bond_types, AtomFragment_database
                )

                next_atom_fragment_logits[~next_ID_mask] = -np.inf
                next_atom_fragment_ID = int(torch.argmax(next_atom_fragment_logits))

            # predict next attachment point
            if (len(ground_truth_sequence) > seq) and (
                ground_truth_sequence[seq][2] != None
            ):
                next_fragment_attachment_index_rel_next = ground_truth_sequence[seq][2]
            elif model is not None:
                next_atom_attachment_indices = torch.arange(fragment_batch.x.shape[0])[
                    fragment_batch.batch == next_atom_fragment_ID
                ]
                next_atom_attachment_batch_index_reindexed = torch.zeros(
                    next_atom_attachment_indices.shape[0], dtype=torch.long
                )  # all zeros, only 1 molecule

                nextAtomFragment_features = fragment_library_features[
                    [next_atom_fragment_ID]
                ]
                graph_subgraph_focal_nextAtomFragment_focalAtom_features_concat = (
                    torch.cat(
                        (
                            graph_subgraph_focal_focalAtom_features_concat,
                            nextAtomFragment_features,
                        ),
                        dim=1,
                    )
                )

                next_fragment_attachment_scores_softmax = model.Decoder.decode_next_attachment_point(
                    graph_subgraph_focal_nextAtomFragment_focalAtom_features_concat=graph_subgraph_focal_nextAtomFragment_focalAtom_features_concat,
                    fragment_library_node_features=fragment_library_node_features,
                    next_atom_attachment_indices=next_atom_attachment_indices,
                    next_atom_attachment_batch_index_reindexed=next_atom_attachment_batch_index_reindexed,
                )
                next_attachment_mask = next_attachment_valency_mask(
                    next_ID=next_atom_fragment_ID,
                    AtomFragment_database=AtomFragment_database,
                )
                next_fragment_attachment_scores_softmax[~next_attachment_mask] = 0.0
                next_fragment_attachment_index_rel_next = int(
                    torch.argmax(next_fragment_attachment_scores_softmax)
                )

            # predict bond type
            equivalent_atoms = AtomFragment_database.iloc[
                next_atom_fragment_ID
            ].equiv_atoms
            equiv_attachments = np.array(
                np.array(equivalent_atoms)
                == equivalent_atoms[next_fragment_attachment_index_rel_next],
                dtype=int,
            )

            next_atom_attachment_batch_index_reindexed = torch.zeros(
                equiv_attachments.shape[0], dtype=torch.long
            )

            if (len(ground_truth_sequence) > seq) and (
                ground_truth_sequence[seq][3] != None
            ):
                bond_type_idx = ground_truth_sequence[seq][3]
            elif model is not None:
                bond_types_logits = model.Decoder.decode_bond_type(
                    graph_subgraph_focal_nextAtomFragment_focalAtom_features_concat=graph_subgraph_focal_nextAtomFragment_focalAtom_features_concat,
                    multihot_next_attachments=equiv_attachments,
                    next_atom_attachment_batch_index_reindexed=next_atom_attachment_batch_index_reindexed,
                    fragment_library_node_features=fragment_library_node_features,
                    next_atom_attachment_indices=next_atom_attachment_indices,
                ).squeeze()
                bond_type_mask = bond_type_valency_mask(
                    focal_bond_types=focal_bond_types,
                    next_ID=next_atom_fragment_ID,
                    AtomFragment_database=AtomFragment_database,
                )

                bond_types_logits[~bond_type_mask] = -np.inf
                bond_type_idx = int(torch.argmax(bond_types_logits))

            node_features = np.array(data.x.detach().numpy())

        bond_type = [
            rdkit.Chem.rdchem.BondType.SINGLE,
            rdkit.Chem.rdchem.BondType.DOUBLE,
            rdkit.Chem.rdchem.BondType.TRIPLE,
            rdkit.Chem.rdchem.BondType.AROMATIC,
        ][bond_type_idx]

        (
            updated_partial_mol,
            next_fragment_attachment_index,
            next_fragment_indices,
        ) = make_3D_bond_attachments(
            mol=partial_mol,
            positioned_atoms_indices=positioned_atoms_indices,
            node_features=node_features,
            next_atom_fragment_ID=next_atom_fragment_ID,
            focal_attachment_index=int(focal_attachment_index),
            next_fragment_attachment_index_rel_next=next_fragment_attachment_index_rel_next,
            bond_type=bond_type,
            AtomFragment_database=AtomFragment_database,
            unique_atoms=unique_atoms,
            bond_lookup=bond_lookup,
            ignore_3D=True,
        )

        seq += 1

        partial_mol = updated_partial_mol

        # update mol indices
        positioned_atoms_indices += [next_fragment_attachment_index]
        add_to_queue.append(next_fragment_attachment_index)

        # changed from 0 to -1
        if (next_atom_fragment_ID != -1) and (
            AtomFragment_database.iloc[next_atom_fragment_ID].is_fragment == 0
        ):
            atom_to_library_ID_map[
                next_fragment_attachment_index
            ] = next_atom_fragment_ID

    node_features = getNodeFeatures(updated_partial_mol.GetAtoms())
    ring_fragments = get_ring_fragments(updated_partial_mol)
    fragment_library_atom_features = np.concatenate(
        AtomFragment_database["atom_features"], axis=0
    ).reshape((len(AtomFragment_database), -1))

    updated_queue = add_to_queue_BFS(
        updated_queue, add_to_queue, canonical=canonical, mol=updated_partial_mol
    )

    # update mol and all indices, including those in the queue and positioned_atoms_indices
    updated_partial_mol = updated_partial_mol
    positioned_atoms_indices = list(
        set(positioned_atoms_indices).union(set(focal_indices))
    )
    updated_queue = updated_queue[1:]

    return (
        updated_partial_mol,
        updated_queue,
        positioned_atoms_indices,
        atom_to_library_ID_map,
        seq,
    )


def update_3D_mol_BFS_switched(
    partial_mol,
    queue,
    positioned_atoms_indices,
    AtomFragment_database,
    unique_atoms,
    bond_lookup,
    fragment_batch,
    atom_to_library_ID_map,
    model,
    Z_equi,
    Z_inv,
    N_points,
    fragment_library_node_features,
    fragment_library_features,
    rocs_model,
    Z_equi_rocs,
    Z_inv_rocs,
    N_points_rocs,
    fragment_library_features_rocs,
    fragment_library_node_features_rocs,
    mask_first_stop=False,
    ground_truth_sequence=[],
    seq=0,
    canonical=False,
    random_dihedrals=False,
    N_rocs_decisions=0,
    use_mol_before_3D_scoring=None,
    stochastic=False,
    chirality_scoring=True,
    stop_threshold=0.01,
    steric_mask=False,
    pointCloudVar=1.0 / (4.0 * 1.7),
    rocs_pointCloudVar=1.0 / (4.0 * 1.7),
):
    all_fragments = get_ring_fragments(partial_mol)

    focal_atom = queue[0]
    positioned_source, positioned_source_of_source = get_source_atom(
        partial_mol, focal_atom, positioned_atoms_indices
    )

    focal_indices = [list(f) for f in all_fragments if focal_atom in f]
    focal_indices.append([focal_atom])
    focal_indices = [item for sublist in focal_indices for item in sublist]
    focal_indices = list(set(focal_indices).union(set([focal_atom])))

    query_indices = [d for d in deepcopy(focal_indices) if d != focal_atom]

    query_atom_to_library_ID_map = {}

    old_positioned_atom_indices = list(
        set(list(set(positioned_atoms_indices) - set(focal_indices)) + [focal_atom])
    )
    positioned_atoms_indices = list(
        set(positioned_atoms_indices).union(set(focal_indices))
    )

    updated_queue = queue

    STOP = False
    all_STOP = True
    mask_stop = mask_first_stop

    chirality_scored = False
    add_to_queue = []
    while not STOP:
        data = process_partial_mol(
            partial_mol,
            positioned_atoms_indices,  # this gets updated with the newly predicted atoms at each iteration
            focal_indices,
            AtomFragment_database,
            N_points,
            atom_to_library_ID_map,
            pointCloudVar=pointCloudVar,
        )

        batch_index = torch.zeros(
            data.x.shape[0], dtype=torch.long
        )  # only 1 molecule per run
        batch_size = 1

        focal_index_rel_partial = data.focal_index_subgraph
        focal_batch_index = batch_index[data.focal_index_subgraph]
        focal_reindexing_map = {
            int(j): int(i)
            for i, j in zip(
                torch.arange(torch.unique(focal_batch_index).shape[0]),
                torch.unique(focal_batch_index),
            )
        }
        focal_batch_index_reindexed = torch.tensor(
            [focal_reindexing_map[int(i)] for i in focal_batch_index], dtype=torch.long
        )

        # encode partial molecule
        if model is not None:
            model_out = model.Encoder.encode(
                x=torch.cat(
                    (
                        data.x_subgraph,
                        fragment_library_features[
                            data.atom_fragment_associations_subgraph
                        ],
                    ),
                    dim=1,
                ),
                edge_index=data.edge_index_subgraph,
                pos=data.pos_subgraph,
                points=data.cloud_subgraph,
                points_atom_index=data.cloud_indices_subgraph,
                edge_attr=data.edge_attr_subgraph,
                batch_size=1,
                select_indices=focal_index_rel_partial,
                select_indices_batch=focal_batch_index,
                shared_encoders=model.shared_encoders,
            )
            (
                x_inv_subgraph,
                Z_equi_subgraph,
                Z_inv_subgraph,
                Z_equi_select,
                Z_inv_select,
            ) = (model_out[0], model_out[1], model_out[2], model_out[3], model_out[4])

            graph_subgraph_focal_features_concat = model.Encoder.mix_codes(
                batch_size=1,
                Z_equi=Z_equi,
                Z_inv=Z_inv,
                Z_equi_subgraph=Z_equi_subgraph,
                Z_inv_subgraph=Z_inv_subgraph,
                Z_equi_select=Z_equi_select,
                Z_inv_select=Z_inv_select,
            )
            h_partial = x_inv_subgraph.permute(0, 2, 1).reshape(
                -1, x_inv_subgraph.shape[1]
            )
            h_focal = h_partial[focal_index_rel_partial]

            (
                stop_logit_mask,
                focal_attachment_mask,
                available_bond_types_per_focal_atom,
            ) = focal_attachment_valency_mask(
                mol=partial_mol,
                focal_indices=focal_indices,
                node_features=data.x.detach().numpy(),
            )

            stop_logits = model.Decoder.decode_stop(
                graph_subgraph_focal_features_concat
            )

            if stop_logit_mask is not None:
                if stop_logit_mask == 1:  # force stop
                    stop_decision = True
                elif stop_logit_mask == 0:  # force continuation
                    stop_decision = False
            else:
                stop_decision = (
                    torch.sigmoid(stop_logits) < stop_threshold
                )  # if True, STOP. Else, continue with generation.

            if mask_stop:
                stop_decision = False

        # use ground truth sequence
        if (len(ground_truth_sequence) > seq) and (
            ground_truth_sequence[seq][0] != None
        ):
            next_atom_fragment_ID = ground_truth_sequence[seq][0]
            stop_decision = next_atom_fragment_ID == -1

        # update STOP
        if stop_decision:
            seq += 1
            STOP = True
            updated_partial_mol = partial_mol

            if len(focal_indices) > 1:
                all_STOP = False

            break
        all_STOP = False

        # predict focal attachment point
        if model is not None:
            fragment_attachment_scores_softmax = model.Decoder.decode_focal_attachment_point(
                graph_subgraph_focal_features_concat=graph_subgraph_focal_features_concat,
                h_focal=h_focal,
                focal_batch_index_reindexed=focal_batch_index_reindexed,
            )

            fragment_attachment_scores_softmax[~focal_attachment_mask] = 0.0

            # if stochastic, sample from multinomial (rather than being greedy)
            if stochastic:
                focal_attachment_index_rel_focal = int(
                    torch.multinomial(fragment_attachment_scores_softmax, 1).item()
                )
            else:
                focal_attachment_index_rel_focal = int(
                    torch.argmax(fragment_attachment_scores_softmax)
                )

            focal_attachment_index_rel_partial = focal_index_rel_partial[
                focal_attachment_index_rel_focal
            ]
            focal_attachment_index = data.focal_index[focal_attachment_index_rel_focal]
            focal_bond_types = (
                available_bond_types_per_focal_atom[focal_attachment_index_rel_focal]
                > 0.0
            )

        if (len(ground_truth_sequence) > seq) and (
            ground_truth_sequence[seq][1] != None
        ):
            if len(focal_index_rel_partial) > 1:
                G_subgraph, focal_attachment_index_ground_truth = ground_truth_sequence[
                    seq
                ][1]

                G_partial = get_substructure_graph(
                    partial_mol, positioned_atoms_indices, node_features=data.x.numpy()
                )

                nm = nx.algorithms.isomorphism.generic_node_match(
                    ["atom_features"], [None], [np.allclose]
                )
                em = nx.algorithms.isomorphism.numerical_edge_match("bond_type", 1.0)
                GM = nx.algorithms.isomorphism.GraphMatcher(
                    G_subgraph, G_partial, node_match=nm, edge_match=em
                )

                assert (
                    GM.is_isomorphic()
                )  # THIS NEEDS TO BE CALLED FOR GM.mapping to appear
                focal_attachment_index = GM.mapping[focal_attachment_index_ground_truth]
                focal_attachment_index_rel_partial = torch.tensor(
                    focal_attachment_index
                )

            else:
                focal_attachment_index_rel_focal = 0
                focal_attachment_index_rel_partial = focal_index_rel_partial[
                    focal_attachment_index_rel_focal
                ]
                focal_attachment_index = data.focal_index[
                    focal_attachment_index_rel_focal
                ]

        # predict next atom/fragment type
        if model is not None:
            focal_atom_features = h_partial[
                focal_attachment_index_rel_partial.unsqueeze(0)
            ]
            graph_subgraph_focal_focalAtom_features_concat = torch.cat(
                (graph_subgraph_focal_features_concat, focal_atom_features), dim=1
            )

            next_atom_fragment_logits = model.Decoder.decode_next_atom_fragment(
                graph_subgraph_focal_focalAtom_features_concat,
                fragment_library_features,
            ).squeeze()
            next_ID_mask = next_ID_valency_mask(focal_bond_types, AtomFragment_database)

            next_atom_fragment_logits[~next_ID_mask] = -np.inf

            # stochastic sampling
            if stochastic:
                next_atom_fragment_ID = int(
                    torch.multinomial(
                        torch.softmax(next_atom_fragment_logits, dim=0), 1
                    ).item()
                )
            else:
                next_atom_fragment_ID = int(torch.argmax(next_atom_fragment_logits))

        if (len(ground_truth_sequence) > seq) and (
            ground_truth_sequence[seq][0] != None
        ):
            next_atom_fragment_ID = ground_truth_sequence[seq][0]

        # predict next attachment point
        if model is not None:
            next_atom_attachment_indices = torch.arange(fragment_batch.x.shape[0])[
                fragment_batch.batch == next_atom_fragment_ID
            ]
            next_atom_attachment_batch_index_reindexed = torch.zeros(
                next_atom_attachment_indices.shape[0], dtype=torch.long
            )  # all zeros, only 1 molecule

            nextAtomFragment_features = fragment_library_features[
                [next_atom_fragment_ID]
            ]
            graph_subgraph_focal_nextAtomFragment_focalAtom_features_concat = torch.cat(
                (
                    graph_subgraph_focal_focalAtom_features_concat,
                    nextAtomFragment_features,
                ),
                dim=1,
            )

            next_fragment_attachment_scores_softmax = model.Decoder.decode_next_attachment_point(
                graph_subgraph_focal_nextAtomFragment_focalAtom_features_concat=graph_subgraph_focal_nextAtomFragment_focalAtom_features_concat,
                fragment_library_node_features=fragment_library_node_features,
                next_atom_attachment_indices=next_atom_attachment_indices,
                next_atom_attachment_batch_index_reindexed=next_atom_attachment_batch_index_reindexed,
            )
            next_attachment_mask = next_attachment_valency_mask(
                next_ID=next_atom_fragment_ID,
                AtomFragment_database=AtomFragment_database,
            )
            next_fragment_attachment_scores_softmax[~next_attachment_mask] = 0.0

            # stochastic samples
            if stochastic:
                next_fragment_attachment_index_rel_next = int(
                    torch.multinomial(next_fragment_attachment_scores_softmax, 1).item()
                )
            else:
                next_fragment_attachment_index_rel_next = int(
                    torch.argmax(next_fragment_attachment_scores_softmax)
                )

        if (len(ground_truth_sequence) > seq) and (
            ground_truth_sequence[seq][2] != None
        ):
            next_fragment_attachment_index_rel_next = ground_truth_sequence[seq][2]

        # predict bond type
        if model is not None:
            equivalent_atoms = AtomFragment_database.iloc[
                next_atom_fragment_ID
            ].equiv_atoms
            equiv_attachments = np.array(
                np.array(equivalent_atoms)
                == equivalent_atoms[next_fragment_attachment_index_rel_next],
                dtype=int,
            )
            next_atom_attachment_batch_index_reindexed = torch.zeros(
                equiv_attachments.shape[0], dtype=torch.long
            )  # only 1 molecule during inference

            bond_types_logits = model.Decoder.decode_bond_type(
                graph_subgraph_focal_nextAtomFragment_focalAtom_features_concat=graph_subgraph_focal_nextAtomFragment_focalAtom_features_concat,
                multihot_next_attachments=equiv_attachments,
                next_atom_attachment_batch_index_reindexed=next_atom_attachment_batch_index_reindexed,
                fragment_library_node_features=fragment_library_node_features,
                next_atom_attachment_indices=next_atom_attachment_indices,
            ).squeeze()

            bond_type_mask = bond_type_valency_mask(
                focal_bond_types=focal_bond_types,
                next_ID=next_atom_fragment_ID,
                AtomFragment_database=AtomFragment_database,
            )

            bond_types_logits[~bond_type_mask] = -np.inf
            bond_type_idx = int(torch.argmax(bond_types_logits))

        if (len(ground_truth_sequence) > seq) and (
            ground_truth_sequence[seq][3] != None
        ):
            bond_type_idx = ground_truth_sequence[seq][3]

        bond_type = [
            rdkit.Chem.rdchem.BondType.SINGLE,
            rdkit.Chem.rdchem.BondType.DOUBLE,
            rdkit.Chem.rdchem.BondType.TRIPLE,
            rdkit.Chem.rdchem.BondType.AROMATIC,
        ][bond_type_idx]

        node_features = np.array(data.x.detach().numpy())
        (
            updated_partial_mol,
            next_fragment_attachment_index,
            next_fragment_indices,
        ) = make_3D_bond_attachments(
            mol=partial_mol,
            positioned_atoms_indices=positioned_atoms_indices,
            node_features=node_features,
            next_atom_fragment_ID=next_atom_fragment_ID,
            focal_attachment_index=int(focal_attachment_index),
            next_fragment_attachment_index_rel_next=next_fragment_attachment_index_rel_next,
            bond_type=bond_type,
            AtomFragment_database=AtomFragment_database,
            unique_atoms=unique_atoms,
            bond_lookup=bond_lookup,
            ignore_3D=False,
        )

        seq += 1

        partial_mol = updated_partial_mol

        # update mol indices
        positioned_atoms_indices += [next_fragment_attachment_index]
        query_indices += [next_fragment_attachment_index]

        add_to_queue.append(next_fragment_attachment_index)

        if (next_atom_fragment_ID != -1) and (
            AtomFragment_database.iloc[next_atom_fragment_ID].is_fragment == 0
        ):
            atom_to_library_ID_map[
                next_fragment_attachment_index
            ] = next_atom_fragment_ID

        if next_atom_fragment_ID != -1:
            query_atom_to_library_ID_map[
                next_fragment_attachment_index
            ] = next_atom_fragment_ID

    # need to score E/Z isomerism here, since it won't be flagged as a rotatable bond below.

    if (
        (
            updated_partial_mol.GetBondBetweenAtoms(
                focal_atom, positioned_source
            ).GetBondTypeAsDouble()
            == 1.0
        )
        and (all_STOP == False)
        and (
            updated_partial_mol.GetBondBetweenAtoms(
                focal_atom, positioned_source
            ).IsInRing()
            == False
        )
    ):
        N_rocs_decisions += 1
        dihedral_indices = get_dihedral_indices(
            updated_partial_mol, positioned_source, focal_atom
        )

        if use_mol_before_3D_scoring is not None:
            data_ = process_partial_mol(
                updated_partial_mol,
                positioned_atoms_indices,
                focal_indices,
                AtomFragment_database,
                2,
                atom_to_library_ID_map,
                pointCloudVar=pointCloudVar,
            )

            try:
                G = get_substructure_graph(
                    use_mol_before_3D_scoring,
                    list(range(0, use_mol_before_3D_scoring.GetNumAtoms())),
                )
                G_partial = get_substructure_graph(
                    updated_partial_mol,
                    list(range(0, updated_partial_mol.GetNumAtoms())),
                    node_features=data_.x.detach().numpy(),
                )

                nm = nx.algorithms.isomorphism.generic_node_match(
                    ["atom_features"], [None], [np.allclose]
                )
                em = nx.algorithms.isomorphism.numerical_edge_match("bond_type", 1.0)
                GM = nx.algorithms.isomorphism.GraphMatcher(
                    G, G_partial, node_match=nm, edge_match=em
                )
                assert GM.subgraph_is_isomorphic()
                rocs_idx_map = GM.mapping
                rocs_idx_map_reversed = {v: k for k, v in rocs_idx_map.items()}

                # fixing all positions according to ground truth molecule
                for k in rocs_idx_map:
                    x, y, z = use_mol_before_3D_scoring.GetConformer().GetPositions()[k]
                    updated_partial_mol.GetConformer().SetAtomPosition(
                        rocs_idx_map[k], Point3D(x, y, z)
                    )

            except Exception as e:
                print("can't find isomorphic match:", e)
                pass

        # need to enumerate all stereoisomers here for chirality scoring
        if chirality_scoring:
            chiral_actions = []
            if len(focal_indices) == 1:
                focal_root_atom = updated_partial_mol.GetAtomWithIdx(focal_indices[0])
                neighbors = [a.GetIdx() for a in focal_root_atom.GetNeighbors()]
                fixed_source = positioned_source
                new_neighbors = sorted(list(set(query_indices) - set(focal_indices)))

                if len(new_neighbors) >= 2:
                    new_neighbors_IDs = [
                        query_atom_to_library_ID_map[n] for n in new_neighbors
                    ]

                    if len(set(new_neighbors_IDs)) < len(
                        new_neighbors_IDs
                    ):  # graph equivalency -> not a chiral center (yet)
                        skip_enumeration = True
                    else:
                        skip_enumeration = False

                    new_atoms = [
                        AtomFragment_database.iloc[ID].atom_objects.GetAtomWithIdx(0)
                        for ID in new_neighbors_IDs
                        if (AtomFragment_database.iloc[ID].is_fragment == 0)
                    ]
                    new_atoms_N_rad_electrons_bool = True in [
                        (n.GetNumRadicalElectrons() in (2, 3)) for n in new_atoms
                    ]
                    new_atoms_sp_sp2_bool = True in [
                        (str(n.GetHybridization()) in ("SP", "SP2")) for n in new_atoms
                    ]
                    new_atoms_in_fragments_sp_sp2_bool = True in [
                        (
                            str(
                                updated_partial_mol.GetAtomWithIdx(n).GetHybridization()
                            )
                            in ("SP", "SP2")
                        )
                        for n in new_neighbors
                    ]
                    if (
                        (focal_root_atom.GetSymbol() in ("O", "N"))
                        & (str(focal_root_atom.GetHybridization()) == "SP3")
                    ) & (
                        new_atoms_N_rad_electrons_bool
                        | new_atoms_sp_sp2_bool
                        | new_atoms_in_fragments_sp_sp2_bool
                    ):
                        focal_hybridization_fixed = "SP2"
                    else:
                        focal_hybridization_fixed = None

                    if (
                        (
                            str(
                                updated_partial_mol.GetAtomWithIdx(
                                    focal_indices[0]
                                ).GetHybridization()
                            )
                            == "SP3"
                        )
                        & (focal_hybridization_fixed != "SP2")
                        & (skip_enumeration == False)
                    ):
                        chiral_actions.append(
                            (
                                "swap_chirality_atoms",
                                focal_indices[0],
                                fixed_source,
                                new_neighbors[0],
                                new_neighbors[1],
                            )
                        )

            if len(focal_indices) > 1:
                new_fragment_neighbors = sorted(
                    list(set(query_indices) - set(focal_indices))
                )

                for f in focal_indices:
                    neighbors = [
                        a.GetIdx()
                        for a in updated_partial_mol.GetAtomWithIdx(f).GetNeighbors()
                    ]
                    new_neighbors = list(
                        set(neighbors).intersection(set(new_fragment_neighbors))
                    )
                    new_neighbors_IDs = [
                        query_atom_to_library_ID_map[n] for n in new_neighbors
                    ]

                    if f != focal_atom:
                        if len(new_neighbors) == 0:
                            continue

                        if (len(new_neighbors) == 1) & (len(neighbors) == 3):
                            new_atoms = [
                                AtomFragment_database.iloc[
                                    ID
                                ].atom_objects.GetAtomWithIdx(0)
                                for ID in new_neighbors_IDs
                                if (AtomFragment_database.iloc[ID].is_fragment == 0)
                            ]
                            new_atoms_N_rad_electrons_bool = True in [
                                (n.GetNumRadicalElectrons() in (2, 3))
                                for n in new_atoms
                            ]
                            new_atoms_sp_sp2_bool = True in [
                                (str(n.GetHybridization()) in ("SP", "SP2"))
                                for n in new_atoms
                            ]
                            new_atoms_in_fragments_sp_sp2_bool = True in [
                                (
                                    str(
                                        updated_partial_mol.GetAtomWithIdx(
                                            n
                                        ).GetHybridization()
                                    )
                                    in ("SP", "SP2")
                                )
                                for n in new_neighbors
                            ]
                            if (
                                (
                                    updated_partial_mol.GetAtomWithIdx(f).GetSymbol()
                                    in ("O", "N")
                                )
                                & (
                                    str(
                                        updated_partial_mol.GetAtomWithIdx(
                                            f
                                        ).GetHybridization()
                                    )
                                    == "SP3"
                                )
                            ) & (
                                new_atoms_N_rad_electrons_bool
                                | new_atoms_sp_sp2_bool
                                | new_atoms_in_fragments_sp_sp2_bool
                            ):
                                focal_hybridization_fixed = "SP2"
                            else:
                                focal_hybridization_fixed = None

                            if (
                                str(
                                    updated_partial_mol.GetAtomWithIdx(
                                        f
                                    ).GetHybridization()
                                )
                                == "SP3"
                            ) & (focal_hybridization_fixed != "SP2"):
                                chiral_actions.append(
                                    (
                                        "swap_chirality_focal_ring_atoms",
                                        f,
                                        new_neighbors[0],
                                        None,
                                    )
                                )

                        elif len(new_neighbors) == 2:
                            if (
                                new_neighbors_IDs[0] == new_neighbors_IDs[1]
                            ):  # atom f is not a chiral center due to graph equivalency
                                continue
                            chiral_actions.append(
                                (
                                    "swap_chirality_focal_ring_atoms",
                                    f,
                                    new_neighbors[0],
                                    new_neighbors[1],
                                )
                            )

                    else:
                        assert len(new_neighbors) <= 1  # sanity check
                        fixed_source = positioned_source
                        if (
                            str(
                                updated_partial_mol.GetAtomWithIdx(f).GetHybridization()
                            )
                            == "SP3"
                        ):
                            rings = updated_partial_mol.GetRingInfo().AtomRings()
                            rings = [set(r) for r in rings]
                            ring = [r for r in rings if f in r][0]

                            ring_neighbor = [r for r in ring if r in neighbors][0]

                            if len(new_neighbors) == 1:
                                chiral_actions.append(
                                    (
                                        "swap_chirality_focal_ring",
                                        f,
                                        fixed_source,
                                        ring_neighbor,
                                        new_neighbors[0],
                                    )
                                )
                            elif (len(new_neighbors) == 0) & (len(neighbors) == 3):
                                chiral_actions.append(
                                    (
                                        "swap_chirality_focal_ring",
                                        f,
                                        fixed_source,
                                        ring_neighbor,
                                        None,
                                    )
                                )

            if len(chiral_actions) > 0:
                chiral_actions_bits = [
                    "".join(seq)
                    for seq in itertools.product("01", repeat=len(chiral_actions))
                ]
                chirality_scored = True
            else:
                chiral_actions_bits = []

            query_rotations = np.arange(0, 360, 10, dtype=float)
            list_rotated_positions = [
                torch.tensor(
                    get_aligned_positions_with_rotated_dihedral(
                        updated_partial_mol,
                        old_positioned_atom_indices,
                        dihedral_indices,
                        r,
                        absolute_deg=True,
                    )
                )
                for r in query_rotations
            ]

            max_N_stereoisomers = 32  # avoiding memory overflow
            if len(chiral_actions_bits) > max_N_stereoisomers:
                logger("limiting number of enumerated stereoisomers")
                random.shuffle(chiral_actions_bits)
                chiral_actions_bits = chiral_actions_bits[0:max_N_stereoisomers]

            for bit_sequence in chiral_actions_bits:
                if bit_sequence == "0" * len(bit_sequence):
                    continue

                updated_partial_mol_chiral = deepcopy(updated_partial_mol)

                for b, bit in enumerate(bit_sequence):
                    if bit == "1":
                        chiral_action = chiral_actions[b]
                        if chiral_action[0] == "swap_chirality_atoms":
                            updated_partial_mol_chiral = swap_chirality_atoms(
                                updated_partial_mol_chiral, *chiral_action[1:]
                            )
                        elif chiral_action[0] == "swap_chirality_focal_ring_atoms":
                            updated_partial_mol_chiral = (
                                swap_chirality_focal_ring_atoms(
                                    updated_partial_mol_chiral, *chiral_action[1:]
                                )
                            )
                        elif chiral_action[0] == "swap_chirality_focal_ring":
                            updated_partial_mol_chiral = swap_chirality_focal_ring(
                                updated_partial_mol_chiral, *chiral_action[1:]
                            )

                list_rotated_positions_chiral = [
                    torch.tensor(
                        get_aligned_positions_with_rotated_dihedral(
                            updated_partial_mol_chiral,
                            old_positioned_atom_indices,
                            dihedral_indices,
                            r,
                            absolute_deg=True,
                        )
                    )
                    for r in query_rotations
                ]
                list_rotated_positions += list_rotated_positions_chiral

        else:  # chirality scoring == False
            query_rotations = np.arange(0, 360, 10, dtype=float)
            list_rotated_positions = [
                torch.tensor(
                    get_aligned_positions_with_rotated_dihedral(
                        updated_partial_mol,
                        old_positioned_atom_indices,
                        dihedral_indices,
                        r,
                        absolute_deg=True,
                    )
                )
                for r in query_rotations
            ]

        if random_dihedrals == False:
            rotated_positions = torch.cat(list_rotated_positions, dim=0)

            data_rotated = process_partial_mol_rotated(
                updated_partial_mol,
                positioned_atoms_indices,
                query_indices,
                AtomFragment_database,
                atom_to_library_ID_map,
                list_rotated_positions,
                N_points_rocs,
                pointCloudVar=rocs_pointCloudVar,
            )

            query_indices_rel_to_partial = data_rotated.query_index_subgraph
            query_indices_batch = data_rotated.new_batch_subgraph[
                query_indices_rel_to_partial
            ]

            rocs_out = rocs_model.Encoder.encode(
                x=torch.cat(
                    (
                        data_rotated.x_subgraph,
                        fragment_library_features_rocs[
                            data_rotated.atom_fragment_associations_subgraph
                        ],
                    ),
                    dim=1,
                ),
                edge_index=data_rotated.edge_index_subgraph,
                pos=data_rotated.pos_subgraph,
                points=data_rotated.cloud_subgraph,
                points_atom_index=data_rotated.cloud_indices_subgraph,
                edge_attr=data_rotated.edge_attr_subgraph,
                batch_size=data_rotated.subgraph_size.shape[0],
                select_indices=query_indices_rel_to_partial,
                select_indices_batch=query_indices_batch,
                shared_encoders=rocs_model.shared_encoders,
            )
            (
                x_inv_subgraph_rocs,
                Z_equi_subgraph_rocs,
                Z_inv_subgraph_rocs,
                Z_equi_select_rocs,
                Z_inv_select_rocs,
            ) = (rocs_out[0], rocs_out[1], rocs_out[2], rocs_out[3], rocs_out[4])

            graph_subgraph_focal_features_concat_rocs = rocs_model.Encoder.mix_codes(
                batch_size=data_rotated.subgraph_size.shape[0],
                Z_equi=Z_equi_rocs.expand(len(list_rotated_positions), -1, -1).squeeze(
                    0
                ),
                Z_inv=Z_inv_rocs.expand(len(list_rotated_positions), -1),
                Z_equi_subgraph=Z_equi_subgraph_rocs,
                Z_inv_subgraph=Z_inv_subgraph_rocs,
                Z_equi_select=Z_equi_select_rocs,
                Z_inv_select=Z_inv_select_rocs,
            )

            scores = rocs_model.ROCS_scorer(graph_subgraph_focal_features_concat_rocs)
            scores = scores.squeeze().sigmoid()

            if steric_mask == True:
                rotated_positions_sterics = torch.cat(
                    [p.unsqueeze(0) for p in list_rotated_positions]
                )
                distance_mat = torch.cdist(
                    rotated_positions_sterics, rotated_positions_sterics
                )
                distance_mat[distance_mat < 1e-8] = np.inf  # mask out self interactions

                steric_distances = distance_mat.min(dim=1).values.min(dim=1).values
                steric_distances_mask = steric_distances < 1.0

                if (
                    False not in steric_distances_mask
                ):  # all scored rotations have steric clashes
                    steric_mask = False  # don't try to mask any of them
                    # logger('All orientations contain steric clash.')

            if steric_mask == True:
                scores[
                    steric_distances_mask
                ] = (
                    -np.inf
                )  # don't let model select orientation that leads to steric clash

            best_pos_idx = torch.argmax(scores)

        else:  # random_dihedrals == True
            best_pos_idx = random.choice(list(range(0, len(list_rotated_positions))))

        # update positions based on scored dihedral
        if use_mol_before_3D_scoring is None:
            for k in range(updated_partial_mol.GetNumAtoms()):
                x, y, z = list_rotated_positions[best_pos_idx][k].detach().numpy()
                updated_partial_mol.GetConformer().SetAtomPosition(k, Point3D(x, y, z))

    # updating queue with canonical ordering of the newly generated atoms/fragments
    node_features = getNodeFeatures(updated_partial_mol.GetAtoms())
    ring_fragments = get_ring_fragments(updated_partial_mol)
    fragment_library_atom_features = np.concatenate(
        AtomFragment_database["atom_features"], axis=0
    ).reshape((len(AtomFragment_database), -1))

    updated_queue = add_to_queue_BFS(
        updated_queue, add_to_queue, canonical=canonical, mol=updated_partial_mol
    )

    # update mol and all indices, including those in the queue and positioned_atoms_indices
    updated_partial_mol = updated_partial_mol
    positioned_atoms_indices = list(
        set(positioned_atoms_indices).union(set(focal_indices))
    )
    updated_queue = updated_queue[1:]

    return (
        updated_partial_mol,
        updated_queue,
        positioned_atoms_indices,
        atom_to_library_ID_map,
        seq,
        N_rocs_decisions,
        chirality_scored,
    )


def get_frame_terminalSeeds(mol, seed, AtomFragment_database, include_rocs=True):
    fragment_library_atom_features = np.concatenate(
        AtomFragment_database["atom_features"], axis=0
    ).reshape((len(AtomFragment_database), -1))

    results = get_partial_subgraphs_BFS(mol, seed[0], canonical=True)

    (
        list_2D_partial_graphs,
        list_2D_focal_atom_fragment,
        list_2D_focal_attachment,
        list_2D_next_atom_fragment,
        list_2D_next_atom_fragment_indices,
        list_2D_focal_root_node,
    ) = results[0]

    positions_before, positions_after, dihedral = results[1], results[2], results[3]

    graph_construction_database = []
    for j in range(len(results[0][0])):
        graph_construction_database.append(
            [
                -1,
                list_2D_partial_graphs[j],
                list_2D_focal_atom_fragment[j],
                list_2D_focal_attachment[j],
                list_2D_next_atom_fragment[j],
                list_2D_next_atom_fragment_indices[j],
                list_2D_focal_root_node[j],
                sorted(seed),
            ]
        )

    future_rocs_partial_subgraph_database = []
    for j in range(len(positions_before)):
        if (-1) in dihedral[j]:
            continue
        future_rocs_partial_subgraph_database.append(
            [-1, positions_before[j], positions_after[j], dihedral[j], sorted(seed)]
        )

    frame_df = pd.DataFrame()
    frame_df[
        [
            "original_index",
            "partial_graph_indices",
            "focal_indices",
            "focal_attachment_index",
            "next_atom_index",
            "next_atom_fragment_indices",
            "focal_root_node",
            "seed",
        ]
    ] = graph_construction_database

    # delete after testing
    frame_df["rdkit_mol_cistrans_stereo"] = [mol] * len(frame_df)
    frame_df["SMILES_nostereo"] = [
        rdkit.Chem.MolToSmiles(mol, isomericSmiles=False)
    ] * len(frame_df)
    frame_df["ID"] = [rdkit.Chem.MolToSmiles(mol)] * len(frame_df)

    frame_df["seed_idx"] = [s[0] for s in frame_df.seed]

    frame_df["partial_graph_indices_sorted"] = [
        sorted(l) for l in frame_df.partial_graph_indices
    ]
    frame_df["focal_indices_sorted"] = [sorted(l) for l in frame_df.focal_indices]
    frame_df["next_atom_fragment_indices_sorted"] = [
        sorted(l) if (l != -1) else l for l in frame_df.next_atom_fragment_indices
    ]

    if include_rocs:
        frame_rocs_df = pd.DataFrame()
        frame_rocs_df[
            [
                "original_index",
                "positions_before",
                "positions_after",
                "dihedral_indices",
                "seed",
            ]
        ] = future_rocs_partial_subgraph_database
        frame_rocs_df["positions_before_sorted"] = [
            sorted(l) for l in frame_rocs_df.positions_before
        ]
        frame_rocs_df["positions_after_sorted"] = [
            sorted(l) for l in frame_rocs_df.positions_after
        ]
        return frame_df, frame_rocs_df

    return frame_df, None


def get_all_possible_seeds(mol, ring_fragments):
    non_fragment_atoms = set(list(range(0, mol.GetNumAtoms()))) - set([]).union(
        *ring_fragments
    )
    return (
        *[[i] for i in list(non_fragment_atoms)],
        *[list(r) for r in ring_fragments],
    )


def filter_terminal_seeds(all_seeds, mol):
    terminal_seeds = []
    for seed in all_seeds:
        if len(seed) == 1:
            atom = mol.GetAtomWithIdx(seed[0])
            N_bonds = len(atom.GetBonds())
            if N_bonds == 1:
                terminal_seeds.append(seed)
        else:  # fragment -> "terminal" fragment, NOT a fragment inside the overall structure
            atoms_bonded_to_fragment = [
                [a.GetIdx() for a in mol.GetAtomWithIdx(f).GetNeighbors()] for f in seed
            ]
            atoms_bonded_to_fragment = set(
                [a for sublist in atoms_bonded_to_fragment for a in sublist]
            )
            if (
                len(atoms_bonded_to_fragment - set(seed)) == 1
            ):  # fragment only has 1 outside bond
                terminal_seeds.append(seed)
    return tuple(terminal_seeds)


def generate_2D_mol_from_sequence(
    sequence,
    partial_mol,
    mol,
    positioned_atoms_indices,
    queue,
    atom_to_library_ID_map,
    model,
    AtomFragment_database,
    unique_atoms,
    bond_lookup,
    stop_after_sequence=False,
    mask_first_stop=False,
    pointCloudVar=1.0 / (4.0 * 1.7),
):
    fragment_library_atom_features = np.concatenate(
        AtomFragment_database["atom_features"], axis=0
    ).reshape((len(AtomFragment_database), -1))

    node_features = getNodeFeatures(mol.GetAtoms())
    ring_fragments = get_ring_fragments(mol)

    data_mol = process_mol(
        mol, AtomFragment_database, N_points=2, pointCloudVar=pointCloudVar
    )

    library_dataset = AtomFragmentLibrary(AtomFragment_database)
    library_loader = torch_geometric.data.DataLoader(
        library_dataset, shuffle=False, batch_size=len(library_dataset), num_workers=0
    )
    fragment_batch = next(iter(library_loader))

    (
        fragment_library_features,
        fragment_library_node_features,
        fragment_library_batch,
    ) = model.Encoder.encode_fragment_library(fragment_batch)

    _, Z_inv, _ = model.Encoder.encode(
        x=torch.cat(
            (
                data_mol.x,
                fragment_library_features[data_mol.atom_fragment_associations],
            ),
            dim=1,
        ),
        edge_index=data_mol.edge_index,
        edge_attr=data_mol.edge_attr,
        batch_size=1,
        select_indices=None,
        select_indices_batch=None,
        shared_encoders=True,
    )

    if True:
        updated_mol = deepcopy(partial_mol)
        # atom_to_library_ID_map = {}
        q = 0
        unconverged = 0

        mask_first_stop = mask_first_stop

        seq = 0
        while len(queue) > 0:
            (
                updated_mol,
                queue,
                positioned_atoms_indices,
                atom_to_library_ID_map,
                seq,
            ) = update_2D_mol_BFS_switched(
                model=model,
                Z_inv=Z_inv,
                partial_mol=updated_mol,
                queue=queue,
                positioned_atoms_indices=positioned_atoms_indices,
                AtomFragment_database=AtomFragment_database,
                fragment_batch=fragment_batch,
                fragment_library_node_features=fragment_library_node_features,
                fragment_library_features=fragment_library_features,
                atom_to_library_ID_map=atom_to_library_ID_map,
                unique_atoms=unique_atoms,
                bond_lookup=bond_lookup,
                N_points=2,
                mask_first_stop=mask_first_stop,
                ground_truth_sequence=sequence,
                seq=seq,
                canonical=True,
            )
            mask_first_stop = False

            if stop_after_sequence:
                if seq >= len(sequence):
                    break

            q += 1
            if q > 30:
                logger("failed to converge")
                unconverged += 1
                break

    return (
        mol,
        updated_mol,
        queue,
        positioned_atoms_indices,
        atom_to_library_ID_map,
        0,
        0,
        0,
    )


def generate_3D_mol_from_sequence(
    sequence,
    partial_mol,
    mol,
    positioned_atoms_indices,
    queue,
    atom_to_library_ID_map,
    model,
    rocs_model,
    AtomFragment_database,
    unique_atoms,
    bond_lookup,
    N_points=5,
    N_points_rocs=5,
    stop_after_sequence=False,
    random_dihedrals=False,
    mask_first_stop=False,
    use_mol_before_3D_scoring=False,
    stochastic=False,
    chirality_scoring=True,
    stop_threshold=0.01,
    steric_mask=False,
    variational_factor_equi=0.0,
    variational_factor_inv=0.0,
    interpolate_to_prior_equi=0.0,
    interpolate_to_prior_inv=0.0,
    use_variational_GNN=False,
    variational_GNN_factor=1.0,
    interpolate_to_GNN_prior=0.0,
    rocs_use_variational_GNN=False,
    rocs_variational_GNN_factor=0.0,
    rocs_interpolate_to_GNN_prior=0.0,
    pointCloudVar=1.0 / (12.0 * 1.7),
    rocs_pointCloudVar=1.0 / (12.0 * 1.7),
    h_interpolate=None,
):
    fragment_library_atom_features = np.concatenate(
        AtomFragment_database["atom_features"], axis=0
    ).reshape((len(AtomFragment_database), -1))

    node_features = getNodeFeatures(mol.GetAtoms())
    ring_fragments = get_ring_fragments(mol)

    data_mol = process_mol(
        mol, AtomFragment_database, N_points=N_points, pointCloudVar=pointCloudVar
    )
    data_mol_rocs = process_mol(
        mol,
        AtomFragment_database,
        N_points=N_points_rocs,
        pointCloudVar=rocs_pointCloudVar,
    )

    library_dataset = AtomFragmentLibrary(AtomFragment_database)
    library_loader = torch_geometric.data.DataLoader(
        library_dataset, shuffle=False, batch_size=len(library_dataset), num_workers=0
    )
    fragment_batch = next(iter(library_loader))

    if model is not None:
        (
            fragment_library_features,
            fragment_library_node_features,
            fragment_library_batch,
        ) = model.Encoder.encode_fragment_library(fragment_batch)

        model_out = model.Encoder.encode(
            x=torch.cat(
                (
                    data_mol.x,
                    fragment_library_features[data_mol.atom_fragment_associations],
                ),
                dim=1,
            ),
            edge_index=data_mol.edge_index,
            edge_attr=data_mol.edge_attr,
            pos=data_mol.pos,
            points=data_mol.cloud,
            points_atom_index=data_mol.cloud_indices,
            batch_size=1,
            select_indices=None,
            select_indices_batch=None,
            shared_encoders=True,
            use_variational_GNN=use_variational_GNN,
            variational_GNN_factor=variational_GNN_factor,
            interpolate_to_GNN_prior=interpolate_to_GNN_prior,
            h_interpolate=h_interpolate,
        )
        Z_equi, Z_inv = model_out[1], model_out[2]

    else:
        fragment_library_features = None
        fragment_library_node_features = None
        fragment_library_batch = None
        Z_equi = None
        Z_inv = None

    if model.variational:
        if (model.variational_mode == "both") | (model.variational_mode == "equi"):
            batch_size = 1
            Z_equi = model.Encoder.VariationalEncoder_equi(
                Z_equi
            )  # equivariant, shape [B,C*2,3]
            Z_equi_mean, Z_equi_logvar = Z_equi.chunk(
                2, dim=1
            )  # equivariant, shape [B,C,3]
            Z_equi_logvar, _ = model.Encoder.VariationalEncoder_equi_T(
                Z_equi_logvar
            )  # invariant, shape [B, C, 3]
            Z_equi_logvar = Z_equi_logvar.reshape(
                batch_size, -1
            )  # flattened to shape [B, C*3]
            Z_equi_logvar = (
                model.Encoder.VariationEncoder_equi_linear(Z_equi_logvar)
                .unsqueeze(2)
                .expand((-1, -1, 3))
            )  # invariant, shape [B, C, 3]
            Z_equi_std = torch.exp(0.5 * Z_equi_logvar)  # invariant, shape [B, C, 1]

            if interpolate_to_prior_equi > 0.0:
                Z_equi_mean = torch.lerp(
                    Z_equi_mean,
                    torch.zeros_like(Z_equi_mean),
                    interpolate_to_prior_equi,
                )
                Z_equi_std = torch.lerp(
                    Z_equi_std, torch.ones_like(Z_equi_std), interpolate_to_prior_equi
                )

            Z_equi_eps = (
                torch.randn_like(Z_equi_mean) * variational_factor_equi
            )  # normal noise with shape [B,C,3]
            Z_equi = (
                Z_equi_mean + Z_equi_std * Z_equi_eps
            )  # equivariant mean + isotropic noise (equivariant)
        else:
            Z_equi_mean = None
            Z_equi_std = None

        if (model.variational_mode == "both") | (model.variational_mode == "inv"):
            Z_inv = model.Encoder.VariationalEncoder_inv(Z_inv)
            Z_inv_mean, Z_inv_logvar = Z_inv.chunk(2, dim=1)
            Z_inv_std = torch.exp(0.5 * Z_inv_logvar)

            if interpolate_to_prior_inv > 0.0:
                Z_inv_mean = torch.lerp(
                    Z_inv_mean, torch.zeros_like(Z_inv_mean), interpolate_to_prior_inv
                )
                Z_inv_std = torch.lerp(
                    Z_inv_std, torch.ones_like(Z_inv_std), interpolate_to_prior_inv
                )

            Z_inv_eps = torch.randn_like(Z_inv_mean) * variational_factor_inv
            Z_inv = Z_inv_mean + Z_inv_std * Z_inv_eps

        else:
            Z_inv_mean = None
            Z_inv_std = None
    else:
        Z_equi_mean = None
        Z_equi_std = None
        Z_inv_mean = None
        Z_inv_std = None

    (
        fragment_library_features_rocs,
        fragment_library_node_features_rocs,
        fragment_library_batch_rocs,
    ) = rocs_model.Encoder.encode_fragment_library(fragment_batch)

    rocs_out = rocs_model.Encoder.encode(
        x=torch.cat(
            (
                data_mol_rocs.x,
                fragment_library_features_rocs[
                    data_mol_rocs.atom_fragment_associations
                ],
            ),
            dim=1,
        ),
        edge_index=data_mol_rocs.edge_index,
        edge_attr=data_mol_rocs.edge_attr,
        pos=data_mol_rocs.pos,
        points=data_mol_rocs.cloud,
        points_atom_index=data_mol_rocs.cloud_indices,
        batch_size=1,
        select_indices=None,
        select_indices_batch=None,
        shared_encoders=True,
        use_variational_GNN=rocs_use_variational_GNN,
        variational_GNN_factor=rocs_variational_GNN_factor,
        interpolate_to_GNN_prior=rocs_interpolate_to_GNN_prior,
    )
    Z_equi_rocs, Z_inv_rocs = rocs_out[1], rocs_out[2]

    try:
        unconverged = 0
        updated_mol = deepcopy(partial_mol)
        # atom_to_library_ID_map = {}
        q = 0
        mask_first_stop = mask_first_stop

        seq = 0
        N_rocs_decisions = 0
        is_chirality_scored = False
        while len(queue) > 0:
            (
                updated_mol,
                queue,
                positioned_atoms_indices,
                atom_to_library_ID_map,
                seq,
                N_rocs_decisions,
                chirality_scored,
            ) = update_3D_mol_BFS_switched(
                partial_mol=updated_mol,
                queue=queue,
                positioned_atoms_indices=positioned_atoms_indices,
                AtomFragment_database=AtomFragment_database,
                unique_atoms=unique_atoms,
                bond_lookup=bond_lookup,
                fragment_batch=fragment_batch,
                atom_to_library_ID_map=atom_to_library_ID_map,
                model=model,
                Z_equi=Z_equi,
                Z_inv=Z_inv,
                N_points=N_points,
                fragment_library_node_features=fragment_library_node_features,
                fragment_library_features=fragment_library_features,
                rocs_model=rocs_model,
                Z_equi_rocs=Z_equi_rocs,
                Z_inv_rocs=Z_inv_rocs,
                N_points_rocs=N_points_rocs,
                fragment_library_features_rocs=fragment_library_features_rocs,
                fragment_library_node_features_rocs=fragment_library_node_features_rocs,
                mask_first_stop=mask_first_stop,
                ground_truth_sequence=sequence,
                seq=seq,
                canonical=True,
                random_dihedrals=random_dihedrals,
                N_rocs_decisions=N_rocs_decisions,
                use_mol_before_3D_scoring=deepcopy(mol)
                if use_mol_before_3D_scoring
                else None,  # None
                stochastic=stochastic,
                chirality_scoring=chirality_scoring,
                stop_threshold=stop_threshold,
                steric_mask=steric_mask,
            )

            if chirality_scored == True:
                is_chirality_scored = True

            mask_first_stop = False

            if stop_after_sequence:
                if seq >= len(sequence):
                    break

            q += 1
            if q > 30:
                logger("failed to converge")
                unconverged += 1
                break

    except Exception as e:
        logger(f"failed to generate: {e}")
        return None, None, None, None, None, None, 0, 1, 0, None  # failed to generate

    return (
        mol,
        updated_mol,
        queue,
        positioned_atoms_indices,
        atom_to_library_ID_map,
        N_rocs_decisions,
        0,
        0,
        0,
        int(is_chirality_scored),
    )


def generate_seed_from_sequence(
    sequence,
    mol,
    partial_indices,
    queue_indices,
    AtomFragment_database,
    unique_atoms,
    bond_lookup,
    stop_after_sequence=True,
):
    fragment_library_atom_features = np.concatenate(
        AtomFragment_database["atom_features"], axis=0
    ).reshape((len(AtomFragment_database), -1))

    node_features = getNodeFeatures(mol.GetAtoms())
    ring_fragments = get_ring_fragments(mol)

    seed_partial_indices = deepcopy(partial_indices)
    add_to_partial = [
        list(f) for p in seed_partial_indices for f in ring_fragments if p in f
    ]
    add_to_partial = [item for sublist in add_to_partial for item in sublist]
    seeding_partial_indices = list(set(seed_partial_indices).union(add_to_partial))

    try:
        if len(seed_partial_indices) == 1:  # atom
            atom_ID = np.where(
                np.all(
                    fragment_library_atom_features
                    == node_features[seed_partial_indices[0]],
                    axis=1,
                )
            )[0][0]
            partial_mol = AtomFragment_database.iloc[atom_ID].atom_objects
            num_Hs = sum([a.GetAtomicNum() == 1.0 for a in partial_mol.GetAtoms()])
            partial_mol = rdkit.Chem.RemoveHs(partial_mol)
            partial_mol.GetAtomWithIdx(0).SetNumExplicitHs(num_Hs)
            idx_map = {seed_partial_indices[0]: 0}  # only 1 atom!

            rdkit.Chem.AllChem.EmbedMolecule(
                partial_mol, randomSeed=0xF00D
            )  # we just need a conformer object later. Since this is an atom, its position should be just at the origin.
            atom_to_library_ID_map = {0: atom_ID}

        else:  # fragment
            seed_smiles = get_fragment_smiles(mol, seed_partial_indices)
            frag_ID = AtomFragment_database.index[
                AtomFragment_database["smiles"] == seed_smiles
            ].tolist()[0]
            partial_mol = AtomFragment_database.iloc[frag_ID].mol
            idx_map = get_reindexing_map(mol, seeding_partial_indices, partial_mol)
            atom_to_library_ID_map = {}
    except Exception as e:
        logger(f"failed to create seed: {e}")
        return None, None, None, None, None, 1, 0, 0  # failed to start

    queue = [idx_map[q] for q in queue_indices]

    positioned_atoms_indices = list(range(0, len(partial_indices)))

    updated_mol = deepcopy(partial_mol)
    # atom_to_library_ID_map = {}
    q = 0
    mask_first_stop = False

    seq = 0
    unconverged = 0
    while len(queue) > 0:
        (
            updated_mol,
            queue,
            positioned_atoms_indices,
            atom_to_library_ID_map,
            seq,
        ) = update_2D_mol_BFS_switched(
            model=None,
            Z_inv=None,
            partial_mol=updated_mol,
            queue=queue,
            positioned_atoms_indices=positioned_atoms_indices,
            AtomFragment_database=AtomFragment_database,
            fragment_batch=None,
            fragment_library_node_features=None,
            fragment_library_features=None,
            atom_to_library_ID_map=atom_to_library_ID_map,
            unique_atoms=unique_atoms,
            bond_lookup=bond_lookup,
            N_points=2,
            mask_first_stop=mask_first_stop,
            ground_truth_sequence=sequence,
            seq=seq,
            canonical=True,
        )
        mask_first_stop = False

        if stop_after_sequence:
            if seq >= len(sequence):
                break

        q += 1
        if q > 30:
            logger("failed to converge")
            unconverged += 1
            break

    return (
        mol,
        updated_mol,
        queue,
        positioned_atoms_indices,
        atom_to_library_ID_map,
        0,
        0,
        0,
    )


from numpy.linalg import norm
from scipy.spatial.transform import Rotation


def swap_cis_trans(
    mol, focal, source
):  # focal, source form a double bond. Flips the cis/trans configuration at the focal atom.
    mol2 = deepcopy(mol)

    v = (
        np.array(mol2.GetConformer().GetPositions())
        - mol2.GetConformer().GetPositions()[source]
    )
    axis = v[focal] - v[source]
    theta = 180.0

    axis = axis / norm(axis)
    rot = Rotation.from_rotvec(axis * theta, degrees=True)

    new_v = rot.apply(v) + mol2.GetConformer().GetPositions()[source]

    G = deepcopy(
        get_substructure_graph_for_matching(mol2, list(range(0, mol2.GetNumAtoms())))
    )
    G.remove_edge(focal, source)

    disjoint_graphs = [
        list(G.subgraph(c).copy().nodes()) for c in nx.connected_components(G)
    ]
    query_indices = [graph for graph in disjoint_graphs if focal in graph][0]

    for i in query_indices:
        x, y, z = new_v[i]
        mol2.GetConformer().SetAtomPosition(i, Point3D(x, y, z))

    return mol2


def swap_chirality_focal_ring_atoms(
    mol_, focal, A, B=None
):  # focal in ring system. A, B (if present) are bonded to focal, but not in the focal's ring system.
    mol = deepcopy(mol_)

    rings = mol.GetRingInfo().AtomRings()
    rings = [set(r) for r in rings]
    ring = [r for r in rings if focal in r][0]

    source = [
        a.GetIdx()
        for a in mol.GetAtomWithIdx(focal).GetNeighbors()
        if ((a.GetIdx() in ring) & (a.GetIdx() not in (A, B)))
    ][0]
    source_of_source = [
        a.GetIdx()
        for a in mol.GetAtomWithIdx(source).GetNeighbors()
        if a.GetIdx() != focal
    ]
    assert len(source_of_source) > 0
    source_of_source = source_of_source[0]

    cut_neighbor = [
        a.GetIdx()
        for a in mol.GetAtomWithIdx(focal).GetNeighbors()
        if ((a.GetIdx() in ring) & (a.GetIdx() not in (A, B, source)))
    ]
    assert len(cut_neighbor) > 0
    cut_neighbor = cut_neighbor[0]

    d_A = rdkit.Chem.rdMolTransforms.GetDihedralDeg(
        mol.GetConformer(), source_of_source, source, focal, A
    )
    if B is not None:
        d_B = rdkit.Chem.rdMolTransforms.GetDihedralDeg(
            mol.GetConformer(), source_of_source, source, focal, B
        )
    else:
        d_cut_neighbor = rdkit.Chem.rdMolTransforms.GetDihedralDeg(
            mol.GetConformer(), source_of_source, source, focal, cut_neighbor
        )
        dif = (d_cut_neighbor - d_A + 180) % 360 - 180
        if np.abs(dif - (-120.0)) < np.abs(dif - 120.0):
            sign = 1
        else:
            sign = -1

        d_B_vacant = d_A + sign * 120.0

    em = rdkit.Chem.RWMol(mol)
    em.RemoveBond(focal, cut_neighbor)
    rdkit.Chem.SanitizeMol(em)

    em_1 = deepcopy(em)
    if B is not None:
        em_2 = deepcopy(em)

    G = deepcopy(
        get_substructure_graph_for_matching(mol, list(range(0, mol.GetNumAtoms())))
    )
    G.remove_edge(focal, A)
    if B is not None:
        G.remove_edge(focal, B)
    disjoint_graphs = [
        list(G.subgraph(c).copy().nodes()) for c in nx.connected_components(G)
    ]
    A_indices = [graph for graph in disjoint_graphs if A in graph][0]
    if B is not None:
        B_indices = [graph for graph in disjoint_graphs if B in graph][0]

    if B is not None:
        rdkit.Chem.rdMolTransforms.SetDihedralDeg(
            em_1.GetConformer(), source_of_source, source, focal, A, d_B
        )
    else:
        rdkit.Chem.rdMolTransforms.SetDihedralDeg(
            em_1.GetConformer(), source_of_source, source, focal, A, d_B_vacant
        )
    if B is not None:
        rdkit.Chem.rdMolTransforms.SetDihedralDeg(
            em_2.GetConformer(), source_of_source, source, focal, B, d_A
        )

    for i in A_indices:
        x, y, z = em_1.GetConformer().GetPositions()[i]
        mol.GetConformer().SetAtomPosition(i, Point3D(x, y, z))

    if B is not None:
        for i in B_indices:
            x, y, z = em_2.GetConformer().GetPositions()[i]
            mol.GetConformer().SetAtomPosition(i, Point3D(x, y, z))
    return mol


def swap_chirality_atoms(
    mol_, focal, source, A, B
):  # A and B cannot be part of the same ring system, and focal is not in a ring.
    mol = deepcopy(mol_)
    mol1 = deepcopy(mol_)
    mol2 = deepcopy(mol_)

    source_of_source = [
        a.GetIdx()
        for a in mol.GetAtomWithIdx(source).GetNeighbors()
        if a.GetIdx() != focal
    ]
    assert len(source_of_source) > 0
    source_of_source = source_of_source[0]

    d_A = rdkit.Chem.rdMolTransforms.GetDihedralDeg(
        mol.GetConformer(), source_of_source, source, focal, A
    )
    d_B = rdkit.Chem.rdMolTransforms.GetDihedralDeg(
        mol.GetConformer(), source_of_source, source, focal, B
    )

    rdkit.Chem.rdMolTransforms.SetDihedralDeg(
        mol1.GetConformer(), source_of_source, source, focal, A, d_B
    )
    rdkit.Chem.rdMolTransforms.SetDihedralDeg(
        mol2.GetConformer(), source_of_source, source, focal, B, d_A
    )

    G = deepcopy(
        get_substructure_graph_for_matching(mol, list(range(0, mol.GetNumAtoms())))
    )
    G.remove_edge(focal, A)
    G.remove_edge(focal, B)
    disjoint_graphs = [
        list(G.subgraph(c).copy().nodes()) for c in nx.connected_components(G)
    ]
    A_indices = [graph for graph in disjoint_graphs if A in graph][0]
    B_indices = [graph for graph in disjoint_graphs if B in graph][0]

    for i in A_indices:
        x, y, z = mol1.GetConformer().GetPositions()[i]
        mol.GetConformer().SetAtomPosition(i, Point3D(x, y, z))

    for i in B_indices:
        x, y, z = mol2.GetConformer().GetPositions()[i]
        mol.GetConformer().SetAtomPosition(i, Point3D(x, y, z))

    return mol


def swap_chirality_focal_ring(mol, focal, source, A, B=None):
    # A, focal are in the same ring. Source is bonded to focal, but not in the same ring system. B, if present, is bonded to the focal BUT is not in the ring system. Swaps the chirality of the entire ring (focal atom included), with the source atom's position remaining fixed.
    mol2 = deepcopy(mol)

    source_of_source = [
        a.GetIdx()
        for a in mol.GetAtomWithIdx(source).GetNeighbors()
        if a.GetIdx() != focal
    ]
    assert len(source_of_source) > 0
    source_of_source = source_of_source[0]

    neighbor = [
        a.GetIdx()
        for a in mol.GetAtomWithIdx(A).GetNeighbors()
        if ((a.GetIdx() != focal) & (a.IsInRing() == True))
    ][0]
    dihedral_A = (source, focal, A, neighbor)
    if B is not None:
        dihedral_B = (neighbor, A, focal, B)
    cut_neighbor = [
        a.GetIdx()
        for a in mol.GetAtomWithIdx(focal).GetNeighbors()
        if ((a.GetIdx() != A) & (a.IsInRing() == True))
    ][0]

    em = rdkit.Chem.RWMol(mol)
    em.RemoveBond(focal, cut_neighbor)
    rdkit.Chem.SanitizeMol(em)

    d1 = rdkit.Chem.rdMolTransforms.GetDihedralDeg(
        em.GetConformer(), source_of_source, source, focal, A
    )
    d2 = rdkit.Chem.rdMolTransforms.GetDihedralDeg(
        em.GetConformer(), source_of_source, source, focal, cut_neighbor
    )
    dif = (d2 - d1 + 180) % 360 - 180
    if np.abs(dif - (-120.0)) < np.abs(dif - 120.0):
        sign = -1
    else:
        sign = 1

    dihedral_1 = rdkit.Chem.rdMolTransforms.GetDihedralDeg(
        em.GetConformer(), *dihedral_A
    )
    if B is not None:
        dihedral_2 = rdkit.Chem.rdMolTransforms.GetDihedralDeg(
            em.GetConformer(), *dihedral_B
        )

    em_1 = deepcopy(em)
    rdkit.Chem.rdMolTransforms.SetDihedralDeg(
        em_1.GetConformer(), *dihedral_A, dihedral_1 + sign * 120.0
    )
    if B is not None:
        em_2 = deepcopy(em)
        rdkit.Chem.rdMolTransforms.SetDihedralDeg(
            em_2.GetConformer(), *dihedral_B, dihedral_2 + sign * 120.0
        )

    G = deepcopy(
        get_substructure_graph_for_matching(mol2, list(range(0, mol2.GetNumAtoms())))
    )
    G.remove_edge(focal, source)
    if B is not None:
        G.remove_edge(focal, B)
    disjoint_graphs = [
        list(G.subgraph(c).copy().nodes()) for c in nx.connected_components(G)
    ]
    A_indices = [graph for graph in disjoint_graphs if A in graph][0]
    if B is not None:
        B_indices = [graph for graph in disjoint_graphs if B in graph][0]

    for i in A_indices:
        x, y, z = em_1.GetConformer().GetPositions()[i]
        mol2.GetConformer().SetAtomPosition(i, Point3D(x, y, z))

    if B is not None:
        for i in B_indices:
            x, y, z = em_2.GetConformer().GetPositions()[i]
            mol2.GetConformer().SetAtomPosition(i, Point3D(x, y, z))

    return mol2


def getNodeFeaturesForGraphMatching(list_rdkit_atoms):
    atomTypes = ["H", "C", "B", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"]

    def one_hot_embedding(value, options):
        embedding = [0] * (len(options) + 1)
        index = options.index(value) if value in options else -1
        embedding[index] = 1
        return embedding

    F_v = len(atomTypes) + 1

    node_features = np.zeros((len(list_rdkit_atoms), F_v))
    for node_index, node in enumerate(list_rdkit_atoms):
        features = one_hot_embedding(
            node.GetSymbol(), atomTypes
        )  # atom symbol, dim=12 + 1

        node_features[node_index, :] = features

    return np.array(node_features, dtype=np.float32)


def get_substructure_graph_for_matching(mol, atom_indices, node_features=None):
    G = nx.Graph()
    bonds = list(mol.GetBonds())
    bond_indices = [sorted([b.GetBeginAtomIdx(), b.GetEndAtomIdx()]) for b in bonds]

    for atom_idx in atom_indices:
        atom = mol.GetAtomWithIdx(int(atom_idx))
        if node_features is None:
            atom_features = getNodeFeaturesForGraphMatching([atom])[0]
        else:
            atom_features = node_features[atom_idx]
        G.add_node(atom_idx, atom_features=atom_features)

    for i in atom_indices:
        for j in atom_indices:
            if sorted([i, j]) in bond_indices:
                G.add_edge(
                    i,
                    j,
                    bond_type=mol.GetBondBetweenAtoms(
                        int(i), int(j)
                    ).GetBondTypeAsDouble(),
                )
    return G


def get_available_dihedrals(
    mol, focal_idx, source_idx, source_of_source_idx, focal_hybridization_fixed=None
):
    # returns available dihedral positions (e.g., that may be occupied by implicit hydrogens)
    # - uses hybridization symmetry considerations along with currently occupied positions

    focal_source_bond_type = mol.GetBondBetweenAtoms(
        int(focal_idx), int(source_idx)
    ).GetBondTypeAsDouble()

    focal_hybridization = (
        str(mol.GetAtomWithIdx(focal_idx).GetHybridization())
        if focal_hybridization_fixed is None
        else focal_hybridization_fixed
    )

    other_focal_bonds = get_bonded_connections(
        mol, atoms=[focal_idx], completed_atoms=[source_idx]
    )[1]

    occupied_dihedrals = [
        rdkit.Chem.rdMolTransforms.GetDihedralDeg(
            mol.GetConformer(),
            int(source_of_source_idx),
            int(source_idx),
            int(focal_idx),
            int(b),
        )
        for b in other_focal_bonds
    ]

    if len(occupied_dihedrals) > 0:
        if (focal_hybridization == "SP") | (
            mol.GetAtomWithIdx(focal_idx).GetNumRadicalElectrons() == 3
        ):
            raise Exception("Valency exception for SP hybridization")

        elif (focal_hybridization == "SP2") | (
            mol.GetAtomWithIdx(focal_idx).GetNumRadicalElectrons() == 2
        ):
            if len(occupied_dihedrals) == 2:
                raise Exception("Valency exception for SP2 hybridization")
            else:
                return [occupied_dihedrals[0] + 180.0]

        elif focal_hybridization == "SP3":
            if len(occupied_dihedrals) == 1:
                return [occupied_dihedrals[0] - 120.0, occupied_dihedrals[0] + 120.0]

            elif len(occupied_dihedrals) == 2:
                dihedral_coupled_1 = occupied_dihedrals[0]  # * np.pi / 180.
                dihedral_coupled_2 = occupied_dihedrals[1]  # * np.pi / 180.

                avail_dihedral_1 = (dihedral_coupled_1) - 120.0
                avail_dihedral_2 = (dihedral_coupled_1) + 120.0

                # these similarities give a list of the angles between the query dihedral, and the existing occupied dihedrals
                # we should accept the query dihedral which gives the maximal minimum angle difference
                similarity_1 = [
                    np.abs(((avail_dihedral_1 - dihedral_coupled_1) + 180) % 360 - 180),
                    np.abs(((avail_dihedral_1 - dihedral_coupled_2) + 180) % 360 - 180),
                ]
                similarity_2 = [
                    np.abs(((avail_dihedral_2 - dihedral_coupled_1) + 180) % 360 - 180),
                    np.abs(((avail_dihedral_2 - dihedral_coupled_2) + 180) % 360 - 180),
                ]

                if min(similarity_1) < min(similarity_2):
                    return [avail_dihedral_2]
                else:
                    return [avail_dihedral_1]

            else:
                raise Exception("Valency exception for SP3 hybridization")

        else:
            raise Exception(f"Hybridization {focal_hybridization} not implemented")

    else:
        if (focal_hybridization == "SP") | (
            mol.GetAtomWithIdx(focal_idx).GetNumRadicalElectrons() == 3
        ):
            return False  # position must be specially specified by SP geometry
        elif (focal_hybridization == "SP2") | (
            mol.GetAtomWithIdx(focal_idx).GetNumRadicalElectrons() == 2
        ):
            return [0.0, 180.0]
        elif focal_hybridization == "SP3":
            return True  # any dihedral is valid -- will be determined during scoring
        else:
            raise Exception(f"Hybridization {focal_hybridization} not implemented")


def get_bond_angle(mol, focal_idx, focal_hybridization_fixed=None):
    # returns rough bond angles based on atom hybridization
    # we also use NumRadicalElectrons to anticipate whether a double or triple bond will be formed to the focal atom (but has not yet been generated)

    focal_hybridization = (
        str(mol.GetAtomWithIdx(focal_idx).GetHybridization())
        if focal_hybridization_fixed is None
        else focal_hybridization_fixed
    )
    if (focal_hybridization == "SP") | (
        mol.GetAtomWithIdx(focal_idx).GetNumRadicalElectrons() == 3
    ):
        return 180.0
    elif (focal_hybridization == "SP2") | (
        mol.GetAtomWithIdx(focal_idx).GetNumRadicalElectrons() == 2
    ):
        return 120.0
    elif focal_hybridization == "SP3":
        return 109.5
    else:
        return None  # not implemented for other hybridization states


def make_3D_bond_attachments(
    mol,
    positioned_atoms_indices,
    node_features,
    next_atom_fragment_ID,
    focal_attachment_index,
    next_fragment_attachment_index_rel_next,
    bond_type,
    AtomFragment_database,
    unique_atoms,
    bond_lookup,
    ignore_3D=False,
):
    num_atoms = mol.GetNumAtoms()

    source, source_of_source = get_source_atom(
        mol,
        focal_attachment_index,
        list(set(positioned_atoms_indices) - set([focal_attachment_index])),
    )

    if not ignore_3D:
        focal_atom = mol.GetAtomWithIdx(focal_attachment_index)
        focal_hybridization_fixed = None

        # refining bond angles around sp3 nitrogens or oxygens when they are newly bonded to atoms that form double or triple bonds, thus leading to resonance and SP2 hybridization on the N/O atoms

        neighbors = [a.GetIdx() for a in focal_atom.GetNeighbors()]
        fixed_source = [n for n in neighbors if n < focal_attachment_index]
        unfixed_neighbors = [n for n in neighbors if n > focal_attachment_index]

        if AtomFragment_database.iloc[next_atom_fragment_ID].is_fragment == 0:
            next_atom = AtomFragment_database.iloc[next_atom_fragment_ID].atom_objects
            if (
                (focal_atom.GetSymbol() in ("O", "N"))
                & (str(focal_atom.GetHybridization()) == "SP3")
            ) & (
                (next_atom.GetAtomWithIdx(0).GetNumRadicalElectrons() in (2, 3))
                | (
                    True
                    in [
                        (
                            (
                                str(mol.GetAtomWithIdx(n).GetHybridization())
                                in ("SP2", "SP")
                            )
                            | (mol.GetAtomWithIdx(n).GetNumRadicalElectrons() in (2, 3))
                        )
                        for n in unfixed_neighbors
                    ]
                )
            ):
                focal_hybridization_fixed = "SP2"
        else:
            next_fragment_mol = deepcopy(
                AtomFragment_database.iloc[next_atom_fragment_ID].mol
            )
            next_fragment_mol = rdkit.Chem.RemoveHs(next_fragment_mol)
            next_atom = next_fragment_mol.GetAtomWithIdx(
                next_fragment_attachment_index_rel_next
            )
            if (
                (focal_atom.GetSymbol() in ("O", "N"))
                & (str(focal_atom.GetHybridization()) == "SP3")
            ) & (
                (str(next_atom.GetHybridization()) == "SP2")
                | (
                    True
                    in [
                        (
                            (
                                str(mol.GetAtomWithIdx(n).GetHybridization())
                                in ("SP2", "SP")
                            )
                            | (mol.GetAtomWithIdx(n).GetNumRadicalElectrons() in (2, 3))
                        )
                        for n in unfixed_neighbors
                    ]
                )
            ):
                focal_hybridization_fixed = "SP2"

        if (focal_hybridization_fixed is not None) & (focal_atom.IsInRing() == False):
            for n in neighbors:
                if (
                    (n > focal_attachment_index)
                    & (len(fixed_source) > 0)
                    & (focal_hybridization_fixed == "SP2")
                ):
                    rdkit.Chem.rdMolTransforms.SetAngleDeg(
                        mol.GetConformer(),
                        *(fixed_source[0], focal_attachment_index, n),
                        120.0,
                    )

    if not ignore_3D:
        # counting number of positioned neighbors prior to additions
        available_dihedrals = get_available_dihedrals(
            mol,
            focal_attachment_index,
            source,
            source_of_source,
            focal_hybridization_fixed=focal_hybridization_fixed,
        )

    next_fragment_attachment_index = num_atoms + next_fragment_attachment_index_rel_next

    if (
        AtomFragment_database.iloc[next_atom_fragment_ID].is_fragment == 0
    ):  # adding a single atom only
        next_atom = deepcopy(
            AtomFragment_database.iloc[next_atom_fragment_ID].atom_objects
        )

        assert next_atom != None

        em_mol = rdkit.Chem.RWMol(mol)

        # em_mol = rdkit.Chem.RemoveHs(em_mol)
        # em_mol = rdkit.Chem.AddHs(em_mol)
        em_mol = rdkit.Chem.CombineMols(em_mol, next_atom)
        em_mol = rdkit.Chem.AddHs(em_mol)

        em_mol = rdkit.Chem.RWMol(em_mol)
        em_mol.AddBond(focal_attachment_index, next_fragment_attachment_index)

        if bond_type == rdkit.Chem.rdchem.BondType.SINGLE:
            focal_H_bonds = [
                b.GetOtherAtomIdx(focal_attachment_index)
                for b in em_mol.GetAtomWithIdx(focal_attachment_index).GetBonds()
            ]
            focal_remove_H_idx = [
                h for h in focal_H_bonds if em_mol.GetAtomWithIdx(h).GetAtomicNum() == 1
            ][0]
            next_H_bonds = [
                b.GetOtherAtomIdx(next_fragment_attachment_index)
                for b in em_mol.GetAtomWithIdx(
                    next_fragment_attachment_index
                ).GetBonds()
            ]
            next_remove_H_idx = [
                h for h in next_H_bonds if em_mol.GetAtomWithIdx(h).GetAtomicNum() == 1
            ][0]

            if next_remove_H_idx > focal_remove_H_idx:
                em_mol.RemoveAtom(next_remove_H_idx)
                em_mol.RemoveAtom(focal_remove_H_idx)
            else:
                em_mol.RemoveAtom(focal_remove_H_idx)
                em_mol.RemoveAtom(next_remove_H_idx)

            em_mol.GetBondBetweenAtoms(
                focal_attachment_index, next_fragment_attachment_index
            ).SetBondType(bond_type)
            em_mol = rdkit.Chem.RemoveHs(em_mol)

        else:
            em_mol.GetBondBetweenAtoms(
                focal_attachment_index, next_fragment_attachment_index
            ).SetBondType(bond_type)
            # em_mol = rdkit.Chem.RemoveHs(em_mol)
            f_rad_count = em_mol.GetAtomWithIdx(
                focal_attachment_index
            ).GetNumRadicalElectrons()
            next_rad_count = em_mol.GetAtomWithIdx(
                next_fragment_attachment_index
            ).GetNumRadicalElectrons()

            N_rad = 2 if bond_type == rdkit.Chem.rdchem.BondType.DOUBLE else 3
            updated_f_rad_count = (
                f_rad_count - N_rad if (f_rad_count - N_rad) >= 0 else f_rad_count
            )  # sometimes, there aren't any radicals on an atom registered as needing double/triple bonds (e.g., in SH2 -> SO2H2)
            updated_next_rad_count = (
                next_rad_count - N_rad
                if (next_rad_count - N_rad) >= 0
                else next_rad_count
            )  # sometimes, there aren't any radicals on an atom registered as needing double/triple bonds (e.g., in SH2 -> SO2H2)

            em_mol.GetAtomWithIdx(focal_attachment_index).SetNumRadicalElectrons(
                updated_f_rad_count
            )
            em_mol.GetAtomWithIdx(
                next_fragment_attachment_index
            ).SetNumRadicalElectrons(updated_next_rad_count)

            em_mol = rdkit.Chem.RemoveHs(em_mol)

        if ignore_3D:
            next_fragment_indices = [next_fragment_attachment_index]
            return em_mol, next_fragment_attachment_index, next_fragment_indices

        bond_angle = get_bond_angle(
            em_mol,
            focal_attachment_index,
            focal_hybridization_fixed=focal_hybridization_fixed,
        )

        bond = em_mol.GetBondBetweenAtoms(
            focal_attachment_index, next_fragment_attachment_index
        )
        atom1_ID = retrieve_atom_ID(
            node_features[focal_attachment_index], unique_atoms[1:]
        )
        atom2_ID = retrieve_atom_ID(
            AtomFragment_database.iloc[next_atom_fragment_ID].atom_features,
            unique_atoms[1:],
        )
        bond_properties = [*sorted([atom1_ID, atom2_ID]), bond.GetBondTypeAsDouble()]
        bond_ID = retrieve_bond_ID(bond_properties, bond_lookup)
        if bond_ID != None:
            bond_distance = bond_lookup.iloc[bond_ID][3]
        else:
            logger(
                f"warning: bond distance between atoms {bond_properties[0]} and {bond_properties[1]} unknown"
            )
            bond_distance = 1.8  # we need a better way of estimating weird bond distances that aren't in the training set

        if available_dihedrals == True:
            # any dihedral is valid
            psi = np.random.uniform(
                0, 2 * np.pi
            )  # randomly sampling placeholder dihedral
            theta = bond_angle * np.pi / 180.0
            d = bond_distance
            p1, p2, p3 = em_mol.GetConformer().GetPositions()[
                [source_of_source, source, focal_attachment_index], :
            ]

            p4 = get_xyz(p1, p2, p3, d, theta, psi)

        elif available_dihedrals == False:
            # sp focal atom --> use bond distance and vector geometry to get next position
            p2, p3 = em_mol.GetConformer().GetPositions()[
                [source, focal_attachment_index], :
            ]
            vector = (p3 - p2) / np.linalg.norm((p3 - p2))

            # adding a little bit of noise to avoid colinearity problems
            p4 = (
                p3
                + vector * bond_distance
                + np.clip(np.random.randn(3) * 0.01, -0.01, 0.01)
            )

        else:
            dihedral = np.random.choice(available_dihedrals)
            psi = dihedral * np.pi / 180.0
            theta = bond_angle * np.pi / 180.0
            d = bond_distance
            p1, p2, p3 = em_mol.GetConformer().GetPositions()[
                [source_of_source, source, focal_attachment_index], :
            ]

            p4 = get_xyz(p1, p2, p3, d, theta, psi)

        x, y, z = p4
        em_mol.GetConformer().SetAtomPosition(
            next_fragment_attachment_index, Point3D(x, y, z)
        )

        next_fragment_indices = [next_fragment_attachment_index]

    else:  # adding a fragment (single bonds only)
        next_fragment_mol = deepcopy(
            AtomFragment_database.iloc[next_atom_fragment_ID].mol
        )
        next_fragment_mol = rdkit.Chem.RemoveHs(next_fragment_mol)
        next_mol_Hs = deepcopy(AtomFragment_database.iloc[next_atom_fragment_ID].mol_Hs)

        bond_type = rdkit.Chem.rdchem.BondType.SINGLE

        em_mol = rdkit.Chem.RWMol(mol)
        em_mol = rdkit.Chem.CombineMols(em_mol, next_fragment_mol)

        em_mol = rdkit.Chem.AddHs(em_mol)
        em_mol = rdkit.Chem.RWMol(em_mol)
        em_mol.AddBond(focal_attachment_index, next_fragment_attachment_index)

        focal_H_bonds = [
            b.GetOtherAtomIdx(focal_attachment_index)
            for b in em_mol.GetAtomWithIdx(focal_attachment_index).GetBonds()
        ]
        focal_remove_H_idx = [
            h for h in focal_H_bonds if em_mol.GetAtomWithIdx(h).GetAtomicNum() == 1
        ][0]
        next_H_bonds = [
            b.GetOtherAtomIdx(next_fragment_attachment_index)
            for b in em_mol.GetAtomWithIdx(next_fragment_attachment_index).GetBonds()
        ]
        next_remove_H_idx = [
            h for h in next_H_bonds if em_mol.GetAtomWithIdx(h).GetAtomicNum() == 1
        ][0]

        if focal_remove_H_idx > next_remove_H_idx:
            em_mol.RemoveAtom(focal_remove_H_idx)
            em_mol.RemoveAtom(next_remove_H_idx)
        else:
            em_mol.RemoveAtom(next_remove_H_idx)
            em_mol.RemoveAtom(focal_remove_H_idx)

        em_mol.GetBondBetweenAtoms(
            focal_attachment_index, next_fragment_attachment_index
        ).SetBondType(bond_type)
        em_mol = rdkit.Chem.RemoveHs(em_mol)

        next_atom = em_mol.GetAtoms()[next_fragment_attachment_index]

        if ignore_3D:
            next_fragment_indices = [
                num_atoms + k for k in range(next_fragment_mol.GetNumAtoms())
            ]
            return em_mol, next_fragment_attachment_index, next_fragment_indices

        # now, make bond distances, angles, positions, etc..
        bond_angle = get_bond_angle(
            em_mol,
            focal_attachment_index,
            focal_hybridization_fixed=focal_hybridization_fixed,
        )

        bond = em_mol.GetBondBetweenAtoms(
            focal_attachment_index, next_fragment_attachment_index
        )
        atom1_ID = retrieve_atom_ID(
            node_features[focal_attachment_index], unique_atoms[1:]
        )
        atom2_ID = retrieve_atom_ID(
            getNodeFeatures(
                [
                    next_fragment_mol.GetAtomWithIdx(
                        next_fragment_attachment_index_rel_next
                    )
                ]
            )[0, :],
            unique_atoms[1:],
        )
        bond_properties = [*sorted([atom1_ID, atom2_ID]), bond.GetBondTypeAsDouble()]
        bond_ID = retrieve_bond_ID(bond_properties, bond_lookup)
        if bond_ID != None:
            bond_distance = bond_lookup.iloc[bond_ID][3]
        else:
            logger(
                f"warning: bond distance between atoms {bond_properties[0]} and {bond_properties[1]} unknown"
            )
            bond_distance = 1.8

        if available_dihedrals == True:
            # any dihedral is valid
            psi = np.random.uniform(
                0, 2 * np.pi
            )  # randomly sampling placeholder dihedral
            theta = bond_angle * np.pi / 180.0
            d = bond_distance
            p1, p2, p3 = em_mol.GetConformer().GetPositions()[
                [source_of_source, source, focal_attachment_index], :
            ]
            p4 = get_xyz(p1, p2, p3, d, theta, psi)

        elif available_dihedrals == False:
            # sp focal atom --> use bond distance and vector geometry to get next position
            p2, p3 = em_mol.GetConformer().GetPositions()[
                [source, focal_attachment_index], :
            ]
            vector = (p3 - p2) / np.linalg.norm((p3 - p2))
            d = bond_distance
            # adding noise to fix colinearity bug.
            p4 = p3 + vector * d + np.clip(np.random.randn(3) * 0.01, -0.01, 0.01)

        else:
            dihedral = np.random.choice(available_dihedrals)
            psi = dihedral * np.pi / 180.0
            theta = bond_angle * np.pi / 180.0
            d = bond_distance
            p1, p2, p3 = em_mol.GetConformer().GetPositions()[
                [source_of_source, source, focal_attachment_index], :
            ]
            p4 = get_xyz(p1, p2, p3, d, theta, psi)

        next_mol_bonds_to_attach_idx = [
            b.GetOtherAtomIdx(next_fragment_attachment_index_rel_next)
            for b in next_mol_Hs.GetAtomWithIdx(
                next_fragment_attachment_index_rel_next
            ).GetBonds()
        ]
        next_attachment_Hs = [
            h
            for h in next_mol_bonds_to_attach_idx
            if next_mol_Hs.GetAtomWithIdx(h).GetAtomicNum() == 1
        ]

        # determines chirality of attachment in the next fragment (explicitly scored later)
        next_H_idx = np.random.choice(next_attachment_Hs)

        # altering the bond angle of the hydrogen attached to the sp3 oxygen or nitrogen so that the hydrogen is now in an sp2 configuration
        if (
            (
                (focal_hybridization_fixed in ("SP2", "SP"))
                | (str(focal_atom.GetHybridization()) in ("SP2", "SP"))
                | (focal_atom.GetNumRadicalElectrons() in (2, 3))
            )
            & (next_atom.GetSymbol() in ("N", "O"))
            & (str(next_atom.GetHybridization()) in ("SP2", "SP3"))
        ):
            neighbor_1_hop = [
                a.GetIdx()
                for a in next_fragment_mol.GetAtomWithIdx(
                    next_fragment_attachment_index_rel_next
                ).GetNeighbors()
            ][0]
            neighbor_2_hop = [
                a.GetIdx()
                for a in next_fragment_mol.GetAtomWithIdx(neighbor_1_hop).GetNeighbors()
                if a.GetIdx() != next_fragment_attachment_index_rel_next
            ][0]
            dihedral_tuple = (
                next_fragment_attachment_index_rel_next,
                neighbor_1_hop,
                neighbor_2_hop,
            )
            available_dihedrals_fragment = get_available_dihedrals(
                next_fragment_mol, *dihedral_tuple, focal_hybridization_fixed="SP2"
            )
            dihedral_frag = available_dihedrals_fragment[
                0
            ]  # there should only be one hydrogen if we have sp2 hybridization
            psi_frag = dihedral_frag * np.pi / 180.0
            theta_frag = 120.0 * np.pi / 180.0
            d_frag = rdkit.Chem.rdMolTransforms.GetBondLength(
                next_mol_Hs.GetConformer(),
                int(next_fragment_attachment_index_rel_next),
                int(next_H_idx),
            )
            p1_frag, p2_frag, p3_frag = next_mol_Hs.GetConformer().GetPositions()[
                list(reversed(dihedral_tuple)), :
            ]
            p4_frag = get_xyz(p1_frag, p2_frag, p3_frag, d_frag, theta_frag, psi_frag)
            x_frag, y_frag, z_frag = p4_frag

            next_mol_Hs.GetConformer().SetAtomPosition(
                int(next_H_idx), Point3D(x_frag, y_frag, z_frag)
            )

        next_vector = (
            next_mol_Hs.GetConformer().GetPositions()[next_H_idx]
            - next_mol_Hs.GetConformer().GetPositions()[
                next_fragment_attachment_index_rel_next
            ]
        )
        next_vector_norm = next_vector / np.linalg.norm(next_vector)
        p4_next = next_mol_Hs.GetConformer().GetPositions()[
            next_fragment_attachment_index_rel_next
        ]
        p3_next = p4_next + next_vector_norm * d

        R, t, err = rigid_transform_3D(
            np.vstack([p3_next, p4_next]).T, np.vstack([p3, p4]).T
        )
        next_positions_transformed = (
            R @ next_mol_Hs.GetConformer().GetPositions().T + t
        ).T

        for k in range(next_fragment_mol.GetNumAtoms()):
            x, y, z = next_positions_transformed[k]
            em_mol.GetConformer().SetAtomPosition(num_atoms + k, Point3D(x, y, z))

        next_fragment_indices = [
            num_atoms + k for k in range(next_fragment_mol.GetNumAtoms())
        ]

    rdkit.Chem.SanitizeMol(em_mol)

    return em_mol, next_fragment_attachment_index, next_fragment_indices


def get_aligned_positions_with_rotated_dihedral(
    mol, partial_graph_indices, dihedral_indices, rotation, absolute_deg=False
):
    mol3D = deepcopy(mol)
    mol_3D_rotated = deepcopy(mol3D)

    if dihedral_indices[0] in partial_graph_indices:
        oriented_dihedral_indices = dihedral_indices
    else:
        assert dihedral_indices[-1] in partial_graph_indices
        oriented_dihedral_indices = dihedral_indices[::-1]

    if not absolute_deg:
        dihedral = rdkit.Chem.rdMolTransforms.GetDihedralDeg(
            mol_3D_rotated.GetConformer(), *dihedral_indices
        )
        rdkit.Chem.rdMolTransforms.SetDihedralDeg(
            mol_3D_rotated.GetConformer(), *dihedral_indices, dihedral + rotation
        )
    else:
        rdkit.Chem.rdMolTransforms.SetDihedralDeg(
            mol_3D_rotated.GetConformer(), *dihedral_indices, rotation
        )

    atom_positions = mol3D.GetConformer().GetPositions()
    rotated_atom_positions = mol_3D_rotated.GetConformer().GetPositions()

    partial_graph_positions = atom_positions[np.array(partial_graph_indices)]
    rotated_partial_graph_positions = rotated_atom_positions[
        np.array(partial_graph_indices)
    ]

    R, t, error = rigid_transform_3D(
        rotated_partial_graph_positions.T, partial_graph_positions.T
    )

    assert error < 1e-5

    # new positions of the molecule with its rotated dihedral, but snapped to the initial coordinates of the inputted partial graph
    aligned_rotated_atom_positions = R @ rotated_atom_positions.T + t

    return aligned_rotated_atom_positions.T


def get_singly_bonded_atoms_to_ring_fragment(mol, ring_fragment):
    bonds_indices = [
        b.GetIdx()
        for b in mol.GetBonds()
        if len(
            set([b.GetBeginAtomIdx(), b.GetEndAtomIdx()]).intersection(
                set(ring_fragment)
            )
        )
        == 1
    ]
    bonded_atom_indices = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        for b in mol.GetBonds()
        if len(
            set([b.GetBeginAtomIdx(), b.GetEndAtomIdx()]).intersection(
                set(ring_fragment)
            )
        )
        == 1
    ]
    bonded_atom_indices_sorted = [
        (b[0], b[1]) if (b[0] in ring_fragment) else (b[1], b[0])
        for b in bonded_atom_indices
    ]
    atoms = [b[1] for b in bonded_atom_indices_sorted]

    return bonds_indices, bonded_atom_indices_sorted, atoms


def get_fragment_smiles(mol, ring_fragment):
    ring_fragment = [int(r) for r in ring_fragment]

    (
        bonds_indices,
        bonded_atom_indices_sorted,
        atoms_bonded_to_ring,
    ) = get_singly_bonded_atoms_to_ring_fragment(mol, ring_fragment)

    pieces = rdkit.Chem.FragmentOnSomeBonds(
        mol, bonds_indices, numToBreak=len(bonds_indices), addDummies=False
    )

    fragsMolAtomMapping = []
    fragments = rdkit.Chem.GetMolFrags(
        pieces[0],
        asMols=True,
        sanitizeFrags=True,
        fragsMolAtomMapping=fragsMolAtomMapping,
    )

    frag_mol = [
        m_
        for i, m_ in enumerate(fragments)
        if (set(fragsMolAtomMapping[i]) == set(ring_fragment))
    ][0]

    for a in range(frag_mol.GetNumAtoms()):
        N_rads = frag_mol.GetAtomWithIdx(a).GetNumRadicalElectrons()
        N_Hs = frag_mol.GetAtomWithIdx(a).GetTotalNumHs()
        if N_rads > 0:
            frag_mol.GetAtomWithIdx(a).SetNumExplicitHs(N_rads + N_Hs)
            frag_mol.GetAtomWithIdx(a).SetNumRadicalElectrons(0)

    smiles = rdkit.Chem.MolToSmiles(frag_mol, isomericSmiles=False)

    smiles_mol = rdkit.Chem.MolFromSmiles(smiles)
    if not smiles_mol:
        logger(f"failed to extract fragment smiles: {smiles}, {ring_fragment}")

        return None

    reduced_smiles = rdkit.Chem.MolToSmiles(smiles_mol, isomericSmiles=False)

    return reduced_smiles


def append_PEG(
    mol,
    attachment_site,
    AtomFragment_database,
    unique_atoms,
    bond_lookup,
    PEG_ID_sequence=[0, 0, 7],
):
    new_atom_idx = attachment_site
    mol_PEG = deepcopy(mol)
    for PEG_atom_ID in PEG_ID_sequence:
        mol_PEG, new_atom_idx, _ = make_3D_bond_attachments(
            mol_PEG,
            list(range(0, mol_PEG.GetNumAtoms())),
            getNodeFeatures(mol_PEG.GetAtoms()),
            PEG_atom_ID,
            new_atom_idx,
            0,
            rdkit.Chem.rdchem.BondType.SINGLE,
            AtomFragment_database,
            unique_atoms,
            bond_lookup,
            ignore_3D=False,
        )

    PEG_seed_idx = new_atom_idx

    return mol_PEG, PEG_seed_idx


def encode_molecule_with_generator(
    mol,
    model,
    AtomFragment_database,
    N_points=5,
    pointCloudVar=1.0 / (12.0 * 1.7),
    variational_factor_equi=0.0,
    variational_factor_inv=0.0,
    interpolate_to_prior_equi=0.0,
    interpolate_to_prior_inv=0.0,
    use_variational_GNN=True,
    variational_GNN_factor=0.0,
    interpolate_to_GNN_prior=0.0,
    h_interpolate=None,
):
    fragment_library_atom_features = np.concatenate(
        AtomFragment_database["atom_features"], axis=0
    ).reshape((len(AtomFragment_database), -1))

    node_features = getNodeFeatures(mol.GetAtoms())
    ring_fragments = get_ring_fragments(mol)

    data_mol = process_mol(
        mol, AtomFragment_database, N_points=N_points, pointCloudVar=pointCloudVar
    )

    library_dataset = AtomFragmentLibrary(AtomFragment_database)
    library_loader = torch_geometric.data.DataLoader(
        library_dataset, shuffle=False, batch_size=len(library_dataset), num_workers=0
    )
    fragment_batch = next(iter(library_loader))

    (
        fragment_library_features,
        fragment_library_node_features,
        fragment_library_batch,
    ) = model.Encoder.encode_fragment_library(fragment_batch)

    model_out = model.Encoder.encode(
        x=torch.cat(
            (
                data_mol.x,
                fragment_library_features[data_mol.atom_fragment_associations],
            ),
            dim=1,
        ),
        edge_index=data_mol.edge_index,
        edge_attr=data_mol.edge_attr,
        pos=data_mol.pos,
        points=data_mol.cloud,
        points_atom_index=data_mol.cloud_indices,
        batch_size=1,
        select_indices=None,
        select_indices_batch=None,
        shared_encoders=True,
        use_variational_GNN=use_variational_GNN,
        variational_GNN_factor=variational_GNN_factor,
        interpolate_to_GNN_prior=interpolate_to_GNN_prior,
        h_interpolate=h_interpolate,  # ((BxN) X F) = (N X F)
    )

    Z_equi, Z_inv = model_out[1], model_out[2]

    h_mean, h_std = model_out[5], model_out[6]

    h_reshaped = model_out[9]

    if model.variational:
        if (model.variational_mode == "both") | (model.variational_mode == "equi"):
            batch_size = 1
            Z_equi = model.Encoder.VariationalEncoder_equi(
                Z_equi
            )  # equivariant, shape [B,C*2,3]
            Z_equi_mean, Z_equi_logvar = Z_equi.chunk(
                2, dim=1
            )  # equivariant, shape [B,C,3]
            Z_equi_logvar, _ = model.Encoder.VariationalEncoder_equi_T(
                Z_equi_logvar
            )  # invariant, shape [B, C, 3]
            Z_equi_logvar = Z_equi_logvar.reshape(
                batch_size, -1
            )  # flattened to shape [B, C*3]
            Z_equi_logvar = (
                model.Encoder.VariationEncoder_equi_linear(Z_equi_logvar)
                .unsqueeze(2)
                .expand((-1, -1, 3))
            )  # invariant, shape [B, C, 3]
            Z_equi_std = torch.exp(0.5 * Z_equi_logvar)  # invariant, shape [B, C, 1]

            if interpolate_to_prior_equi > 0.0:
                Z_equi_mean = torch.lerp(
                    Z_equi_mean,
                    torch.zeros_like(Z_equi_mean),
                    interpolate_to_prior_equi,
                )
                Z_equi_std = torch.lerp(
                    Z_equi_std, torch.ones_like(Z_equi_std), interpolate_to_prior_equi
                )

            Z_equi_eps = (
                torch.randn_like(Z_equi_mean) * variational_factor_equi
            )  # normal noise with shape [B,C,3]
            Z_equi = Z_equi_mean + Z_equi_std * Z_equi_eps
        else:
            Z_equi_mean = None
            Z_equi_std = None

        if (model.variational_mode == "both") | (model.variational_mode == "inv"):
            Z_inv = model.Encoder.VariationalEncoder_inv(Z_inv)
            Z_inv_mean, Z_inv_logvar = Z_inv.chunk(2, dim=1)
            Z_inv_std = torch.exp(0.5 * Z_inv_logvar)

            if interpolate_to_prior_inv > 0.0:
                Z_inv_mean = torch.lerp(
                    Z_inv_mean, torch.zeros_like(Z_inv_mean), interpolate_to_prior_inv
                )
                Z_inv_std = torch.lerp(
                    Z_inv_std, torch.ones_like(Z_inv_std), interpolate_to_prior_inv
                )

            Z_inv_eps = torch.randn_like(Z_inv_mean) * variational_factor_inv
            Z_inv = Z_inv_mean + Z_inv_std * Z_inv_eps

        else:
            Z_inv_mean = None
            Z_inv_std = None
    else:
        Z_equi_mean = None
        Z_equi_std = None
        Z_inv_mean = None
        Z_inv_std = None

    return Z_equi, Z_inv, h_mean, h_std, h_reshaped


def get_starting_seeds(
    mol,
    AtomFragment_database,
    fragment_library_atom_features,
    unique_atoms,
    bond_lookup,
):
    ring_fragments = get_ring_fragments(mol)
    all_possible_seeds = get_all_possible_seeds(mol, ring_fragments)
    terminal_seeds = filter_terminal_seeds(all_possible_seeds, mol)

    select_seeds = []
    for seed in terminal_seeds:
        try:
            frame_generation, frame_rocs = get_frame_terminalSeeds(
                mol, seed, AtomFragment_database, include_rocs=True
            )
            positions = list(frame_rocs.iloc[0].positions_before)
            start = 0
            for i in range(len(frame_generation)):
                if (
                    set(frame_generation.iloc[i].partial_graph_indices)
                    == set(positions)
                ) & (frame_generation.iloc[i].next_atom_index == -1):
                    start = i + 1
                    break
            terminalSeed_frame = frame_generation.iloc[0:start].reset_index(drop=True)
            sequence = get_ground_truth_generation_sequence(
                terminalSeed_frame,
                AtomFragment_database,
                fragment_library_atom_features,
            )
            mol = deepcopy(terminalSeed_frame.iloc[0].rdkit_mol_cistrans_stereo)
            partial_indices = deepcopy(
                terminalSeed_frame.iloc[0].partial_graph_indices_sorted
            )
            queue_indices = deepcopy(terminalSeed_frame.iloc[0].focal_indices_sorted)
            _, seed_mol, _, _, _, _, _, _ = generate_seed_from_sequence(
                sequence,
                mol,
                partial_indices,
                queue_indices,
                AtomFragment_database,
                unique_atoms,
                bond_lookup,
                stop_after_sequence=True,
            )

            if seed_mol.GetNumAtoms() <= 6:
                select_seeds.append(seed)
                continue
        except Exception as e:
            # logger(e)
            continue

    if len(select_seeds) == 0:
        # logger('No suitable seed structure.')
        return []

    return select_seeds


def decode(
    mol,
    select_seeds,
    model_3D,
    rocs_model_3D,
    AtomFragment_database,
    fragment_library_atom_features,
    unique_atoms,
    bond_lookup,
    N_points_3D,
    N_points_rocs,
    stop_threshold,
    variational_GNN_factor,
    interpolate_to_GNN_prior,
    h_interpolate,
    rocs_use_variational_GNN,
    rocs_variational_GNN_factor,
    rocs_interpolate_to_GNN_prior,
    pointCloudVar,
    rocs_pointCloudVar,
):
    generated_mols = []
    failed_to_start_rocs = 0
    failed_to_get_sequence = 0
    for seed in select_seeds:
        try:
            frame_generation, frame_rocs = get_frame_terminalSeeds(
                mol, seed, AtomFragment_database, include_rocs=True
            )
        except Exception as e:
            logger(f"failed to get frame -- {e}")
            continue

        if frame_rocs is not None:
            if len(frame_rocs) > 0:
                positions = list(frame_rocs.iloc[0].positions_before)
                start = 0
                for i in range(len(frame_generation)):
                    if (
                        set(frame_generation.iloc[i].partial_graph_indices)
                        == set(positions)
                    ) & (frame_generation.iloc[i].next_atom_index == -1):
                        start = i + 1
                        break
            else:
                failed_to_start_rocs += 1
                continue
        else:
            logger("failed to get rocs dataframe")
            failed_to_start_rocs += 1
            continue

        if (
            len(frame_generation.iloc[0].partial_graph_indices) == 1
        ):  # seed is a terminal ATOM
            terminalSeed_frame = frame_generation.iloc[0:start].reset_index(drop=True)

            try:
                sequence = get_ground_truth_generation_sequence(
                    terminalSeed_frame,
                    AtomFragment_database,
                    fragment_library_atom_features,
                )
            except Exception as e:
                logger(f"failed to get seeding sequence -- {e}")
                failed_to_get_sequence += 1
                continue

            mol = deepcopy(terminalSeed_frame.iloc[0].rdkit_mol_cistrans_stereo)
            partial_indices = deepcopy(
                terminalSeed_frame.iloc[0].partial_graph_indices_sorted
            )

            final_partial_indices = deepcopy(
                terminalSeed_frame.iloc[-1].partial_graph_indices_sorted
            )
            ring_fragments = get_ring_fragments(mol)
            add_to_partial = [
                list(f) for p in final_partial_indices for f in ring_fragments if p in f
            ]
            add_to_partial = [item for sublist in add_to_partial for item in sublist]
            final_partial_indices = list(
                set(final_partial_indices).union(add_to_partial)
            )

            queue_indices = deepcopy(terminalSeed_frame.iloc[0].focal_indices_sorted)

            (
                _,
                seed_mol,
                queue,
                positioned_atoms_indices,
                atom_to_library_ID_map,
                _,
                _,
                _,
            ) = generate_seed_from_sequence(
                sequence,
                mol,
                partial_indices,
                queue_indices,
                AtomFragment_database,
                unique_atoms,
                bond_lookup,
                stop_after_sequence=True,
            )

            seed_node_features = getNodeFeatures(seed_mol.GetAtoms())

            for k in atom_to_library_ID_map:
                seed_node_features[k] = AtomFragment_database.iloc[
                    atom_to_library_ID_map[k]
                ].atom_features

            G = get_substructure_graph(mol, final_partial_indices)
            G_seed = get_substructure_graph(
                seed_mol,
                list(range(0, seed_mol.GetNumAtoms())),
                node_features=seed_node_features,
            )
            nm = nx.algorithms.isomorphism.generic_node_match(
                ["atom_features"], [None], [np.allclose]
            )
            em = nx.algorithms.isomorphism.numerical_edge_match("bond_type", 1.0)
            GM = nx.algorithms.isomorphism.GraphMatcher(
                G, G_seed, node_match=nm, edge_match=em
            )
            assert GM.is_isomorphic()
            idx_map = GM.mapping

        else:  # seed is a terminal FRAGMENT
            partial_indices = deepcopy(
                frame_generation.iloc[0].partial_graph_indices_sorted
            )
            final_partial_indices = partial_indices
            seed_mol = generate_conformer(get_fragment_smiles(mol, partial_indices))
            idx_map = get_reindexing_map(mol, partial_indices, seed_mol)
            positioned_atoms_indices = sorted(
                [idx_map[f] for f in final_partial_indices]
            )

            atom_to_library_ID_map = {}  # no individual atoms generated
            queue = [0]  # 0 can be considered the focal root node

        assert len(final_partial_indices) == len(seed_mol.GetAtoms())
        for i in final_partial_indices:
            x, y, z = mol.GetConformer().GetPositions()[i]
            seed_mol.GetConformer().SetAtomPosition(idx_map[i], Point3D(x, y, z))

        starting_queue = deepcopy(queue)
        try:
            _, updated_mol, _, _, _, _, _, _, _, _ = generate_3D_mol_from_sequence(
                sequence=[],
                partial_mol=deepcopy(seed_mol),
                mol=deepcopy(mol),
                positioned_atoms_indices=deepcopy(positioned_atoms_indices),
                queue=starting_queue,
                atom_to_library_ID_map=deepcopy(atom_to_library_ID_map),
                model=model_3D,
                rocs_model=rocs_model_3D,
                AtomFragment_database=AtomFragment_database,
                unique_atoms=unique_atoms,
                bond_lookup=bond_lookup,
                N_points=N_points_3D,
                N_points_rocs=N_points_rocs,
                stop_after_sequence=False,
                mask_first_stop=False,
                stochastic=False,
                chirality_scoring=True,
                stop_threshold=stop_threshold,
                steric_mask=True,
                variational_factor_equi=0.0,
                variational_factor_inv=0.0,
                interpolate_to_prior_equi=0.0,
                interpolate_to_prior_inv=0.0,
                use_variational_GNN=True,
                variational_GNN_factor=variational_GNN_factor,
                interpolate_to_GNN_prior=interpolate_to_GNN_prior,
                rocs_use_variational_GNN=rocs_use_variational_GNN,
                rocs_variational_GNN_factor=rocs_variational_GNN_factor,
                rocs_interpolate_to_GNN_prior=rocs_interpolate_to_GNN_prior,
                pointCloudVar=pointCloudVar,
                rocs_pointCloudVar=rocs_pointCloudVar,
                h_interpolate=h_interpolate,
            )

            generated_mols.append(updated_mol)

        except Exception as e:
            logger(f"failed to generate: {e}")

    return generated_mols

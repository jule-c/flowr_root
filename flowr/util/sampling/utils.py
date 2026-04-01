import logging
import math
import os
import sys
from collections import Counter
from typing import Any, Collection, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import (
    QED,
    AllChem,
    Crippen,
    Descriptors,
    Lipinski,
    Mol,
    RDConfig,
    rdMolDescriptors,
)

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.ML.Descriptors import MoleculeDescriptors

try:
    from scipy import histogram
except:
    pass
from scipy.stats import entropy, gaussian_kde

# Mute RDKit logger
RDLogger.logger().setLevel(RDLogger.CRITICAL)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

margin1, margin2, margin3 = 10, 5, 3

bonds1 = {
    "H": {
        "H": 74,
        "C": 109,
        "N": 101,
        "O": 96,
        "F": 92,
        "B": 119,
        "Si": 148,
        "P": 144,
        "As": 152,
        "S": 134,
        "Cl": 127,
        "Br": 141,
        "I": 161,
    },
    "C": {
        "H": 109,
        "C": 154,
        "N": 147,
        "O": 143,
        "F": 135,
        "Si": 185,
        "P": 184,
        "S": 182,
        "Cl": 177,
        "Br": 194,
        "I": 214,
    },
    "N": {
        "H": 101,
        "C": 147,
        "N": 145,
        "O": 140,
        "F": 136,
        "Cl": 175,
        "Br": 214,
        "S": 168,
        "I": 222,
        "P": 177,
    },
    "O": {
        "H": 96,
        "C": 143,
        "N": 140,
        "O": 148,
        "F": 142,
        "Br": 172,
        "S": 151,
        "P": 163,
        "Si": 163,
        "Cl": 164,
        "I": 194,
    },
    "F": {
        "H": 92,
        "C": 135,
        "N": 136,
        "O": 142,
        "F": 142,
        "S": 158,
        "Si": 160,
        "Cl": 166,
        "Br": 178,
        "P": 156,
        "I": 187,
    },
    "B": {"H": 119, "Cl": 175},
    "Si": {
        "Si": 233,
        "H": 148,
        "C": 185,
        "O": 163,
        "S": 200,
        "F": 160,
        "Cl": 202,
        "Br": 215,
        "I": 243,
    },
    "Cl": {
        "Cl": 199,
        "H": 127,
        "C": 177,
        "N": 175,
        "O": 164,
        "P": 203,
        "S": 207,
        "B": 175,
        "Si": 202,
        "F": 166,
        "Br": 214,
    },
    "S": {
        "H": 134,
        "C": 182,
        "N": 168,
        "O": 151,
        "S": 204,
        "F": 158,
        "Cl": 207,
        "Br": 225,
        "Si": 200,
        "P": 210,
        "I": 234,
    },
    "Br": {
        "Br": 228,
        "H": 141,
        "C": 194,
        "O": 172,
        "N": 214,
        "Si": 215,
        "S": 225,
        "F": 178,
        "Cl": 214,
        "P": 222,
    },
    "P": {
        "P": 221,
        "H": 144,
        "C": 184,
        "O": 163,
        "Cl": 203,
        "S": 210,
        "F": 156,
        "N": 177,
        "Br": 222,
    },
    "I": {
        "H": 161,
        "C": 214,
        "Si": 243,
        "N": 222,
        "O": 194,
        "S": 234,
        "F": 187,
        "I": 266,
    },
    "As": {"H": 152},
}

bonds2 = {
    "C": {"C": 134, "N": 129, "O": 120, "S": 160},
    "N": {"C": 129, "N": 125, "O": 121},
    "O": {"C": 120, "N": 121, "O": 121, "P": 150},
    "P": {"O": 150, "S": 186},
    "S": {"P": 186},
}


bonds3 = {
    "C": {"C": 120, "N": 116, "O": 113},
    "N": {"C": 116, "N": 110},
    "O": {"C": 113},
}

allowed_bonds = {
    "H": {0: 1, 1: 0, -1: 0},
    "C": {0: [3, 4], 1: 3, -1: 3},
    "N": {
        0: [2, 3],
        1: [2, 3, 4],
        -1: 2,
    },  # In QM9, N+ seems to be present in the form NH+ and NH2+
    "O": {0: 2, 1: 3, -1: 1},
    "F": {0: 1, -1: 0},
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": {0: [3, 5], 1: 4},
    "S": {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
    "Cl": 1,
    "As": 3,
    "Br": {0: 1, 1: 2},
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
    "Se": [2, 4, 6],
}

allowed_bonds_nocharges = {
    "H": 1,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": [3, 5],
    "S": 4,
    "Cl": 1,
    "As": 3,
    "Br": 1,
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
}

bond_dict = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}


def check_stability(
    molecule,
    atom_decoder,
    debug=False,
    smiles=None,
):
    atom_types = molecule.atom_types
    edge_types = molecule.bond_types

    edge_types[edge_types == 4] = 1.5
    edge_types[edge_types < 0] = 0

    valencies = torch.sum(edge_types, dim=-1).long()

    n_stable_bonds = 0
    mol_stable = True
    for i, (atom_type, valency, charge) in enumerate(
        zip(atom_types, valencies, molecule.charges)
    ):
        atom_type = atom_type.item()
        valency = valency.item()
        charge = charge.item()
        possible_bonds = allowed_bonds[atom_decoder[atom_type]]
        if isinstance(possible_bonds, int):
            is_stable = possible_bonds == valency
        elif isinstance(possible_bonds, dict):
            expected_bonds = (
                possible_bonds[charge]
                if charge in possible_bonds.keys()
                else possible_bonds[0]
            )
            is_stable = (
                expected_bonds == valency
                if isinstance(expected_bonds, int)
                else valency in expected_bonds
            )
        else:
            is_stable = valency in possible_bonds
        if not is_stable:
            mol_stable = False
        if not is_stable and debug:
            if smiles is not None:
                print(smiles)
            print(
                f"Invalid atom {atom_decoder[atom_type]}: valency={valency}, charge={charge}"
            )
            print()
        n_stable_bonds += int(is_stable)

    return float(mol_stable), n_stable_bonds, len(atom_types)


def get_bond_order(atom1, atom2, distance, check_exists=False):
    distance = 100 * distance  # We change the metric

    # Check exists for large molecules where some atom pairs do not have a
    # typical bond length.
    if check_exists:
        if atom1 not in bonds1:
            return 0
        if atom2 not in bonds1[atom1]:
            return 0

    # margin1, margin2 and margin3 have been tuned to maximize the stability of
    # the QM9 true samples.
    if distance < bonds1[atom1][atom2] + margin1:
        # Check if atoms in bonds2 dictionary.
        if atom1 in bonds2 and atom2 in bonds2[atom1]:
            thr_bond2 = bonds2[atom1][atom2] + margin2
            if distance < thr_bond2:
                if atom1 in bonds3 and atom2 in bonds3[atom1]:
                    thr_bond3 = bonds3[atom1][atom2] + margin3
                    if distance < thr_bond3:
                        return 3  # Triple
                return 2  # Double
        return 1  # Single
    return 0  # No bond


def check_stability_without_bonds(molecule, atom_decoder, debug=False):
    positions = molecule.positions
    atom_type = molecule.atom_types
    charges = molecule.charges
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    nr_bonds = np.zeros(len(x), dtype="int")

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            pair = sorted([atom_type[i], atom_type[j]])
            order = get_bond_order(
                atom_decoder[pair[0]], atom_decoder[pair[1]], dist, check_exists=True
            )
            nr_bonds[i] += order
            nr_bonds[j] += order
    nr_stable_bonds = 0
    for i, (atom_type_i, nr_bonds_i) in enumerate(zip(atom_type, nr_bonds)):
        if charges is not None:
            possible_bonds = allowed_bonds[atom_decoder[atom_type_i]]
        else:
            possible_bonds = allowed_bonds_nocharges[atom_decoder[atom_type_i]]
        if isinstance(possible_bonds, int):
            is_stable = possible_bonds == nr_bonds_i
        elif isinstance(possible_bonds, dict) and charges is not None:
            expected_bonds = (
                possible_bonds[charges[i]]
                if charges[i] in possible_bonds.keys()
                else possible_bonds[0]
            )
            is_stable = (
                expected_bonds == nr_bonds_i
                if isinstance(expected_bonds, int)
                else nr_bonds_i in expected_bonds
            )
        else:
            is_stable = nr_bonds_i in possible_bonds
        if not is_stable and debug:
            print(
                "Invalid bonds for molecule %s with %d bonds"
                % (atom_decoder[atom_type_i], nr_bonds_i)
            )
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == len(x)

    return float(molecule_stable), nr_stable_bonds, len(x)


def number_nodes_distance(molecules, dataset_counts):
    max_number_nodes = max(dataset_counts.keys())
    reference_n = torch.zeros(max_number_nodes + 1)
    for n, count in dataset_counts.items():
        reference_n[n] = count

    c = Counter()
    for molecule in molecules:
        c[molecule.num_nodes] += 1

    generated_n = counter_to_tensor(c)
    return wasserstein1d(generated_n, reference_n)


def atom_types_distance(molecules, target, save_histogram=False):
    generated_distribution = torch.zeros_like(target)
    for molecule in molecules:
        for atom_type in molecule.atom_types:
            generated_distribution[atom_type] += 1
    if save_histogram:
        np.save("generated_atom_types.npy", generated_distribution.cpu().numpy())
    return total_variation1d(generated_distribution, target)


def bond_types_distance(molecules, target, save_histogram=False):
    device = molecules[0].bond_types.device
    generated_distribution = torch.zeros_like(target).to(device)
    for molecule in molecules:
        bond_types = molecule.bond_types
        mask = torch.ones_like(bond_types)
        mask = torch.triu(mask, diagonal=1).bool()
        bond_types = bond_types[mask]
        unique_edge_types, counts = torch.unique(bond_types, return_counts=True)
        for type, count in zip(unique_edge_types, counts):
            generated_distribution[type] += count
    if save_histogram:
        np.save("generated_bond_types.npy", generated_distribution.cpu().numpy())
    sparsity_level = generated_distribution[0] / torch.sum(generated_distribution)
    tv, tv_per_class = total_variation1d(generated_distribution, target.to(device))
    return tv, tv_per_class, sparsity_level


def charge_distance(molecules, target, atom_types_probabilities, dataset_infos):
    device = molecules[0].charges.device
    generated_distribution = torch.zeros_like(target).to(device)
    for molecule in molecules:
        for atom_type in range(target.shape[0]):
            mask = molecule.atom_types == atom_type
            if mask.sum() > 0:
                at_charges = dataset_infos.one_hot_charges(molecule.charges[mask])
                generated_distribution[atom_type] += at_charges.sum(dim=0)

    s = generated_distribution.sum(dim=1, keepdim=True)
    s[s == 0] = 1
    generated_distribution = generated_distribution / s

    cs_generated = torch.cumsum(generated_distribution, dim=1)
    cs_target = torch.cumsum(target, dim=1).to(device)

    w1_per_class = torch.sum(torch.abs(cs_generated - cs_target), dim=1)

    w1 = torch.sum(w1_per_class * atom_types_probabilities.to(device)).item()

    return w1, w1_per_class


def valency_distance(
    molecules, target_valencies, atom_types_probabilities, atom_encoder
):
    # Build a dict for the generated molecules that is similar to the target one
    num_atom_types = len(atom_types_probabilities)
    generated_valencies = {i: Counter() for i in range(num_atom_types)}
    for molecule in molecules:
        edge_types = molecule.bond_types
        edge_types[edge_types == 4] = 1.5
        valencies = torch.sum(edge_types, dim=0)
        for atom, val in zip(molecule.atom_types, valencies):
            generated_valencies[atom.item()][val.item()] += 1

    # Convert the valencies to a tensor of shape (num_atom_types, max_valency)
    max_valency_target = max(
        max(vals.keys()) if len(vals) > 0 else -1 for vals in target_valencies.values()
    )
    max_valency_generated = max(
        max(vals.keys()) if len(vals) > 0 else -1
        for vals in generated_valencies.values()
    )
    max_valency = max(max_valency_target, max_valency_generated)

    valencies_target_tensor = torch.zeros(num_atom_types, max_valency + 1)
    for atom_type, valencies in target_valencies.items():
        for valency, count in valencies.items():
            valencies_target_tensor[atom_encoder[atom_type], valency] = count

    valencies_generated_tensor = torch.zeros(num_atom_types, max_valency + 1)
    for atom_type, valencies in generated_valencies.items():
        for valency, count in valencies.items():
            valencies_generated_tensor[atom_type, valency] = count

    # Normalize the distributions
    s1 = torch.sum(valencies_target_tensor, dim=1, keepdim=True)
    s1[s1 == 0] = 1
    valencies_target_tensor = valencies_target_tensor / s1

    s2 = torch.sum(valencies_generated_tensor, dim=1, keepdim=True)
    s2[s2 == 0] = 1
    valencies_generated_tensor = valencies_generated_tensor / s2

    cs_target = torch.cumsum(valencies_target_tensor, dim=1)
    cs_generated = torch.cumsum(valencies_generated_tensor, dim=1)

    w1_per_class = torch.sum(torch.abs(cs_target - cs_generated), dim=1)

    total_w1 = torch.sum(w1_per_class * atom_types_probabilities).item()
    return total_w1, w1_per_class


def get_similarity(fg_pair):
    return TanimotoSimilarity(fg_pair[0], fg_pair[1])


def bond_length_distance(molecules, target, bond_types_probabilities):
    generated_bond_lenghts = {1: Counter(), 2: Counter(), 3: Counter(), 4: Counter()}
    for molecule in molecules:
        cdists = torch.cdist(
            molecule.positions.unsqueeze(0), molecule.positions.unsqueeze(0)
        ).squeeze(0)
        for bond_type in range(1, 5):
            edges = torch.nonzero(molecule.bond_types == bond_type)
            bond_distances = cdists[edges[:, 0], edges[:, 1]]
            distances_to_consider = torch.round(bond_distances, decimals=2)
            for d in distances_to_consider:
                generated_bond_lenghts[bond_type][d.item()] += 1

    # Normalizing the bond lenghts
    for bond_type in range(1, 5):
        s = sum(generated_bond_lenghts[bond_type].values())
        if s == 0:
            s = 1
        for d, count in generated_bond_lenghts[bond_type].items():
            generated_bond_lenghts[bond_type][d] = count / s

    # Convert both dictionaries to tensors
    min_generated_length = min(
        min(d.keys()) if len(d) > 0 else 1e4 for d in generated_bond_lenghts.values()
    )
    min_target_length = min(
        min(d.keys()) if len(d) > 0 else 1e4 for d in target.values()
    )
    min_length = min(min_generated_length, min_target_length)

    max_generated_length = max(
        max(bl.keys()) if len(bl) > 0 else -1 for bl in generated_bond_lenghts.values()
    )
    max_target_length = max(
        max(bl.keys()) if len(bl) > 0 else -1 for bl in target.values()
    )
    max_length = max(max_generated_length, max_target_length)

    num_bins = int((max_length - min_length) * 100) + 1
    generated_bond_lengths = torch.zeros(4, num_bins)
    target_bond_lengths = torch.zeros(4, num_bins)

    for bond_type in range(1, 5):
        for d, count in generated_bond_lenghts[bond_type].items():
            bin = int((d - min_length) * 100)
            generated_bond_lengths[bond_type - 1, bin] = count
        for d, count in target[bond_type].items():
            bin = int((d - min_length) * 100)
            target_bond_lengths[bond_type - 1, bin] = count

    cs_generated = torch.cumsum(generated_bond_lengths, dim=1)
    cs_target = torch.cumsum(target_bond_lengths, dim=1)

    w1_per_class = (
        torch.sum(torch.abs(cs_generated - cs_target), dim=1) / 100
    )  # 100 because of bin size
    weighted = w1_per_class * bond_types_probabilities[1:]
    return torch.sum(weighted).item(), w1_per_class


def angle_distance(
    molecules,
    target_angles,
    atom_types_probabilities,
    valencies,
    atom_decoder,
    save_histogram: bool,
):
    num_atom_types = len(atom_types_probabilities)
    generated_angles = torch.zeros(num_atom_types, 180 * 10 + 1)
    for molecule in molecules:
        adj = molecule.bond_types
        pos = molecule.positions
        for atom in range(adj.shape[0]):
            p_a = pos[atom]
            neighbors = torch.nonzero(adj[atom]).squeeze(1)
            for i in range(len(neighbors)):
                p_i = pos[neighbors[i]]
                for j in range(i + 1, len(neighbors)):
                    p_j = pos[neighbors[j]]
                    v1 = p_i - p_a
                    v2 = p_j - p_a
                    assert not torch.isnan(v1).any()
                    assert not torch.isnan(v2).any()
                    prod = torch.dot(
                        v1 / (torch.norm(v1) + 1e-6), v2 / (torch.norm(v2) + 1e-6)
                    )
                    if prod > 1:
                        print(
                            f"Invalid angle {i} {j} -- {prod} -- {v1 / (torch.norm(v1) + 1e-6)} --"
                            f" {v2 / (torch.norm(v2) + 1e-6)}"
                        )
                    prod.clamp(min=0, max=1)
                    angle = torch.acos(prod)
                    if torch.isnan(angle).any():
                        print(
                            f"Nan obtained in angle {i} {j} -- {prod} -- {v1 / (torch.norm(v1) + 1e-6)} --"
                            f" {v2 / (torch.norm(v2) + 1e-6)}"
                        )
                    else:
                        bin = int(
                            torch.round(angle * 180 / math.pi, decimals=1).item() * 10
                        )
                        generated_angles[molecule.atom_types[atom], bin] += 1

    s = torch.sum(generated_angles, dim=1, keepdim=True)
    s[s == 0] = 1
    generated_angles = generated_angles / s
    if save_histogram:
        np.save("generated_angles_histogram.npy", generated_angles.numpy())

    if type(target_angles) in [np.array, np.ndarray]:
        target_angles = torch.from_numpy(target_angles).float()

    cs_generated = torch.cumsum(generated_angles, dim=1)
    cs_target = torch.cumsum(target_angles, dim=1)

    w1_per_type = torch.sum(torch.abs(cs_generated - cs_target), dim=1) / 10

    # The atoms that have a valency less than 2 should not matter
    valency_weight = torch.zeros(len(w1_per_type), device=w1_per_type.device)
    for i in range(len(w1_per_type)):
        valency_weight[i] = (
            1 - valencies[atom_decoder[i]][0] - valencies[atom_decoder[i]][1]
        )

    weighted = w1_per_type * atom_types_probabilities * valency_weight
    return (
        torch.sum(weighted)
        / (torch.sum(atom_types_probabilities * valency_weight) + 1e-5)
    ).item(), w1_per_type


def dihedral_distance(
    molecules,
    target_dihedrals,
    bond_types_probabilities,
    save_histogram,
):
    def calculate_dihedral_angles(mol):
        def find_dihedrals(mol):
            torsionSmarts = "[!$(*#*)&!D1]~[!$(*#*)&!D1]"
            torsionQuery = Chem.MolFromSmarts(torsionSmarts)
            matches = mol.GetSubstructMatches(torsionQuery)
            torsionList = []
            btype = []
            for match in matches:
                idx2 = match[0]
                idx3 = match[1]
                bond = mol.GetBondBetweenAtoms(idx2, idx3)
                jAtom = mol.GetAtomWithIdx(idx2)
                kAtom = mol.GetAtomWithIdx(idx3)
                if (
                    (jAtom.GetHybridization() != Chem.HybridizationType.SP2)
                    and (jAtom.GetHybridization() != Chem.HybridizationType.SP3)
                ) or (
                    (kAtom.GetHybridization() != Chem.HybridizationType.SP2)
                    and (kAtom.GetHybridization() != Chem.HybridizationType.SP3)
                ):
                    continue
                for b1 in jAtom.GetBonds():
                    if b1.GetIdx() == bond.GetIdx():
                        continue
                    idx1 = b1.GetOtherAtomIdx(idx2)
                    for b2 in kAtom.GetBonds():
                        if (b2.GetIdx() == bond.GetIdx()) or (
                            b2.GetIdx() == b1.GetIdx()
                        ):
                            continue
                        idx4 = b2.GetOtherAtomIdx(idx3)
                        # skip 3-membered rings
                        if idx4 == idx1:
                            continue
                        bt = bond.GetBondTypeAsDouble()
                        # bt = str(bond.GetBondType())
                        # if bond.IsInRing():
                        #     bt += '_R'
                        btype.append(bt)
                        torsionList.append((idx1, idx2, idx3, idx4))
            return np.asarray(torsionList), np.asarray(btype)

        dihedral_idx, dihedral_types = find_dihedrals(mol)

        coords = mol.GetConformer().GetPositions()
        t_angles = []
        for t in dihedral_idx:
            u1, u2, u3, u4 = coords[torch.tensor(t)]

            a1 = u2 - u1
            a2 = u3 - u2
            a3 = u4 - u3

            v1 = np.cross(a1, a2)
            v1 = v1 / (v1 * v1).sum(-1) ** 0.5
            v2 = np.cross(a2, a3)
            v2 = v2 / (v2 * v2).sum(-1) ** 0.5
            porm = np.sign((v1 * a3).sum(-1))
            rad = np.arccos(
                (v1 * v2).sum(-1) / ((v1**2).sum(-1) * (v2**2).sum(-1) + 1e-9) ** 0.5
            )
            if not porm == 0:
                rad = rad * porm
            t_angles.append(rad * 180 / torch.pi)

        return np.asarray(t_angles), dihedral_types

    # forget about none and tripple bonds
    bond_types_probabilities[torch.tensor([0, 3])] = 0
    bond_types_probabilities /= bond_types_probabilities.sum()

    num_bond_types = len(bond_types_probabilities)
    generated_dihedrals = torch.zeros(num_bond_types, 180 * 10 + 1)
    for mol in molecules:
        mol = mol.rdkit_mol
        try:
            Chem.SanitizeMol(mol)
        except:
            continue
        angles, types = calculate_dihedral_angles(mol)
        # transform types to idx
        types[types == 1.5] = 4
        types = types.astype(int)
        for a, t in zip(np.abs(angles), types):
            if np.isnan(a):
                continue
            generated_dihedrals[t, int(np.round(a, decimals=1) * 10)] += 1

    # normalize
    s = generated_dihedrals.sum(axis=1, keepdims=True)
    s[s == 0] = 1
    generated_dihedrals = generated_dihedrals.float() / s

    if save_histogram:
        np.save("generated_dihedrals_historgram.npy", generated_dihedrals.numpy())

    if type(target_dihedrals) in [np.array, np.ndarray]:
        target_dihedrals = torch.from_numpy(target_dihedrals).float()

    cs_generated = torch.cumsum(generated_dihedrals, dim=1)
    cs_target = torch.cumsum(target_dihedrals, dim=1)

    w1_per_type = torch.sum(torch.abs(cs_generated - cs_target), dim=1) / 10

    weighted = w1_per_type * bond_types_probabilities

    return torch.sum(weighted).item(), w1_per_type


def node_counts(data_list):
    print("Computing node counts...")
    all_node_counts = Counter()
    for i, data in enumerate(data_list):
        num_nodes = data.num_nodes
        all_node_counts[num_nodes] += 1
    print("Done.")
    return all_node_counts


def atom_type_counts(data_list, num_classes):
    print("Computing node types distribution...")
    counts = np.zeros(num_classes)
    for data in data_list:
        x = torch.nn.functional.one_hot(data.x, num_classes=num_classes)
        counts += x.sum(dim=0).numpy()

    counts = counts / counts.sum()
    print("Done.")
    return counts


def edge_counts(data_list, num_bond_types=5):
    print("Computing edge counts...")
    d = np.zeros(num_bond_types)

    for data in data_list:
        total_pairs = data.num_nodes * (data.num_nodes - 1)

        num_edges = data.edge_attr.shape[0]
        num_non_edges = total_pairs - num_edges
        assert num_non_edges >= 0

        edge_types = (
            torch.nn.functional.one_hot(
                data.edge_attr - 1, num_classes=num_bond_types - 1
            )
            .sum(dim=0)
            .numpy()
        )
        d[0] += num_non_edges
        d[1:] += edge_types

    d = d / d.sum()
    return d


def charge_counts(data_list, num_classes, charges_dic):
    print("Computing charge counts...")
    d = np.zeros((num_classes, len(charges_dic)))

    for data in data_list:
        for atom, charge in zip(data.x, data.charges):
            assert charge in [-2, -1, 0, 1, 2, 3]
            d[atom.item(), charges_dic[charge.item()]] += 1

    s = np.sum(d, axis=1, keepdims=True)
    s[s == 0] = 1
    d = d / s
    print("Done.")
    return d


def valency_count(data_list, atom_encoder):
    atom_decoder = {v: k for k, v in atom_encoder.items()}
    print("Computing valency counts...")
    valencies = {atom_type: Counter() for atom_type in atom_encoder.keys()}

    for data in data_list:
        edge_attr = data.edge_attr
        edge_attr[edge_attr == 4] = 1.5
        bond_orders = edge_attr

        for atom in range(data.num_nodes):
            edges = bond_orders[data.edge_index[0] == atom]
            valency = edges.sum(dim=0)
            valencies[atom_decoder[data.x[atom].item()]][valency.item()] += 1

    # Normalizing the valency counts
    for atom_type in valencies.keys():
        s = sum(valencies[atom_type].values())
        for valency, count in valencies[atom_type].items():
            valencies[atom_type][valency] = count / s
    print("Done.")
    return valencies


def bond_lengths_counts(data_list, num_bond_types=5):
    """Compute the bond lenghts separetely for each bond type."""
    print("Computing bond lengths...")
    all_bond_lenghts = {1: Counter(), 2: Counter(), 3: Counter(), 4: Counter()}
    for data in data_list:
        cdists = torch.cdist(data.pos.unsqueeze(0), data.pos.unsqueeze(0)).squeeze(0)
        bond_distances = cdists[data.edge_index[0], data.edge_index[1]]
        for bond_type in range(1, num_bond_types):
            bond_type_mask = data.edge_attr == bond_type
            distances_to_consider = bond_distances[bond_type_mask]
            distances_to_consider = torch.round(distances_to_consider, decimals=2)
            for d in distances_to_consider:
                all_bond_lenghts[bond_type][d.item()] += 1

    # Normalizing the bond lenghts
    for bond_type in range(1, num_bond_types):
        s = sum(all_bond_lenghts[bond_type].values())
        for d, count in all_bond_lenghts[bond_type].items():
            all_bond_lenghts[bond_type][d] = count / s
    print("Done.")
    return all_bond_lenghts


def bond_angles(data_list, atom_encoder):
    atom_decoder = {v: k for k, v in atom_encoder.items()}
    print("Computing bond angles...")
    all_bond_angles = np.zeros((len(atom_encoder.keys()), 180 * 10 + 1))
    for data in data_list:
        assert not torch.isnan(data.pos).any()
        for i in range(data.num_nodes):
            neighbors = data.edge_index[1][data.edge_index[0] == i]
            for j in neighbors:
                for k in neighbors:
                    if j == k:
                        continue
                    assert i != j and i != k and j != k, "i, j, k: {}, {}, {}".format(
                        i, j, k
                    )
                    a = data.pos[j] - data.pos[i]
                    b = data.pos[k] - data.pos[i]

                    # print(a, b, torch.norm(a) * torch.norm(b))
                    angle = torch.acos(
                        torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-6)
                    )
                    angle = angle * 180 / math.pi

                    bin = int(torch.round(angle, decimals=1) * 10)
                    all_bond_angles[data.x[i].item(), bin] += 1

    # Normalizing the angles
    s = all_bond_angles.sum(axis=1, keepdims=True)
    s[s == 0] = 1
    all_bond_angles = all_bond_angles / s
    print("Done.")
    return all_bond_angles


def counter_to_tensor(c: Counter):
    max_key = max(c.keys())
    assert type(max_key) == int
    arr = torch.zeros(max_key + 1, dtype=torch.float)
    for k, v in c.items():
        arr[k] = v
    arr / torch.sum(arr)
    return arr


def wasserstein1d(preds, target, step_size=1):
    """preds and target are 1d tensors. They contain histograms for bins that are regularly spaced"""
    target = normalize(target) / step_size
    preds = normalize(preds) / step_size
    max_len = max(len(preds), len(target))
    preds = F.pad(preds, (0, max_len - len(preds)))
    target = F.pad(target, (0, max_len - len(target)))

    cs_target = torch.cumsum(target, dim=0)
    cs_preds = torch.cumsum(preds, dim=0)
    return torch.sum(torch.abs(cs_preds - cs_target)).item()


def total_variation1d(preds, target):
    assert (
        target.dim() == 1 and preds.shape == target.shape
    ), f"preds: {preds.shape}, target: {target.shape}"
    target = normalize(target)
    preds = normalize(preds)
    return torch.sum(torch.abs(preds - target)).item(), torch.abs(preds - target)


def normalize(tensor):
    s = tensor.sum()
    assert s > 0
    return tensor / s


def canonicalize(smiles: str, include_stereocenters=True, remove_hs=False):
    mol = Chem.MolFromSmiles(smiles)
    if remove_hs:
        mol = Chem.RemoveHs(mol)

    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)
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
    canonicalized_smiles = [s for s in canonicalized_smiles if s is not None]

    return remove_duplicates(canonicalized_smiles)


def remove_duplicates(list_with_duplicates):
    unique_set = set()
    unique_list = []
    ids = []
    for i, element in enumerate(list_with_duplicates):
        if element not in unique_set:
            unique_set.add(element)
            unique_list.append(element)
        else:
            ids.append(i)

    return unique_list, ids


def continuous_kldiv(X_baseline: np.ndarray, X_sampled: np.ndarray) -> float:
    kde_P = gaussian_kde(X_baseline)
    kde_Q = gaussian_kde(X_sampled)
    x_eval = np.linspace(
        np.hstack([X_baseline, X_sampled]).min(),
        np.hstack([X_baseline, X_sampled]).max(),
        num=1000,
    )
    P = kde_P(x_eval) + 1e-10
    Q = kde_Q(x_eval) + 1e-10

    return entropy(P, Q)


def discrete_kldiv(X_baseline: np.ndarray, X_sampled: np.ndarray) -> float:
    P, bins = histogram(X_baseline, bins=10, density=True)
    P += 1e-10
    Q, _ = histogram(X_sampled, bins=bins, density=True)
    Q += 1e-10

    return entropy(P, Q)


def calculate_pc_descriptors(
    smiles: Iterable[str], pc_descriptors: List[str]
) -> np.ndarray:
    output = []

    for i in smiles:
        d = _calculate_pc_descriptors(i, pc_descriptors)
        if d is not None:
            output.append(d)

    return np.array(output)


def _calculate_pc_descriptors(smiles: str, pc_descriptors: List[str]):
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(pc_descriptors)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    _fp = calc.CalcDescriptors(mol)
    _fp = np.array(_fp)
    mask = np.isfinite(_fp)
    if (mask == 0).sum() > 0:
        logger.warning(f"{smiles} contains an NaN physchem descriptor")
        _fp[~mask] = 0

    return _fp


def calculate_internal_pairwise_similarities(
    smiles_list: Collection[str],
) -> np.ndarray:
    """
    Computes the pairwise similarities of the provided list of smiles against itself.

    Returns:
        Symmetric matrix of pairwise similarities. Diagonal is set to zero.
    """
    if len(smiles_list) > 10000:
        logger.warning(
            f"Calculating internal similarity on large set of "
            f"SMILES strings ({len(smiles_list)})"
        )

    mols = get_mols(smiles_list)
    fps = get_fingerprints(mols)
    nfps = len(fps)

    similarities = np.zeros((nfps, nfps))

    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        similarities[i, :i] = sims
        similarities[:i, i] = sims

    return similarities


def calculate_bulk_diversity(
    mol_list: Collection[str],
    rdkit_fp=True,
):
    """
    Computes the pairwise diversity of the provided list of smiles against itself.

    Returns:
        Mean of pairwise diversity.
    """
    if len(mol_list) > 10000:
        logger.warning(
            f"Calculating internal similarity on large set of "
            f"RDKit molecules ({len(mol_list)})"
        )

    if rdkit_fp:
        fps = get_rdkit_fingerprints(mol_list)
    else:
        fps = get_fingerprints(mol_list)
    nfps = len(fps)

    diversities = np.zeros(nfps - 1)

    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        diversities[i - 1] = np.mean([1 - sim for sim in sims])

    return np.mean(diversities)


def get_fingerprints_from_smileslist(smiles_list):
    """
    Converts the provided smiles into ECFP4 bitvectors of length 4096.

    Args:
        smiles_list: list of SMILES strings

    Returns: ECFP4 bitvectors of length 4096.

    """
    return get_fingerprints(get_mols(smiles_list))


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


def get_rdkit_fingerprints(mols: Iterable[Chem.Mol]):
    """
    Converts molecules to RDKit bitvectors.

    Returns: a list of fingerprints
    """
    return [Chem.RDKFingerprint(m) for m in mols]


def get_mols(smiles_list: Iterable[str]) -> Iterable[Chem.Mol]:
    for i in smiles_list:
        try:
            mol = Chem.MolFromSmiles(i)
            if mol is not None:
                yield mol
        except Exception as e:
            logger.warning(e)


def get_mols_list(smiles_list: Iterable[str]) -> Iterable[Chem.Mol]:
    mols = []
    for i in smiles_list:
        try:
            mol = Chem.MolFromSmiles(i)
            if mol is not None:
                mols.append(mol)
        except Exception as e:
            logger.warning(e)
    return mols


def get_random_subset(
    dataset: List[Any], subset_size: int, seed: Optional[int] = None
) -> List[Any]:
    if len(dataset) < subset_size:
        raise Exception(
            f"The dataset to extract a subset from is too small: "
            f"{len(dataset)} < {subset_size}"
        )

    # save random number generator state
    rng_state = np.random.get_state()

    if seed is not None:
        # extract a subset (for a given training set, the subset will always be identical).
        np.random.seed(seed)

    subset = np.random.choice(dataset, subset_size, replace=False)

    if seed is not None:
        # reset random number generator state, only if needed
        np.random.set_state(rng_state)

    return list(subset)


def logP(mol: Mol) -> float:
    return Descriptors.MolLogP(mol)


def qed(mol: Mol) -> float:
    return Descriptors.qed(mol)


def num_rings(mol: Mol) -> int:
    return rdMolDescriptors.CalcNumRings(mol)


def num_aromatic_rings(mol: Mol) -> int:
    return rdMolDescriptors.CalcNumAromaticRings(mol)


import os.path as op
import pickle

_fscores = None


def readFragmentScores(name="fpscores"):
    import gzip

    global _fscores
    # generate the full path filename:
    if name == "fpscores":
        name = op.join(op.dirname(__file__), name)
    data = pickle.load(gzip.open("%s.pkl.gz" % name))
    outDict = {}
    for i in data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict


def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro


def calculateScore(m):
    if _fscores is None:
        readFragmentScores()

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(
        m, 2
    )  # <- 2 is the *radius* of the circular fingerprint
    fps = fp.GetNonzeroElements()
    score1 = 0.0
    nf = 0
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.0
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = (
        0.0
        - sizePenalty
        - stereoPenalty
        - spiroPenalty
        - bridgePenalty
        - macrocyclePenalty
    )

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.0
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * 0.5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11.0 - (sascore - min + 1) / (max - min) * 9.0
    # smooth the 10-end
    if sascore > 8.0:
        sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
    if sascore > 10.0:
        sascore = 10.0
    elif sascore < 1.0:
        sascore = 1.0

    return sascore


def processMols(mols):
    print("smiles\tName\tsa_score")
    for i, m in enumerate(mols):
        if m is None:
            continue

        s = calculateScore(m)

        smiles = Chem.MolToSmiles(m)
        print(smiles + "\t" + m.GetProp("_Name") + "\t%3f" % s)


def calculate_sa(rdmol):
    sa = np.array([sascorer.calculateScore(Chem.RemoveHs(rdmol))])
    sa = (sa - 1.0) / (10.0 - 1.0)
    sa = 1.0 - sa
    sa = torch.from_numpy(sa).float()
    return sa


def calculate_logp(rdmol):
    return Crippen.MolLogP(rdmol)


def calculate_hdonors(rdmol):
    num_hdonors = Lipinski.NumHDonors(rdmol)
    return num_hdonors


def calculate_hacceptors(rdmol):
    num_hacceptors = Lipinski.NumHAcceptors(rdmol)
    return num_hacceptors


def calculate_molwt(rdmol):
    mol_weight = Descriptors.MolWt(rdmol)
    return mol_weight


def calculate_qed(rdmol):
    return QED.qed(rdmol)


def calculate_lipinski(rdmol):
    rule_1 = Descriptors.ExactMolWt(rdmol) < 500
    rule_2 = Lipinski.NumHDonors(rdmol) <= 5
    rule_3 = Lipinski.NumHAcceptors(rdmol) <= 10
    rule_4 = (logp := Crippen.MolLogP(rdmol) >= -2) & (logp <= 5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(rdmol) <= 10
    return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])

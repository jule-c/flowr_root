import logging
import os
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import biotite.structure.io.pdb as pdb
import hydride
import MDAnalysis as mda
import numpy as np
import pdbinf
import prolif as plf
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Mol, rdDetermineBonds, rdForceFieldHelpers
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

_T = torch.Tensor
TupleRot = tuple[float, float, float]


# *************************************************************************************************
# ********************************** Data Util Functions ****************************************
# *************************************************************************************************


def symmetrize_bonds(bond_dist: torch.Tensor, is_one_hot: bool = False) -> torch.Tensor:
    """
    Symmetrize bond tensor by ensuring B[i,j] == B[j,i].

    Args:
        bond_dist: Bond tensor of shape:
            - (B, N, N, n_bond_types) if batched one-hot
            - (B, N, N) if batched adjacency indices
            - (N, N, n_bond_types) if single molecule one-hot
            - (N, N) if single molecule adjacency indices
        is_one_hot: If True, input is one-hot encoded bonds

    Returns:
        Symmetrized bond tensor (same shape as input)
    """

    # Detect if input is batched or not
    if is_one_hot:
        # One-hot: (B, N, N, K) or (N, N, K)
        single_mol = len(bond_dist.shape) == 3
        if single_mol:
            bond_dist = bond_dist.unsqueeze(0)  # Add batch dim: (1, N, N, K)
        bond_adj = torch.argmax(bond_dist, dim=-1)  # (B, N, N)
    else:
        # Adjacency indices: (B, N, N) or (N, N)
        single_mol = len(bond_dist.shape) == 2
        if single_mol:
            bond_dist = bond_dist.unsqueeze(0)  # Add batch dim: (1, N, N)
        bond_adj = bond_dist  # (B, N, N)

    # Create upper triangle mask (excludes diagonal - no self-loops)
    n_atoms = bond_adj.shape[-1]
    device = bond_adj.device
    upper_tri_mask = torch.triu(
        torch.ones((n_atoms, n_atoms), dtype=torch.bool, device=device), diagonal=1
    )  # (N, N)

    # Vectorized symmetrization for entire batch
    # Keep only upper triangle, zero out lower
    bond_adj_upper = bond_adj.clone()
    # Broadcast upper_tri_mask to batch dimension
    bond_adj_upper[:, ~upper_tri_mask] = 0  # (B, N, N)

    # Symmetrize by adding transpose (diagonal stays 0)
    bond_adj_sym = bond_adj_upper + bond_adj_upper.transpose(1, 2)

    # Convert back to one-hot if needed
    if is_one_hot:
        n_bond_types = bond_dist.shape[-1]
        result = one_hot_encode_tensor(bond_adj_sym, n_bond_types)
    else:
        result = bond_adj_sym

    # Remove batch dimension if input was single molecule
    if single_mol:
        result = result.squeeze(0)

    return result


def prepare_complex_data(
    gen_ligs: list[Chem.Mol],
    native_lig: Chem.Mol,
    pdb_file: str,
    add_optimize_gen_lig_hs: bool = True,
    add_optimize_ref_lig_hs: bool = False,
    optimize_pocket_hs: bool = False,
    process_pocket: bool = False,
    optimization_method: str = "prolif_mmff",
    pocket_cutoff: float = 6.0,
    strip_invalid: bool = True,
):
    optimizer = LigandPocketOptimization(
        pocket_cutoff=pocket_cutoff, strip_invalid=strip_invalid
    )

    complex_id = Path(pdb_file).stem
    if add_optimize_gen_lig_hs:
        if isinstance(gen_ligs, Chem.Mol):
            gen_lig_mol = add_and_optimize_hs(
                gen_ligs,
                pdb_file,
                optimizer=optimizer,
                optimize_pocket_hs=optimize_pocket_hs,
                process_pocket=process_pocket,
            )
            if gen_lig_mol is None:
                print(
                    f"Could not find any generated ligand that could be optimized for complex: {complex_id}"
                )
                return
        elif isinstance(gen_ligs, list):
            gen_lig_mol = [
                add_and_optimize_hs(
                    lig,
                    pdb_file,
                    optimizer=optimizer,
                    optimize_pocket_hs=optimize_pocket_hs,
                    process_pocket=process_pocket,
                )
                for lig in gen_ligs
            ]
            gen_lig_mol = [lig for lig in gen_lig_mol if lig is not None]
            if len(gen_lig_mol) == 0:
                print(
                    f"Could not find any generated ligand that could be optimized for complex: {complex_id}"
                )
                return
        else:
            raise ValueError("Invalid ligand format")
    else:
        if isinstance(gen_ligs, Chem.Mol):
            gen_lig_mol = ligand_from_mol(gen_ligs, add_hydrogens=False)
            if gen_lig_mol is None:
                print(
                    f"Could not find any generated ligand that could be optimized for complex: {complex_id}"
                )
                return
        elif isinstance(gen_ligs, list):
            gen_lig_mol = [
                ligand_from_mol(lig, add_hydrogens=False) for lig in gen_ligs
            ]
            gen_lig_mol = [lig for lig in gen_lig_mol if lig is not None]
            if len(gen_lig_mol) == 0:
                print(
                    f"Could not find any generated ligand that could be optimized for complex: {complex_id}"
                )
                return
        else:
            raise ValueError("Invalid ligand format")

    if add_optimize_ref_lig_hs:
        native_lig = Chem.RemoveHs(native_lig)
        native_lig_mol = add_and_optimize_hs(
            native_lig,
            pdb_file,
            optimizer=optimizer,
            process_pocket=process_pocket,
            optimize_pocket_hs=optimize_pocket_hs,
        )
        if native_lig_mol is None:
            print(f"Failed to optimize native ligand in complex: {complex_id}")
            return
    else:
        native_lig_mol = ligand_from_mol(native_lig, add_hydrogens=False)

    # per ligand-pocket prolif calculation currently not expected, thus initiate pocket only once
    pocket_mol = optimizer.pocket_from_pdb(
        pdb_file, native_lig_mol, process_pocket=process_pocket
    )
    return gen_lig_mol, native_lig_mol, pocket_mol


# *************************************************************************************************
# ********************************** Tensor Util Functions ****************************************
# *************************************************************************************************


def pad_tensors(tensors: list[_T], pad_dim: int = 0) -> _T:
    """Pad a list of tensors with zeros

    All dimensions other than pad_dim must have the same shape. A single tensor is returned with the batch dimension
    first, where the batch dimension is the length of the tensors list.

    Args:
        tensors (list[torch.Tensor]): List of tensors
        pad_dim (int): Dimension on tensors to pad. All other dimensions must be the same size.

    Returns:
        torch.Tensor: Batched, padded tensor, if pad_dim is 0 then shape [B, L, *] where L is length of longest tensor.
    """

    if pad_dim != 0:
        # TODO
        raise NotImplementedError()

    padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    return padded


# TODO replace with tensor version below
def one_hot_encode(indices: list[int], vocab_size: int) -> _T:
    """Create one-hot encodings from a list of indices

    Args:
        indices (list[int]): List of indices into one-hot vectors
        vocab_size (int): Length of returned vectors

    Returns:
        torch.Tensor: One-hot encoded vectors, shape [L, vocab_size] where L is length of indices list
    """

    one_hots = torch.zeros((len(indices), vocab_size), dtype=torch.int64)

    for batch_idx, vocab_idx in enumerate(indices):
        one_hots[batch_idx, vocab_idx] = 1

    return one_hots


# TODO test
def one_hot_encode_tensor(indices: _T, vocab_size: int) -> _T:
    """Create one-hot encodings from indices

    Args:
        indices (torch.Tensor): Indices into one-hot vectors, shape [*, L]
        vocab_size (int): Length of returned vectors

    Returns:
        torch.Tensor: One-hot encoded vectors, shape [*, L, vocab_size]
    """

    one_hot_shape = (*indices.shape, vocab_size)
    one_hots = torch.zeros(one_hot_shape, dtype=torch.int64, device=indices.device)
    one_hots.scatter_(-1, indices.unsqueeze(-1), 1)
    return one_hots


def pairwise_concat(t: _T) -> _T:
    """Concatenates two representations from all possible pairings in dimension 1

    Computes all possible pairs of indices into dimension 1 and concatenates whatever representation they have in
    higher dimensions. Note that all higher dimensions will be flattened. The output will have its shape for
    dimension 1 duplicated in dimension 2.

    Example:
    Input shape [100, 16, 128]
    Output shape [100, 16, 16, 256]
    """

    idx_pairs = torch.cartesian_prod(*((torch.arange(t.shape[1]),) * 2))
    output = t[:, idx_pairs].view(t.shape[0], t.shape[1], t.shape[1], -1)
    return output


def segment_sum(data, segment_ids, num_segments):
    """Computes the sum of data elements that are in each segment

    The inputs must have shapes that look like the following:
    data [batch_size, seq_length, num_features]
    segment_ids [batch_size, seq_length], must contain integers

    Then the output will have the following shape:
    output [batch_size, num_segments, num_features]
    """

    err_msg = (
        "data and segment_ids must have the same shape in the first two dimensions"
    )
    assert data.shape[0:2] == segment_ids.shape[0:2], err_msg

    result_shape = (data.shape[0], num_segments, data.shape[2])
    result = data.new_full(result_shape, 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, -1, data.shape[2])
    result.scatter_add_(1, segment_ids, data)
    return result


def retrieve_ligand(
    coords, pred_coords, type_logits, charge_logits, lig_mask, pred_edges=None
):
    true_coords = []
    lig_coords = []
    lig_type_logits = []
    lig_charge_logits = []
    max_atoms = lig_mask.sum(dim=1).max().item()
    if pred_edges is not None:
        lig_edge_logits = []
    for i in range(coords.size(0)):
        true_coords.append(coords[i, lig_mask[i], :])
        lig_coords.append(pred_coords[i, lig_mask[i], :])
        lig_type_logits.append(type_logits[i, lig_mask[i], :])
        lig_charge_logits.append(charge_logits[i, lig_mask[i], :])
        if pred_edges is not None:
            bond_probs = torch.zeros(max_atoms, max_atoms, pred_edges.shape[-1]).to(
                lig_mask.device
            )
            num_atoms = lig_mask[i].sum().item()
            bond_indices = lig_mask[i].nonzero(as_tuple=True)[0]
            bond_probs[:num_atoms, :num_atoms, :] = pred_edges[i][
                bond_indices[:, None], bond_indices
            ]
            lig_edge_logits.append(bond_probs)
    atom_mask = (
        pad_tensors([torch.ones(len(coords)) for coords in lig_coords])
        .to(lig_mask.device)
        .int()
    )
    true_coords = pad_tensors(true_coords)
    lig_coords = pad_tensors(lig_coords)
    lig_type_logits = pad_tensors(lig_type_logits)
    lig_charge_logits = pad_tensors(lig_charge_logits)

    if pred_edges is not None:
        lig_edge_logits = torch.stack(lig_edge_logits)
    return (
        true_coords,
        lig_coords,
        lig_type_logits,
        lig_charge_logits,
        lig_edge_logits,
        atom_mask,
    )


# *************************************************************************************************
# ******************************* Functions for handling edges ************************************
# *************************************************************************************************


def adj_from_node_mask(node_mask, self_connect=False):
    """Creates an edge mask from a given node mask assuming all nodes are fully connected excluding self-connections

    Args:
        node_mask (torch.Tensor): Node mask tensor, shape [batch_size, num_nodes], 1 for real node 0 otherwise
        self_connect (bool): Whether to include self connections in the adjacency

    Returns:
        torch.Tensor: Adjacency tensor, shape [batch_size, num_nodes, num_nodes], 1 for real edge 0 otherwise
    """
    node_mask = node_mask.long()
    num_nodes = node_mask.size()[1]

    # Matrix mult gives us an outer product on the node mask, which is an edge mask
    mask = node_mask.float()
    adjacency = torch.bmm(mask.unsqueeze(2), mask.unsqueeze(1))
    adjacency = adjacency.long()

    # Set diagonal connections
    node_idxs = torch.arange(num_nodes)
    self_mask = node_mask if self_connect else torch.zeros_like(node_mask)

    adjacency[:, node_idxs, node_idxs] = self_mask

    return adjacency


def _pad_edges(edges, max_edges, value=0):
    """Add fake edges to an edge tensor so that the shape matches max_edges

    Args:
        edges (torch.Tensor): Unbatched edge tensor, shape [num_edges, 2], each element is a node index for the edge
        max_edges (int): The number of edges the output tensor should have
        value (int): Padding value, default 0

    Returns:
        (torch.Tensor, torch.Tensor): Tuple of padded edge tensor and padding mask. Shapes [max_edges, 2] for edge
                tensor and [max_edges] for mask. Mask is one for pad elements, 0 otherwise.
    """

    num_edges = edges.size(0)
    mask_kwargs = {"dtype": torch.int64, "device": edges.device}

    if num_edges > max_edges:
        raise ValueError(
            "Number of edges in edge tensor to be padded cannot be greater than max_edges."
        )

    add_edges = max_edges - num_edges

    if add_edges == 0:
        pad_mask = torch.zeros(num_edges, **mask_kwargs)
        return edges, pad_mask

    pad = (0, 0, 0, add_edges)
    padded = torch.nn.functional.pad(edges, pad, mode="constant", value=value)

    zeros_mask = torch.zeros(num_edges, **mask_kwargs)
    ones_mask = torch.ones(add_edges, **mask_kwargs)
    pad_mask = torch.cat((zeros_mask, ones_mask), dim=0)

    return padded, pad_mask


# TODO change callers to use bonds_from_adj
def edges_from_adj(adj_matrix):
    """Flatten an adjacency matrix into a 1D edge representation

    Args:
        adj_matrix (torch.Tensor): Batched adjacency matrix, shape [batch_size, num_nodes, num_nodes]. It can contain
                any non-zero integer for connected nodes but must be 0 for unconnected nodes.

    Returns:
        A tuple of the edge tensor and the edge mask tensor. The edge tensor has shape [batch_size, max_num_edges, 2]
        and the mask [batch_size, max_num_edges]. The mask contains 1 for real edges, 0 otherwise.
    """

    adj_ones = torch.zeros_like(adj_matrix).int()
    adj_ones[adj_matrix != 0] = 1

    # Pad each batch element by a seperate amount so that they can all be packed into a tensor
    # It might be possible to do this in batch form without iterating, but for now this will do
    num_edges = adj_ones.sum(dim=(1, 2)).tolist()
    edge_tuples = list(adj_matrix.nonzero()[:, 1:].split(num_edges))
    padded = [_pad_edges(edges, max(num_edges), value=0) for edges in edge_tuples]

    # Unravel the padded tuples and stack them into batches
    edge_tuples_padded, pad_masks = tuple(zip(*padded))
    edges = torch.stack(edge_tuples_padded).long()
    edges = (edges[:, :, 0], edges[:, :, 1])
    edge_mask = (torch.stack(pad_masks) == 0).long()
    return edges, edge_mask


# TODO test and merge with edges_from_adj
def bonds_from_adj(adj_matrix, lower_tri=True):
    """Flatten an adjacency matrix into a 1D edge representation

    Args:
        adj_matrix (torch.Tensor): Adjacency matrix, can be batched or not, shape [batch_size, num_nodes, num_nodes].
            Each item in the matrix corrsponds to the bond type and will be placed into index 2 on dim 1 in bonds.
        lower_tri (bool): Whether to only consider bonds which sit in the lower triangular of adj_matrix.

    Returns:
        An bond list tensor, shape [batch_size, num_bonds, 3]. If an item is a padding bond index 2 on the last
            dimension will be 0.
    """

    batched = True
    if len(adj_matrix.shape) == 2:
        adj_matrix = adj_matrix.unsqueeze(0)
        batched = False

    if lower_tri:
        adj_matrix = torch.tril(adj_matrix, diagonal=-1)

    bonds = []
    for adj in list(adj_matrix):
        bond_indices = adj.nonzero()
        bond_types = adj[bond_indices[:, 0], bond_indices[:, 1]]
        bond_list = torch.cat((bond_indices, bond_types.unsqueeze(-1)), dim=-1)
        bonds.append(bond_list)

    # Bonds will be padded with 0s so the bond type will tell whether the bond is real or not
    bonds = pad_tensors(bonds, pad_dim=0)
    if not batched:
        bonds = bonds.squeeze(0)

    return bonds


def adj_from_edges(
    edge_indices: _T, edge_types: _T, n_nodes: int, symmetric: bool = False
):
    """Create adjacency matrix from a list of edge indices and types

    If an edge pair appears multiple times with different edge types, the adj element for that edge is undefined.

    Args:
        edge_indices (torch.Tensor): Edge list tensor, shape [n_edges, 2]. Pairs of (from_idx, to_idx).
        edge_types (torch.Tensor): Edge types, shape either [n_edges] or [n_edges, edge_types].
        n_nodes (int): Number of nodes in the adjacency matrix. This must be >= to the max node index in edges.
        symmetric (bool): Whether edges are considered symmetric. If True the adjacency matrix will also be symmetric,
                otherwise only the exact node indices within edges will be used to create the adjacency.

    Returns:
        torch.Tensor: Adjacency matrix tensor, shape [n_nodes, n_nodes] or
                [n_nodes, n_nodes, edge_types] if distributions over edge types are provided.
    """

    assert len(edge_indices.shape) == 2
    assert edge_indices.shape[0] == edge_types.shape[0]
    assert edge_indices.size(1) == 2

    adj_dist = len(edge_types.shape) == 2

    edge_indices = edge_indices.long()
    edge_types = edge_types.float() if adj_dist else edge_types.long()

    if adj_dist:
        shape = (n_nodes, n_nodes, edge_types.size(-1))
        adj = torch.zeros(shape, device=edge_indices.device, dtype=torch.float)

    else:
        shape = (n_nodes, n_nodes)
        adj = torch.zeros(shape, device=edge_indices.device, dtype=torch.long)

    from_indices = edge_indices[:, 0]
    to_indices = edge_indices[:, 1]

    adj[from_indices, to_indices] = edge_types
    if symmetric:
        adj[to_indices, from_indices] = edge_types

    return adj


def edges_from_nodes(coords, k=None, node_mask=None, edge_format="adjacency"):
    """Constuct edges from node coords

    Connects a node to its k nearest nodes. If k is None then connects each node to all its neighbours. A node is
    never connected to itself.

    Args:
        coords (torch.Tensor): Node coords, shape [batch_size, num_nodes, 3]
        k (int): Number of neighbours to connect each node to, None means connect to all nodes except itself
        node_mask (torch.Tensor): Node mask, shape [batch_size, num_nodes], 1 for real nodes 0 otherwise
        edge_format (str): Edge format, should be either 'adjacency' or 'list'

    Returns:
        If format is 'adjacency' this returns an adjacency matrix, shape [batch_size, num_nodes, num_nodes] which
        contains 1 for connected nodes and 0 otherwise. Note that if a value for k is provided the adjacency matrix
        may not be symmetric and should always be used s.t. 'from nodes' are in dim 1 and 'to nodes' are in dim 2.

        If format is 'list' this returns the tuple (edges, edge mask), edges is also a two-tuple of tensors, each of
        shape [batch_size, num_edges], specifying node indices for each edge. The edge mask has shape
        [batch_size, num_edges] and contains 1 for 'real' edges and 0 otherwise.
    """

    if edge_format not in ["adjacency", "list"]:
        raise ValueError(f"Unrecognised edge format '{edge_format}'")

    adj_format = edge_format == "adjacency"
    batch_size, num_nodes, _ = coords.size()

    # If node mask is None all nodes are real
    if node_mask is None:
        node_mask = torch.ones(
            (batch_size, num_nodes), device=coords.device, dtype=torch.int64
        )

    adj_matrix = adj_from_node_mask(node_mask)

    if k is not None:
        # Find k closest nodes for each node
        dists = calc_distances(coords)
        dists[adj_matrix == 0] = float("inf")
        _, best_idxs = dists.topk(k, dim=2, largest=False)

        # Adjust adj matrix to only have k connections per node
        k_adj_matrix = torch.zeros_like(adj_matrix)
        batch_idxs = torch.arange(batch_size).view(-1, 1, 1).expand(-1, num_nodes, k)
        node_idxs = torch.arange(num_nodes).view(1, -1, 1).expand(batch_size, -1, k)
        k_adj_matrix[batch_idxs, node_idxs, best_idxs] = 1

        # Ensure that there are no connections to fake nodes
        k_adj_matrix[adj_matrix == 0] = 0
        adj_matrix = k_adj_matrix

    if adj_format:
        return adj_matrix

    edges, edge_mask = edges_from_adj(adj_matrix)
    return edges, edge_mask


def gather_edge_features(pairwise_feats, adj_matrix):
    """Gather edge features for each node from pairwise features using the adjacency matrix

    All 'from nodes' (dimension 1 on the adj matrix) must have the same number of edges to 'to nodes'. Practically
    this means that the number of non-zero elements in dimension 2 of the adjacency matrix must always be the same.

    Args:
        pairwise_feats (torch.Tensor): Pairwise features tensor, shape [batch_size, num_nodes, num_nodes, num_feats]
        adj_matrix (torch.Tensor): Batched adjacency matrix, shape [batch_size, num_nodes, num_nodes]. It can contain
                any non-zero integer for connected nodes but must be 0 for unconnected nodes.

    Returns:
        torch.Tensor: Dense feature matrix, shape [batch_size, num_nodes, edges_per_node, num_feats]
    """

    # In case some of the connections don't use 1, create a 1s adjacency matrix
    adj_ones = torch.zeros_like(adj_matrix).int()
    adj_ones[adj_matrix != 0] = 1

    num_neighbours = adj_ones.sum(dim=2)
    feats_per_node = num_neighbours[0, 0].item()

    assert (
        num_neighbours == feats_per_node
    ).all(), "All nodes must have the same number of connections"

    if len(pairwise_feats.size()) == 3:
        batch_size, num_nodes, _ = pairwise_feats.size()
        pairwise_feats = pairwise_feats.unsqueeze(3)

    elif len(pairwise_feats.size()) == 4:
        batch_size, num_nodes, _, _ = pairwise_feats.size()

    # nonzero() orders indices lexicographically with the last index changing the fastest, so we can reshape the
    # indices into a dense form with nodes along the outer axis and features along the inner
    gather_idxs = adj_ones.nonzero()[:, 2].reshape(
        (batch_size, num_nodes, feats_per_node)
    )
    batch_idxs = torch.arange(batch_size).view(-1, 1, 1)
    node_idxs = torch.arange(num_nodes).view(1, -1, 1)
    dense_feats = pairwise_feats[batch_idxs, node_idxs, gather_idxs, :]
    if dense_feats.size(-1) == 1:
        return dense_feats.squeeze(-1)

    return dense_feats


# *************************************************************************************************
# ********************************* Geometric Util Functions **************************************
# *************************************************************************************************


# TODO rename? Maybe also merge with inter_distances
# TODO test unbatched and coord sets inputs
def calc_distances(coords, edges=None, sqrd=False, eps=1e-6):
    """Computes distances between connected nodes

    Takes an optional edges argument. If edges is None this will calculate distances between all nodes and return the
    distances in a batched square matrix [batch_size, num_nodes, num_nodes]. If edges is provided the distances are
    returned for each edge in a batched 1D format [batch_size, num_edges].

    Args:
        coords (torch.Tensor): Coordinate tensor, shape [batch_size, num_nodes, 3]
        edges (tuple): Two-tuple of connected node indices, each tensor has shape [batch_size, num_edges]
        sqrd (bool): Whether to return the squared distances
        eps (float): Epsilon to add before taking the square root for numical stability in the gradients

    Returns:
        torch.Tensor: Distances tensor, the shape depends on whether edges is provided (see above).
    """

    # TODO add checks

    # Create fake batch dim if unbatched
    unbatched = False
    if len(coords.size()) == 2:
        coords = coords.unsqueeze(0)
        unbatched = True

    if edges is None:
        coord_diffs = coords.unsqueeze(-2) - coords.unsqueeze(-3)
        sqrd_dists = torch.sum(coord_diffs * coord_diffs, dim=-1)

    else:
        edge_is, edge_js = edges
        batch_index = torch.arange(coords.size(0)).unsqueeze(1)
        coord_diffs = coords[batch_index, edge_js, :] - coords[batch_index, edge_is, :]
        sqrd_dists = torch.sum(coord_diffs * coord_diffs, dim=2)

    sqrd_dists = sqrd_dists.squeeze(0) if unbatched else sqrd_dists

    if sqrd:
        return sqrd_dists

    return torch.sqrt(sqrd_dists + eps)


def inter_distances(coords1, coords2, sqrd=False, eps=1e-6):
    # TODO add checks and doc

    # Create fake batch dim if unbatched
    unbatched = False
    if len(coords1.size()) == 2:
        coords1 = coords1.unsqueeze(0)
        coords2 = coords2.unsqueeze(0)
        unbatched = True

    coord_diffs = coords1.unsqueeze(2) - coords2.unsqueeze(1)
    sqrd_dists = torch.sum(coord_diffs * coord_diffs, dim=3)
    sqrd_dists = sqrd_dists.squeeze(0) if unbatched else sqrd_dists

    if sqrd:
        return sqrd_dists

    return torch.sqrt(sqrd_dists + eps)


def calc_com(coords, node_mask=None):
    """Calculates the centre of mass of a pointcloud

    Args:
        coords (torch.Tensor): Coordinate tensor, shape [*, num_nodes, 3]
        node_mask (torch.Tensor): Mask for points, shape [*, num_nodes], 1 for real node, 0 otherwise

    Returns:
        torch.Tensor: CoM of pointclouds with imaginary nodes excluded, shape [*, 1, 3]
    """

    node_mask = torch.ones_like(coords[..., 0]) if node_mask is None else node_mask

    assert node_mask.shape == coords[..., 0].shape

    num_nodes = node_mask.sum(dim=-1)
    real_coords = coords * node_mask.unsqueeze(-1)
    com = real_coords.sum(dim=-2) / num_nodes.unsqueeze(-1)
    return com.unsqueeze(-2)


def zero_com(coords, node_mask=None):
    """Sets the centre of mass for a batch of pointclouds to zero for each pointcloud

    Args:
        coords (torch.Tensor): Coordinate tensor, shape [*, num_nodes, 3]
        node_mask (torch.Tensor): Mask for points, shape [*, num_nodes], 1 for real node, 0 otherwise

    Returns:
        torch.Tensor: CoM-free coordinates, where imaginary nodes are excluded from CoM calculation
    """

    com = calc_com(coords, node_mask=node_mask)
    shifted = coords - com
    return shifted


def standardise_coords(coords, node_mask=None):
    """Convert coords into a standard normal distribution

    This will first remove the centre of mass from all pointclouds in the batch, then calculate the (biased) variance
    of the shifted coords and use this to produce a standard normal distribution.

    Args:
        coords (torch.Tensor):  Coordinate tensor, shape [batch_size, num_nodes, 3]
        node_mask (torch.Tensor): Mask for points, shape [batch_size, num_nodes], 1 for real node, 0 otherwise

    Returns:
        Tuple[torch.Tensor, float]: The standardised coords and the variance of the original coords
    """

    if node_mask is None:
        node_mask = torch.ones_like(coords)[:, :, 0]

    coord_idxs = node_mask.nonzero()
    real_coords = coords[coord_idxs[:, 0], coord_idxs[:, 1], :]

    variance = torch.var(real_coords, correction=0)
    std_dev = torch.sqrt(variance)

    result = (coords / std_dev) * node_mask.unsqueeze(2)
    return result, std_dev.item()


def rotate(coords: torch.Tensor, rotation: Union[Rotation, TupleRot]):
    """Rotate coordinates for a single molecule

    Args:
        coords (torch.Tensor): Unbatched coordinate tensor, shape [num_atoms, 3]
        rotation (Union[Rotation, Tuple[float, float, float]]): Can be either a scipy Rotation object or a tuple of
                rotation values in radians, (x, y, z). These are treated as extrinsic rotations. See the scipy docs
                (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html) for info.

    Returns:
        torch.Tensor: Rotated coordinates
    """

    if not isinstance(rotation, Rotation):
        rotation = Rotation.from_euler("xyz", rotation)

    device = coords.device
    coords = coords.cpu().numpy()

    rotated = rotation.apply(coords)
    rotated = torch.tensor(rotated, device=device)
    return rotated


def cartesian_to_spherical(coords):
    sqrd_dists = (coords * coords).sum(dim=-1)
    radii = torch.sqrt(sqrd_dists)
    inclination = torch.acos(coords[..., 2] / radii).unsqueeze(2)
    azimuth = torch.atan2(coords[..., 1], coords[..., 0]).unsqueeze(2)
    spherical = torch.cat((radii.unsqueeze(2), inclination, azimuth), dim=-1)
    return spherical


# *************************************************************************************************
# ************************************** Util Classes *********************************************
# *************************************************************************************************


class SparseFeatures:
    def __init__(self, dense, idxs):
        assert len(dense.size()) == 3
        assert dense.size() == idxs.size()

        batch_size, num_nodes, num_feats = dense.size()

        self.bs = batch_size
        self.num_nodes = num_nodes
        self.num_feats = num_feats

        self._dense = dense
        self._idxs = idxs

    @staticmethod
    def from_sparse(sparse_feats, adj_matrix, feats_per_node):
        err_msg = "adj_matrix must have feats_per_node ones in each row"
        assert (
            sparse_feats.size() == adj_matrix.size()
        ), "sparse_feats and adj_matrix must have the same shape"
        assert adj_matrix.size()[1] == adj_matrix.size()[2], "adj_matrix must be square"
        assert (adj_matrix.sum(dim=2) == feats_per_node).all().item(), err_msg

        batch_size, num_nodes, _ = adj_matrix.size()
        feat_idxs = adj_matrix.nonzero()[:, 2].reshape(
            (batch_size, num_nodes, feats_per_node)
        )
        dense_feats = torch.gather(sparse_feats, 2, feat_idxs)
        return SparseFeatures(dense_feats, feat_idxs)

    @staticmethod
    def from_dense(dense_feats, idxs):
        return SparseFeatures(dense_feats, idxs)

    def to_tensor(self):
        sparse_matrix = torch.zeros(
            (self.bs, self.num_nodes, self.num_nodes), device=self._dense.device
        )
        sparse_matrix.scatter_(2, self._idxs, self._dense)
        return sparse_matrix

    def mult(self, other):
        if isinstance(other, (int, float)):
            return self.from_dense(self._dense * other, self._idxs)

        if not torch.is_tensor(other):
            raise TypeError(
                f"Object to multiply by must be an int, float or torch.Tensor"
            )

        assert other.size() == (self.bs, self.num_nodes, self.num_nodes)

        other_dense = torch.gather(other, 2, self._idxs)
        return self.from_dense(self._dense * other_dense, self._idxs)

    def matmul(self, other):
        if not torch.is_tensor(other):
            raise TypeError(f"Object to multiply by must be a torch.Tensor")

        assert tuple(other.size()[:2]) == (self.bs, self.num_nodes)

        # There doesn't seem to be an efficient implementation of sparse batched matmul available atm, so just do
        # regular matmul instead. We will still get some speed benefit from having lots of zeros.
        tensor = self.to_tensor()
        return torch.bmm(tensor, other)

    def softmax(self):
        dense_softmax = torch.softmax(self._dense, dim=2)
        return self.from_dense(dense_softmax, self._idxs)

    def dropout(self, p, train=False):
        dense_dropout = torch.dropout(self._dense, p, train=train)
        return self.from_dense(dense_dropout, self._idxs)

    def add(self, other):
        """Add a matrix only at elements which are not sparse in self"""

        assert len(other.size()) == 3

        other_dense = torch.gather(other, 2, self._idxs)
        return self.from_dense(self._dense + other_dense, self._idxs)

    def sum(self, dim=None):
        if dim == 1:
            return self.to_tensor().sum(dim=1)

        return self._dense.sum(dim=dim)


def get_rdkit_mol(fname_xyz):
    """
    Convert XYZ file to RDKit molecule using openbabel module directly.
    """
    try:
        from openbabel import openbabel as ob

        try:
            # Create OpenBabel molecule object
            obmol = ob.OBMol()

            # Create OpenBabel conversion object
            obConversion = ob.OBConversion()
            obConversion.SetInFormat("xyz")
            obConversion.SetOutFormat("pdb")

            # Read XYZ file
            obConversion.ReadFile(obmol, str(fname_xyz))

            # Convert to PDB format string
            pdb_string = obConversion.WriteString(obmol)

            # Create RDKit molecule from PDB string
            mol = Chem.MolFromPDBBlock(
                molBlock=pdb_string,
                sanitize=False,
                removeHs=False,
                proximityBonding=True,
            )

            return mol

        except Exception as e:
            print(f"Error converting XYZ file {fname_xyz} using openbabel: {e}")
            return None
    except ImportError:
        print("OpenBabel is not installed. Please install it to use this function.")
        return None


def write_xyz_file(coords, atom_types, filename):
    out = f"{len(coords)}\n\n"
    assert len(coords) == len(atom_types)
    for i in range(len(coords)):
        out += f"{atom_types[i]} {coords[i, 0]:.3f} {coords[i, 1]:.3f} {coords[i, 2]:.3f}\n"
    with open(filename, "w") as f:
        f.write(out)


class LigandPocketOptimization:
    def __init__(
        self,
        pocket_cutoff: float = 6.0,
        sanitize: bool = True,
        strip_invalid: bool = True,
    ):
        self.pocket_cutoff = pocket_cutoff
        self.sanitize = sanitize
        self.strip_invalid = strip_invalid
        self.radical_replaces_charge = {"C", "N", "OXT"}

    def __call__(
        self,
        ligand: Chem.Mol,
        pdb_file: str,
        method: str = "prolif_mmff",
        add_ligand_hs: bool = True,
        process_pocket: bool = False,
        return_pocket: bool = False,
        optimize_pocket_hs: bool = False,
        only_ligand_hs: bool = True,
    ):
        if method == "mmff":
            ligand = self._mmff_add_and_optimize_ligand_hs(ligand, pdb_file)
        elif method == "openmm":
            ligand = self._openmm_add_and_optimize_ligand_hs(ligand, pdb_file)
        elif method == "prolif_mmff":
            if return_pocket:
                ligand, pocket = self._prolif_mmff_optimize_ligand(
                    ligand,
                    pdb_file,
                    add_ligand_hs=add_ligand_hs,
                    process_pocket=process_pocket,
                    return_pocket=True,
                    optimize_pocket_hs=optimize_pocket_hs,
                    only_ligand_hs=only_ligand_hs,
                )
                return ligand, pocket
            else:
                ligand = self._prolif_mmff_optimize_ligand(
                    ligand,
                    pdb_file,
                    add_ligand_hs=add_ligand_hs,
                    process_pocket=process_pocket,
                    optimize_pocket_hs=optimize_pocket_hs,
                    only_ligand_hs=only_ligand_hs,
                )
        return ligand

    def _openmm_add_and_optimize_ligand_hs(
        self,
        ligand,
        protein_pdb,
        forcefield_files=("amber14-all.xml", "amber14/tip3pfb.xml"),
        restraint_force_constant: float = None,
        restrain_ligand_only: bool = True,
        max_iter: int = 200,
    ):
        """
        Read a protein (PDB) that is fully resolved (with hydrogens),
        read a ligand (Chem.Mol) missing hydrogens, add them, then combine
        both into an OpenMM system using Amber14. Freeze everything except
        the ligand's hydrogen atoms and run a short minimization so that
        only the ligand hydrogens are optimized.

        Parameters
        ----------
        protein_pdb : str
            Path to the protein PDB (fully resolved, containing hydrogens).
        ligand_sdf : str
            Path to the ligand SDF (hydrogens will be added).
        output_pdb : str
            Output PDB with minimized complex.
        forcefield_files : str
            Path to the Amber force field XML file.
        restraint_force_constant : openmm.unit.Quantity
            Force constant for positional restraints (everything except ligand H).
        restrain_ligand_only: If False, both ligand and pocket hydrogens are optimized (default: True)
        """
        import openmm
        from openmm import app, unit
        from openmm.app import PDBFile

        restraint_force_constant = 10.0 * unit.kilocalories_per_mole / unit.angstroms**2

        protein = PDBFile(str(protein_pdb))
        if protein.topology.getNumAtoms() == 0:
            raise ValueError(f"Could not parse PDB file at {protein_pdb}")

        # Add missing hydrogens
        ligand_h = Chem.AddHs(ligand, addCoords=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            ligand_pdb_path = os.path.join(tmpdir, "ligand_tmp.pdb")
            # Write out the ligand with current coords
            Chem.MolToPDBFile(ligand_h, ligand_pdb_path)
            ligand_pdb = app.PDBFile(ligand_pdb_path)  # read it back in

        # Create a Modeller object from the protein
        modeller = app.Modeller(protein.topology, protein.positions)
        modeller.add(ligand_pdb.topology, ligand_pdb.positions)

        # Create force field system
        forcefield = app.ForceField(*forcefield_files)
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=None,
        )

        # Restrain everything except ligand hydrogens
        restraint_force = openmm.CustomExternalForce(
            "0.5 * k * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)"
        )
        restraint_force.addPerParticleParameter("k")
        restraint_force.addPerParticleParameter("x0")
        restraint_force.addPerParticleParameter("y0")
        restraint_force.addPerParticleParameter("z0")
        k_value = (
            restraint_force_constant * unit.kilojoule_per_mole / (unit.nanometer**2)
        )

        positions = modeller.positions
        topology = modeller.topology
        start_idx = protein.topology.getNumAtoms()
        end_idx = modeller.topology.getNumAtoms()
        combined_atoms = list(topology.atoms())

        if restrain_ligand_only:
            for idx, atom in enumerate(combined_atoms):
                # If it's a ligand hydrogen, skip
                if start_idx <= idx < end_idx:
                    if atom.element is not None and atom.element.symbol == "H":
                        continue
                pos = positions[idx]
                restraint_force.addParticle(idx, [k_value, pos[0], pos[1], pos[2]])
        else:
            atom_idx = 0
            for chain in topology.chains():
                for residue in chain.residues():
                    for atom in residue.atoms():
                        pos = positions[atom_idx]
                        if pos is None:
                            raise ValueError("Missing coordinate for an atom.")
                        if atom.element.symbol != "H":
                            # Add this atom to the restraint
                            restraint_force.addParticle(
                                atom_idx, [k_value, pos.x, pos.y, pos.z]
                            )

                        atom_idx += 1

        system.addForce(restraint_force)

        # Set up the simulation
        integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
        platform = openmm.Platform.getPlatformByName(
            "Reference"
        )  # or "CPU" or "CUDA" etc.

        simulation = app.Simulation(modeller.topology, system, integrator, platform)
        simulation.context.setPositions(modeller.positions)

        # Minimize
        simulation.minimizeEnergy(
            tolerance=10.0 * unit.kilojoule_per_mole, maxIterations=max_iter
        )

        final_positions = simulation.context.getState(getPositions=True).getPositions(
            asNumpy=True
        )

        ligand_atom_indices = []
        ligand_atom_indices = list(range(start_idx, end_idx))

        optimized_ligand = Chem.Mol(ligand_h)  # copy
        conf = optimized_ligand.GetConformer()
        for i, idx in enumerate(ligand_atom_indices):
            pos = final_positions[idx]
            # OpenMM uses nanometers; RDKit uses Angstroms => 1 nm = 10 Å
            x = pos.x * 10.0
            y = pos.y * 10.0
            z = pos.z * 10.0
            conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(x, y, z))

        return optimized_ligand

    def _mmff_add_and_optimize_ligand_hs(
        self, ligand: Chem.Mol, pocket_pdb_path: str, maxIters: int = 200
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
            raise ValueError(f"Could not parse PDB file at: {str(pocket_pdb_path)}")

        ligand_with_H = Chem.AddHs(ligand, addCoords=True)

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
            conf.SetAtomPosition(
                protein_num_atoms + i, Chem.rdGeometry.Point3D(x, y, z)
            )

        combined_mol.AddConformer(conf)
        combined_mol = combined_mol.GetMol()

        # Set up the MMFF force field
        mp = AllChem.MMFFGetMoleculeProperties(combined_mol, mmffVariant="MMFF94s")
        if mp is None:
            raise ValueError("Could not get MMFF properties for combined molecule.")
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
            ff.Minimize(maxIters=maxIters)
        except Exception as e:
            print(f"[Warning] Minimization failed: {e}")
            return ligand_with_H

        # Extract the updated coordinates for the ligand
        new_coords = combined_mol.GetConformer().GetPositions()
        for i in range(ligand_num_atoms):
            x, y, z = new_coords[protein_num_atoms + i]
            ligand_with_H.GetConformer().SetAtomPosition(
                i, Chem.rdGeometry.Point3D(x, y, z)
            )

        return ligand_with_H

    def _prolif_mmff_optimize_ligand(
        self,
        ligand: Chem.Mol,
        pdb_file: str,
        max_iter: int = 200,
        add_ligand_hs: bool = True,
        process_pocket: bool = False,
        return_pocket: bool = False,
        only_ligand_hs: bool = False,
        optimize_pocket_hs: bool = False,
        distance_constraint: float = 1.0,
    ):
        # create complex from pocket residues and ligand including cleaning up stuff and sanitizing
        ligand_mol = ligand_from_mol(ligand, add_hydrogens=add_ligand_hs)
        pocket_mol = self.pocket_from_pdb(
            pdb_file,
            ligand_mol,
            process_pocket=process_pocket,
            add_hydrogens=optimize_pocket_hs,
        )

        cpx = Chem.CombineMols(ligand_mol, pocket_mol)
        Chem.SanitizeMol(cpx)
        # parametrize ff
        props = rdForceFieldHelpers.MMFFGetMoleculeProperties(cpx, "MMFF94s")
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(
            cpx, props, ignoreInterfragInteractions=False
        )
        if ff is None:
            print(
                f"Returning non-optimized ligand. Could not get MMFF properties for combined molecule with PDB {pdb_file}"
            )
            if return_pocket:
                return Chem.Mol(ligand_mol), pocket_mol
            return Chem.Mol(ligand_mol)
        # constrain position of heavy atoms and certain hydrogens
        num_ligand_atoms = ligand_mol.GetNumAtoms()
        for atom in cpx.GetAtoms():
            if only_ligand_hs:
                # constrain everything except ligand hydrogens
                if atom.GetIdx() < num_ligand_atoms:
                    if atom.GetAtomicNum() != 1:
                        ff.AddFixedPoint(atom.GetIdx())
            else:
                if atom.GetIdx() < num_ligand_atoms and atom.GetSymbol() != "H":
                    ff.MMFFAddPositionConstraint(
                        atom.GetIdx(), distance_constraint, 999.0
                    )

            # constrain pocket atoms
            if atom.GetIdx() >= num_ligand_atoms:
                if optimize_pocket_hs:
                    if atom.GetAtomicNum() != 1:
                        ff.AddFixedPoint(atom.GetIdx())
                    # constrain hydrogen bound to backbone nitrogen fragment
                    elif (
                        atom.GetNeighbors()[0].GetPDBResidueInfo().GetName().strip()
                        == "N"
                        and atom.GetIdx() >= num_ligand_atoms
                    ):
                        ff.AddFixedPoint(atom.GetIdx())
                else:
                    ff.AddFixedPoint(atom.GetIdx())

        # minimize
        ff.Initialize()
        ff.Minimize(maxIts=max_iter)
        # update coordinates
        cpx_conf = cpx.GetConformer()
        lig_conf = ligand_mol.GetConformer()
        prot_conf = pocket_mol.GetConformer()
        lig_idx = 0
        prot_idx = 0
        for atom in cpx.GetAtoms():
            idx = atom.GetIdx()
            xyz = cpx_conf.GetAtomPosition(idx)
            if idx < num_ligand_atoms:
                lig_conf.SetAtomPosition(lig_idx, xyz)
                lig_idx += 1
            else:
                prot_conf.SetAtomPosition(prot_idx, xyz)
                prot_idx += 1

        if return_pocket:
            return Chem.Mol(ligand_mol), pocket_mol
        return Chem.Mol(ligand_mol)

    def pocket_from_pdb(
        self,
        protein_file: str,
        ligand_mol=None,
        process_pocket: bool = False,
        add_hydrogens: bool = False,
    ):
        """
        Read a protein PDB file, optionally add hydrogens, and return a prolif pocket molecule
        Args:
            protein_file: path to the protein PDB file
            ligand_mol: a prolif molecule of the ligand, if None, will not process the pocket
            process_pocket: whether to process the pocket around the ligand
            add_hydrogens: whether to add hydrogens to the protein structure
        Return: a prolif pocket molecule
        """

        if add_hydrogens:
            pdb_file = pdb.PDBFile.read(str(protein_file))
            read_fn = pdb.get_structure
            extra = ["charge"]
            structure = read_fn(
                pdb_file, model=1, extra_fields=extra, include_bonds=True
            )
            structure, mask = hydride.add_hydrogen(structure)
            structure.coord = hydride.relax_hydrogen(structure)
            protein_file = Path(protein_file.replace(".pdb", "with_hs_optim.pdb"))
            pdb_file = pdb.PDBFile()
            pdb.set_structure(pdb_file, structure)
            pdb_file.write(protein_file)
        if process_pocket:
            protein_mol = Chem.MolFromPDBFile(
                str(protein_file), removeHs=False, proximityBonding=False
            )
            return self.pocket_from_mol(protein_mol, ligand_mol)
        else:
            return prepare_prolif_mols(pdb_file=protein_file)

    def pocket_from_mol(self, protein_mol: Chem.Mol, ligand_mol):
        """
        Create a prolif pocket molecule from a protein molecule and a ligand molecule.
        Args:
            protein_mol: RDKit molecule of the protein
            ligand_mol: Prolif molecule of the ligand
        Returns: a prolif pocket molecule
        """

        self._skipped = []
        pocket_mol = self.subset_around_ligand(protein_mol, ligand_mol)
        assert (
            pocket_mol.GetNumAtoms() > 0
        ), f"No atoms within {self.pocket_cutoff} of ligand!"

        residues = self.group_by_resid(pocket_mol)
        for resid, atoms_subset in residues.items():
            pocket_mol = self.assign_bond_orders(pocket_mol, resid, atoms_subset)

        if self._skipped:
            with Chem.RWMol(pocket_mol) as mw:
                for atom in self._skipped:
                    mw.RemoveAtom(atom.GetIdx())
            pocket_mol = mw.GetMol()
            pocket_mol.UpdatePropertyCache(False)

        self.assign_charges(pocket_mol)
        try:
            Chem.SanitizeMol(pocket_mol, catchErrors=not self.sanitize)
        except Chem.KekulizeException:
            try:
                Chem.Kekulize(pocket_mol, clearAromaticFlags=True)
                Chem.SanitizeMol(pocket_mol, catchErrors=not self.sanitize)
            except Chem.KekulizeException:
                pocket_mol = Chem.RemoveHs(pocket_mol)
                Chem.Kekulize(pocket_mol, clearAromaticFlags=True)
                Chem.SanitizeMol(pocket_mol, catchErrors=not self.sanitize)

        return plf.Molecule(pocket_mol)

    def _subset_from_residues(self, mol: Chem.Mol, residues) -> Chem.Mol:
        """Get a subset of the molecule that contains only the atoms from the specified residues.
        Args:
            mol: RDKit molecule
            residues: a set of ResidueId objects to include in the subset : set[plf.ResidueId]
        Returns: a new RDKit molecule containing only the atoms from the specified residues
        """
        residues_to_atoms = self.group_by_resid(mol, subset=residues)
        atoms_subset = []
        for resid in residues:
            try:
                atoms = residues_to_atoms[resid]
            except KeyError:
                # HIS residue in input file might have been renamed to HID/HIP
                if resid.name != "HIS":
                    raise
                for resname in ("HID", "HIP"):
                    try:
                        atoms = residues_to_atoms[
                            plf.ResidueId(resname, resid.number, resid.chain)
                        ]
                    except KeyError:
                        continue
                    break
                else:
                    raise
            atoms_subset.extend(atoms)
        skipped = {a.GetIdx() for a in self._skipped}
        atoms_subset = list(filter(lambda a: a.GetIdx() not in skipped, atoms_subset))
        self._skipped.clear()
        return self.mol_from_subset(mol, atoms_subset)

    def group_by_resid(self, mol: Chem.Mol, subset: Optional = None):
        """Group atoms in the molecule by their residue ID.
        Args:
            mol: RDKit molecule
            subset: a set of ResidueId objects to include in the subset, if None, all residues are included set[plf.ResidueId]
        Returns: a dictionary mapping ResidueId to a list of atoms in that residue  -> dict[plf.ResidueId, list[Chem.Atom]]
        """
        residues: defaultdict[plf.ResidueId, list[Chem.Atom]] = defaultdict(list)
        for atom in mol.GetAtoms():
            resid = plf.ResidueId.from_atom(atom)
            if subset is None or resid in subset:
                residues[resid].append(atom)
        return dict(self._fix_non_standard(residues))

    def subset_around_ligand(self, prot_mol: Chem.Mol, ligand_mol) -> Chem.Mol:
        """Get a subset of the protein molecule that is within a certain distance
        from the ligand molecule.
        Args:
            prot_mol: RDKit molecule of the protein
            ligand_mol: Prolif molecule of the ligand
        Returns: a subset of the protein molecule that is within the pocket_cutoff distance from the ligand
        molecule.
        """
        tree = cKDTree(prot_mol.GetConformer().GetPositions())
        ix = tree.query_ball_point(ligand_mol.xyz, self.pocket_cutoff)
        ix = {i for lst in ix for i in lst}
        pocket_resids = {
            plf.ResidueId.from_atom(prot_mol.GetAtomWithIdx(i)) for i in ix
        }
        return self._subset_from_residues(prot_mol, pocket_resids)

    def mol_from_subset(self, refmol: Chem.Mol, atoms: list[Chem.Atom]) -> Chem.Mol:
        mw = Chem.RWMol()
        refconf = refmol.GetConformer()
        conf = Chem.Conformer(len(atoms))
        for atom in atoms:
            ix = mw.AddAtom(atom)
            mw.GetAtomWithIdx(ix).SetUnsignedProp("_idx", atom.GetIdx())
            conf.SetAtomPosition(ix, refconf.GetAtomPosition(atom.GetIdx()))
        mol = mw.GetMol()
        mol.AddConformer(conf, assignId=True)
        return mol

    def _fix_non_standard(
        self,
        residues,
    ):
        """Fix non-standard residues in the molecule, e.g. HIS residues that have been renamed to HID/HIP.
        Args:
            residues: a dictionary mapping ResidueId to a list of atoms in that residue : defaultdict[plf.ResidueId, list[Chem.Atom]]
        Returns: an iterator over tuples of ResidueId and list of atoms in that residue, with non-standard residues fixed
                     -> Iterator[tuple[plf.ResidueId, list[Chem.Atom]]]
        """
        for resid, atoms in residues.items():
            if resid.name == "HIS":
                # only HIS and HID have templates, the rest will use the RDKitConverter
                # from MDAnalysis
                names = {
                    atom.GetPDBResidueInfo().GetName().strip(): atom for atom in atoms
                }
                rename_resid = None
                rename_atoms = None
                if len({"1HD2", "2HD2"}.intersection(names)) == 2:
                    # weird non-aromatic histidine with sp3 CD2 from yasara minimization
                    warnings.warn(f"Non-aromatic histidine {resid!s} detected, fixing.")
                    # delete extra H on CD2, reposition remaining one on ring plane
                    # with corrected distance, and rename atoms to correspond to
                    # their actual position in the ring system
                    if "2HD2" in names:
                        self._skipped.append(names["2HD2"])
                    if "HG" in names:
                        self._skipped.append(names["HG"])
                    if "HE2" in names:
                        rename_resid = "HID"

                    # modify 1HD2 position so that distance CD2-1HD2 = 1.09
                    conf = names["1HD2"].GetOwningMol().GetConformer()
                    coords = np.array(
                        [
                            conf.GetAtomPosition(names["1HD2"].GetIdx()),
                            conf.GetAtomPosition(names["2HD2"].GetIdx()),
                        ]
                    )
                    cd2xyz = np.array(conf.GetAtomPosition(names["CD2"].GetIdx()))
                    v = np.mean(coords, axis=0) - cd2xyz
                    xyz = cd2xyz + (1.09 * v / np.linalg.norm(v))
                    conf.SetAtomPosition(names["1HD2"].GetIdx(), xyz.tolist())

                    rename_atoms = {
                        "1HD2": " HD2",
                        "ND1": " NE2",
                        "NE2": " ND1",
                        "HD1": " HE2",
                        "HE2": " HD1",
                    }

                elif "HD1" in names:
                    if "HE2" in names:
                        resid.name = "HIP"
                        rename_resid = "HIP"
                    else:
                        resid.name = "HID"
                        rename_resid = "HID"

                # check for inversion of ND1 and NE2 by yasara...
                if rename_atoms is None:
                    conf = names["CG"].GetOwningMol().GetConformer()
                    cg_nd1_dist = conf.GetAtomPosition(names["CG"].GetIdx()).Distance(
                        conf.GetAtomPosition(names["ND1"].GetIdx())
                    )
                    if cg_nd1_dist >= 2:
                        rename_atoms = {
                            "ND1": " NE2",
                            "HD1": " HE2",
                            "NE2": " ND1",
                            "HE2": " HD1",
                        }

                if rename_resid:
                    for atom in atoms:
                        atom.GetPDBResidueInfo().SetResidueName(rename_resid)
                        atom.SetFormalCharge(0)

                if rename_atoms:
                    for atom_name, new_name in rename_atoms.items():
                        if atom := names.get(atom_name):
                            atom.GetPDBResidueInfo().SetName(new_name)

            yield resid, atoms

    def assign_bond_orders(
        self, mol: Chem.Mol, resid, atoms_subset: list[Chem.Atom]
    ) -> Chem.Mol:
        """Assign bond orders to the atoms in the subset of the molecule.
        Args:
            mol: RDKit molecule
            resid: ResidueId of the residue to assign bond orders for : plf.ResidueId
            atoms_subset: list of atoms in the subset to assign bond orders for
        Returns: a new RDKit molecule with the bond orders assigned
        """
        # reuse HIS block for HIP
        resname = "HIS" if resid.name == "HIP" else resid.name
        try:
            # infer sidechain bonds from template
            block = pdbinf.STANDARD_AA_DOC[resname]
        except KeyError:
            from rdkit.Chem import GetPeriodicTable

            _periodic_table = GetPeriodicTable()

            # skip ions
            if len(atoms_subset) == 1:
                # strip lone hydrogens and other atoms that aren't delt with by rdkit
                if self.strip_invalid and (
                    (ion := atoms_subset[0]).GetAtomicNum() == 1
                    or _periodic_table.GetDefaultValence(ion.GetAtomicNum()) == -1
                ):
                    warnings.warn(
                        f"Found lone {ion.GetSymbol()} atom for residue {resid!s},"
                        " stripping it."
                    )
                    self._skipped.append(ion)
                return mol
            # infer from topology with explicit Hs
            mol = self.infer_bond_orders_subset(mol, atoms_subset)
        else:
            span = [a.GetIdx() for a in atoms_subset]
            mol, _ = pdbinf._pdbinf.assign_intra_props(mol, span, block)
        mol.UpdatePropertyCache(False)
        return mol

    def infer_bond_orders_subset(
        self,
        mol: Chem.Mol,
        atoms: list[Chem.Atom],
    ) -> Chem.Mol:
        bond: Chem.Bond

        # make separate subset mol
        subset = self.mol_from_subset(mol, atoms)
        # assign bonds
        rdDetermineBonds.DetermineConnectivity(subset)
        # assign bond orders
        try:
            subset = self.mdanalysis_inferer(subset)
        except Exception:
            if not self.strip_invalid:
                raise
            warnings.warn(
                "Unable to determine bond orders for"
                f" {plf.ResidueId.from_atom(atoms[0])!s}, stripping it."
            )
            self._skipped.extend(atoms)
            return mol

        # sanity check: charged carbon atom
        if any(
            atom.GetAtomicNum() == 6 and atom.GetFormalCharge() != 0
            for atom in subset.GetAtoms()
        ):
            msg = (
                "Invalid bond orders detected for"
                f" {plf.ResidueId.from_atom(atoms[0])!s}:"
                f" SMILES={Chem.MolToSmiles(subset)!r}"
            )
            if not self.strip_invalid:
                raise ValueError(msg)
            warnings.warn(f"{msg}, stripping it.")
            self._skipped.extend(atoms)
            return mol

        # transfer to mol
        refmol = Chem.RWMol(mol)
        for bond in subset.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            refmol.AddBond(
                a1.GetUnsignedProp("_idx"),
                a2.GetUnsignedProp("_idx"),
                bond.GetBondType(),
            )
        return refmol.GetMol()

    def mdanalysis_inferer(
        self, mol: Chem.Mol, sanitize: bool = True, reorder: bool = True
    ) -> Chem.Mol:

        from MDAnalysis.converters.RDKitInferring import MDAnalysisInferrer

        try:
            MDAnalysisInferrer._infer_bo_and_charges(mol)
            mol = MDAnalysisInferrer(sanitize=sanitize)._standardize_patterns(mol)
        except Exception:
            if sanitize:
                raise
        if reorder:
            order = np.argsort(
                [atom.GetUnsignedProp("_idx") for atom in mol.GetAtoms()]
            )
            mol = Chem.RenumberAtoms(mol, order.astype(int).tolist())
        return mol

    def assign_charges(self, mol: Chem.Mol) -> None:
        from MDAnalysis.converters.RDKitInferring import MDAnalysisInferrer
        from rdkit.Chem import GetPeriodicTable

        _periodic_table = GetPeriodicTable()
        MONATOMIC_CATION_CHARGES = MDAnalysisInferrer.MONATOMIC_CATION_CHARGES

        cysteines = {"CYS", "CYX"}

        for atom in mol.GetAtoms():
            atom.SetNoImplicit(True)
            mi = atom.GetPDBResidueInfo()
            pdb_name = mi.GetName().strip()

            if pdb_name in self.radical_replaces_charge:
                unpaired = (
                    _periodic_table.GetDefaultValence(atom.GetAtomicNum())
                    - atom.GetTotalValence()
                )
                if unpaired < 0:
                    # with RFAA, N-term nitrogen is not capped and has 3 explicit H
                    # so require a formal charge
                    atom.SetFormalCharge(-unpaired)
                else:
                    atom.SetFormalCharge(0)
                    atom.SetNumRadicalElectrons(unpaired)

            elif pdb_name == "SG" and mi.GetResidueName() in cysteines:
                resid = plf.ResidueId.from_atom(atom)
                for na in atom.GetNeighbors():
                    if (
                        nr := plf.ResidueId.from_atom(na)
                    ) != resid and nr.name in cysteines:
                        # S involved in cysteine bridge shouldn't be charged
                        atom.SetFormalCharge(0)
                        atom.SetNumRadicalElectrons(1)

            elif pdb_name == "ND1" and mi.GetResidueName() == "HIP":
                atom.SetFormalCharge(1)

            else:
                if (
                    atom.GetDegree() == 0
                    and atom.GetAtomicNum() in MONATOMIC_CATION_CHARGES
                ):
                    chg = MONATOMIC_CATION_CHARGES[atom.GetAtomicNum()]
                else:
                    chg = atom.GetTotalValence() - _periodic_table.GetDefaultValence(
                        atom.GetAtomicNum()
                    )
                atom.SetFormalCharge(chg)
                atom.SetNumRadicalElectrons(0)

            mol.UpdatePropertyCache(False)


def to_prolif(pdb_file):
    """Convert a PDB file to a prolif Molecule object"""
    try:
        import MDAnalysis as mda

        pocket = mda.Universe(str(pdb_file), guess_bonds=True)
        pocket = plf.Molecule.from_mda(pocket)
    except Exception:
        protein_mol = Chem.MolFromPDBFile(
            str(pdb_file), removeHs=False, proximityBonding=True
        )
        pocket = plf.Molecule.from_rdkit(protein_mol)
    return pocket


def prepare_prolif_mols(
    ligand: Chem.Mol = None,
    pdb_file: str = None,
    lig_resname="LIG",
    lig_resnumber=-1,
    add_hydrogens=False,
):
    """Create prolif molecules for pocket and ligand.
    Args:
        ligand: RDKit molecule of the ligand, if None, will not process the ligand
        pdb_file: path to the protein PDB file, if None, will not process the pocket
        lig_resname: residue name for the ligand, default is "LIG"
        lig_resnumber: residue number for the ligand, default is -1
        add_hydrogens: whether to add hydrogens to the ligand and pocket
    Returns:
        tuple of prolif Molecule objects for pocket and ligand
    """

    pocket_mol = to_prolif(pdb_file) if pdb_file is not None else None
    # ligand_mol = (
    #     plf.Molecule.from_rdkit(ligand, resname=lig_resname, resnumber=lig_resnumber)
    #     if ligand is not None
    #     else None
    # )
    ligand_mol = (
        ligand_from_mol(ligand, add_hydrogens=add_hydrogens)
        if ligand is not None
        else None
    )

    if pocket_mol is not None and ligand_mol is not None:
        return pocket_mol, ligand_mol
    elif pocket_mol is not None:
        return pocket_mol
    elif ligand_mol is not None:
        return ligand_mol
    else:
        raise ValueError("Either ligand or pocket must be provided.")


def ligand_from_mol(mol: Chem.Mol, add_hydrogens: bool = False):
    try:
        if isinstance(mol, Chem.Mol):
            with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
                sdf_path = tmp.name
                with Chem.SDWriter(sdf_path) as w:
                    w.write(mol)
                mol = plf.sdf_supplier(sdf_path)[0]
        elif isinstance(mol, str):
            mol = plf.sdf_supplier(mol)[0]
        else:
            raise ValueError(f"Invalid input type {type(mol)}")

        if add_hydrogens:
            resid: plf.ResidueId = mol[0].resid
            mol = Chem.AddHs(mol, addCoords=True)
            # bug with addResidueInfo in some cases so add manually
            for atom in mol.GetAtoms():
                if atom.GetPDBResidueInfo() is None:
                    mi = Chem.AtomPDBResidueInfo(
                        f" {atom.GetSymbol():<3.3}",
                        residueName=resid.name,
                        residueNumber=resid.number,
                        chainId=resid.chain or "",
                    )
                    atom.SetMonomerInfo(mi)
        return plf.Molecule(mol)
    except Exception as e:
        print(f"Failed to create prolif molecule from mol: {e}. Skipping ligand.")
        return


def add_and_optimize_hs(
    lig: Chem.Mol,
    pdb_file: str,
    optimizer: LigandPocketOptimization,
    optim_method: str = "prolif_mmff",
    process_pocket: bool = False,
    optimize_pocket_hs: bool = False,
):
    try:
        return optimizer(
            lig,
            pdb_file,
            method=optim_method,
            process_pocket=process_pocket,
            optimize_pocket_hs=optimize_pocket_hs,
            only_ligand_hs=True,
        )
    except Exception:
        return None


def optimize_ligand_in_pocket(
    ligand: Chem.Mol,
    pdb_file: str,
    optimizer: LigandPocketOptimization,
    method: str = "prolif_mmff",
    add_ligand_hs: bool = True,
    process_pocket: bool = False,
    optimize_pocket_hs: bool = False,
    only_ligand_hs: bool = False,
):
    """Optimize ligand in the context of the protein pocket using the specified method.
    Args:
        ligand: RDKit molecule of the ligand
        pdb_file: path to the protein PDB file
        method: optimization method, either "prolif_mmff" or "mmff_add_ligand_hs"
        process_pocket: whether to process the pocket around the ligand
        optimize_pocket_hs: whether to optimize pocket hydrogens
    Returns:
        optimized RDKit molecule of the ligand
    """
    try:
        return optimizer(
            ligand,
            pdb_file,
            method=method,
            add_ligand_hs=add_ligand_hs,
            process_pocket=process_pocket,
            optimize_pocket_hs=optimize_pocket_hs,
            only_ligand_hs=only_ligand_hs,
        )
    except Exception as e:
        logging.warning(f"Ligand optimization failed: {e}")
        logging.warning("Returning non-optimized ligand.")
        return ligand


# ==============================================================================
# ==============================================================================
# Pocket-constrained Ligand minimization!
# ==============================================================================
# ==============================================================================


# ==============================================================================
# Pocket Extraction
# ==============================================================================


def extract_pocket_mol(
    protein_filepath: str,
    native_ligand: Mol,
    distance_from_ligand: float = 5.0,
    ligand_resname: str = "UNL",
) -> Optional[Mol]:
    """
    Extract pocket residues around the native ligand as an RDKit molecule.

    Args:
        protein_filepath: Path to the protein PDB file.
        native_ligand: RDKit molecule of the native ligand (used to define pocket center).
        distance_from_ligand: Distance threshold in Angstroms for pocket extraction.
        ligand_resname: Residue name to assign to the ligand in the merged universe.

    Returns:
        RDKit molecule representing the pocket, or None if extraction fails.
    """
    try:
        universe = mda.Universe(protein_filepath)
        ligand = mda.Universe(native_ligand)
        ligand.add_TopologyAttr("resname", [ligand_resname])

        complx = mda.Merge(universe.atoms, ligand.atoms)

        # Select protein atoms within distance of ligand, excluding hydrogens
        selections = [
            "protein",
            f"around {distance_from_ligand} resname {ligand_resname}",
            "not type H",
        ]
        selection = "(" + ") and (".join(selections) + ")"
        atom_group: mda.AtomGroup = complx.select_atoms(selection)

        pocket_mol = None
        if len(atom_group) > 10:
            # Build selection for complete residues
            segids = {}
            for residue in atom_group.residues:
                segid = residue.segid
                resid = residue.resid
                if segid in segids:
                    segids[segid].append(resid)
                else:
                    segids[segid] = [resid]

            selections = []
            for segid, resids in segids.items():
                resids_str = " ".join([str(resid) for resid in set(resids)])
                selections.append(f"((resid {resids_str}) and (segid {segid}))")

            pocket_selection = " or ".join(selections)
            protein_pocket: mda.AtomGroup = universe.select_atoms(pocket_selection)
            pocket_mol = protein_pocket.atoms.convert_to("RDKIT")
        else:
            logging.warning(
                "Pocket quite small (<10 atoms), falling back to PDB parsing"
            )

        # Fallback: try to load directly from PDB
        if pocket_mol is None:
            pocket_mol = Chem.MolFromPDBFile(
                protein_filepath, removeHs=False, proximityBonding=False
            )

        return pocket_mol

    except Exception as e:
        logging.warning(f"Failed to extract pocket: {e}")
        return None


# ==============================================================================
# Complex Minimizer
# ==============================================================================


class ComplexMinimizer:
    """
    Minimizes a ligand within a protein pocket context using MMFF force field.

    The protein pocket atoms are kept fixed while the ligand is minimized
    with position constraints on heavy atoms.
    """

    def __init__(
        self,
        pocket_mol: Mol,
        n_steps: int = 200,
        distance_constraint: float = 1.0,
    ) -> None:
        """
        Initialize the minimizer.

        Args:
            pocket_mol: RDKit molecule representing the pocket (with hydrogens).
            n_steps: Maximum number of minimization steps.
            distance_constraint: Position constraint distance in Angstroms.
        """
        self.pocket_mol = pocket_mol
        self.n_steps = n_steps
        self.distance_constraint = distance_constraint

    def minimize_ligand(
        self,
        ligand_mol: Mol,
        add_hs: bool = True,
        ignore_pocket: bool = False,
    ) -> Optional[Mol]:
        """
        Minimize a ligand within the pocket context.

        Args:
            ligand_mol: RDKit molecule of the ligand to minimize.
            ignore_pocket: If True, minimize ligand in isolation.

        Returns:
            Minimized ligand molecule with hydrogens, or None if minimization fails.
        """
        if add_hs:
            ligand = Chem.AddHs(ligand_mol, addCoords=True)
        else:
            ligand = ligand_mol

        if not ignore_pocket and self.pocket_mol is not None:
            complx = Chem.CombineMols(self.pocket_mol, ligand)
            n_pocket_atoms = self.pocket_mol.GetNumAtoms()
        elif ignore_pocket:
            logging.warning("Ignoring pocket for minimization.")
            complx = ligand
            n_pocket_atoms = 0
        else:
            logging.warning("No pocket provided for minimization.")
            return None

        try:
            Chem.SanitizeMol(complx)
        except Exception as e:
            logging.warning(f"Failed to sanitize complex: {e}")
            return None

        try:
            mol_properties = AllChem.MMFFGetMoleculeProperties(
                complx, mmffVariant="MMFF94s"
            )
            if mol_properties is None:
                logging.warning("Could not get MMFF properties for complex")
                return None

            mmff = AllChem.MMFFGetMoleculeForceField(
                complx,
                mol_properties,
                confId=0,
                nonBondedThresh=10.0,
                ignoreInterfragInteractions=False,
            )
            if mmff is None:
                logging.warning("Could not create MMFF force field")
                return None

            mmff.Initialize()

            # Add position constraints on ligand heavy atoms
            for idx in range(n_pocket_atoms, complx.GetNumAtoms()):
                atom = complx.GetAtomWithIdx(idx)
                if atom.GetSymbol() != "H":
                    mmff.MMFFAddPositionConstraint(idx, self.distance_constraint, 999.0)

            # Fix pocket atoms in place
            if not ignore_pocket and n_pocket_atoms > 0:
                for i in range(n_pocket_atoms):
                    mmff.AddFixedPoint(i)

            # Perform minimization
            result = mmff.Minimize(maxIts=self.n_steps)
            if result != 0:
                logging.debug("Minimization did not converge")

            # Extract minimized ligand
            minimized_frags = Chem.GetMolFrags(complx, asMols=True)
            minimized_ligand = minimized_frags[-1]

            return minimized_ligand

        except Exception as e:
            logging.warning(f"MMFF minimization exception: {e}")
            logging.warning("Returning non-optimized ligand.")
            return ligand


def setup_minimize(
    gen_lig: Mol,
    ref_lig: Mol,
    pdb_file: str,
    add_ligand_hs: bool = True,
    pocket_distance: float = 5.0,
    n_steps: int = 1000,
    distance_constraint: float = 1.0,
) -> Optional[Mol]:
    """
    Set up and run minimization for a generated ligand.

    Args:
        gen_lig: Generated ligand molecule to minimize.
        ref_lig: Reference ligand used to define the pocket.
        pdb_file: Path to the protein PDB file.
        pocket_distance: Distance from ligand for pocket extraction (Angstroms).
        n_steps: Maximum number of minimization steps.
        distance_constraint: Position constraint distance in Angstroms.

    Returns:
    Molecules after minimization -> list[Optional[Mol]]
    """
    # Extract pocket around reference ligand
    pocket_mol = extract_pocket_mol(
        protein_filepath=pdb_file,
        native_ligand=ref_lig,
        distance_from_ligand=pocket_distance,
    )

    if pocket_mol is None:
        logging.warning(f"Could not extract pocket from {pdb_file}")
        logging.warning("Returning non-optimized ligands.")
        return gen_lig

    # Run minimization
    minimizer = ComplexMinimizer(
        pocket_mol,
        n_steps=n_steps,
        distance_constraint=distance_constraint,
    )
    mol_optim = minimizer.minimize_ligand(gen_lig, add_hs=add_ligand_hs)
    return mol_optim


def setup_minimize_list(
    gen_ligs: list[Mol],
    ref_lig: Mol,
    pdb_file: str,
    add_ligand_hs: bool = True,
    pocket_distance: float = 5.0,
    n_steps: int = 1000,
    distance_constraint: float = 1.0,
) -> Optional[Mol]:
    """
    Set up and run minimization for a list of generated ligands.

    Args:
        gen_ligs: List of generated ligand molecules to minimize.
        ref_lig: Reference ligand used to define the pocket.
        pdb_file: Path to the protein PDB file.
        pocket_distance: Distance from ligand for pocket extraction (Angstroms).
        n_steps: Maximum number of minimization steps.
        distance_constraint: Position constraint distance in Angstroms.

    Returns:
    Molecules after minimization -> list[Optional[Mol]]
    """
    # Extract pocket around reference ligand
    pocket_mol = extract_pocket_mol(
        protein_filepath=pdb_file,
        native_ligand=ref_lig,
        distance_from_ligand=pocket_distance,
    )

    if pocket_mol is None:
        logging.warning(f"Could not extract pocket from {pdb_file}.")
        logging.warning("Returning non-optimized ligands.")
        return gen_ligs

    # Run minimization
    minimizer = ComplexMinimizer(
        pocket_mol,
        n_steps=n_steps,
        distance_constraint=distance_constraint,
    )
    ligs_optim = [
        minimizer.minimize_ligand(gen_lig, add_hs=add_ligand_hs) for gen_lig in gen_ligs
    ]
    return ligs_optim

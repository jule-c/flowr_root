import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import pi

import flowr.util.functional as smolF
from flowr.models.semla import LengthsMLP, adj_to_attn_mask


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim, bias=bias),
            torch.nn.SiLU(),
            torch.nn.Linear(out_dim, out_dim, bias=bias),
        )

    def forward(self, x):
        return self.fc(x)


class _CoordNorm(nn.Module):
    def __init__(self, d_equi, zero_com=True, eps=1e-6):
        super().__init__()

        self.d_equi = d_equi
        self.zero_com = zero_com
        self.eps = eps

        self.set_weights = torch.nn.Parameter(torch.ones((1, 1, 1, d_equi)))

    def forward(self, coord_sets, node_mask):
        """Apply coordinate normlisation layer

        Args:
            coord_sets (torch.Tensor): Coordinate tensor, shape [B, N, 3, d_equi]
            node_mask (torch.Tensor): Mask for nodes, shape [B, N], 1 for real, 0 otherwise

        Returns:
            torch.Tensor: Normalised coords, shape [B, N, 3, d_equi]
        """

        if self.zero_com:
            coord_sets = smolF.zero_com(coord_sets, node_mask)

        n_atoms = node_mask.sum(dim=-1).view(-1, 1, 1, 1)
        lengths = torch.linalg.vector_norm(coord_sets, dim=2, keepdim=True)
        scaled_lengths = lengths.sum(dim=1, keepdim=True) / n_atoms
        coord_sets = (coord_sets * self.set_weights) / (scaled_lengths + self.eps)
        coord_sets = coord_sets * node_mask.unsqueeze(-1).unsqueeze(-1)

        return coord_sets

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)


# *****************************************************************************
# ******************************* Model  **************************************
# *****************************************************************************


class _EquivariantMLP(nn.Module):
    def __init__(self, d_equi, d_inv):
        super().__init__()

        self.node_proj = torch.nn.Sequential(
            torch.nn.Linear(d_equi + d_inv, d_equi),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_equi, d_equi),
            torch.nn.Sigmoid(),
        )
        self.coord_proj = torch.nn.Linear(d_equi, d_equi, bias=False)
        self.attn_proj = torch.nn.Linear(d_equi, d_equi, bias=False)

    def forward(self, equis, invs):
        """Pass data through the layer

        Assumes coords and node_feats have already been normalised

        Args:
            equis (torch.Tensor): Equivariant features, shape [B, N, 3, d_equi]
            invs (torch.Tensor): Invariant features, shape [B, N, d_inv]

        Returns:
            torch.Tensor: Updated equivariant features, shape [B, N, 3, d_equi]
        """

        lengths = torch.linalg.vector_norm(equis, dim=2)
        inv_feats = torch.cat((invs, lengths), dim=-1)

        # inv_feats shape [B, N, 1, d_equi]
        # proj_sets shape [B, N, 3, d_equi]
        inv_feats = self.node_proj(inv_feats).unsqueeze(2)
        proj_sets = self.coord_proj(equis)

        gated_equis = proj_sets * inv_feats
        equis_out = self.attn_proj(gated_equis)
        return equis_out


class _PairwiseMessages(torch.nn.Module):
    """Compute pairwise features for a set of query and a set of key nodes"""

    def __init__(
        self,
        d_equi,
        d_q_inv,
        d_kv_inv,
        d_message,
        d_out,
        d_ff,
        d_edge=None,
        include_distances=False,
        include_crossproducts=False,
    ):
        super().__init__()

        in_feats = (d_message * 2) + d_equi
        in_feats = in_feats + d_edge if d_edge is not None else in_feats
        in_feats = in_feats + d_equi if include_distances else in_feats
        in_feats = in_feats + d_equi if include_crossproducts else in_feats

        self.d_equi = d_equi
        self.d_edge = d_edge
        self.include_distances = include_distances
        self.include_crossproducts = include_crossproducts

        self.q_message_proj = torch.nn.Linear(d_q_inv, d_message)
        self.k_message_proj = torch.nn.Linear(d_kv_inv, d_message)

        self.message_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_feats, d_ff),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_ff, d_out),
        )

    def forward(
        self,
        q_equi,
        q_inv,
        k_equi,
        k_inv,
        edge_feats=None,
    ):
        """Produce messages between query and key

        Args:
            q_equi (torch.Tensor): Equivariant query features, shape [B, N_q, 3, d_equi]
            q_inv (torch.Tensor): Invariant query features, shape [B, N_q, d_q_inv]
            k_equi (torch.Tensor): Equivariant key features, shape [B, N_kv, 3, d_equi]
            k_inv (torch.Tensor): Invariant key features, shape [B, N_kv, 3, d_kv_inv]
            edge_feats (torch.Tensor): Edge features, shape [B, N_q, N_kv, d_edge]

        Returns:
            torch.Tensor: Message matrix, shape [B, N_q, N_k, d_out]
        """

        if edge_feats is not None and self.d_edge is None:
            raise ValueError(
                "edge_feats was provided but the model was initialised with d_edge as None."
            )

        if edge_feats is None and self.d_edge is not None:
            raise ValueError(
                "The model was initialised with d_edge but no edge feats were provided to forward fn."
            )

        q_equi_batched = q_equi.movedim(-1, 1).flatten(0, 1)
        k_equi_batched = k_equi.movedim(-1, 1).flatten(0, 1)

        dotprods = torch.bmm(q_equi_batched, k_equi_batched.transpose(1, 2))
        dotprods = dotprods.unflatten(0, (-1, self.d_equi)).movedim(1, -1)

        q_messages = (
            self.q_message_proj(q_inv).unsqueeze(2).expand(-1, -1, k_inv.size(1), -1)
        )
        k_messages = (
            self.k_message_proj(k_inv).unsqueeze(1).expand(-1, q_inv.size(1), -1, -1)
        )

        pairwise_feats = torch.cat((q_messages, k_messages, dotprods), dim=-1)

        if self.include_distances:
            vec_dists = q_equi.unsqueeze(2) - k_equi.unsqueeze(1)
            dists = torch.linalg.vector_norm(vec_dists, dim=3)
            pairwise_feats = torch.cat((pairwise_feats, dists), dim=-1)

        if self.include_crossproducts:
            # Compute cross products between query and key equivariant features
            # Shape: [B, N_q, N_kv, 3, d_equi]
            q_expanded = q_equi.unsqueeze(2).expand(-1, -1, k_equi.size(1), -1, -1)
            k_expanded = k_equi.unsqueeze(1).expand(-1, q_equi.size(1), -1, -1, -1)
            cross_products = torch.cross(
                q_expanded, k_expanded, dim=3
            )  # [B, N_q, N_kv, 3, d_equi]
            cross_norms = torch.linalg.vector_norm(
                cross_products, dim=3
            )  # [B, N_q, N_kv, d_equi]
            pairwise_feats = torch.cat((pairwise_feats, cross_norms), dim=-1)

        if edge_feats is not None:
            pairwise_feats = torch.cat((pairwise_feats, edge_feats), dim=-1)

        pairwise_messages = self.message_mlp(pairwise_feats)

        return pairwise_messages


class _EquiAttention(torch.nn.Module):
    def __init__(self, d_equi, eps=1e-6):
        super().__init__()

        self.d_equi = d_equi
        self.eps = eps

        self.coord_proj = torch.nn.Linear(d_equi, d_equi, bias=False)
        self.attn_proj = torch.nn.Linear(d_equi, d_equi, bias=False)

    def forward(self, v_equi, messages, adj_matrix):
        """Compute an attention update for equivariant features

        Args:
            v_equi (torch.Tensor): Coordinate tensor, shape [B, N_kv, 3, d_equi]
            messages (torch.Tensor): Messages tensor, shape [B, N_q, N_kv, d_equi]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [B, N_q, N_kv]

        Returns:
            torch.Tensor: Updates for equi features, shape [B, N_q, 3, d_equi]
        """

        proj_equi = self.coord_proj(v_equi)

        attn_mask = adj_to_attn_mask(adj_matrix)
        messages = messages + attn_mask.unsqueeze(3)
        attentions = torch.softmax(messages, dim=2)

        # Attentions shape now [B * d_equi, N_q, N_kv]
        # proj_equi shape now [B * d_equi, N_kv, 3]
        attentions = attentions.movedim(-1, 1).flatten(0, 1)
        proj_equi = proj_equi.movedim(-1, 1).flatten(0, 1)
        attn_out = torch.bmm(attentions, proj_equi)

        # Apply variance preserving updates as proposed in GNN-VPA (https://arxiv.org/abs/2403.04747)
        weights = torch.sqrt((attentions**2).sum(dim=-1))
        attn_out = attn_out * weights.unsqueeze(-1)

        attn_out = attn_out.unflatten(0, (-1, self.d_equi)).movedim(1, -1)
        return self.attn_proj(attn_out)


class _InvAttention(torch.nn.Module):
    def __init__(self, d_inv, n_attn_heads, d_inv_cond=None):
        super().__init__()

        d_inv_in = d_inv_cond if d_inv_cond is not None else d_inv

        d_head = d_inv_in // n_attn_heads

        if d_inv_in % n_attn_heads != 0:
            raise ValueError(
                f"n_attn_heads must divide d_inv or d_inv_cond (if provided) exactly."
            )

        self.d_inv = d_inv
        self.n_attn_heads = n_attn_heads
        self.d_head = d_head

        self.in_proj = torch.nn.Linear(d_inv_in, d_inv_in)
        self.out_proj = torch.nn.Linear(d_inv_in, d_inv)

    def forward(self, v_inv, messages, adj_matrix):
        """Accumulate edge messages to each node using attention-based message passing

        Args:
            v_inv (torch.Tensor): Node feature tensor, shape [B, N_kv, d_inv or d_inv_cond if provided]
            messages (torch.Tensor): Messages tensor, shape [B, N_q, N_kv, d_message]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [B, N_q, N_kv]

        Returns:
            torch.Tensor: Updates to invariant features, shape [B, N_q, d_inv]
        """

        attn_mask = adj_to_attn_mask(adj_matrix)
        messages = messages + attn_mask.unsqueeze(-1)
        attentions = torch.softmax(messages, dim=2)

        proj_feats = self.in_proj(v_inv)
        head_feats = proj_feats.unflatten(-1, (self.n_attn_heads, self.d_head))

        # Put n_heads into the batch dim for both the features and the attentions
        # head_feats shape [B * n_heads, N_kv, d_head]
        # attentions shape [B * n_heads, N_q, N_kv]
        head_feats = head_feats.movedim(-2, 1).flatten(0, 1)
        attentions = attentions.movedim(-1, 1).flatten(0, 1)
        attn_out = torch.bmm(attentions, head_feats)

        # Apply variance preserving updates as proposed in GNN-VPA (https://arxiv.org/abs/2403.04747)
        weights = torch.sqrt((attentions**2).sum(dim=-1))
        attn_out = attn_out * weights.unsqueeze(-1)

        attn_out = attn_out.unflatten(0, (-1, self.n_attn_heads))
        attn_out = attn_out.movedim(1, -2).flatten(2, 3)
        return self.out_proj(attn_out)


class SemlaSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_equi,
        d_inv,
        d_message,
        n_heads,
        d_ff,
        d_edge_in=None,
        d_edge_out=None,
        fixed_equi=False,
        include_distances=False,
        include_crossproducts=False,
        eps=1e-6,
    ):
        super().__init__()

        d_out = n_heads if fixed_equi else d_equi + n_heads
        d_out = d_out + d_edge_out if d_edge_out is not None else d_out

        messages = _PairwiseMessages(
            d_equi,
            d_inv,
            d_inv,
            d_message,
            d_out,
            d_ff,
            d_edge=d_edge_in,
            include_distances=include_distances,
            include_crossproducts=include_crossproducts,
        )

        inv_attn = _InvAttention(d_inv, n_attn_heads=n_heads)

        self.d_equi = d_equi
        self.n_heads = n_heads
        self.d_edge_in = d_edge_in
        self.d_edge_out = d_edge_out
        self.fixed_equi = fixed_equi

        self.messages = messages
        self.inv_attn = inv_attn

        if not fixed_equi:
            self.equi_attn = _EquiAttention(d_equi, eps=eps)

    def forward(self, equis, invs, edges, adj_matrix):
        """Compute output of self attention layer

        Args:
            equis (torch.Tensor): Equivariant features, shape [B, N, 3, d_equi]
            invs (torch.Tensor): Invariant features, shape [B, N, d_inv]
            edges (torch.Tensor): Edge feature matrix, shape [B, N, N, d_edge]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [B, N, N], 1 is connected, 0 otherwise

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Updates to equi features, inv feats, edge features
            Note that equi features are None if fixed_equi is specified, and edge features are None if d_edge_out
            is None. This ordering is used to maintain consistency with the ordering in other modules and to help to
            ensure that errors will be thrown if the wrong output is taken.
        """

        messages = self.messages(
            equis,
            invs,
            equis,
            invs,
            edge_feats=edges,
        )

        inv_messages = messages[..., : self.n_heads]
        inv_updates = self.inv_attn(invs, inv_messages, adj_matrix)

        equi_updates = None
        if not self.fixed_equi:
            equi_messages = messages[..., self.n_heads : self.n_heads + self.d_equi]
            equi_updates = self.equi_attn(equis, equi_messages, adj_matrix)

        edge_feats = None
        if self.d_edge_out is not None:
            edge_feats = messages[..., self.n_heads + self.d_equi :]

        return equi_updates, inv_updates, edge_feats


class SemlaCondAttention(torch.nn.Module):
    def __init__(
        self,
        d_equi,
        d_inv,
        d_message,
        n_heads,
        d_ff,
        d_inv_cond=None,
        d_edge_in=None,
        d_edge_out=None,
        include_distances=False,
        include_crossproducts=False,
        eps=1e-6,
    ):
        super().__init__()

        # Set the number of pairwise output features depending on whether edge features are generated or not
        d_out = d_equi + n_heads
        d_out = d_out if d_edge_out is None else d_out + d_edge_out

        # Use d_inv for the conditional inviariant features by default
        d_inv_cond = d_inv if d_inv_cond is None else d_inv_cond

        messages = _PairwiseMessages(
            d_equi,
            d_inv,
            d_inv_cond,
            d_message,
            d_out,
            d_ff,
            d_edge=d_edge_in,
            include_distances=include_distances,
            include_crossproducts=include_crossproducts,
        )

        equi_attn = _EquiAttention(d_equi, eps=eps)
        inv_attn = _InvAttention(d_inv, n_attn_heads=n_heads, d_inv_cond=d_inv_cond)

        self.d_equi = d_equi
        self.n_heads = n_heads
        self.d_edge_in = d_edge_in
        self.d_edge_out = d_edge_out

        self.messages = messages
        self.equi_attn = equi_attn
        self.inv_attn = inv_attn

    def forward(
        self,
        equis,
        invs,
        cond_equis,
        cond_invs,
        edges,
        adj_matrix,
        # coords=None,
        # coords_cond=None,
    ):
        """Compute output of conditional attention layer

        Args:
            equis (torch.Tensor): Equivariant features, shape [B, N, 3, d_equi]
            invs (torch.Tensor): Invariant features, shape [B, N, d_inv]
            cond_equis (torch.Tensor): Conditional equivariant features, shape [B, N_c, 3, d_equi]
            cond_invs (torch.Tensor): Conditional invariant features, shape [B, N_c, d_inv_cond]
            edges (torch.Tensor): Edge feature matrix, shape [B, N, N_c, d_edge]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [B, N, N_c], 1 is connected, 0 otherwise
            coords (torch.Tensor): Coordinate tensor, shape [B, N, 3]
            coords_cond (torch.Tensor): Conditional coordinate tensor, shape [B, N_c, 3]

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Updates to equi feats, inv feats, and edge feats,
            respectively. Note that the edge features will be None is d_edge_out is None.
        """

        messages = self.messages(
            equis,
            invs,
            cond_equis,
            cond_invs,
            edge_feats=edges,
            # coords_q=coords,
            # coords_k=coords_cond,
        )
        equi_messages = messages[..., : self.d_equi]
        inv_messages = messages[..., self.d_equi : self.d_equi + self.n_heads]

        edge_feats = None
        if self.d_edge_out is not None:
            edge_feats = messages[..., self.d_equi + self.n_heads :]

        equi_updates = self.equi_attn(cond_equis, equi_messages, adj_matrix)
        inv_updates = self.inv_attn(cond_invs, inv_messages, adj_matrix)

        return equi_updates, inv_updates, edge_feats


# *****************************************************************************
# ********************************* Semla Layer *******************************
# *****************************************************************************


class SemlaLayer(torch.nn.Module):
    """Core layer of the Semla architecture.

    The layer contains a self-attention component and a feedforward component, by default. To turn on the conditional
    -attention component in addition to the others, set d_inv_cond to the number of invariant features in the
    conditional input. Note that currently d_equi must be the same for both attention inputs.
    """

    def __init__(
        self,
        d_equi,
        d_inv,
        d_message,
        n_attn_heads,
        d_message_ff,
        d_inv_cond=None,
        d_self_edge_in=None,
        d_self_edge_out=None,
        d_cond_edge_in=None,
        d_cond_edge_out=None,
        fixed_equi=False,
        use_distances=False,
        use_crossproducts=False,
        intermediate_coord_updates=False,
        zero_com=False,
        eps=1e-6,
    ):
        super().__init__()

        if d_inv_cond is not None and fixed_equi:
            raise ValueError(
                "Equivariant features cannot be fixed when using conditional attention."
            )

        self.d_inv_cond = d_inv_cond
        self.d_self_edge_out = d_self_edge_out
        self.d_cond_edge_out = d_cond_edge_out
        self.fixed_equi = fixed_equi
        self.intermediate_coord_updates = intermediate_coord_updates

        # *** Self attention components ***
        self.self_attn_inv_norm = torch.nn.LayerNorm(d_inv)

        if not fixed_equi:
            self.self_attn_equi_norm = _CoordNorm(d_equi, zero_com=zero_com, eps=eps)

        self.self_attention = SemlaSelfAttention(
            d_equi,
            d_inv,
            d_message,
            n_attn_heads,
            d_message_ff,
            d_edge_in=d_self_edge_in,
            d_edge_out=d_self_edge_out,
            fixed_equi=fixed_equi,
            include_distances=use_distances,
            include_crossproducts=use_crossproducts,
            eps=eps,
        )

        # *** Cross attention components ***
        if d_inv_cond is not None:
            self.cond_attn_self_inv_norm = torch.nn.LayerNorm(d_inv)
            self.cond_attn_cond_inv_norm = torch.nn.LayerNorm(d_inv_cond)
            self.cond_attn_equi_norm = _CoordNorm(d_equi, zero_com=zero_com, eps=eps)

            self.cond_attention = SemlaCondAttention(
                d_equi,
                d_inv,
                d_message,
                n_attn_heads,
                d_message_ff,
                d_inv_cond=d_inv_cond,
                d_edge_in=d_cond_edge_in,
                d_edge_out=d_cond_edge_out,
                include_distances=use_distances,
                include_crossproducts=use_crossproducts,
                eps=eps,
            )

        # *** Feedforward components ***
        self.ff_inv_norm = torch.nn.LayerNorm(d_inv)
        self.inv_ff = LengthsMLP(d_inv, d_equi)

        if not fixed_equi:
            self.ff_equi_norm = _CoordNorm(d_equi, zero_com=zero_com, eps=eps)
            self.equi_ff = _EquivariantMLP(d_equi, d_inv)

        if intermediate_coord_updates:
            assert (
                not fixed_equi
            ), "Intermediate coordinate updates are not allowed with fixed equivariant features."
            self.coord_down_proj = torch.nn.Linear(d_equi, 1, bias=False)
            self.coord_up_proj = torch.nn.Linear(1, d_equi, bias=False)
            self.rbf_embed = RadialBasisEmbedding(
                d_edge=None,
                num_rbf=16,
                cutoff=5.0,
                learnable_cutoff=False,
                eps=1e-6,
                rbf_type="center",
            )
            self.edge_proj = torch.nn.Linear(
                (
                    16 + d_self_edge_out
                    if d_self_edge_out is not None
                    else 16 + d_self_edge_in
                ),
                d_self_edge_out if d_self_edge_out is not None else d_self_edge_in,
            )
            self.edge_norm = torch.nn.LayerNorm(
                d_self_edge_out if d_self_edge_out is not None else d_self_edge_in
            )

            if d_inv_cond is not None:
                self.rbf_embed_cond = RadialBasisEmbeddingPL(
                    d_edge=(None if d_cond_edge_out is not None else d_cond_edge_in),
                    num_rbf=16,
                    cutoff=5.0,
                    learnable_cutoff=False,
                    eps=1e-6,
                    rbf_type="center",
                )
                if d_cond_edge_out is not None:
                    self.edge_proj_cond = torch.nn.Linear(
                        16 + d_cond_edge_out, d_cond_edge_out
                    )
                    self.edge_norm_cond = torch.nn.LayerNorm(d_cond_edge_out)

    def forward(
        self,
        equis,
        invs,
        edges,
        adj_matrix,
        node_mask,
        coords=None,
        cond_coords=None,
        cond_equis=None,
        cond_invs=None,
        cond_edges=None,
        cond_node_mask=None,
        cond_adj_matrix=None,
    ):
        """Compute output of Semla layer
        Args:
            equis (torch.Tensor): Equivariant features, shape [B, N, 3, d_equi]
            invs (torch.Tensor): Invariant features, shape [B, N, d_inv]
            edges (torch.Tensor): Edge features, shape [B, N, N, d_self_edge_in]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [B, N, N], 1 is connected, 0 otherwise
            node_mask (torch.Tensor): Mask for nodes, shape [B, N], 1 for real, 0 otherwise
            cond_equis (torch.Tensor): Cond equivariant features, shape [B, N, 3, d_equi]
            cond_invs (torch.Tensor): Cond invariant features, shape [B, N, d_inv_cond]
            cond_edges (torch.Tensor): Edge features between self and cond, shape [B, N, N_c, d_cond_edge_in]
            cond_adj_matrix (torch.Tensor): Adj matrix to cond data, shape [B, N, N_c], 1 is connected, 0 otherwise

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            Updated equivariant features, updated invariant features, self pairwise features, self-conditional
            pairwise features. Note that self pairwise features will be None if d_self_edge_out is None, and self
            -conditional pairwise features will be None if d_cond_edge_out is None.
            Tensor shapes: [B, N, 3, d_equi], [B, N, d_inv], [B, N, N, d_self_edge_out], [B, N, N_c, d_cond_edge_out]
        """

        if self.d_inv_cond is not None and cond_equis is None:
            raise ValueError(
                "The layer was initialised with conditional attention but cond_equis is missing."
            )

        if self.d_inv_cond is not None and cond_invs is None:
            raise ValueError(
                "The layer was initialised with conditional attention but cond_invs is missing."
            )

        if self.d_inv_cond is not None and cond_adj_matrix is None:
            raise ValueError(
                "The layer was initialised with conditional attention but cond_adj_matrix is missing."
            )

        # *** Self attention component ***
        invs_norm = self.self_attn_inv_norm(invs)
        equis_norm = (
            equis if self.fixed_equi else self.self_attn_equi_norm(equis, node_mask)
        )
        equi_updates, inv_updates, self_edge_feats = self.self_attention(
            equis_norm,
            invs_norm,
            edges,
            adj_matrix,
        )

        invs = invs + inv_updates
        if not self.fixed_equi:
            equis = equis + equi_updates

        # *** Conditional attention component ***
        cond_edge_feats = None
        if self.d_inv_cond is not None:
            equis, invs, cond_edge_feats = self._compute_cond_attention(
                equis,
                invs,
                cond_equis,
                cond_invs,
                cond_edges,
                node_mask,
                cond_adj_matrix,
            )

        # *** Feedforward component ***
        invs_norm = self.ff_inv_norm(invs)
        equis_norm = equis if self.fixed_equi else self.ff_equi_norm(equis, node_mask)

        inv_update = self.inv_ff(equis_norm.movedim(-1, 1), invs_norm)
        invs = invs + inv_update

        if not self.fixed_equi:
            equi_update = self.equi_ff(equis_norm, invs_norm)
            equis = equis + equi_update

        if self.intermediate_coord_updates and not self.fixed_equi:
            equis, coords, self_edge_feats = self._compute_intermediate_updates(
                equis,
                coords,
                edges=self_edge_feats if self_edge_feats is not None else edges,
                node_mask=node_mask,
                adj_matrix=adj_matrix,
            )
            if self.d_inv_cond is not None:
                cond_edge_feats = self._compute_intermediate_cond_updates(
                    coords,
                    cond_coords,
                    cond_edges=cond_edge_feats,
                    node_mask=node_mask,
                    cond_node_mask=cond_node_mask,
                    cond_adj_matrix=cond_adj_matrix,
                )
        else:
            if self_edge_feats is None:
                self_edge_feats = edges

        return equis, invs, coords, self_edge_feats, cond_edge_feats

    def _compute_cond_attention(
        self,
        equis,
        invs,
        cond_equis,
        cond_invs,
        cond_edges,
        node_mask,
        cond_adj_matrix,
    ):
        self_invs_norm = self.cond_attn_self_inv_norm(invs)
        cond_invs_norm = self.cond_attn_cond_inv_norm(cond_invs)
        equis_norm = self.cond_attn_equi_norm(equis, node_mask)

        equi_updates, inv_updates, cond_edge_feats = self.cond_attention(
            equis_norm,
            self_invs_norm,
            cond_equis,
            cond_invs_norm,
            cond_edges,
            cond_adj_matrix,
        )

        equis = equis + equi_updates
        invs = invs + inv_updates

        return equis, invs, cond_edge_feats

    def _compute_intermediate_updates(
        self, equis, coords, edges, node_mask, adj_matrix
    ):
        coords = coords + self.coord_down_proj(equis).squeeze(-1)
        coords = coords * node_mask.unsqueeze(-1)
        equis = self.coord_up_proj(coords.unsqueeze(-1))

        edges = torch.cat([edges, self.rbf_embed(coords, mask=node_mask)], dim=-1)
        edges = self.edge_norm(self.edge_proj(edges)) * adj_matrix.unsqueeze(-1)

        return equis, coords, edges

    def _compute_intermediate_cond_updates(
        self,
        coords,
        cond_coords,
        cond_edges,
        node_mask,
        cond_node_mask,
        cond_adj_matrix,
    ):
        rbf_embeds = self.rbf_embed_cond(
            coords,
            cond_coords,
            ligand_mask=node_mask,
            pocket_mask=cond_node_mask,
        )
        cond_edges = (
            self.edge_norm_cond(
                self.edge_proj_cond(torch.cat([cond_edges, rbf_embeds], dim=-1))
            )
            if cond_edges is not None
            else rbf_embeds
        )
        cond_edges = cond_edges * cond_adj_matrix.unsqueeze(-1)
        return cond_edges

    def _axis_angle_to_rotation_matrix(self, axis_angle):
        """Convert axis-angle representation to rotation matrix using Rodrigues' formula"""
        # axis_angle: [B, N, 3]
        angle = torch.norm(axis_angle, dim=-1, keepdim=True)  # [B, N, 1]
        axis = axis_angle / (angle + 1e-8)  # [B, N, 3]

        cos_angle = torch.cos(angle)  # [B, N, 1]
        sin_angle = torch.sin(angle)  # [B, N, 1]

        # Skew-symmetric matrix
        K = self._skew_symmetric_matrix(axis)  # [B, N, 3, 3]

        # Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
        I = torch.eye(3, device=axis_angle.device).unsqueeze(0).unsqueeze(0)
        R = (
            I
            + sin_angle.unsqueeze(-1) * K
            + (1 - cos_angle).unsqueeze(-1)
            * torch.bmm(K.view(-1, 3, 3), K.view(-1, 3, 3)).view(*K.shape)
        )

        return R

    def _skew_symmetric_matrix(self, v):
        """Create skew-symmetric matrix from vector [x, y, z]"""
        B, N, _ = v.shape
        zeros = torch.zeros(B, N, device=v.device)

        K = torch.stack(
            [
                torch.stack([zeros, -v[..., 2], v[..., 1]], dim=-1),
                torch.stack([v[..., 2], zeros, -v[..., 0]], dim=-1),
                torch.stack([-v[..., 1], v[..., 0], zeros], dim=-1),
            ],
            dim=-2,
        )

        return K


# *****************************************************************************
# ************************ Encoder and Decoder Stacks *************************
# *****************************************************************************


class _InvariantEmbedding(torch.nn.Module):
    def __init__(
        self,
        d_inv,
        n_atom_types,
        n_bond_types,
        emb_size,
        n_charge_types=None,
        n_time_feats=None,
        n_extra_feats=None,
        n_res_types=None,
        self_cond=False,
        max_size=None,
        use_fourier_time_embed=False,
    ):
        super().__init__()

        if n_time_feats is not None:
            assert n_time_feats == emb_size, "n_time_feats must be equal to emb_size."

        n_embeddings = 2 if max_size is not None else 1  # atom type (+pot. max_size)
        n_embeddings = n_embeddings + 1 if n_charge_types is not None else n_embeddings
        n_embeddings = n_embeddings + 1 if n_res_types is not None else n_embeddings
        n_embeddings = n_embeddings + 1 if n_extra_feats is not None else n_embeddings
        n_embeddings = n_embeddings + 1 if n_time_feats is not None else n_embeddings

        atom_in_feats = emb_size * n_embeddings
        atom_in_feats = atom_in_feats + n_atom_types if self_cond else atom_in_feats

        self.n_charge_types = n_charge_types
        self.n_extra_feats = n_extra_feats
        self.n_res_types = n_res_types
        self.self_cond = self_cond
        self.max_size = max_size
        self.n_time_feats = n_time_feats
        self.use_fourier_time_embed = use_fourier_time_embed

        self.atom_type_emb = torch.nn.Embedding(n_atom_types, emb_size)

        if n_charge_types is not None:
            self.atom_charge_emb = torch.nn.Embedding(n_charge_types, emb_size)

        # Embed time
        if n_time_feats is not None:
            if use_fourier_time_embed:
                self.time_fourier = TimeFourierEncoding(
                    posenc_dim=n_time_feats, max_len=200, random_permute=False
                )
                self.time_emb_disc = MLP(n_time_feats, n_time_feats)
            else:
                self.time_emb_disc = MLP(1, n_time_feats)

        if n_extra_feats is not None:
            self.extra_feats_emb = torch.nn.Embedding(n_extra_feats, emb_size)

        if n_res_types is not None:
            self.res_type_emb = torch.nn.Embedding(n_res_types, emb_size)

        if max_size is not None:
            self.size_emb = torch.nn.Embedding(max_size, emb_size)

        self.atom_emb = torch.nn.Sequential(
            torch.nn.Linear(atom_in_feats, d_inv),
            torch.nn.SiLU(),
            torch.nn.Linear(d_inv, d_inv),
        )

        self.bond_emb = torch.nn.Embedding(n_bond_types, emb_size)

        if self_cond:
            self.bond_proj = torch.nn.Linear(emb_size + n_bond_types, emb_size)

    def forward(
        self,
        atom_types,
        bond_types,
        atom_mask,
        atom_charges=None,
        times=None,
        extra_feats=None,
        res_types=None,
        cond_types=None,
        cond_bonds=None,
    ):
        if (cond_types is not None or cond_bonds is not None) and not self.self_cond:
            raise ValueError(
                "Conditional inputs were provided but the model was initialised with self_cond as False."
            )

        if (cond_types is None or cond_bonds is None) and self.self_cond:
            raise ValueError(
                "Conditional inputs must be provided if using self conditioning."
            )

        if self.n_charge_types is not None and atom_charges is None:
            raise ValueError(
                "The invariant embedding was initialised for charge embeddings but none were provided."
            )

        if self.n_extra_feats is not None and (extra_feats is None and times is None):
            raise ValueError(
                "The invariant embedding was initialised with extra feats but none were provided."
            )

        invs = self.atom_type_emb(atom_types)

        if self.n_charge_types is not None:
            charge_feats = self.atom_charge_emb(atom_charges)
            invs = torch.cat((invs, charge_feats), dim=-1)

        if times is not None:
            if self.use_fourier_time_embed:
                times = self.time_fourier(times)
            times = self.time_emb_disc(times)
            invs = torch.cat((invs, times), dim=-1)

        if self.n_extra_feats is not None:
            extra_feats = self.extra_feats_emb(extra_feats)
            invs = torch.cat((invs, extra_feats), dim=-1)

        if self.n_res_types is not None:
            residue_type_feats = self.res_type_emb(res_types)
            invs = torch.cat((invs, residue_type_feats), dim=-1)

        if self.max_size is not None:
            n_atoms = atom_mask.sum(dim=-1, keepdim=True)
            size_emb = self.size_emb(n_atoms).expand(-1, atom_mask.size(1), -1)
            invs = torch.cat((invs, size_emb), dim=-1)

        if self.self_cond:
            invs = torch.cat((invs, cond_types), dim=-1)

        invs = self.atom_emb(invs)

        edges = self.bond_emb(bond_types)
        if self.self_cond:
            edges = torch.cat((edges, cond_bonds), dim=-1)
            edges = self.bond_proj(edges)

        return invs, edges


# *****************************************************************************
# *****************************************************************************
# ******************************* Helper Modules ******************************
# *****************************************************************************
# *****************************************************************************


class TimeFourierEncoding(nn.Module):
    """Encoder for continuous timesteps in `[0, 1]`"""

    def __init__(self, posenc_dim, max_len=100, random_permute=False):
        super().__init__()
        self.posenc_dim = posenc_dim
        self.random_permute = random_permute
        self.max_len = max_len

    def forward(self, t: torch.Tensor):
        """Encode a tensor of timesteps.

        Args:
            t: Tensor with values in `[0, 1]`. Can be 1-D (B,) or 2-D (B, num_atoms).

        Returns:
            Tensor with sine/cosine features:
            - If input is 1-D (B,): returns (B, posenc_dim)
            - If input is 2-D (B, num_atoms): returns (B, num_atoms, posenc_dim)
        """
        if t.dim() == 3:
            t = t.squeeze(-1)

        original_shape = t.shape

        # Flatten to 1-D for processing if needed
        if t.dim() == 2:
            t_flat = t.reshape(-1)  # (B * num_atoms,)
        else:
            t_flat = t  # (B,)

        t_scaled = t_flat * self.max_len
        half_dim = self.posenc_dim // 2
        emb = math.log(self.max_len) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb
        )
        emb = torch.outer(
            t_scaled.float(), emb
        )  # (B*num_atoms, half_dim) or (B, half_dim)
        emb = torch.cat(
            [torch.sin(emb), torch.cos(emb)], dim=-1
        )  # (B*num_atoms, posenc_dim) or (B, posenc_dim)

        if self.posenc_dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode="constant")

        # Reshape back to original batch structure if input was 2-D
        if len(original_shape) == 2:
            emb = emb.reshape(
                original_shape[0], original_shape[1], self.posenc_dim
            )  # (B, num_atoms, posenc_dim)
            expected_shape = (original_shape[0], original_shape[1], self.posenc_dim)
        else:
            expected_shape = (original_shape[0], self.posenc_dim)

        assert (
            emb.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {emb.shape}"
        return emb

    def out_dim(self):
        return self.posenc_dim


class RBFTimeEmbed(torch.nn.Module):
    def __init__(
        self,
        d_equi: int,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        time_dim: int = 64,
        learnable_cutoff: bool = False,
        exclude_self: bool = True,
        eps: float = 1e-6,
        rbf_type: str = "gaussian",  # simple
    ):
        super().__init__()

        self.d_equi = d_equi
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.time_dim = time_dim
        self.exclude_self = exclude_self
        self.eps = eps
        self.rbf_type = rbf_type

        # Learnable cutoff
        if learnable_cutoff:
            self.cutoff_param = nn.Parameter(torch.tensor(cutoff))
        else:
            self.register_buffer("cutoff_param", torch.tensor(cutoff))

        # Choose RBF expansion type
        if rbf_type == "gaussian":
            self.rbf_expansion = GaussianExpansion(max_value=cutoff, K=num_rbf)
        else:  # simple/legacy
            centers = torch.linspace(0, cutoff, num_rbf)
            self.register_buffer("centers", centers)
            if num_rbf > 1:
                width = centers[1] - centers[0]
            else:
                width = 1.0
            self.register_buffer("width", torch.tensor(width))

        # Time modulation networks - simplified
        self.time_to_rbf_modulation = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, num_rbf * 2),  # scale and shift together
        )

        self.time_to_global_gate = nn.Sequential(
            nn.Linear(time_dim, time_dim // 2),
            nn.SiLU(),
            nn.Linear(time_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Project RBF features to equivariant space
        self.rbf_to_equi = nn.Linear(num_rbf, d_equi, bias=False)

        # Improved coordinate mixing
        self.coord_mixing = nn.Sequential(
            nn.Linear(time_dim, d_equi),
            nn.Tanh(),  # bounded output
        )

    def forward(
        self,
        coords: torch.Tensor,  # [B, N, 3]
        time_emb: torch.Tensor,  # [B, N, time_dim]
        node_mask: torch.Tensor = None,  # [B, N]
    ) -> torch.Tensor:  # [B, N, 3, d_equi]
        """Compute RBF features with time modulation and coordinate mixing.
        Args:
            coords (torch.Tensor): Node coordinates, shape [B, N, 3].
            time_emb (torch.Tensor): Time embeddings, shape [B, N, time_dim].
            node_mask (torch.Tensor, optional): Node mask, shape [B, N]. Defaults to None.
        Returns:
            torch.Tensor: Final features, shape [B, N, 3, d_equi].
        """

        B, N, _ = coords.shape
        device = coords.device

        if node_mask is None:
            node_mask = torch.ones(B, N, device=device, dtype=torch.bool)

        # Compute pairwise coordinate differences (equivariant)
        coord_diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # [B, N, N, 3]

        # Compute pairwise distances (invariant)
        distances = torch.linalg.norm(coord_diff + self.eps, dim=-1)  # [B, N, N]

        # Apply cutoff function for smooth decay with effective cutoff
        cutoff_weights = 0.5 * (torch.cos(distances * pi / self.cutoff_param) + 1.0)
        cutoff_weights = cutoff_weights * (distances < self.cutoff_param)

        if self.exclude_self:
            eye_mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
            cutoff_weights = cutoff_weights * (~eye_mask).float()
            coord_diff = coord_diff * (~eye_mask).unsqueeze(-1).float()

        # Generate RBF features based on type
        if self.rbf_type == "gaussian":
            rbf_features = self.rbf_expansion(distances)  # [B, N, N, num_rbf]
        else:
            # Legacy simple RBF - FIXED
            distances_exp = distances.unsqueeze(-1)  # [B, N, N, 1]
            centers = self.centers.view(1, 1, 1, self.num_rbf)
            rbf_features = torch.exp(
                -0.5 * ((distances_exp - centers) / self.width) ** 2
            )

        # Apply cutoff to RBF features
        rbf_features = rbf_features * cutoff_weights.unsqueeze(-1)

        # Time modulation - more efficient
        time_modulation = self.time_to_rbf_modulation(time_emb)  # [B, N, num_rbf*2]
        rbf_scale, rbf_shift = torch.chunk(
            time_modulation, 2, dim=-1
        )  # [B, N, num_rbf] each

        # Apply softplus to scale for positive values
        rbf_scale = F.softplus(rbf_scale) + 0.1  # ensure minimum scale
        rbf_shift = torch.tanh(rbf_shift)  # bounded shift

        global_gate = self.time_to_global_gate(time_emb)  # [B, N, 1]

        # Apply time modulation efficiently
        rbf_scale_expanded = rbf_scale.unsqueeze(2)  # [B, N, 1, num_rbf]
        rbf_shift_expanded = rbf_shift.unsqueeze(2)  # [B, N, 1, num_rbf]
        modulated_rbf = rbf_features * rbf_scale_expanded + rbf_shift_expanded
        modulated_rbf = modulated_rbf * global_gate.unsqueeze(2)

        # Apply node mask
        mask_2d = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        modulated_rbf = modulated_rbf * mask_2d.unsqueeze(-1)

        # Project to equivariant dimension
        rbf_projected = self.rbf_to_equi(modulated_rbf)  # [B, N, N, d_equi]

        # Aggregate equivariant features
        equi_features = torch.einsum(
            "bijd,bijf->bidf",
            coord_diff,
            rbf_projected,
        )  # [B, N, 3, d_equi]

        # Time-modulated coordinate mixing
        coord_mixing_weights = self.coord_mixing(time_emb)  # [B, N, d_equi]
        coord_identity = coords.unsqueeze(-1) * coord_mixing_weights.unsqueeze(2)

        # Combine features
        final_features = equi_features + coord_identity

        # Apply node mask
        final_features = final_features * node_mask.view(B, N, 1, 1)

        return final_features


class GaussianExpansion(torch.nn.Module):
    def __init__(self, max_value=5.0, K=20):
        super(GaussianExpansion, self).__init__()
        offset = torch.linspace(0.0, max_value, K)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        # Input: [B, N, N] -> Output: [B, N, N, K]
        dist_expanded = dist.unsqueeze(-1)  # [B, N, N, 1]
        offset_expanded = (
            self.offset.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )  # [1, 1, 1, K]

        # Broadcast subtraction: [B, N, N, 1] - [1, 1, 1, K] -> [B, N, N, K]
        diff = dist_expanded - offset_expanded
        return torch.exp(self.coeff * torch.pow(diff, 2))


class RadialBasisEmbeddingPL(nn.Module):
    def __init__(
        self,
        d_edge: int,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        learnable_cutoff: bool = False,
        eps: float = 1e-6,
        rbf_type: str = "gaussian",  # "gaussian" or "simple"
    ):
        """
        Args:
            d_edge (int): Output embedding dimension.
            num_rbf (int): Number of radial basis functions.
            cutoff (float): Distance cutoff in Angstroms.
            learnable_cutoff (bool): Whether cutoff should be learnable.
            eps (float): Small value for numerical stability.
            rbf_type (str): Type of RBF expansion ("gaussian" or "simple").
        """
        super().__init__()
        self.d_edge = d_edge
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.eps = eps
        self.rbf_type = rbf_type

        # Learnable cutoff
        if learnable_cutoff:
            self.cutoff_param = nn.Parameter(torch.tensor(cutoff))
        else:
            self.register_buffer("cutoff_param", torch.tensor(cutoff))

        # Choose RBF expansion type
        if rbf_type == "gaussian":
            self.rbf_expansion = GaussianExpansion(max_value=cutoff, K=num_rbf)
        elif rbf_type == "center":  # simple
            centers = torch.linspace(0, cutoff, num_rbf)
            if num_rbf > 1:
                width = centers[1] - centers[0]
            else:
                width = 1.0
            self.register_buffer("centers", centers)
            self.width = width
        else:
            raise ValueError(f"Unknown RBF type: {rbf_type}")

        # Project RBF features to edge dimension
        if d_edge is not None:
            self.rbf_to_edge = nn.Linear(num_rbf, d_edge)

    def forward(
        self,
        ligand_coords: torch.Tensor,  # [B, N, 3]
        pocket_coords: torch.Tensor,  # [B, N_p, 3]
        ligand_mask: torch.Tensor,  # [B, N]
        pocket_mask: torch.Tensor,  # [B, N_p]
    ) -> torch.Tensor:  # [B, N, N_p, d_edge]
        """
        Args:
            ligand_coords: Tensor of shape [B, N, 3]
            pocket_coords: Tensor of shape [B, N_p, 3]
            ligand_mask: Boolean mask of shape [B, N] for valid ligand atoms
            pocket_mask: Boolean mask of shape [B, N_p] for valid pocket atoms

        Returns:
            Tensor of shape [B, N, N_p, d_edge]
        """
        B, N, _ = ligand_coords.shape

        # Compute pairwise distances between ligand and pocket atoms
        # Using cdist for cross-distances: ligand (queries) to pocket (keys)
        distances = torch.cdist(ligand_coords, pocket_coords, p=2)  # [B, N, N_p]

        # Generate RBF features based on type
        if self.rbf_type == "gaussian":
            # Apply smooth cosine cutoff for better gradients
            cutoff_weights = 0.5 * (torch.cos(distances * pi / self.cutoff_param) + 1.0)
            cutoff_weights = cutoff_weights * (distances < self.cutoff_param)

            rbf_features = self.rbf_expansion(distances)  # [B, N, N_p, num_rbf]
            rbf_features = rbf_features * cutoff_weights.unsqueeze(-1)
        else:
            # Simple RBF
            distances = torch.clamp(distances, max=self.cutoff)
            centers = self.centers.view(1, 1, 1, self.num_rbf)
            dists_exp = distances.unsqueeze(-1)
            rbf_features = torch.exp(
                -0.5 * ((dists_exp - centers) / self.width) ** 2
            )  # [B, N, N_p, num_rbf]

        # Apply masks: ligand atoms (dim 1) and pocket atoms (dim 2)
        mask = ligand_mask.unsqueeze(-1).unsqueeze(-1) * pocket_mask.unsqueeze(
            1
        ).unsqueeze(-1)

        if self.d_edge is not None:
            # Project to edge dimension
            rbf_features = self.rbf_to_edge(rbf_features)  # [B, N, N_p, d_edge]
        return rbf_features * mask


class RadialBasisEmbedding(nn.Module):
    def __init__(
        self,
        d_edge: int,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        learnable_cutoff: bool = False,
        eps: float = 1e-6,
        rbf_type: str = "gaussian",  # "gaussian" or "simple"
    ):
        """
        Args:
            d_edge (int): Output embedding dimension.
            num_rbf (int): Number of radial basis functions.
            cutoff (float): Distance cutoff in Angstroms.
            learnable_cutoff (bool): Whether cutoff should be learnable.
            eps (float): Small value for numerical stability.
            rbf_type (str): Type of RBF expansion ("gaussian" or "simple").
        """
        super().__init__()
        self.d_edge = d_edge
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.eps = eps
        self.rbf_type = rbf_type

        # Learnable cutoff
        if learnable_cutoff:
            self.cutoff_param = nn.Parameter(torch.tensor(cutoff))
        else:
            self.register_buffer("cutoff_param", torch.tensor(cutoff))

        # Choose RBF expansion type
        if rbf_type == "gaussian":
            self.rbf_expansion = GaussianExpansion(max_value=cutoff, K=num_rbf)
        elif rbf_type == "center":  # simple/legacy
            centers = torch.linspace(0, cutoff, num_rbf)
            if num_rbf > 1:
                width = centers[1] - centers[0]
            else:
                width = 1.0
            self.register_buffer("centers", centers)
            self.width = width
        else:
            raise ValueError(f"Unknown RBF type: {rbf_type}")

        if d_edge is not None:
            # Project RBF features to edge dimension
            self.rbf_to_edge = nn.Linear(num_rbf, d_edge)

    def forward(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            coords: Tensor of shape [B, N, 3]
            mask: Boolean mask of shape [B, N] for valid ligand atoms

        Returns:
            Tensor of shape [B, N, N, d_edge]
        """
        B, N, _ = coords.shape

        # Compute pairwise distances within ligand
        distances = torch.cdist(coords, coords, p=2)  # [B, N, N]

        # Generate RBF features based on type
        if self.rbf_type == "gaussian":
            # Apply smooth cosine cutoff for better gradients
            cutoff_weights = 0.5 * (torch.cos(distances * pi / self.cutoff_param) + 1.0)
            cutoff_weights = cutoff_weights * (distances < self.cutoff_param)

            rbf_features = self.rbf_expansion(distances)  # [B, N, N, num_rbf]
            rbf_features = rbf_features * cutoff_weights.unsqueeze(-1)
        else:
            # Simple RBF
            distances = torch.clamp(distances, max=self.cutoff)
            centers = self.centers.view(1, 1, 1, self.num_rbf)
            dists_exp = distances.unsqueeze(-1)
            rbf_features = torch.exp(
                -0.5 * ((dists_exp - centers) / self.width) ** 2
            )  # [B, N, N_p, num_rbf]

        # Apply mask for valid ligand atoms
        mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)  # [B, N, N]

        if self.d_edge is not None:
            # Project to edge dimension
            rbf_features = self.rbf_to_edge(rbf_features)  # [B, N, N, d_edge]
        return rbf_features * mask_2d.unsqueeze(-1)


class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra

    Modified to work with batched input (B, N, ...) instead of PyTorch Geometric input (N, ...)
    """

    def __init__(
        self,
        d_inv,
        d_equi,
        d_out,
        scalar_activation=False,
        return_vector=False,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = d_out
        self.return_vector = return_vector

        self.vec1_proj = nn.Linear(d_equi, d_inv, bias=False)
        self.vec2_proj = nn.Linear(d_equi, d_out, bias=False)

        self.update_net = nn.Sequential(
            nn.Linear(d_inv * 2, d_inv),
            nn.SiLU(),
            nn.Linear(d_inv, d_out * 2),
        )

        self.act = nn.SiLU() if scalar_activation else None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v, node_mask=None):
        """
        Args:
            x: Scalar features of shape [B, N, hidden_channels]
            v: Vector features of shape [B, N, 3, hidden_channels]
            node_mask: Boolean mask of shape [B, N], True for valid nodes

        Returns:
            x_out: Updated scalar features of shape [B, N, out_channels]
            v_out: Updated vector features of shape [B, N, 3, out_channels] (if return_vector=True)
        """
        B, N, _ = x.shape

        if node_mask is None:
            node_mask = torch.ones(B, N, dtype=torch.bool, device=x.device)

        # Apply mask to input features to ensure masked positions don't contribute
        mask_scalar = node_mask.unsqueeze(-1)  # [B, N, 1]
        mask_vector = node_mask.unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1]

        # Project vector features: [B, N, 3, hidden_channels] -> [B, N, 3, hidden_channels]
        vec1_buffer = self.vec1_proj(v)

        # Compute vector norms: [B, N, 3, hidden_channels] -> [B, N, hidden_channels]
        # Use safe norm computation to avoid NaN gradients
        vec1 = torch.norm(vec1_buffer, dim=2)  # [B, N, hidden_channels]
        # Project vectors for gating: [B, N, 3, hidden_channels] -> [B, N, 3, out_channels]
        vec2 = self.vec2_proj(v)

        # Concatenate scalar and vector norm features
        x_combined = torch.cat([x, vec1], dim=-1)  # [B, N, hidden_channels * 2]

        # Update through MLP and split into scalar and vector gate
        updates = self.update_net(x_combined)  # [B, N, out_channels * 2]
        x_update, v_gate = torch.split(updates, self.out_channels, dim=-1)

        # Apply gating to vectors: [B, N, 1, out_channels] * [B, N, 3, out_channels]
        v_out = v_gate.unsqueeze(2) * vec2  # [B, N, 3, out_channels]

        # Apply activation to scalar features if specified
        if self.act is not None:
            x_update = self.act(x_update)

        # Apply final masking to outputs
        x_update = x_update * mask_scalar
        v_out = v_out * mask_vector

        if self.return_vector:
            return x_update, v_out
        else:
            return x_update + v_out.sum() * 0


# Helper: return π as a tensor matching the input's dtype and device.
def _pi(x: torch.Tensor) -> torch.Tensor:
    return x.new_tensor(math.pi)


# --- Spherical Harmonics Definitions ---


# l = 0
def fn_Y0(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    # Compute the constant value.
    val = 0.5 * torch.sqrt(1.0 / (_pi(x)))
    # Return a tensor of the same shape as x (i.e. [B, N, N, 1]) filled with that value.
    return x.new_full(x.shape, val.item())


# l = 1
def fn_Y1(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    # Each of _Y1_1, _Y10, _Y11 will be applied elementwise.
    _Y1_1 = lambda x, y, z: torch.sqrt(3.0 / (4 * _pi(x))) * y
    _Y10 = lambda x, y, z: torch.sqrt(3.0 / (4 * _pi(x))) * z
    _Y11 = lambda x, y, z: torch.sqrt(3.0 / (4 * _pi(x))) * x
    return torch.cat(
        [_Y1_1(x, y, z), _Y10(x, y, z), _Y11(x, y, z)], dim=-1
    )  # Result shape: [B, N, N, 3]


# l = 2
def fn_Y2(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    _Y2_2 = lambda x, y, z: 0.5 * torch.sqrt(15.0 / (_pi(x))) * x * y
    _Y2_1 = lambda x, y, z: 0.5 * torch.sqrt(15.0 / (_pi(x))) * y * z
    _Y20 = lambda x, y, z: 0.25 * torch.sqrt(5.0 / (_pi(x))) * (3 * z**2 - 1)
    _Y21 = lambda x, y, z: 0.5 * torch.sqrt(15.0 / (_pi(x))) * x * z
    _Y22 = lambda x, y, z: 0.25 * torch.sqrt(15.0 / (_pi(x))) * (x**2 - y**2)
    return torch.cat(
        [_Y2_2(x, y, z), _Y2_1(x, y, z), _Y20(x, y, z), _Y21(x, y, z), _Y22(x, y, z)],
        dim=-1,
    )  # [B, N, N, 5]


# l = 3
def fn_Y3(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    _Y3_3 = (
        lambda x, y, z: 0.25 * torch.sqrt(35.0 / (2 * _pi(x))) * y * (3 * x**2 - y**2)
    )
    _Y3_2 = lambda x, y, z: 0.5 * torch.sqrt(105.0 / (_pi(x))) * x * y * z
    _Y3_1 = lambda x, y, z: 0.25 * torch.sqrt(21.0 / (2 * _pi(x))) * y * (5 * z**2 - 1)
    _Y30 = lambda x, y, z: 0.25 * torch.sqrt(7.0 / (_pi(x))) * (5 * z**3 - 3 * z)
    _Y31 = lambda x, y, z: 0.25 * torch.sqrt(21.0 / (2 * _pi(x))) * x * (5 * z**2 - 1)
    _Y32 = lambda x, y, z: 0.25 * torch.sqrt(105.0 / (_pi(x))) * (x**2 - y**2) * z
    _Y33 = (
        lambda x, y, z: 0.25 * torch.sqrt(35.0 / (2 * _pi(x))) * x * (x**2 - 3 * y**2)
    )
    return torch.cat(
        [
            _Y3_3(x, y, z),
            _Y3_2(x, y, z),
            _Y3_1(x, y, z),
            _Y30(x, y, z),
            _Y31(x, y, z),
            _Y32(x, y, z),
            _Y33(x, y, z),
        ],
        dim=-1,
    )  # [B, N, N, 7]


def init_sph_fn(l: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns a function that computes the spherical harmonics expansion of order l.
    Expects an input tensor of shape (..., 3) and returns a tensor of shape (..., 2l+1).
    """
    if l == 0:
        return lambda r: fn_Y0(*torch.chunk(r, chunks=3, dim=-1))
    elif l == 1:
        return lambda r: fn_Y1(*torch.chunk(r, chunks=3, dim=-1))
    elif l == 2:
        return lambda r: fn_Y2(*torch.chunk(r, chunks=3, dim=-1))
    elif l == 3:
        return lambda r: fn_Y3(*torch.chunk(r, chunks=3, dim=-1))
    else:
        raise NotImplementedError(
            "Spherical harmonics are only defined up to order l = 3."
        )


def poly_cutoff_chi(x: torch.Tensor, p: int) -> torch.Tensor:
    """
    Polynomial cutoff function as in Eq. (29):
      φ_χ_cut(x) = 1 - ((p+1)(p+2)/2) * x^p + p(p+2)*x^(p+1) - (p(p+1)/2)*x^(p+2)
    The function is applied for all x.
    Here, x = (χ̃_ij) / (χ_cut).
    """
    poly_val = (
        1
        - ((p + 1) * (p + 2) / 2) * x**p
        + p * (p + 2) * x ** (p + 1)
        - (p * (p + 1) / 2) * x ** (p + 2)
    )
    return poly_val


class EquisSphc(nn.Module):
    """
    Computes SPHC features for each point in a point cloud.

    For each point i the feature is computed as
      χ_i^(l) = (1 / C_i) ∑_{j ≠ i} φ(‖R(j)-R(i)‖) · Y^(l)(ˆr_{ij}),
    where ˆr_{ij} = (R(j)-R(i)) / ‖R(j)-R(i)‖ and
          C_i = ∑_{j ≠ i} φ(‖R(j)-R(i)‖),
          φ(r) = 0.5 * [cos(π*r/r_cut)+1]  if r < r_cut, else 0.

    The final SPHC feature for each point is the concatenation over degrees l (l_min to l_max)
    resulting in a vector of dimension ∑_{l=l_min}^{l_max}(2l+1).

    Optionally, if return_sphc_distance_matrix is True, the module also returns a SPHC distance matrix.
    This matrix is computed as follows:
      1. For each pair (i,j), compute X_ij = ||χ_i - χ_j||₂.
      2. Compute Ẋ = softmax(X) along each row.
      3. Define χ_cut = κ / n, where n is the number of valid atoms per molecule.
      4. Compute x = Ẋ / χ_cut and apply the polynomial cutoff φ_χ_cut(x) as in Eq. (29).
    """

    def __init__(
        self,
        l_min: int,
        l_max: int,
        r_cut: float,
        eps: float = 1e-8,
        return_sphc_distance_matrix: bool = False,
        p: int = 1,
        kappa: float = 1.0,
    ):
        """
        Args:
            l_min: Minimum spherical harmonic degree.
            l_max: Maximum spherical harmonic degree (must be ≤ 3).
            r_cut: Cutoff radius in Euclidean space.
            eps: Small value to avoid division by zero.
            return_sphc_distance_matrix: If True, also return the SPHC distance matrix.
            p: Polynomial order parameter for the distance cutoff function.
            kappa: Scaling factor for the SPHC cutoff, with χ_cut = κ / n.
        """
        super(EquisSphc, self).__init__()
        assert (
            0 <= l_min <= l_max <= 3
        ), "l_min and l_max must satisfy 0 ≤ l_min ≤ l_max ≤ 3."
        self.l_min = l_min
        self.l_max = l_max
        self.r_cut = r_cut
        self.eps = eps
        self.return_sphc_distance_matrix = return_sphc_distance_matrix
        self.p = p
        self.kappa = kappa

        # Create spherical harmonics functions for each degree.
        self.sph_fns = {l: init_sph_fn(l) for l in range(l_min, l_max + 1)}
        # Total output dimension: ∑_{l=l_min}^{l_max} (2l+1)
        self.out_dim = sum(2 * l + 1 for l in range(l_min, l_max + 1))

    def forward(
        self, coords: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            coords: Tensor of shape [B, N, 3] representing the point cloud.
            mask: Optional boolean tensor of shape [B, N] indicating valid points.

        Returns:
            If return_sphc_distance_matrix is False:
                Tensor of shape [B, N, out_dim] containing SPHC features per point.
            else:
                Tuple (sphc, sphc_distance_matrix) where:
                  - sphc is [B, N, out_dim],
                  - sphc_distance_matrix is [B, N, N] with the rescaled distances.
        """
        B, N, _ = coords.shape
        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=coords.device)

        # Compute pairwise difference vectors: r_ij = R(j) - R(i)
        diff = coords.unsqueeze(1) - coords.unsqueeze(2)  # [B, N, N, 3]
        dists = torch.norm(diff, dim=-1)  # [B, N, N]
        unit_diff = diff / (dists.unsqueeze(-1) + self.eps)  # [B, N, N, 3]

        # Build neighbor mask: valid if both points are valid and exclude self.
        valid_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        eye = torch.eye(N, dtype=torch.bool, device=coords.device).unsqueeze(0)
        neighbor_mask = valid_mask & (~eye)

        # Compute cosine cutoff φ(r): [B, N, N]
        phi = 0.5 * (torch.cos(math.pi * dists / self.r_cut) + 1.0)
        phi = torch.where(dists < self.r_cut, phi, torch.zeros_like(phi))
        phi = phi * neighbor_mask.float()

        C = phi.sum(dim=-1, keepdim=True)  # [B, N, 1]
        C_safe = torch.where(C < self.eps, torch.ones_like(C), C)

        chi_per_l = []
        for l in range(self.l_min, self.l_max + 1):
            sph_fn = self.sph_fns[l]  # Function mapping [..., 3] -> [..., (2l+1)]
            # Evaluate spherical harmonics on each unit vector.
            Y_l = sph_fn(unit_diff)  # [B, N, N, (2l+1)]

            weighted_Y = phi.unsqueeze(-1) * Y_l  # [B, N, N, (2l+1)]
            sum_Y = weighted_Y.sum(dim=2)  # [B, N, (2l+1)]
            chi_l = sum_Y / C_safe  # [B, N, (2l+1)]
            chi_l = torch.where(C < self.eps, torch.zeros_like(chi_l), chi_l)
            chi_per_l.append(chi_l)

        # Concatenate over degrees l → [B, N, out_dim]
        sphc = torch.cat(chi_per_l, dim=-1)

        if self.return_sphc_distance_matrix:
            # Compute pairwise Euclidean distances in SPHC space.
            chi_diff = sphc.unsqueeze(2) - sphc.unsqueeze(1)  # [B, N, N, out_dim]
            X = torch.norm(chi_diff, dim=-1)  # [B, N, N]
            X_soft = torch.softmax(X, dim=-1)  # [B, N, N]
            n_valid = mask.sum(dim=1, keepdim=True).float()  # [B, 1]
            chi_cut = self.kappa / n_valid  # [B, 1]
            chi_cut = chi_cut.unsqueeze(-1)  # [B, 1, 1]
            x = X_soft / chi_cut  # [B, N, N]
            X_soft_cut = poly_cutoff_chi(x, self.p)  # [B, N, N]
            return sphc, X_soft_cut

        return sphc

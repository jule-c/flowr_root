import torch

# *******************************************************************************
# ******************************* Helper Functions ******************************
# *******************************************************************************


def adj_to_attn_mask(adj_matrix, pos_inf=False):
    """Assumes adj_matrix is only 0s and 1s"""

    inf = float("inf") if pos_inf else float("-inf")
    attn_mask = torch.zeros_like(adj_matrix.float())
    attn_mask[adj_matrix == 0] = inf

    # Ensure nodes with no connections (fake nodes) don't have all -inf in the attn mask
    # Otherwise we would have problems when softmaxing
    n_nodes = adj_matrix.sum(dim=-1)
    attn_mask[n_nodes == 0] = 0.0

    return attn_mask


# *****************************************************************************
# ******************************* Helper Modules ******************************
# *****************************************************************************


class RadialBasisFeatures(torch.nn.Module):
    def __init__(self, n_rbf, max_dist=5.0):
        super().__init__()

        centres = torch.linspace(0, max_dist, n_rbf)
        width = centres[1] - centres[0] if n_rbf > 1 else 1.0
        widths = torch.tensor([width] * n_rbf)

        self.n_rbf = n_rbf
        self.max_dist = max_dist

        self.centres = torch.nn.Parameter(centres)
        self.widths = torch.nn.Parameter(widths)

    def forward(self, dists):
        dists = dists.clamp(max=self.max_dist) if self.max_dist is not None else dists
        centres = self.centres.view(1, 1, 1, self.n_rbf)
        widths = self.widths.view(1, 1, 1, self.n_rbf)
        rbfs = torch.exp(-0.5 * ((dists.unsqueeze(-1) - centres) / widths) ** 2)
        return rbfs


class CoordNorm(torch.nn.Module):
    def __init__(self, d_equi, zero_com=False, eps=1e-5):
        super().__init__()

        self.d_equi = d_equi
        self.zero_com = zero_com
        self.eps = eps

        self.set_weights = torch.nn.Parameter(torch.ones((1, 1, 1, d_equi)))

    def forward(self, equis, mask):
        """Apply coordinate normlisation layer

        Args:
            equis (torch.Tensor): Coordinate tensor, shape [B, N, 3, d_equi]
            mask (torch.Tensor): Mask for nodes, shape [B, N], 1 for real, 0 otherwise

        Returns:
            torch.Tensor: Normalised coords, shape [B, N, 3, d_equi]
        """

        n_atoms = mask.sum(dim=-1).view(-1, 1, 1, 1)
        mask = mask.unsqueeze(-1).unsqueeze(-1)

        if self.zero_com:
            real_coords = equis * mask
            com = real_coords.sum(dim=1, keepdim=True) / n_atoms
            equis = equis - com

        lengths = torch.linalg.vector_norm(equis, dim=2, keepdim=True)

        # scaled_lengths = lengths.sum(dim=1, keepdim=True) / n_atoms
        # equis = (equis * self.set_weights) / (scaled_lengths + self.eps)

        # Equivalent to RMSNorm but applied to the norm of vector features
        scales = torch.sqrt((lengths**2).sum(dim=-1, keepdim=True) + self.eps)
        equis = (equis * self.set_weights) / scales

        equis = equis * mask
        return equis

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)


class PairwiseMessages(torch.nn.Module):
    """Compute pairwise features for a set of query and a set of key nodes"""

    def __init__(
        self,
        d_equi,
        d_q_inv,
        d_k_inv,
        d_message,
        d_out,
        d_ff,
        d_edge=None,
        include_dists=False,
    ):
        super().__init__()

        # in_feats = (d_message * 2) + d_equi

        in_feats = d_message + d_equi
        in_feats = in_feats + d_edge if d_edge is not None else in_feats
        in_feats = in_feats + d_equi if include_dists else in_feats

        self.d_equi = d_equi
        self.d_edge = d_edge
        self.include_dists = include_dists

        self.q_message_proj = torch.nn.Linear(d_q_inv, d_message)
        self.k_message_proj = torch.nn.Linear(d_k_inv, d_message)

        self.message_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_feats, d_ff),
            torch.nn.SiLU(),
            torch.nn.Linear(d_ff, d_out),
        )

    def forward(self, q_equi, q_inv, k_equi, k_inv, edges=None):
        """Produce messages between query and key

        Args:
            q_equi (torch.Tensor): Equivariant query features, shape [B, N_q, 3, d_equi]
            q_inv (torch.Tensor): Invariant query features, shape [B, N_q, d_q_inv]
            k_equi (torch.Tensor): Equivariant key features, shape [B, N_kv, 3, d_equi]
            k_inv (torch.Tensor): Invariant key features, shape [B, N_kv, 3, d_k_inv]
            edges (torch.Tensor): Edge features, shape [B, N_q, N_kv, d_edge]

        Returns:
            torch.Tensor: Message matrix, shape [B, N_q, N_k, d_out]
        """

        if edges is not None and self.d_edge is None:
            raise ValueError(
                "edges was provided but the model was initialised with d_edge as None."
            )

        if edges is None and self.d_edge is not None:
            raise ValueError(
                "The model was initialised with d_edge but no edge feats were provided to forward fn."
            )

        # q_equi_batched = q_equi.movedim(-1, 1).flatten(0, 1)
        # k_equi_batched = k_equi.movedim(-1, 1).flatten(0, 1)

        # dotprods = torch.bmm(q_equi_batched, k_equi_batched.transpose(1, 2))
        # dotprods = dotprods.unflatten(0, (-1, self.d_equi)).movedim(1, -1)

        dotprods = torch.einsum("bqed,bked->bqkd", q_equi, k_equi)

        # q_messages = self.q_message_proj(q_inv).unsqueeze(2).expand(-1, -1, k_inv.size(1), -1)
        # k_messages = self.k_message_proj(k_inv).unsqueeze(1).expand(-1, q_inv.size(1), -1, -1)

        # Outer product between query and key invariant features
        q_messages = self.q_message_proj(q_inv).unsqueeze(2)
        k_messages = self.k_message_proj(k_inv).unsqueeze(1)
        pairwise_qk_feats = q_messages * k_messages

        features = [pairwise_qk_feats, dotprods]

        if self.include_dists:
            vec_dists = q_equi.unsqueeze(2) - k_equi.unsqueeze(1)
            dists = torch.linalg.vector_norm(vec_dists, dim=3)
            features.append(dists)

        if edges is not None:
            features.append(edges)

        pairwise_feats = torch.cat(features, dim=-1)
        return self.message_mlp(pairwise_feats)


class BondRefine(torch.nn.Module):
    def __init__(self, d_equi, d_inv, d_edge):
        super().__init__()

        in_feats = d_inv + (d_edge * 2) + (d_equi * 2)
        d_ff = d_edge * 4

        self.d_inv = d_inv

        self.edge_norm = torch.nn.LayerNorm(d_edge)
        self.orig_edge_norm = torch.nn.LayerNorm(d_edge)

        self.node_proj = torch.nn.Linear(d_inv, d_inv * 2)
        self.message_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_feats, d_ff),
            torch.nn.SiLU(),
            torch.nn.Linear(d_ff, d_edge),
        )

    def forward(self, equis, invs, edges, orig_edges):
        """Refine the bond predictions with a message passing layer that only updates bonds

        Args:
            equis (torch.Tensor): Coordinate tensor without coord sets, shape [B, N, 3, d_equi]
            invs (torch.Tensor): Node feature tensor, shape [B, N, d_inv]
            edges (torch.Tensor): Current edge features, shape [B, N, N, d_edge]
            orig_edges (torch.Tensor): Original edge features, shape [B, N, N, d_edge]

        Returns:
            torch.Tensor: Bond predictions tensor, shape [B, N, N, d_edge]
        """

        edges = self.edge_norm(edges)
        orig_edges = self.orig_edge_norm(orig_edges)

        vec_dists = equis.unsqueeze(2) - equis.unsqueeze(1)
        dists = torch.linalg.vector_norm(vec_dists, dim=-2)
        dotprods = torch.einsum("bqed,bked->bqkd", equis, equis)

        invs_A, invs_B = torch.split(self.node_proj(invs), self.d_inv, dim=-1)
        pairwise_feats = invs_A.unsqueeze(2) * invs_B.unsqueeze(1)

        in_feats = torch.cat(
            (pairwise_feats, dists, dotprods, edges, orig_edges), dim=-1
        )
        return self.message_mlp(in_feats)

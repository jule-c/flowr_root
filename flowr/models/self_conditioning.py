import math

import torch
import torch.nn as nn

from flowr.models.pocket_util import GaussianExpansion


class SelfConditioningResidualLayer(nn.Module):
    """Residual self-conditioning layer that modifies input features
    based on the difference between current state and predicted final state.

    Computes displacement-based residual corrections for atom types,
    bond types, and coordinates.
    """

    def __init__(
        self,
        n_atom_types: int = 118,
        n_bond_types: int = 5,
        n_charge_types: int = 6,
        n_extra_atom_feats: int | None = None,
        rbf_dim: int = 20,
        rbf_dmax: float = 10.0,
    ):
        super().__init__()

        self.rbf_dim = rbf_dim
        self.rbf_dmax = rbf_dmax
        self.rbf = GaussianExpansion(max_value=rbf_dmax, K=rbf_dim)

        # Node residual MLP: current atomics + predicted atomics + charges + distance RBF
        node_input_dim = n_atom_types + n_atom_types + n_charge_types + rbf_dim
        if n_extra_atom_feats is not None and n_extra_atom_feats > 0:
            node_input_dim += n_extra_atom_feats

        self.node_residual_mlp = nn.Sequential(
            nn.Linear(node_input_dim, n_atom_types * 2),
            nn.SiLU(),
            nn.Linear(n_atom_types * 2, n_atom_types),
        )

        # Edge residual MLP: current bonds + predicted bonds + edge distance change RBF
        edge_input_dim = n_bond_types + n_bond_types + rbf_dim
        self.edge_residual_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, n_bond_types * 2),
            nn.SiLU(),
            nn.Linear(n_bond_types * 2, n_bond_types),
        )

        # Coordinate gating: learn a per-atom scale for coordinate displacement
        self.coord_gate = nn.Sequential(
            nn.Linear(rbf_dim, rbf_dim),
            nn.SiLU(),
            nn.Linear(rbf_dim, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        current_batch: dict,
        predicted_final: dict,
    ) -> dict:
        """Apply residual corrections to current batch features.

        Args:
            current_batch: Current interpolated state with keys:
                coords [B, N, 3], atomics [B, N, n_atom_types],
                bonds [B, N, N, n_bond_types], mask [B, N]
            predicted_final: Predicted final state (softmax-normalized for categoricals):
                coords [B, N, 3], atomics [B, N, n_atom_types],
                bonds [B, N, N, n_bond_types], charges [B, N, n_charge_types],
                mask [B, N]

        Returns:
            Updated batch dict with residual corrections applied to coords, atomics, bonds
        """
        # NOTE: GaussianExpansion(0) produces non-zero output (peak at 0th basis).
        # This layer should only be called when SC actually produced predictions
        # (i.e., _sc_fired=True). Calling with zero inputs gives spurious features.
        coords_current = current_batch["coords"]
        atomics_current = current_batch["atomics"]
        bonds_current = current_batch["bonds"]
        mask = current_batch["mask"]

        coords_pred = predicted_final["coords"]
        atomics_pred = predicted_final["atomics"]
        bonds_pred = predicted_final["bonds"]
        charges_pred = predicted_final["charges"]

        # --- Node residuals ---
        coord_displacement = coords_pred - coords_current  # [B, N, 3]
        node_distances = torch.norm(coord_displacement, dim=-1, keepdim=False)  # [B, N]
        # GaussianExpansion expects [B, N, N], so unsqueeze and squeeze for [B, N] input
        node_distances_rbf = self.rbf(node_distances.unsqueeze(1)).squeeze(
            1
        )  # [B, N, rbf_dim]

        node_inputs = [atomics_current, atomics_pred, charges_pred, node_distances_rbf]
        if "hybridization" in predicted_final:
            node_inputs.append(predicted_final["hybridization"])

        node_residual_features = torch.cat(node_inputs, dim=-1)
        node_residuals = self.node_residual_mlp(node_residual_features)

        # --- Coordinate residuals ---
        coord_scale = self.coord_gate(node_distances_rbf)  # [B, N, 1]
        coord_residuals = coord_displacement * coord_scale  # [B, N, 3]

        # --- Edge residuals ---
        mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)  # [B, N, N]
        dists_current = torch.cdist(coords_current, coords_current)  # [B, N, N]
        dists_pred = torch.cdist(coords_pred, coords_pred)  # [B, N, N]
        edge_distance_change = (dists_pred - dists_current) * mask_2d
        edge_distance_change_rbf = self.rbf(edge_distance_change)

        edge_inputs = [bonds_current, bonds_pred, edge_distance_change_rbf]
        edge_residual_features = torch.cat(edge_inputs, dim=-1)
        edge_residuals = self.edge_residual_mlp(edge_residual_features)

        # Apply residuals with masking
        mask_node = mask.unsqueeze(-1)  # [B, N, 1]
        mask_edge = mask_2d.unsqueeze(-1)  # [B, N, N, 1]

        updated_batch = dict(current_batch)
        updated_batch["coords"] = coords_current + coord_residuals * mask_node
        updated_batch["atomics"] = atomics_current + node_residuals * mask_node
        updated_batch["bonds"] = bonds_current + edge_residuals * mask_edge

        return updated_batch


class DistanceEdgeSelfCond(nn.Module):
    """HarmonicFlow-style distance self-conditioning for edges.

    Computes pairwise distances from predicted coordinates,
    applies RBF expansion, and projects to edge feature dimension.
    """

    def __init__(self, num_rbf: int = 32, cutoff: float = 10.0, d_edge: int = 64):
        super().__init__()
        self.rbf = GaussianExpansion(max_value=cutoff, K=num_rbf)
        self.proj = nn.Linear(num_rbf, d_edge)
        self.num_rbf = num_rbf

    def forward(self, pred_coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute RBF edge features from predicted pairwise distances.

        Args:
            pred_coords: [B, N, 3] predicted final coordinates
            mask: [B, N] atom mask

        Returns:
            edge_sc_feats: [B, N, N, d_edge] distance-based SC edge features
        """
        dists = torch.cdist(pred_coords, pred_coords)  # [B, N, N]
        rbf_feats = self.rbf(dists)  # [B, N, N, num_rbf]
        edge_feats = self.proj(rbf_feats)  # [B, N, N, d_edge]
        adj_mask = (mask.unsqueeze(1) * mask.unsqueeze(2)).unsqueeze(-1)
        return edge_feats * adj_mask


class AdaptiveSCSchedule:
    """Schedule for self-conditioning probability during training.

    Supports constant (default), linear warmup, and cosine warmup schedules.
    """

    def __init__(
        self,
        mode: str = "constant",
        start_prob: float = 0.0,
        end_prob: float = 0.5,
        warmup_steps: int = 10000,
    ):
        self.mode = mode
        self.start_prob = start_prob
        self.end_prob = end_prob
        self.warmup_steps = max(warmup_steps, 1)

    def get_prob(self, global_step: int) -> float:
        if self.mode == "constant":
            return self.end_prob
        t = min(global_step / self.warmup_steps, 1.0)
        if self.mode == "linear_warmup":
            return self.start_prob + t * (self.end_prob - self.start_prob)
        elif self.mode == "cosine_warmup":
            return self.start_prob + (self.end_prob - self.start_prob) * 0.5 * (
                1 - math.cos(math.pi * t)
            )
        return self.end_prob

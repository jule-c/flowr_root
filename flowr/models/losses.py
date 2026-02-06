import math
from typing import Optional

import scipy.constants as const
import torch
import torch.nn.functional as F

import flowr.util.functional as smolF

# Physical constants for electrostatics
ELEC_FACTOR = 1 / (4 * math.pi * const.epsilon_0)  # Coulomb's constant
ELEC_FACTOR *= const.elementary_charge**2  # Convert elementary charges to Coulombs
ELEC_FACTOR /= const.angstrom  # Convert Angstroms to meters
ELEC_FACTOR *= const.Avogadro / (const.kilo * const.calorie)  # Convert J to kcal/mol


def _get_pair_mask(mask):
    pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)  # [B, N, N]
    batch_size, max_atoms = mask.shape
    eye = (
        torch.eye(max_atoms, device=mask.device).unsqueeze(0).expand(batch_size, -1, -1)
    )
    pair_mask = pair_mask * (1 - eye)
    return pair_mask


def _get_distance_matrix(coords, mask):
    coords = coords * mask.unsqueeze(-1)
    dists = torch.cdist(coords, coords, p=2)  # [B, N, N]
    return dists


def _get_cross_distance_matrix(coords1, coords2, mask1, mask2):
    coords1 = coords1 * mask1.unsqueeze(-1)
    coords2 = coords2 * mask2.unsqueeze(-1)
    dists = torch.cdist(coords1, coords2, p=2)  # [B, N, M]
    return dists


def _get_cross_pair_mask(mask1, mask2):
    pair_mask = mask1.unsqueeze(2) * mask2.unsqueeze(1)  # [B, N, M]
    # no cleaning of diagonal since we assume mask1 and mask2 are different
    return pair_mask


def _compute_bond_lengths(coords, bond_indices, mask):
    """
    Compute bond lengths from coordinates and bond indices.

    Args:
        coords: [B, N, 3] coordinates
        bond_indices: [B, N_bonds, 2] indices of bonded atom pairs
        mask: [B, N_bonds] mask for valid bonds

    Returns:
        distances: [B, N_bonds] bond lengths
    """
    # Gather coordinates of bonded atoms
    idx1 = bond_indices[..., 0]  # [B, N_bonds]
    idx2 = bond_indices[..., 1]  # [B, N_bonds]

    # Get coordinates [B, N_bonds, 3]
    coords1 = torch.gather(coords, 1, idx1.unsqueeze(-1).expand(-1, -1, 3))
    coords2 = torch.gather(coords, 1, idx2.unsqueeze(-1).expand(-1, -1, 3))

    # Compute distances
    distances = torch.norm(coords2 - coords1, dim=-1)  # [B, N_bonds]

    return distances


def _compute_angles_from_coords(coords, angle_indices, mask, eps=1e-6):
    """
    Compute angles from coordinates and angle indices.

    Args:
        coords: [B, N, 3] coordinates
        angle_indices: [B, N_angles, 3] indices of atoms forming angles (i, j, k where j is center)
        mask: [B, N_angles] mask for valid angles

    Returns:
        angles: [B, N_angles] angles in radians
    """
    # Get atom indices
    idx_i = angle_indices[..., 0]  # [B, N_angles]
    idx_j = angle_indices[..., 1]  # [B, N_angles] (center atom)
    idx_k = angle_indices[..., 2]  # [B, N_angles]

    # Get coordinates [B, N_angles, 3]
    coords_i = torch.gather(coords, 1, idx_i.unsqueeze(-1).expand(-1, -1, 3))
    coords_j = torch.gather(coords, 1, idx_j.unsqueeze(-1).expand(-1, -1, 3))
    coords_k = torch.gather(coords, 1, idx_k.unsqueeze(-1).expand(-1, -1, 3))

    # Compute vectors from center
    r_ji = coords_i - coords_j  # [B, N_angles, 3]
    r_jk = coords_k - coords_j  # [B, N_angles, 3]

    # Compute angles
    dot_prod = torch.sum(r_ji * r_jk, dim=-1)
    norm_ji = torch.norm(r_ji, dim=-1)
    norm_jk = torch.norm(r_jk, dim=-1)

    cos_angle = dot_prod / (norm_ji * norm_jk + 1e-8)
    cos_angle = torch.clamp(cos_angle, -1.0 + eps, 1.0 - eps)
    angles = torch.acos(cos_angle)

    return angles


def _compute_dihedrals_from_coords(coords, dihedral_indices, mask):
    """
    Compute dihedral angles from coordinates and dihedral indices.

    Args:
        coords: [B, N, 3] coordinates
        dihedral_indices: [B, N_dihedrals, 4] indices of atoms forming dihedrals (i, j, k, l)
        mask: [B, N_dihedrals] mask for valid dihedrals

    Returns:
        dihedrals: [B, N_dihedrals] dihedral angles in radians
    """
    # Get atom indices
    idx_i = dihedral_indices[..., 0]
    idx_j = dihedral_indices[..., 1]
    idx_k = dihedral_indices[..., 2]
    idx_l = dihedral_indices[..., 3]

    # Get coordinates [B, N_dihedrals, 3]
    coords_i = torch.gather(coords, 1, idx_i.unsqueeze(-1).expand(-1, -1, 3))
    coords_j = torch.gather(coords, 1, idx_j.unsqueeze(-1).expand(-1, -1, 3))
    coords_k = torch.gather(coords, 1, idx_k.unsqueeze(-1).expand(-1, -1, 3))
    coords_l = torch.gather(coords, 1, idx_l.unsqueeze(-1).expand(-1, -1, 3))

    # Compute vectors
    r12 = coords_j - coords_i
    r23 = coords_k - coords_j
    r34 = coords_l - coords_k

    # Calculate dihedral angles
    cross_a = torch.cross(r12, r23, dim=-1)
    cross_b = torch.cross(r23, r34, dim=-1)
    cross_c = torch.cross(r23, cross_a, dim=-1)

    norm_a = torch.norm(cross_a, dim=-1)
    norm_b = torch.norm(cross_b, dim=-1)
    norm_c = torch.norm(cross_c, dim=-1)

    norm_cross_b = cross_b / (norm_b.unsqueeze(-1) + 1e-8)
    cos_phi = torch.sum(cross_a * norm_cross_b, dim=-1) / (norm_a + 1e-8)
    sin_phi = torch.sum(cross_c * norm_cross_b, dim=-1) / (norm_c + 1e-8)
    phi = -torch.atan2(sin_phi, cos_phi)

    return phi


def evaluate_bond_energy(dist, bond_params):
    """
    Evaluate harmonic bond energy: E = k0 * (r - r0)^2

    Args:
        dist: [B, N_bonds] bond distances
        bond_params: [B, N_bonds, 2] parameters (k0, d0)

    Returns:
        pot: [B, N_bonds] bond potential energy
    """
    k0 = bond_params[..., 0]
    d0 = bond_params[..., 1]
    x = dist - d0
    pot = k0 * (x**2)
    return pot


def evaluate_angle_energy(angles, angle_params):
    """
    Evaluate harmonic angle energy: E = k0 * (theta - theta0)^2

    Args:
        angles: [B, N_angles] angles in radians
        angle_params: [B, N_angles, 2] parameters (k0, theta0)

    Returns:
        pot: [B, N_angles] angle potential energy
    """
    k0 = angle_params[..., 0]
    theta0 = angle_params[..., 1]
    delta_theta = angles - theta0
    pot = k0 * delta_theta * delta_theta
    return pot


def evaluate_torsion_energy(phi, torsion_params, torsion_type="amber"):
    """
    Evaluate torsion (dihedral) energy.
    AMBER: E = k0 * (1 + cos(n*phi - phi0))
    CHARMM: E = k0 * (phi - phi0)^2

    Args:
        phi: [B, N_dihedrals] dihedral angles in radians
        torsion_params: [B, N_dihedrals, 3] parameters (k0, phi0, periodicity)
        torsion_type: 'amber' or 'charmm'

    Returns:
        pot: [B, N_dihedrals] torsion potential energy
    """
    k0 = torsion_params[..., 0]
    phi0 = torsion_params[..., 1]
    per = torsion_params[..., 2]

    if torsion_type == "amber":
        angle_diff = per * phi - phi0
        pot = k0 * (1 + torch.cos(angle_diff))
    else:  # charmm
        angle_diff = phi - phi0
        # Wrap to [-pi, pi]
        angle_diff = torch.where(
            angle_diff < -math.pi, angle_diff + 2 * math.pi, angle_diff
        )
        angle_diff = torch.where(
            angle_diff > math.pi, angle_diff - 2 * math.pi, angle_diff
        )
        pot = k0 * angle_diff**2

    return pot


def evaluate_lj_energy(dist, lj_params, switch_dist=None, cutoff=None):
    """
    Evaluate Lennard-Jones 12-6 energy: E = A/r^12 - B/r^6

    Args:
        dist: [B, N_pairs] pairwise distances
        lj_params: [B, N_pairs, 2] parameters (A, B) for 12-6 LJ potential
        switch_dist: switching distance for smoothing
        cutoff: cutoff distance

    Returns:
        pot: [B, N_pairs] LJ potential energy
    """
    aa = lj_params[..., 0]
    bb = lj_params[..., 1]

    rinv1 = 1 / (dist + 1e-8)
    rinv6 = rinv1**6
    rinv12 = rinv6 * rinv6

    pot = (aa * rinv12) - (bb * rinv6)

    # Apply switching function if specified
    if switch_dist is not None and cutoff is not None:
        mask = dist > switch_dist
        t = (dist[mask] - switch_dist) / (cutoff - switch_dist)
        switch_val = 1 + t * t * t * (-10 + t * (15 - t * 6))
        pot[mask] = pot[mask] * switch_val

    return pot


def evaluate_electrostatic_energy(
    dist, charges, scale=1.0, cutoff=None, rfa=False, solvent_dielectric=78.5
):
    """
    Evaluate electrostatic energy: E = ELEC_FACTOR * q_i * q_j / r

    Args:
        dist: [B, N_pairs] pairwise distances
        charges: [B, N_pairs, 2] charges of atom pairs
        scale: scaling factor (for 1-4 interactions)
        cutoff: cutoff distance
        rfa: use reaction field approximation
        solvent_dielectric: solvent dielectric constant for RFA

    Returns:
        pot: [B, N_pairs] electrostatic potential energy
    """
    q_i = charges[..., 0]
    q_j = charges[..., 1]

    if rfa and cutoff is not None:
        # Reaction field approximation
        denom = (2 * solvent_dielectric) + 1
        krf = (1 / cutoff**3) * (solvent_dielectric - 1) / denom
        crf = (1 / cutoff) * (3 * solvent_dielectric) / denom
        common = ELEC_FACTOR * q_i * q_j / scale
        dist2 = dist**2
        pot = common * ((1 / (dist + 1e-8)) + krf * dist2 - crf)
    else:
        pot = ELEC_FACTOR * q_i * q_j / ((dist + 1e-8) * scale)

    return pot


class TimeLossWeighting(torch.nn.Module):
    """Weight ~ 1 / (1 - t)^2 as used in the Proteina paper."""

    def __init__(
        self,
        max_value: float = 10.0,
        min_value: float = 0.05,
        zero_before: float = 0.0,
        exp: int = 1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.max_value = max_value
        self.min_value = min_value
        self.zero_before = zero_before
        self.exp = exp
        self.eps = eps

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Return `1 / (1 - t)^exp` clamped to [`min_value`, `max_value`]."""
        norm_scale = 1 / ((1 - t + self.eps) ** self.exp)
        norm_scale = torch.clamp(norm_scale, min=self.min_value, max=self.max_value)
        return norm_scale * (t > self.zero_before)


class LossComputer:
    """
    Centralized loss computation for the LigandPocketCFM model.
    Handles all loss types including coordinate, type, bond, charge, distance, affinity, and docking losses.
    """

    def __init__(
        self,
        coord_loss_weight: float = 1.0,
        coord_pocket_loss_weight: float = None,
        type_loss_weight: float = 1.0,
        bond_loss_weight: float = 1.0,
        charge_loss_weight: float = 1.0,
        hybridization_loss_weight: float = 1.0,
        distance_loss_weight_lig: float = None,
        distance_loss_weight_pocket: float = None,
        distance_loss_weight_lig_pocket: float = None,
        smooth_distance_loss_weight_lig: float | None = None,
        smooth_distance_loss_weight_pocket: float | None = None,
        smooth_distance_loss_weight_lig_pocket: float | None = None,
        plddt_confidence_loss_weight: float | None = None,
        affinity_loss_weight: float = None,
        docking_loss_weight: float = None,
        bond_angle_loss_weight: float | None = None,
        bond_angle_huber_delta: float = 0.5,
        bond_length_loss_weight: float | None = 1.0,
        dihedral_loss_weight: float | None = None,
        dihedral_huber_delta: float = 0.5,
        lj_loss_weight: float | None = 1.0,
        energy_loss_weight: float | None = None,
        energy_loss_weighting: str = "exponential",  # "constant", "inverse", "inverse_squared", "exponential"
        energy_loss_decay_rate: float = 0.5,  # decay rate for exponential weighting
        use_velocity_loss: bool = False,  # If True, use velocity loss instead of data loss for coordinates
        use_t_loss_weights: bool = False,
        predict_interactions: bool = False,
        flow_interactions: bool = False,
        type_strategy: str = "ce",
        bond_strategy: str = "ce",
    ):
        self.coord_loss_weight = coord_loss_weight
        self.coord_pocket_loss_weight = coord_pocket_loss_weight
        self.type_loss_weight = type_loss_weight
        self.bond_loss_weight = bond_loss_weight
        self.charge_loss_weight = charge_loss_weight
        self.hybridization_loss_weight = hybridization_loss_weight
        self.distance_loss_weight_lig = distance_loss_weight_lig
        self.distance_loss_weight_pocket = distance_loss_weight_pocket
        self.distance_loss_weight_lig_pocket = distance_loss_weight_lig_pocket
        self.smooth_distance_loss_weight_lig = smooth_distance_loss_weight_lig
        self.smooth_distance_loss_weight_pocket = smooth_distance_loss_weight_pocket
        self.smooth_distance_loss_weight_lig_pocket = (
            smooth_distance_loss_weight_lig_pocket
        )
        self.plddt_confidence_loss_weight = plddt_confidence_loss_weight
        self.affinity_loss_weight = affinity_loss_weight
        self.docking_loss_weight = docking_loss_weight
        self.bond_angle_loss_weight = bond_angle_loss_weight
        self.bond_angle_huber_delta = bond_angle_huber_delta
        self.bond_length_loss_weight = bond_length_loss_weight
        self.dihedral_loss_weight = dihedral_loss_weight
        self.dihedral_huber_delta = dihedral_huber_delta
        self.lj_loss_weight = lj_loss_weight
        self.energy_loss_weight = energy_loss_weight
        self.energy_loss_weighting = energy_loss_weighting
        self.energy_loss_decay_rate = energy_loss_decay_rate
        self.use_velocity_loss = use_velocity_loss
        self.predict_interactions = predict_interactions
        self.flow_interactions = flow_interactions
        self.type_strategy = type_strategy
        self.bond_strategy = bond_strategy
        self.t_loss_weights = (
            TimeLossWeighting(
                max_value=100,
                min_value=0.05,
                zero_before=0.0,
                exp=2,
                eps=1e-6,
            )
            if use_t_loss_weights
            else None
        )

    def _compute_velocity_from_data(
        self,
        x_pred: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Convert data prediction to velocity prediction.

        v_θ = (x_θ - z_t) / (1 - t)

        Args:
            x_pred: Predicted data (x_θ) with shape [B, N, 3]
            z_t: Interpolated state at time t with shape [B, N, 3]
            t: Time values with shape [B] or [B, 1] or [B, 1, 1]
            eps: Small constant for numerical stability near t=1

        Returns:
            v_pred: Predicted velocity with shape [B, N, 3]
        """
        # Ensure t has the right shape for broadcasting [B, 1, 1]
        if t.dim() == 1:
            t = t.unsqueeze(-1).unsqueeze(-1)
        elif t.dim() == 2:
            t = t.unsqueeze(-1)

        # Compute velocity: v = (x - z_t) / (1 - t)
        v_pred = (x_pred - z_t) / (1 - t + eps)
        return v_pred

    def _compute_target_velocity(
        self,
        x_1: torch.Tensor,
        x_0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute target velocity for linear interpolation.

        For linear interpolation z_t = (1-t) * x_0 + t * x_1, the velocity is:
        v = dz_t/dt = x_1 - x_0

        Args:
            x_1: Target data (ground truth) with shape [B, N, 3]
            x_0: Source/noise data with shape [B, N, 3]

        Returns:
            v_target: Target velocity with shape [B, N, 3]
        """
        return x_1 - x_0

    def compute_losses(
        self,
        data: dict,
        interpolated: dict,
        predicted: dict,
        prior: Optional[dict] = None,
        times: Optional[torch.Tensor] = None,
        pocket_data: Optional[dict] = None,
        inpaint_mode: bool = False,
    ):
        """
        Compute all ligand-related losses and return as a dictionary.

        Args:
            data: Ground truth data
            interpolated: Interpolated data
            predicted: Model predictions
        Returns:
            dict: Dictionary containing all computed losses
        """
        pred_coords = predicted["coords"]
        coords = data["coords"]
        mask = data["mask"]

        # if inpaint_mode:
        #     # NOTE: Needed potentially if t_loss_weights are used to not overexpress inpainted atoms
        #     # Inpainting mask
        #     mask = (~data["fragment_mask"].bool()) & mask  # (B, N_l)

        # Time-dependent Loss weights
        t_loss_weights_cont, t_loss_weights_disc = None, None
        times_cont = None
        if self.t_loss_weights is not None:
            times_atom = predicted.get("times", None)
            if times_atom is not None and len(times_atom) == 2:
                times_cont = times_atom[0].squeeze(-1)[:, 0]
                times_disc = times_atom[1].squeeze(-1)[:, 0]
                if self.t_loss_weights is not None:
                    t_loss_weights_cont = self.t_loss_weights(times_cont)
                    t_loss_weights_disc = self.t_loss_weights(times_disc)
            elif self.t_loss_weights is not None:
                raise AssertionError(
                    "Times must be provided for time-dependent losses for cont. and discr. types!"
                )

        # Coordinate losses - can be data loss or velocity loss
        if self.use_velocity_loss:
            assert (
                prior is not None
            ), "Prior data required for velocity loss computation."
            assert (
                interpolated is not None
            ), "Interpolated data required for velocity loss computation."
            assert times is not None, "Times required for velocity loss computation."

            # Get interpolated coordinates and source coordinates for velocity computation
            z_t_coords = interpolated.get("coords", None)
            x_0_coords = prior.get("coords", None)  # Source/noise coordinates

            if z_t_coords is None:
                raise ValueError(
                    "Interpolated coordinates (z_t) required for velocity loss. "
                    "Make sure 'coords' is in the interpolated dict."
                )
            if x_0_coords is None:
                raise ValueError(
                    "Source coordinates (x_0) required for velocity loss. "
                    "Make sure 'coords' is in the prior dict."
                )

            coord_loss = self.compute_velocity_loss(
                x_pred=pred_coords,
                x_target=coords,
                x_0=x_0_coords,
                z_t=z_t_coords,
                t=times[0],
                mask=mask,
                t_loss_weights=t_loss_weights_cont,
            )
        else:
            coord_loss = self.compute_coordinate_loss(
                coords, pred_coords, mask, t_loss_weights_cont
            )
        # Other losses
        type_loss = self.compute_type_loss(data, predicted, mask, t_loss_weights_disc)
        charge_loss = self.compute_charge_loss(
            data, predicted, mask, t_loss_weights_disc
        )
        bond_loss = (
            self.compute_bond_loss(data, predicted, mask, t_loss_weights_disc)
            if "bonds" in predicted
            else None
        )
        hybridization_loss = (
            self.compute_hybridization_loss(data, predicted, mask, t_loss_weights_disc)
            if "hybridization" in predicted
            else None
        )

        # Distance-based losses
        distance_losses = self.compute_distance_losses(
            data,
            predicted,
            mask,
            pocket_mask=pocket_data["mask"] if pocket_data else None,
            pocket_coords=pocket_data["coords"] if pocket_data else None,
            t_loss_weights=t_loss_weights_cont,
        )

        # pLDDT losses
        plddt_loss = self.compute_plddt_losses(
            data,
            predicted,
            mask,
            pocket_mask=pocket_data["mask"] if pocket_data else None,
            pocket_coords=pocket_data["coords"] if pocket_data else None,
            cutoff=10.0,
            thresholds=[0.5, 1.0, 2.0, 4.0],
        )

        # Smooth LDDT losses
        smooth_distance_losses = self.compute_smooth_lddt_losses(
            data,
            predicted,
            mask,
            pocket_mask=pocket_data["mask"] if pocket_data else None,
            pocket_coords=pocket_data["coords"] if pocket_data else None,
            t_loss_weights=t_loss_weights_cont,
        )

        # Affinity and docking score losses
        affinity_loss = self.compute_affinity_loss(data, predicted, t_loss_weights_cont)
        docking_loss = self.compute_docking_loss(data, predicted, t_loss_weights_cont)

        # Angle losses
        bond_angle_loss = self.compute_bond_angle_losses(
            data,
            predicted,
            mask,
            t_loss_weights=t_loss_weights_cont,
        )

        # Bond length losses
        bond_length_loss = self.compute_bond_length_losses(
            data,
            predicted,
            mask,
            t_loss_weights=t_loss_weights_cont,
        )

        # Energy-based loss
        energy_loss = self.compute_energy_loss(
            data,
            predicted,
            mask,
            t_loss_weights=t_loss_weights_cont,
        )

        # Combine all losses
        losses = {
            "coord-loss": coord_loss,
            "type-loss": type_loss,
            "charge-loss": charge_loss,
            **affinity_loss,
            **docking_loss,
            **distance_losses,
            **plddt_loss,
            **smooth_distance_losses,
        }
        if bond_loss is not None:
            losses["bond-loss"] = bond_loss
        if hybridization_loss is not None:
            losses["hybridization-loss"] = hybridization_loss
        if bond_angle_loss is not None:
            losses["bond-angle-loss"] = bond_angle_loss
        if bond_length_loss is not None:
            losses["bond-length-loss"] = bond_length_loss
        if energy_loss is not None:
            losses["energy-loss"] = energy_loss

        if self.predict_interactions or self.flow_interactions:
            interaction_loss = self._interaction_loss(
                data, interpolated, predicted, t_loss_weights_cont
            )
            losses["interaction-loss"] = interaction_loss

        return losses

    def compute_coordinate_loss(
        self, coords, pred_coords, mask, t_loss_weights=None, eps=1e-3
    ):
        """Compute coordinate prediction loss.
        Args:
            coords: Ground truth coordinates
            pred_coords: Predicted coordinates
            mask: Atom mask indicating valid atoms
            t_loss_weights: Optional time-dependent loss weights
            eps: Small constant for numerical stability
        Returns:
            torch.Tensor: Computed coordinate loss
        """
        coord_loss = F.mse_loss(pred_coords, coords, reduction="none")
        n_atoms = mask.unsqueeze(-1).sum(dim=(1, 2))
        coord_loss = (coord_loss * mask.unsqueeze(-1)).sum(dim=(1, 2)) / n_atoms

        if t_loss_weights is not None:
            coord_loss = coord_loss * t_loss_weights

        coord_loss = coord_loss.mean() * self.coord_loss_weight
        return coord_loss

    def compute_velocity_loss(
        self,
        x_pred: torch.Tensor,
        x_target: torch.Tensor,
        x_0: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor,
        t_loss_weights=None,
        eps: float = 1e-6,
    ):
        """
        Compute velocity prediction loss (v-loss).

        v_θ = (x_θ - z_t) / (1 - t)
        v_target = x_1 - x_0
        Loss = E[||v_θ - v_target||^2]

        Args:
            x_pred: Predicted data coordinates (x_θ) [B, N, 3]
            x_target: Ground truth coordinates (x_1) [B, N, 3]
            x_0: Source/noise coordinates [B, N, 3]
            z_t: Interpolated coordinates at time t [B, N, 3]
            t: Time values [B]
            mask: Atom mask indicating valid atoms [B, N]
            t_loss_weights: Optional time-dependent loss weights
            eps: Small constant for numerical stability

        Returns:
            torch.Tensor: Computed velocity loss
        """
        # Compute predicted velocity from data prediction
        v_pred = self._compute_velocity_from_data(x_pred, z_t, t, eps)

        # Compute target velocity
        v_target = self._compute_target_velocity(x_target, x_0)

        # MSE loss on velocities
        velocity_loss = F.mse_loss(v_pred, v_target, reduction="none")

        # Apply mask and normalize
        n_atoms = mask.unsqueeze(-1).sum(dim=(1, 2))
        velocity_loss = (velocity_loss * mask.unsqueeze(-1)).sum(dim=(1, 2)) / n_atoms

        if t_loss_weights is not None:
            velocity_loss = velocity_loss * t_loss_weights

        velocity_loss = velocity_loss.mean() * self.coord_loss_weight
        return velocity_loss

    def compute_type_loss(self, data, predicted, mask, t_loss_weights=None, eps=1e-3):
        """Compute atom type prediction loss."""
        pred_logits = predicted["atomics"]
        atomics_dist = data["atomics"]
        mask = mask.unsqueeze(2)

        batch_size, num_atoms, _ = pred_logits.size()

        if self.type_strategy == "mse":
            type_loss = F.mse_loss(pred_logits, atomics_dist, reduction="none")
        else:
            atomics = torch.argmax(atomics_dist, dim=-1).flatten(0, 1)
            type_loss = F.cross_entropy(
                pred_logits.flatten(0, 1), atomics, reduction="none"
            )
            type_loss = type_loss.unflatten(0, (batch_size, num_atoms)).unsqueeze(2)

        n_atoms = mask.sum(dim=(1, 2)) + eps
        type_loss = (type_loss * mask).sum(dim=(1, 2)) / n_atoms

        if t_loss_weights is not None:
            type_loss = type_loss * t_loss_weights

        return type_loss.mean() * self.type_loss_weight

    def compute_bond_loss(self, data, predicted, mask, t_loss_weights=None, eps=1e-3):
        """Compute bond prediction loss."""
        pred_logits = predicted["bonds"]
        bonds = torch.argmax(data["bonds"], dim=-1)

        batch_size, num_atoms, _, _ = pred_logits.size()

        bond_loss = F.cross_entropy(
            pred_logits.flatten(0, 2), bonds.flatten(0, 2), reduction="none"
        )
        bond_loss = bond_loss.unflatten(0, (batch_size, num_atoms, num_atoms))

        adj_matrix = smolF.adj_from_node_mask(mask, self_connect=True)
        n_bonds = adj_matrix.sum(dim=(1, 2)) + eps
        bond_loss = (bond_loss * adj_matrix).sum(dim=(1, 2)) / n_bonds

        if t_loss_weights is not None:
            bond_loss = bond_loss * t_loss_weights

        return bond_loss.mean() * self.bond_loss_weight

    def compute_charge_loss(self, data, predicted, mask, t_loss_weights=None, eps=1e-3):
        """Compute charge prediction loss."""
        pred_logits = predicted["charges"]
        charges_dist = data["charges"]
        mask = mask.unsqueeze(2)

        batch_size, num_atoms, _ = pred_logits.size()

        if self.type_strategy == "mse":
            charge_loss = F.mse_loss(pred_logits, charges_dist, reduction="none")
        else:
            charges = torch.argmax(charges_dist, dim=-1).flatten(0, 1)
            charge_loss = F.cross_entropy(
                pred_logits.flatten(0, 1), charges, reduction="none"
            )
            charge_loss = charge_loss.unflatten(0, (batch_size, num_atoms)).unsqueeze(2)

        n_atoms = mask.sum(dim=(1, 2)) + eps
        charge_loss = (charge_loss * mask).sum(dim=(1, 2)) / n_atoms

        if t_loss_weights is not None:
            charge_loss = charge_loss * t_loss_weights

        return charge_loss.mean() * self.charge_loss_weight

    def compute_hybridization_loss(
        self, data, predicted, mask, t_loss_weights=None, eps=1e-3
    ):
        """Compute hybridization prediction loss."""
        pred_logits = predicted["hybridization"]
        hybridization_dist = data["hybridization"]
        mask = mask.unsqueeze(2)

        batch_size, num_atoms, _ = pred_logits.size()

        if self.type_strategy == "mse":
            hybridization_loss = F.mse_loss(
                pred_logits, hybridization_dist, reduction="none"
            )
        else:
            hybridization = torch.argmax(hybridization_dist, dim=-1).flatten(0, 1)
            hybridization_loss = F.cross_entropy(
                pred_logits.flatten(0, 1), hybridization, reduction="none"
            )
            hybridization_loss = hybridization_loss.unflatten(
                0, (batch_size, num_atoms)
            ).unsqueeze(2)

        n_atoms = mask.sum(dim=(1, 2)) + eps
        hybridization_loss = (hybridization_loss * mask).sum(dim=(1, 2)) / n_atoms

        if t_loss_weights is not None:
            hybridization_loss = hybridization_loss * t_loss_weights

        return hybridization_loss.mean() * self.hybridization_loss_weight

    def compute_distance_losses(
        self,
        data,
        predicted,
        lig_mask,
        pocket_mask=None,
        pocket_coords=None,
        t_loss_weights=None,
        eps=1e-3,
    ):
        """
        Compute distance-based losses for ligand-ligand and ligand-pocket interactions.

        Returns:
            dict: Dictionary containing distance loss components
        """
        if not any(
            [
                self.distance_loss_weight_lig,
                self.distance_loss_weight_pocket,
                self.distance_loss_weight_lig_pocket,
            ]
        ):
            return {}

        true_coords = data["coords"]
        pred_coords = predicted["coords"]

        losses = {}

        if self.distance_loss_weight_lig:
            ligand_dist_loss = self._compute_distance_loss(
                pred_coords, true_coords, lig_mask, t_loss_weights=t_loss_weights
            )
            losses["ligand_distance_loss"] = (
                ligand_dist_loss * self.distance_loss_weight_lig
            )

        if self.distance_loss_weight_pocket and pocket_mask is not None:
            pocket_dist_loss = self._compute_distance_loss(
                pred_coords, true_coords, pocket_mask, t_loss_weights=t_loss_weights
            )
            losses["pocket_distance_loss"] = (
                pocket_dist_loss * self.distance_loss_weight_pocket
            )

        if self.distance_loss_weight_lig_pocket:
            if pocket_coords is None:
                lig_pocket_dist_loss = self._compute_complex_distance_loss(
                    pred_coords,
                    true_coords,
                    lig_mask,
                    pocket_mask,
                    t_loss_weights=t_loss_weights,
                )
            else:
                # For ligand-pocket distance loss, we need both ligand and pocket coords
                lig_pocket_dist_loss = self._compute_ligand_pocket_distance_loss(
                    pred_coords,
                    true_coords,
                    pocket_coords,
                    lig_mask,
                    pocket_mask,
                    t_loss_weights=t_loss_weights,
                )
            losses["ligand_pocket_distance_loss"] = (
                lig_pocket_dist_loss * self.distance_loss_weight_lig_pocket
            )

        return losses

    def _compute_distance_loss(
        self, pred_coords, true_coords, mask, t_loss_weights=None
    ):
        """Compute distance loss between atoms selected via masking."""

        # Compute pairwise distances
        pred_dists = torch.cdist(pred_coords, pred_coords, p=2)  # [B, N, N]
        true_dists = torch.cdist(true_coords, true_coords, p=2)  # [B, N, N]

        # Create pairwise mask (exclude self-interactions and non-ligand atoms)
        pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)  # [B, N, N]

        # Create identity matrix with correct dimensions [B, N, N]
        batch_size, max_atoms = mask.shape
        eye = (
            torch.eye(max_atoms, device=mask.device)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        pair_mask = pair_mask * (1 - eye)

        # Huber loss for robustness
        dist_diff = torch.abs(pred_dists - true_dists)
        huber_delta = 0.5
        dist_loss = torch.where(
            dist_diff < huber_delta,
            0.5 * dist_diff**2,
            huber_delta * (dist_diff - 0.5 * huber_delta),
        )

        # Apply mask and compute mean
        masked_loss = dist_loss * pair_mask
        n_pairs = pair_mask.sum(dim=(1, 2))
        loss = masked_loss.sum(dim=(1, 2)) / (n_pairs + 1e-8)

        if t_loss_weights is not None:
            loss = loss * t_loss_weights

        return loss.mean()

    def _compute_ligand_pocket_distance_loss(
        self,
        pred_lig_coords,
        true_lig_coords,
        pocket_coords,
        lig_mask,
        pocket_mask,
        cutoff: float = 10.0,
        sigma: float = 5.0,
        t_loss_weights=None,
    ):
        """
        Compute ligand-pocket distance loss (pocket is rigid/fixed).
        Only compares predicted ligand vs true ligand distances to the same fixed pocket.

        Uses distance-weighted loss to emphasize close contacts which are most
        important for binding pose accuracy.
        """
        # Vectorized implementation
        batch_size, max_lig_atoms, _ = pred_lig_coords.shape
        _, max_pocket_atoms, _ = pocket_coords.shape

        # Compute cross-distances: [B, N_lig, N_pocket]
        true_lig_pocket_dists = torch.cdist(true_lig_coords, pocket_coords, p=2)
        pred_lig_pocket_dists = torch.cdist(pred_lig_coords, pocket_coords, p=2)

        # Create cross mask for valid ligand-pocket pairs
        cross_mask = lig_mask.unsqueeze(2) * pocket_mask.unsqueeze(
            1
        )  # [B, N_lig, N_pocket]

        # Focus on nearby interactions (within cutoff)
        nearby_mask = (true_lig_pocket_dists < cutoff) * cross_mask

        # Distance difference
        dist_diff = torch.abs(pred_lig_pocket_dists - true_lig_pocket_dists)

        # Distance-weighted coefficient: weight closer contacts more heavily
        # k(r) = exp(-r / sigma) where sigma controls decay rate
        distance_weights = torch.exp(-true_lig_pocket_dists / sigma)

        # Huber loss for robustness to outliers
        huber_delta = 2.0
        huber_loss = torch.where(
            dist_diff < huber_delta,
            0.5 * dist_diff**2,
            huber_delta * (dist_diff - 0.5 * huber_delta),
        )

        # Apply weights and masks
        weighted_loss = huber_loss * distance_weights * nearby_mask

        # Normalize per sample
        n_nearby = nearby_mask.sum(dim=(1, 2)) + 1e-8
        per_sample_loss = weighted_loss.sum(dim=(1, 2)) / n_nearby

        # Apply time-dependent weights if provided
        if t_loss_weights is not None:
            per_sample_loss = per_sample_loss * t_loss_weights

        return per_sample_loss.mean()

    def compute_bond_angle_losses(
        self,
        data,
        predicted,
        mask,
        t_loss_weights=None,
        eps: float = 1e-8,
    ):
        if self.bond_angle_loss_weight is None or self.bond_angle_loss_weight == 0.0:
            return None
        bonds = data.get("bonds", None)
        if bonds is None:
            return None

        true_coords = data["coords"]
        pred_coords = predicted["coords"]
        weight = true_coords.new_tensor(float(self.bond_angle_loss_weight))

        bond_types = torch.argmax(bonds, dim=-1)
        adjacency = (bond_types > 0).float()
        batch_size, num_atoms, _ = adjacency.shape
        device = true_coords.device

        eye = torch.eye(num_atoms, device=device, dtype=adjacency.dtype).unsqueeze(0)
        adjacency = adjacency * (1 - eye)

        # triplet_mask[b, j, i, k] = 1 if i-j-k forms a valid angle
        # j is center, i and k are bonded to j
        triplet_mask = adjacency.unsqueeze(3) * adjacency.unsqueeze(2)  # [B, N, N, N]

        # Exclude cases where i == k
        diff_mask = (
            (1 - torch.eye(num_atoms, device=device, dtype=adjacency.dtype))
            .unsqueeze(0)
            .unsqueeze(1)
        )  # [1, 1, N, N]
        triplet_mask = triplet_mask * diff_mask

        # Only keep upper triangle to avoid duplicate angles (i-j-k vs k-j-i)
        upper_mask = (
            torch.triu(
                torch.ones(
                    (num_atoms, num_atoms), device=device, dtype=adjacency.dtype
                ),
                diagonal=1,
            )
            .unsqueeze(0)
            .unsqueeze(1)
        )  # [1, 1, N, N]
        triplet_mask = triplet_mask * upper_mask

        # Apply atom mask: all three atoms (i, j, k) must be valid
        atom_mask = mask.float()  # [B, N]
        triplet_mask = (
            triplet_mask
            * atom_mask.unsqueeze(2).unsqueeze(3)  # j must be valid [B, N, 1, 1]
            * atom_mask.unsqueeze(1).unsqueeze(3)  # i must be valid [B, 1, N, 1]
            * atom_mask.unsqueeze(1).unsqueeze(2)  # k must be valid [B, 1, 1, N]
        )

        if triplet_mask.sum() == 0:
            return weight.new_zeros(())

        angles_true, valid_true = self._compute_triplet_geometry(true_coords, eps=eps)
        angles_pred, valid_pred = self._compute_triplet_geometry(pred_coords, eps=eps)

        # Combine all validity checks
        valid_mask = (triplet_mask > 0) & valid_true & valid_pred

        if valid_mask.sum() == 0:
            return weight.new_zeros(())

        # Only compute angle diff for valid entries
        angle_diff = torch.abs(angles_pred - angles_true)
        # Replace any remaining NaNs with 0 (shouldn't happen but safe)
        angle_diff = torch.where(valid_mask, angle_diff, torch.zeros_like(angle_diff))

        delta = self.bond_angle_huber_delta
        huber_loss = torch.where(
            angle_diff <= delta,
            0.5 * angle_diff**2,
            delta * (angle_diff - 0.5 * delta),
        )
        huber_loss = huber_loss * valid_mask.float()

        valid_counts = valid_mask.float().sum(dim=(1, 2, 3))
        loss_sum = huber_loss.sum(dim=(1, 2, 3))
        per_sample_loss = torch.where(
            valid_counts > 0,
            loss_sum / (valid_counts + eps),
            loss_sum.new_zeros(loss_sum.shape),
        )

        if t_loss_weights is not None:
            per_sample_loss = per_sample_loss * t_loss_weights

        loss = per_sample_loss.mean()
        return loss * weight

    def _compute_triplet_geometry(self, coords, eps: float = 1e-8):
        """
        Compute bond angles for all triplets (i, j, k) where j is the center atom.
        Returns angles in radians and a validity mask.

        Returns:
            angles: [B, N(j), N(i), N(k)] angles in radians
            valid_mask: [B, N(j), N(i), N(k)] boolean mask for valid angles
        """
        # coords: [B, N, 3]
        batch_size, num_atoms, _ = coords.shape

        # Compute all pairwise vectors: vec[b, j, i] = coords[b, i] - coords[b, j]
        # This gives us vectors FROM atom j TO atom i
        coords_i = coords.unsqueeze(2)  # [B, N, 1, 3]
        coords_j = coords.unsqueeze(1)  # [B, 1, N, 3]
        vecs = (
            coords_i - coords_j
        )  # [B, N, N, 3] where vecs[b, j, i] = coords[b,i] - coords[b,j]

        # For angle i-j-k: need vec[j,i] and vec[j,k]
        vec_ji = vecs.unsqueeze(3)  # [B, N(j), N(i), 1, 3]
        vec_jk = vecs.unsqueeze(2)  # [B, N(j), 1, N(k), 3]

        # Compute dot product
        dot = (vec_ji * vec_jk).sum(dim=-1)  # [B, N(j), N(i), N(k)]

        # Compute norms
        norm_ji = torch.linalg.norm(vec_ji, dim=-1)  # [B, N(j), N(i), 1]
        norm_jk = torch.linalg.norm(vec_jk, dim=-1)  # [B, N(j), 1, N(k)]

        # Check validity: both vectors must have non-zero length
        valid_mask = (norm_ji > eps) & (norm_jk > eps)

        # Compute cosine of angle
        denominator = norm_ji * norm_jk
        # Avoid division by zero by using where
        cos_angle = torch.where(
            valid_mask, dot / torch.clamp(denominator, min=eps), torch.zeros_like(dot)
        )

        # Clamp to valid range for acos with extra margin for numerical safety
        cos_angle = torch.clamp(cos_angle, -1.0 + 1e-6, 1.0 - 1e-6)

        # Compute angles - set invalid ones to 0 (they'll be masked out anyway)
        angles = torch.where(
            valid_mask, torch.acos(cos_angle), torch.zeros_like(cos_angle)
        )

        return angles, valid_mask

    def compute_bond_length_losses(
        self,
        data,
        predicted,
        mask,
        t_loss_weights=None,
        eps: float = 1e-8,
    ):
        """
        Compute bond length losses - encourages predicted bonds to have similar lengths to reference.

        Args:
            data: Ground truth data containing bond adjacency information
            predicted: Predicted coordinates
            mask: Atom mask
            t_loss_weights: Optional time-dependent loss weights
            eps: Small constant for numerical stability

        Returns:
            torch.Tensor or None: Bond length loss
        """
        if self.bond_length_loss_weight is None or self.bond_length_loss_weight == 0.0:
            return None

        bonds = data.get("bonds", None)
        if bonds is None:
            return None

        true_coords = data["coords"]
        pred_coords = predicted["coords"]

        # Get bond adjacency matrix: [B, N, N, num_bond_types]
        bond_types = torch.argmax(bonds, dim=-1)  # [B, N, N]
        adjacency = (bond_types > 0).float()  # [B, N, N] binary mask
        batch_size, num_atoms, _ = adjacency.shape
        device = true_coords.device

        # Remove self-connections
        eye = torch.eye(num_atoms, device=device, dtype=adjacency.dtype).unsqueeze(0)
        adjacency = adjacency * (1 - eye)

        # Only keep upper triangle to avoid counting each bond twice
        upper_mask = torch.triu(
            torch.ones((num_atoms, num_atoms), device=device, dtype=adjacency.dtype),
            diagonal=1,
        ).unsqueeze(
            0
        )  # [1, N, N]
        adjacency = adjacency * upper_mask

        # Apply atom mask: both atoms in bond must be valid
        atom_mask = mask.float()  # [B, N]
        bond_mask = atom_mask.unsqueeze(2) * atom_mask.unsqueeze(1)  # [B, N, N]
        adjacency = adjacency * bond_mask

        if adjacency.sum() == 0:
            return torch.tensor(0.0, device=device)

        # Compute pairwise distances
        true_dists = torch.cdist(true_coords, true_coords, p=2)  # [B, N, N]
        pred_dists = torch.cdist(pred_coords, pred_coords, p=2)  # [B, N, N]

        # Simple MSE loss on bond lengths
        bond_length_diff = (pred_dists - true_dists) ** 2
        bond_loss = bond_length_diff * adjacency

        valid_bonds = adjacency.sum(dim=(1, 2))
        loss_sum = bond_loss.sum(dim=(1, 2))
        per_sample_loss = torch.where(
            valid_bonds > 0,
            loss_sum / (valid_bonds + eps),
            loss_sum.new_zeros(loss_sum.shape),
        )

        if t_loss_weights is not None:
            per_sample_loss = per_sample_loss * t_loss_weights

        loss = per_sample_loss.mean()
        return loss * self.bond_length_loss_weight

    def compute_dihedral_losses(
        self,
        data,
        predicted,
        mask,
        t_loss_weights=None,
        eps: float = 1e-8,
    ):
        """
        Compute dihedral angle losses - encourages similar dihedral angles between predicted and reference.
        Identifies proper dihedrals from bond connectivity and computes angular differences.

        Args:
            data: Ground truth data containing bond adjacency information
            predicted: Predicted coordinates
            mask: Atom mask
            t_loss_weights: Optional time-dependent loss weights
            eps: Small constant for numerical stability

        Returns:
            torch.Tensor or None: Dihedral loss
        """
        if self.dihedral_loss_weight is None or self.dihedral_loss_weight == 0.0:
            return None

        bonds = data.get("bonds", None)
        if bonds is None:
            return None

        true_coords = data["coords"]
        pred_coords = predicted["coords"]

        # Get bond adjacency matrix
        bond_types = torch.argmax(bonds, dim=-1)  # [B, N, N]
        adjacency = (bond_types > 0).float()  # [B, N, N]
        batch_size, num_atoms, _ = adjacency.shape
        device = true_coords.device

        # Remove self-connections
        eye = torch.eye(num_atoms, device=device, dtype=adjacency.dtype).unsqueeze(0)
        adjacency = adjacency * (1 - eye)

        # Find proper dihedrals: i-j-k-l where there are bonds i-j, j-k, k-l
        # This is a 4D tensor construction - simplified approach
        # adjacency[b, i, j] = 1 if i-j bonded
        # We want all i-j-k-l where i-j, j-k, k-l are bonded

        # For computational efficiency, we'll identify dihedrals by finding
        # paths of length 3 in the bond graph
        adj_2 = torch.bmm(adjacency, adjacency)  # [B, N, N] paths of length 2
        adj_3 = torch.bmm(adj_2, adjacency)  # [B, N, N] paths of length 3

        # Exclude trivial paths (back and forth)
        adj_3 = adj_3 * (1 - eye)
        # Exclude 1-2 and 1-3 neighbors
        adj_3 = adj_3 * (1 - adjacency) * (1 - adj_2)

        # Create mask for valid 1-4 pairs (ends of proper dihedrals)
        pair_14_mask = (adj_3 > 0).float()

        # Apply atom mask
        atom_mask = mask.float()  # [B, N]
        pair_14_mask = pair_14_mask * atom_mask.unsqueeze(2) * atom_mask.unsqueeze(1)

        if pair_14_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        # Compute all pairwise distances for 1-4 pairs
        # This is a simplified approach: we penalize distance differences for 1-4 pairs
        # which indirectly encourages similar dihedral angles
        true_dists = torch.cdist(true_coords, true_coords, p=2)  # [B, N, N]
        pred_dists = torch.cdist(pred_coords, pred_coords, p=2)  # [B, N, N]

        # MSE on 1-4 distances (proxy for dihedral angles)
        dist_diff = (pred_dists - true_dists) ** 2
        dihedral_loss = dist_diff * pair_14_mask

        valid_pairs = pair_14_mask.sum(dim=(1, 2))
        loss_sum = dihedral_loss.sum(dim=(1, 2))
        per_sample_loss = torch.where(
            valid_pairs > 0,
            loss_sum / (valid_pairs + eps),
            loss_sum.new_zeros(loss_sum.shape),
        )

        if t_loss_weights is not None:
            per_sample_loss = per_sample_loss * t_loss_weights

        loss = per_sample_loss.mean()
        return loss * self.dihedral_loss_weight

    def compute_lj_losses(
        self,
        data,
        predicted,
        mask,
        t_loss_weights=None,
        cutoff: float = 5.0,
        eps: float = 1e-8,
    ):
        """
        Compute LJ-like non-bonded losses - encourages non-bonded atoms to maintain similar distances.
        Focuses on preventing clashes (atoms getting too close) and maintaining favorable interactions.

        Args:
            data: Ground truth data
            predicted: Predicted coordinates
            mask: Atom mask
            t_loss_weights: Optional time-dependent loss weights
            cutoff: Cutoff distance for non-bonded interactions
            eps: Small constant for numerical stability

        Returns:
            torch.Tensor or None: LJ loss
        """
        if self.lj_loss_weight is None or self.lj_loss_weight == 0.0:
            return None

        true_coords = data["coords"]
        pred_coords = predicted["coords"]

        # Get pairwise distances
        pair_mask = _get_pair_mask(mask)
        true_dists = _get_distance_matrix(true_coords, mask)
        pred_dists = _get_distance_matrix(pred_coords, mask)

        # Exclude bonded atoms (1-2 and 1-3 interactions)
        bonds = data.get("bonds", None)
        if bonds is not None:
            bond_types = torch.argmax(bonds, dim=-1)
            bonded_mask = (bond_types > 0).float()
            # Also exclude 1-3 interactions (atoms bonded to same atom)
            bonded_13 = torch.bmm(bonded_mask, bonded_mask)
            exclusion_mask = 1.0 - torch.clamp(bonded_mask + bonded_13, 0, 1)
            pair_mask = pair_mask * exclusion_mask

        # Apply cutoff mask - focus on nearby non-bonded interactions
        cutoff_mask = (true_dists < cutoff) * pair_mask

        if cutoff_mask.sum() == 0:
            return torch.tensor(0.0, device=mask.device)

        # Simple distance preservation for non-bonded pairs
        # This encourages maintaining similar packing and preventing clashes
        dist_diff = (pred_dists - true_dists) ** 2
        lj_loss = dist_diff * cutoff_mask

        valid_counts = cutoff_mask.sum(dim=(1, 2))
        loss_sum = lj_loss.sum(dim=(1, 2))
        per_sample_loss = torch.where(
            valid_counts > 0,
            loss_sum / (valid_counts + eps),
            loss_sum.new_zeros(loss_sum.shape),
        )

        if t_loss_weights is not None:
            per_sample_loss = per_sample_loss * t_loss_weights

        loss = per_sample_loss.mean() * self.lj_loss_weight
        return loss

    def compute_energy_loss(
        self,
        data,
        predicted,
        mask,
        t_loss_weights=None,
        eps: float = 1e-8,
    ):
        """
        Compute energy-based loss function inspired by https://arxiv.org/pdf/2511.02087.

        E(ŷ, y) = Σ_{i,j} 1/2 * k_{ij}(y) * (||y_i - y_j|| - ||ŷ_i - ŷ_j||)^2

        where y is the ground truth coordinates and ŷ is the predicted coordinates.
        The coefficients k_{ij}(y) can be:
        - constant: k_{ij} = 1
        - inverse: k_{ij} = 1 / ||y_i - y_j||
        - inverse_squared: k_{ij} = 1 / ||y_i - y_j||^2
        - exponential: k_{ij} = exp(-decay_rate * ||y_i - y_j||)

        Args:
            data: Ground truth data containing 'coords'
            predicted: Predicted data containing 'coords'
            mask: Atom mask indicating valid atoms [B, N]
            t_loss_weights: Optional time-dependent loss weights
            eps: Small constant for numerical stability

        Returns:
            torch.Tensor or None: Energy loss if weight is set, None otherwise
        """
        if self.energy_loss_weight is None or self.energy_loss_weight == 0.0:
            return None

        true_coords = data["coords"]  # [B, N, 3]
        pred_coords = predicted["coords"]  # [B, N, 3]

        # Compute pairwise distance matrices
        true_dists = _get_distance_matrix(true_coords, mask)  # [B, N, N]
        pred_dists = _get_distance_matrix(pred_coords, mask)  # [B, N, N]

        # Get pair mask (excludes self-interactions and invalid atoms)
        pair_mask = _get_pair_mask(mask)  # [B, N, N]

        # Compute distance-dependent coefficients k_{ij}(y)
        k_ij = self._compute_energy_coefficients(true_dists, pair_mask, eps)

        # Compute the energy loss: k_{ij} * (||y_i - y_j|| - ||ŷ_i - ŷ_j||)^2
        dist_diff_squared = (true_dists - pred_dists) ** 2  # [B, N, N]
        energy_loss = 0.5 * k_ij * dist_diff_squared  # [B, N, N]

        # Apply pair mask
        energy_loss = energy_loss * pair_mask

        # Normalize by number of valid pairs per sample
        n_pairs = pair_mask.sum(dim=(1, 2))  # [B]
        per_sample_loss = energy_loss.sum(dim=(1, 2)) / (n_pairs + eps)  # [B]

        # Apply time-dependent weights if provided
        if t_loss_weights is not None:
            per_sample_loss = per_sample_loss * t_loss_weights

        # Return weighted mean loss
        return per_sample_loss.mean() * self.energy_loss_weight

    def _compute_energy_coefficients(
        self,
        true_dists: torch.Tensor,
        pair_mask: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Compute the distance-dependent coefficients k_{ij}(y) for the energy loss.

        Args:
            true_dists: Ground truth pairwise distances [B, N, N]
            pair_mask: Valid pair mask [B, N, N]
            eps: Small constant for numerical stability

        Returns:
            torch.Tensor: Coefficients k_{ij} with shape [B, N, N]
        """
        if self.energy_loss_weighting == "constant":
            # k_{ij} = 1
            k_ij = torch.ones_like(true_dists)

        elif self.energy_loss_weighting == "inverse":
            # k_{ij} = 1 / ||y_i - y_j||
            # Weight closer atoms more heavily
            k_ij = 1.0 / (true_dists + eps)

        elif self.energy_loss_weighting == "inverse_squared":
            # k_{ij} = 1 / ||y_i - y_j||^2
            # Weight closer atoms even more heavily
            k_ij = 1.0 / (true_dists**2 + eps)

        elif self.energy_loss_weighting == "exponential":
            # k_{ij} = exp(-decay_rate * ||y_i - y_j||)
            # Smooth exponential decay with distance
            k_ij = torch.exp(-self.energy_loss_decay_rate * true_dists)

        else:
            raise ValueError(
                f"Unknown energy_loss_weighting: {self.energy_loss_weighting}. "
                f"Choose from: 'constant', 'inverse', 'inverse_squared', 'exponential'"
            )

        # Zero out invalid pairs
        k_ij = k_ij * pair_mask

        return k_ij

    def compute_smooth_lddt_losses(
        self,
        data,
        predicted,
        lig_mask,
        pocket_mask=None,
        pocket_coords=None,
        cutoff=12.0,
        t_loss_weights=None,
    ):
        """
        Compute smooth lddt distance-based losses for ligand-ligand and ligand-pocket interactions.
        Returns:
            dict: Dictionary containing distance loss components
        """

        if not any(
            [
                self.smooth_distance_loss_weight_lig,
                self.smooth_distance_loss_weight_pocket,
                self.smooth_distance_loss_weight_lig_pocket,
            ]
        ):
            return {}

        true_coords = data["coords"]
        pred_coords = predicted["coords"]

        losses = {}

        if self.smooth_distance_loss_weight_lig:
            # ligand-ligand
            ligand_dist_loss = self._compute_smooth_lddt_loss(
                pred_coords,
                true_coords,
                lig_mask,
                cutoff=cutoff,
                t_loss_weights=t_loss_weights,
            )
            losses["ligand_smooth_lddt_loss"] = (
                ligand_dist_loss * self.smooth_distance_loss_weight_lig
            )

        if self.smooth_distance_loss_weight_pocket and pocket_mask is not None:
            # pocket-pocket
            pocket_dist_loss = self._compute_smooth_lddt_loss(
                pred_coords,
                true_coords,
                pocket_mask,
                cutoff=cutoff,
                t_loss_weights=t_loss_weights,
            )
            losses["pocket_smooth_lddt_loss"] = (
                pocket_dist_loss * self.smooth_distance_loss_weight_pocket
            )

        if self.smooth_distance_loss_weight_lig_pocket and pocket_mask is not None:
            # jointly ligand-pocket
            if pocket_coords is None:
                # jointly
                # lig_mask, pocket_mask with shape [B, N_complex], [B, N_complex]
                joint_mask = lig_mask + pocket_mask
                lig_pocket_dist_loss = self._compute_smooth_lddt_loss(
                    pred_coords,
                    true_coords,
                    joint_mask,
                    cutoff=cutoff,
                    t_loss_weights=t_loss_weights,
                )
            else:
                # only ligand and fixed pocket
                # lig_mask, pocket_mask with shape [B, N_lig], [B, N_pocket]
                joint_mask = torch.cat([lig_mask, pocket_mask], dim=1)
                pred_coords_with_true_pocket = torch.cat(
                    [pred_coords, pocket_coords], dim=1
                )
                true_coords_with_true_pocket = torch.cat(
                    [true_coords, pocket_coords], dim=1
                )
                lig_pocket_dist_loss = self._compute_smooth_lddt_loss(
                    pred_coords_with_true_pocket,
                    true_coords_with_true_pocket,
                    joint_mask,
                    cutoff=cutoff,
                    t_loss_weights=t_loss_weights,
                )
            losses["ligand_pocket_smooth_lddt_loss"] = (
                lig_pocket_dist_loss * self.smooth_distance_loss_weight_lig_pocket
            )

        return losses

    def compute_plddt_losses(
        self,
        data,
        predicted,
        lig_mask,
        pocket_mask,
        pocket_coords,
        cutoff=11.0,
        thresholds=[0.5, 1.0, 2.0, 4.0],
    ):
        if self.plddt_confidence_loss_weight is None or "plddt" not in predicted:
            return {}

        true_coords = data["coords"]
        pred_coords = predicted["coords"]
        pred_lddt = predicted.get("plddt", None)

        losses = {}

        assert pocket_mask is not None, "Pocket mask is required for pLDDT loss"
        assert pocket_coords is not None, "Pocket coords are required for pLDDT loss"
        plddt_confidence_loss = self._compute_plddt_loss(
            coords_true_ligand=true_coords,
            coords_pred_ligand=pred_coords,
            pred_lddt=pred_lddt,
            mask_ligand=lig_mask,
            coords_true_pocket=pocket_coords,
            mask_pocket=pocket_mask,
            cutoff=cutoff,
            dist_thresholds=thresholds,
        )
        losses["plddt_confidence_loss"] = (
            plddt_confidence_loss * self.plddt_confidence_loss_weight
        )

        return losses

    def _compute_smooth_lddt_loss(
        self, pred_coords, true_coords, mask, cutoff: float = 12.0, t_loss_weights=None
    ):
        """Compute smooth lDDT loss based on predicted and true coordinates.
        Algorithm 27 in https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf
        Used in https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/loss/diffusion.py#L97-L171
        """
        pair_mask = _get_pair_mask(mask)
        pred_dists = _get_distance_matrix(pred_coords, mask)
        true_dists = _get_distance_matrix(true_coords, mask)
        nearby_mask = true_dists < cutoff
        dist_diff = torch.abs(pred_dists - true_dists)

        # TODO: experiment with different sigmoid parameters
        eps = (
            F.sigmoid(0.5 - dist_diff)
            + F.sigmoid(1.0 - dist_diff)
            + F.sigmoid(2.0 - dist_diff)
            + F.sigmoid(4.0 - dist_diff)
        ) / 4.0

        chosen_mask = pair_mask * nearby_mask
        masked_loss = eps * chosen_mask
        n_pairs = chosen_mask.sum(dim=(1, 2))

        lddt = masked_loss.sum(dim=(1, 2)) / (n_pairs + 1e-8)
        if t_loss_weights is not None:
            lddt = lddt * t_loss_weights
        loss = 1.0 - lddt.mean()
        return loss

    def _compute_lddt_loss(
        self,
        coords_true_ligand,
        coords_pred_ligand,
        mask_ligand,
        coords_true_pocket,
        mask_pocket,
        cutoff: float = 12.0,
        dist_thresholds: list | None = None,
    ):
        """Compute lDDT loss based on (predicted) coordinates.
        This version uses continuous distances.
        See Section 4.3.1 in https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf

        In practice, assume N_ligand, N_pocket is considered
        """
        cross_pair_mask = _get_cross_pair_mask(
            mask_ligand, mask_pocket
        )  # [B, N_l, N_p]
        dist_true = _get_cross_distance_matrix(
            coords_true_ligand, coords_true_pocket, mask_ligand, mask_pocket
        )  # [B, N_l, N_p]
        dist_pred = _get_cross_distance_matrix(
            coords_pred_ligand, coords_true_pocket, mask_ligand, mask_pocket
        )  # [B, N_l, N_p]

        dist_to_consider = (dist_true < cutoff) * cross_pair_mask  # binary target mask
        dist_loss = (
            torch.abs(dist_pred - dist_true) * cross_pair_mask
        )  # continuous loss
        lddts = []
        if dist_thresholds is None:
            dist_thresholds = [0.5, 1.0, 2.0, 4.0]
        lddts = [
            ((dist_loss < thresh) * cross_pair_mask).float()
            for thresh in dist_thresholds
        ]
        lddts = torch.stack(lddts, dim=-1)  # [B, N_l, N_p, num_bins]
        lddts = lddts.mean(dim=-1)  # Average over bins, # [B, N_l, N_p]
        # lddts \in (0, 1) with shape # [B, N_l, N_p]
        mask_has_no_match = (torch.sum(dist_to_consider, dim=-1) != 0).float()
        # normalize over last dimension (pocket atoms)
        lddts = torch.sum(dist_to_consider * lddts, dim=-1)  # [B, N_l]
        norm = 1.0 / (1e-10 + torch.sum(dist_to_consider, dim=-1))
        lddts = lddts * norm
        return lddts, mask_has_no_match

    def _compute_plddt_loss(
        self,
        coords_true_ligand,
        coords_pred_ligand,
        pred_lddt,
        mask_ligand,
        coords_true_pocket,
        mask_pocket,
        cutoff: float = 12.0,
        dist_thresholds: list | None = None,
    ):
        """Compute binned lDDT loss based on (predicted) coordinates.
        This version bins the distances into discrete intervals.
        See Section 4.3.1 in https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf

        In practice, assume N_ligand, N_pocket is considered

        coords_true_ligand: Ground truth coordinates: [B, N_l, 3]
        coords_pred_ligand: Predicted coordinates: [B, N_l, 3]
        pred_lddt: Predicted lDDT values: [B, N_l, num_bins], in AF3 Supplementary num_bins is set to 50
        mask_ligand: Atom mask indicating valid atoms: [B, N_l]
        coords_true_pocket: Ground truth coordinates: [B, N_p, 3]
        mask_pocket: Atom mask indicating valid atoms: [B, N_p]
        """
        if dist_thresholds is None:
            dist_thresholds = [0.5, 1.0, 2.0, 4.0]

        target_lddt, mask_has_no_match = self._compute_lddt_loss(
            coords_true_ligand,
            coords_pred_ligand,
            mask_ligand,
            coords_true_pocket,
            mask_pocket,
            cutoff=cutoff,
            dist_thresholds=dist_thresholds,
        )
        num_bins = pred_lddt.shape[-1]
        bin_index = torch.floor(target_lddt * num_bins).long()
        bin_index = torch.clamp(bin_index, max=(num_bins - 1))
        lddt_one_hot = torch.nn.functional.one_hot(bin_index, num_classes=num_bins)
        errors = -1 * torch.sum(
            lddt_one_hot * torch.nn.functional.log_softmax(pred_lddt, dim=-1),
            dim=-1,
        )
        loss = torch.sum(errors * mask_has_no_match, dim=-1) / (
            1e-7 + torch.sum(mask_has_no_match, dim=-1)
        )
        loss = torch.mean(loss)  # Average over batch
        return loss

    def compute_affinity_loss(self, data, predicted, t_loss_weights=None, eps=1e-8):
        """
        Compute individual binding affinity losses (Huber loss for pIC50/pKi/pKd/pEC50/Kiba values).

        Args:
            data: Ground truth data containing affinity TensorDict
            predicted: Model predictions containing affinity TensorDict
            t_loss_weights: Optional time-dependent loss weights
            eps: Small constant for numerical stability

        Returns:
            dict or None: Dictionary of individual affinity losses if available, None otherwise
        """
        if "affinity" not in data or predicted["affinity"] is None:
            return {}

        pred_affinity_dict = predicted[
            "affinity"
        ]  # TensorDict with individual predictions
        true_affinity_dict = data[
            "affinity"
        ]  # TensorDict with individual ground truth values

        affinity_losses = {}
        affinity_types = ["pic50", "pkd", "pki", "pec50"]

        for affinity_type in affinity_types:

            pred_values = pred_affinity_dict[affinity_type].flatten()  # [B]
            true_values = true_affinity_dict[affinity_type].float()  # [B]

            # Create mask for valid affinity values (non-NaN, non-negative for log affinity values)
            valid_mask = torch.isfinite(true_values) & (true_values >= 0)

            if valid_mask.sum() == 0:
                # No valid values for this affinity type, but process to avoid unused parameters
                dummy_loss = (pred_values * 0.0).sum()
                affinity_losses[f"{affinity_type}_loss"] = dummy_loss
                continue

            # Huber loss for valid samples only (more robust than MSE)
            pred_valid = pred_values[valid_mask]
            true_valid = true_values[valid_mask]

            # Huber loss with delta=1.0
            huber_loss = F.huber_loss(
                pred_valid, true_valid, reduction="none", delta=1.0
            )

            if t_loss_weights is not None:
                # Apply time weights only to valid samples
                valid_t_weights = t_loss_weights[valid_mask]
                huber_loss = huber_loss * valid_t_weights

            affinity_losses[f"{affinity_type}_loss"] = (
                huber_loss.mean() * self.affinity_loss_weight
            )

        return affinity_losses

    def compute_docking_loss(self, data, predicted, t_loss_weights=None, eps=1e-8):
        """
        Compute individual docking score losses (Huber loss for docking scores).

        Args:
            data: Ground truth data containing docking score TensorDict
            predicted: Model predictions containing docking score TensorDict
            t_loss_weights: Optional time-dependent loss weights
            eps: Small constant for numerical stability

        Returns:
            dict or None: Dictionary of individual docking losses if available, None otherwise
        """
        if "docking_score" not in data or predicted["docking_score"] is None:
            return {}

        pred_docking_dict = predicted[
            "docking_score"
        ]  # TensorDict with individual predictions
        true_docking_dict = data[
            "docking_score"
        ]  # TensorDict with individual ground truth values

        docking_losses = {}

        # Process each docking score type individually
        docking_types = ["vina_score", "gnina_score"]

        for docking_type in docking_types:

            pred_values = pred_docking_dict[docking_type].flatten()  # [B]
            true_values = true_docking_dict[docking_type].float()  # [B]

            # Create mask for valid docking values (non-NaN, reasonable range for docking scores)
            valid_mask = (
                torch.isfinite(true_values)
                & (true_values > -100)
                & (true_values < 50)  # Expanded upper bound to be more inclusive
            )

            if valid_mask.sum() == 0:
                # No valid values for this docking type, but process to avoid unused parameters
                dummy_loss = (pred_values * 0.0).sum()
                docking_losses[f"{docking_type}_loss"] = dummy_loss
                continue

            # Huber loss for valid samples only (more robust than MSE)
            pred_valid = pred_values[valid_mask]
            true_valid = true_values[valid_mask]

            # Huber loss with delta=2.0 (docking scores can have larger variations)
            huber_loss = F.huber_loss(
                pred_valid, true_valid, reduction="none", delta=2.0
            )

            if t_loss_weights is not None:
                # Apply time weights only to valid samples
                valid_t_weights = t_loss_weights[valid_mask]
                huber_loss = huber_loss * valid_t_weights

            docking_losses[f"{docking_type}_loss"] = (
                huber_loss.mean() * self.docking_loss_weight
            )

        return docking_losses

    def _interaction_loss(
        self, data, interpolated, predicted, t_loss_weights=None, eps=1e-3
    ):
        gamma = 2.0
        alpha = (
            torch.tensor([0.5, 1.6, 1.6, 1.6, 1.7, 1.8, 1.8, 1.4, 2.0, 2.0])
            .float()
            .to(self.device)
        )
        lig_mask = data["mask"].bool()
        pocket_mask = data["pocket_mask"].bool()
        combined_mask = pocket_mask[:, :, None] * lig_mask[:, None, :]

        pred_logits = predicted["interactions"].permute(0, 2, 1, 3)
        interactions = torch.argmax(data["interactions"].permute(0, 2, 1, 3), dim=-1)

        num_actual_interactions = max(interactions.count_nonzero(dim=-1).sum(), 1)

        ce_loss = F.cross_entropy(
            pred_logits.flatten(0, 2),
            interactions.flatten(0, 2),
            reduction="none",
        )
        ce_loss = ce_loss.unflatten(
            0, (pred_logits.size(0), pred_logits.size(1), pred_logits.size(2))
        )
        pt = torch.exp(-ce_loss)
        focal_factor = (1 - pt) ** gamma

        if isinstance(alpha, torch.Tensor):
            alpha_factor = alpha[interactions]
        else:
            alpha_factor = alpha

        interaction_loss = focal_factor * alpha_factor * ce_loss
        interaction_loss = (interaction_loss * combined_mask).sum(
            dim=(1, 2)
        ) / num_actual_interactions  # (
        # combined_mask.sum(dim=(1, 2)) + eps
        # )
        return interaction_loss.mean() * self.interaction_loss_weight

    def compute_complex_losses(self, data, interpolated, predicted, times=None):
        """
        Compute all losses and return as a dictionary.

        Args:
            data: Ground truth data
            interpolated: Interpolated data
            predicted: Model predictions

        Returns:
            dict: Dictionary containing all computed losses
        """

        t_loss_weights = (
            self.t_loss_weights(times)
            if self.t_loss_weights and times is not None
            else None
        )

        # Get valid atoms
        lig_mask, pocket_mask = data["lig_mask"], data["pocket_mask"]

        # Basic coordinate losses
        coord_loss_lig, coord_loss_pocket = self.compute_complex_coordinate_losses(
            data, predicted
        )

        # Categorical losses
        type_loss = self.compute_type_loss(
            data, predicted, mask=lig_mask, t_loss_weights=t_loss_weights
        )
        bond_loss = self.compute_bond_loss(
            data, predicted, mask=lig_mask, t_loss_weights=t_loss_weights
        )
        charge_loss = self.compute_charge_loss(
            data, predicted, mask=lig_mask, t_loss_weights=t_loss_weights
        )

        # Distance-based losses
        distance_losses = self.compute_distance_losses(
            data, predicted, lig_mask, pocket_mask, t_loss_weights=t_loss_weights
        )
        # Smooth LDDT losses
        smooth_distance_losses = self.compute_smooth_lddt_losses(
            data,
            predicted,
            lig_mask=lig_mask,
            pocket_mask=pocket_mask,
            t_loss_weights=t_loss_weights,
        )
        angle_loss = self.compute_angle_losses(
            data,
            predicted,
            lig_mask,
            t_loss_weights=t_loss_weights,
        )

        # Affinity and docking score losses
        affinity_loss = self.compute_affinity_loss(
            data, predicted, t_loss_weights=t_loss_weights
        )
        docking_loss = self.compute_docking_loss(
            data, predicted, t_loss_weights=t_loss_weights
        )

        # Combine all losses
        losses = {
            "coord-loss-lig": coord_loss_lig,
            "coord-loss-pocket": coord_loss_pocket,
            "type-loss": type_loss,
            "bond-loss": bond_loss,
            "charge-loss": charge_loss,
            **distance_losses,
            **smooth_distance_losses,
        }
        # Add angle loss if it exists
        if angle_loss is not None:
            losses["angle-loss"] = angle_loss

        # Add affinity and docking losses if they exist
        if affinity_loss is not None:
            losses["affinity-loss"] = affinity_loss
        if docking_loss is not None:
            losses["docking-loss"] = docking_loss

        return losses

    def compute_complex_coordinate_losses(self, data, predicted):
        """Compute coordinate losses for ligand and pocket separately."""
        pred_coords = predicted["coords"]
        coords = data["coords"]
        mask = data["mask"]

        coord_loss = F.mse_loss(pred_coords, coords, reduction="none")
        lig_mask, pocket_mask = data["lig_mask"].bool(), data["pocket_mask"].bool()

        num_lig_atoms, num_pocket_atoms = lig_mask.sum(-1), pocket_mask.sum(-1)
        coord_loss = (coord_loss * mask.unsqueeze(-1)).sum(-1)

        coord_loss_lig = (coord_loss * lig_mask).sum(-1) / num_lig_atoms
        coord_loss_lig = coord_loss_lig.mean() * self.coord_loss_weight

        coord_loss_pocket = (coord_loss * pocket_mask).sum(-1) / num_pocket_atoms
        coord_loss_pocket = coord_loss_pocket.mean() * self.coord_pocket_loss_weight

        return coord_loss_lig, coord_loss_pocket

    def _compute_complex_distance_loss(
        self,
        pred_coords,
        true_coords,
        lig_mask,
        pocket_mask,
        cutoff: float = 12.0,
        t_loss_weights=None,
    ):
        """
        Compute ligand-pocket distance loss (pocket is rigid/fixed).
        Only compares predicted ligand vs true ligand distances to the same fixed pocket.
        """
        batch_size = lig_mask.shape[0]
        total_loss = 0.0

        for b in range(batch_size):
            # Extract valid ligand coordinates for this batch
            lig_indices = lig_mask[b].bool()
            pred_lig = pred_coords[b][lig_indices]  # [n_lig, 3]
            true_lig = true_coords[b][lig_indices]  # [n_lig, 3]

            # Extract pocket coordinates
            pocket_indices = pocket_mask[b].bool()
            pred_pocket = pred_coords[b][pocket_indices]  # [n_pocket, 3]
            true_pocket = true_coords[b][pocket_indices]  # [n_pocket, 3]

            # Compute distances from true and predicted ligand to fixed pocket
            true_lig_pocket_dists = torch.cdist(
                true_lig, true_pocket, p=2
            )  # [n_lig, n_pocket]
            pred_lig_pocket_dists = torch.cdist(
                pred_lig, pred_pocket, p=2
            )  # [n_lig, n_pocket]

            # Focus on nearby interactions only
            nearby_mask = true_lig_pocket_dists < cutoff

            if nearby_mask.sum() == 0:
                continue

            # Distance loss for nearby pairs only
            dist_diff = (
                torch.abs(pred_lig_pocket_dists - true_lig_pocket_dists)
                * nearby_mask.float()
            )

            n_nearby = nearby_mask.sum().float() + 1e-8
            batch_loss = dist_diff.sum() / n_nearby
            if t_loss_weights is not None:
                batch_loss = batch_loss * t_loss_weights[b]
            total_loss += batch_loss

        return total_loss / batch_size


# Additional utility functions for specialized loss computations
class AdversarialLoss:
    """Optional adversarial loss for improved generation quality."""

    def __init__(self, loss_weight: float = 1.0):
        self.loss_weight = loss_weight

    def compute_loss(self, real_data, generated_data):
        """Placeholder for adversarial loss implementation."""
        # This could be implemented if adversarial training is desired
        pass


class PerceptualLoss:
    """Optional perceptual loss based on molecular descriptors."""

    def __init__(self, loss_weight: float = 1.0):
        self.loss_weight = loss_weight

    def compute_loss(self, real_mols, generated_mols):
        """Placeholder for perceptual loss implementation."""
        # This could compute losses based on molecular descriptors
        # like fingerprints, SASA, etc.
        pass


class RegularizationLoss:
    """Additional regularization losses for model training."""

    def __init__(
        self, coord_smoothness_weight: float = 0.0, bond_sparsity_weight: float = 0.0
    ):
        self.coord_smoothness_weight = coord_smoothness_weight
        self.bond_sparsity_weight = bond_sparsity_weight

    def compute_coordinate_smoothness_loss(self, coords, adj_matrix):
        """Encourage smooth coordinate changes for bonded atoms."""
        if self.coord_smoothness_weight == 0.0:
            return 0.0

        # Compute coordinate differences for bonded atoms
        coord_diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # [B, N, N, 3]
        coord_dist = torch.norm(coord_diff, dim=-1)  # [B, N, N]

        # Apply adjacency mask and compute loss
        smoothness_loss = (coord_dist * adj_matrix).sum() / (adj_matrix.sum() + 1e-8)
        return smoothness_loss * self.coord_smoothness_weight

    def compute_bond_sparsity_loss(self, bond_probs):
        """Encourage sparsity in bond predictions."""
        if self.bond_sparsity_weight == 0.0:
            return 0.0

        # L1 regularization on bond probabilities
        sparsity_loss = torch.mean(torch.sum(bond_probs, dim=-1))
        return sparsity_loss * self.bond_sparsity_weight

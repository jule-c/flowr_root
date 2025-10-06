import torch
import torch.nn.functional as F

import flowr.util.functional as smolF


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

    def compute_losses(
        self,
        data,
        interpolated,
        predicted,
        pocket_data=None,
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
        if self.t_loss_weights is not None:
            times = predicted.get("times", None)
            assert (
                times is not None and len(times) == 2
            ), "Times must be provided for time-dependent losses for cont. and discr. types!"

            times_cont = times[0].squeeze(-1)[:, 0]
            times_disc = times[1].squeeze(-1)[:, 0]
            t_loss_weights_cont = self.t_loss_weights(times_cont)
            t_loss_weights_disc = self.t_loss_weights(times_disc)

        # Coordinate losses
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
        cutoff: float = 12.0,
        t_loss_weights=None,
    ):
        """
        Compute ligand-pocket distance loss (pocket is rigid/fixed)
        Only compares predicted ligand vs true ligand distances to the same fixed pocket
        """
        batch_size = lig_mask.shape[0]
        total_loss = 0.0

        for b in range(batch_size):
            # Extract valid ligand coordinates for this batch
            lig_indices = lig_mask[b].bool()
            pred_lig = pred_lig_coords[b][lig_indices]  # [n_lig, 3]
            true_lig = true_lig_coords[b][lig_indices]  # [n_lig, 3]

            # Extract pocket coordinates
            pocket = pocket_coords[b][pocket_mask[b].bool()]  # [n_pocket, 3]

            # Compute distances from true and predicted ligand to fixed pocket
            true_lig_pocket_dists = torch.cdist(
                true_lig, pocket, p=2
            )  # [n_lig, n_pocket]
            pred_lig_pocket_dists = torch.cdist(
                pred_lig, pocket, p=2
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

import torch
from torch import pi

import flowr.util.functional as smolF

_T = torch.Tensor
_BatchT = dict[str, _T]


class SampleNoiseSchedule:
    """Inverse noise scheduler: `scale = 1 / (t + eps)`."""

    def __init__(self, cutoff: float = 0.9, eps: float = 1e-2):
        """Args:
        cutoff: Timesteps above this value return zero noise.
        """
        self.cutoff = cutoff
        self.eps = eps

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Compute inverse noise scaling with small numerical stabilizer."""
        raw_scale = 1 / (t + self.eps)
        return torch.where(t < self.cutoff, raw_scale, torch.zeros_like(t))


class CosineSchedule:
    """Cosine scheduler"""

    def __init__(self, nu: int = 1):
        self.nu = nu

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Compute cosine noise scaling."""
        input = 0.5 * pi * ((1 - t) ** self.nu)
        y = torch.cos(input)
        alpha_t = y**2
        sigma_t = 1.0 - alpha_t
        dalpha_t = (
            2
            * y
            * (-1.0)
            * torch.sin(input)
            * 0.5
            * pi
            * self.nu
            * (1 - t) ** (self.nu - 1)
            * (-1.0)
        )
        return dalpha_t, sigma_t


class Integrator:
    def __init__(
        self,
        steps,
        coord_noise_std=0.01,
        type_strategy="uniform-sample",
        bond_strategy="uniform-sample",
        coord_strategy="continuous",
        pocket_noise=None,
        cat_noise_level=0,
        corrector_sch_a=0.5,
        corrector_sch_b=0.5,
        corrector_iter_weight=5.0,
        type_mask_index=None,
        bond_mask_index=None,
        use_sde_simulation=False,
        use_cosine_scheduler=False,
        eps=1e-5,
    ):

        self._check_cat_sampling_strategy(type_strategy, type_mask_index, "type")
        self._check_cat_sampling_strategy(bond_strategy, bond_mask_index, "bond")

        self.steps = steps
        self.coord_noise_level = coord_noise_std
        self.type_strategy = type_strategy
        self.bond_strategy = bond_strategy
        self.coord_strategy = coord_strategy
        self.pocket_noise = pocket_noise
        self.cat_noise_level = cat_noise_level
        self.corrector_sch_a = corrector_sch_a
        self.corrector_sch_b = corrector_sch_b
        self.corrector_iter_weight = corrector_iter_weight
        self.type_mask_index = type_mask_index
        self.bond_mask_index = bond_mask_index
        self.use_sde_simulation = use_sde_simulation
        self.use_cosine_scheduler = use_cosine_scheduler
        self.eps = eps

        if self.use_cosine_scheduler:
            self.cosine_scheduler = CosineSchedule(nu=1.0)
        # if self.use_sde_simulation:
        self.noise_scheduler = SampleNoiseSchedule(cutoff=0.9, eps=1e-2)

    @property
    def hparams(self):
        return {
            "integration-steps": self.steps,
            "integration-coord-noise-std": self.coord_noise_level,
            "integration-type-strategy": self.type_strategy,
            "integration-bond-strategy": self.bond_strategy,
            "integration-coord-strategy": self.coord_strategy,
            "integration-cat-noise-level": self.cat_noise_level,
            "corrector-sch-a": self.corrector_sch_a,
            "corrector-sch-b": self.corrector_sch_b,
            "corrector-iter-weight": self.corrector_iter_weight,
            "integration-coord-noise-scale": self.coord_noise_level,
            "use-sde-simulation": self.use_sde_simulation,
            "use-cosine-scheduler": self.use_cosine_scheduler,
        }

    def coord_step(
        self,
        curr_coords: _BatchT,
        pred_coords: _BatchT,
        mask: _BatchT,
        times: torch.Tensor,
        step_size: float,
    ):
        # *** Coord update step ***

        if self.coord_strategy == "velocity-sample":
            coords = self._coord_velocity_step(
                curr_coords, pred_coords, times, step_size
            )
        else:
            if self.use_cosine_scheduler:
                d_alpha_t, sigma_t = self.cosine_scheduler(times)
                coord_velocity = (pred_coords - curr_coords) * (
                    d_alpha_t / sigma_t
                ).view(-1, 1, 1)
            else:
                coord_velocity = (pred_coords - curr_coords) / (
                    (1 - times.view(-1, 1, 1))
                )

            # Apply noise scaling
            gt = self.noise_scheduler(times[0])
            if self.use_sde_simulation:
                assert (
                    self.coord_noise_level > 0.0
                ), "coord_noise_level should be greater than 0.0 for SDE simulation."
                scale_ref = 1.0
                gt = self.noise_scheduler(times[0])
                score = (
                    (times.view(-1, 1, 1) * coord_velocity - curr_coords)
                    / (1.0 - times + 1e-6).view(-1, 1, 1)
                    * scale_ref**2
                )  # [*, n, 1]
                eps = torch.randn(
                    curr_coords.shape,
                    dtype=curr_coords.dtype,
                    device=curr_coords.device,
                )  # [*, dim]
                std_eps = torch.sqrt(2 * gt * self.coord_noise_level)
                delta_x = (
                    coord_velocity + gt * score
                ) * step_size + eps * std_eps * step_size
                coords = curr_coords + delta_x
            else:
                coord_velocity += (
                    torch.randn_like(coord_velocity) * self.coord_noise_level * gt
                )
                coords = curr_coords + (step_size * coord_velocity)

        coords = coords * mask.unsqueeze(-1)
        return coords

    def step(
        self,
        curr: _BatchT,
        predicted: _BatchT,
        prior: _BatchT,
        times: list,
        step_size: float,
    ) -> _BatchT:
        """Perform a single integration step.
        Args:
            curr: Current state of the system.
            predicted: Predicted state of the system.
            prior: Prior state of the system.
            times: List of time points for continuous and discrete updates.
            step_size: Size of the integration step.
        Returns:
            Updated state of the system after the integration step.
        """

        lig_times_cont = times[0]
        lig_times_disc = times[1]
        # pocket_times = times[2]
        interaction_times = times[-1]

        # *** Coord update step ***
        coords = self.coord_step(
            curr["coords"],
            predicted["coords"],
            mask=prior["mask"],
            times=lig_times_cont,
            step_size=step_size,
        )

        # *** Atom type update step ***
        if self.type_strategy == "uniform-sample":
            atomics = self._uniform_sample_step(
                curr["atomics"], predicted["atomics"], lig_times_disc, step_size
            )

        # Uniform sampling from discrete flow models paper
        elif self.type_strategy == "velocity-sample":
            atomics = self._velocity_sample_step(
                curr["atomics"], predicted["atomics"], lig_times_disc, step_size
            )

        # *** Charge update step ***
        if self.type_strategy == "uniform-sample":
            charges = self._uniform_sample_step(
                curr["charges"], predicted["charges"], lig_times_disc, step_size
            )
        elif self.type_strategy == "velocity-sample":
            charges = self._velocity_sample_step(
                curr["charges"], predicted["charges"], lig_times_disc, step_size
            )

        # *** Hybridization update step ***
        if "hybridization" in predicted:
            if self.type_strategy == "uniform-sample":
                hybridization = self._uniform_sample_step(
                    curr["hybridization"],
                    predicted["hybridization"],
                    lig_times_disc,
                    step_size,
                )
            elif self.type_strategy == "velocity-sample":
                hybridization = self._velocity_sample_step(
                    curr["hybridization"],
                    predicted["hybridization"],
                    lig_times_disc,
                    step_size,
                )

        # *** Bond update step ***
        if "bonds" in predicted:
            if self.bond_strategy == "uniform-sample":
                bonds = self._uniform_sample_step(
                    curr["bonds"],
                    predicted["bonds"],
                    lig_times_disc,
                    step_size,
                    symmetrize=True,
                )

            # Uniform sampling from discrete flow models paper
            elif self.bond_strategy == "velocity-sample":
                bonds = self._velocity_sample_step(
                    curr["bonds"], predicted["bonds"], lig_times_disc, step_size
                )

        # *** Interaction update step ***
        interactions = None
        if "interactions" in predicted:
            interactions = self._uniform_sample_step(
                curr["interactions"],
                predicted["interactions"],
                interaction_times,
                step_size,
            )

        updated = {
            "coords": coords,
            "atomics": atomics.float(),
            "charges": charges.float(),
            "mask": curr["mask"],
        }
        if "bonds" in predicted:
            updated["bonds"] = bonds.float()
        if "hybridization" in predicted:
            updated["hybridization"] = hybridization.float()
        if interactions is not None:
            updated["interactions"] = interactions.float()

        if "res_names" in curr:
            updated["res_names"] = curr["res_names"]
        if "lig_mask" in curr:
            updated["lig_mask"] = curr["lig_mask"]
        if "pocket_mask" in curr:
            updated["pocket_mask"] = curr["pocket_mask"]

        return updated

    def corrector_iter(
        self,
        curr: _BatchT,
        predicted: _BatchT,
        prior: _BatchT,
        times: list,
        step_size: float,
    ) -> _BatchT:

        # Add coordinate corrector
        if self.coord_strategy == "velocity-sample":
            lig_times_cont = times[0]
            coords = self._coord_velocity_step(
                curr["coords"],
                predicted["coords"],
                lig_times_cont,
                step_size * self.corrector_iter_weight,
            )
        else:
            coords = predicted["coords"]

        lig_times_disc = times[1]
        if self.type_strategy == "velocity-sample":
            atomics = self._corrector_iter_step(
                curr["atomics"], predicted["atomics"], lig_times_disc, step_size
            )
        else:
            raise ValueError(
                "Type strategy must be velocity-sample for corrector iterations"
            )

        charges = self._corrector_iter_step(
            curr["charges"], predicted["charges"], lig_times_disc, step_size
        )

        if "hybridization" in predicted:
            hybridization = self._corrector_iter_step(
                curr["hybridization"],
                predicted["hybridization"],
                lig_times_disc,
                step_size,
            )

        if "bonds" in predicted:
            bonds = self._corrector_iter_step(
                curr["bonds"], predicted["bonds"], lig_times_disc, step_size
            )

        updated = {
            "coords": coords,
            "atomics": atomics.float(),
            "charges": charges.float(),
            "mask": curr["mask"],
        }
        if "bonds" in predicted:
            updated["bonds"] = bonds.float()
        if "hybridization" in predicted:
            updated["hybridization"] = hybridization.float()
        if "res_names" in curr:
            updated["res_names"] = curr["res_names"]
        if "lig_mask" in curr:
            updated["lig_mask"] = curr["lig_mask"]
        if "pocket_mask" in curr:
            updated["pocket_mask"] = curr["pocket_mask"]

        return updated

    def _uniform_sample_step(
        self, curr_dist, pred_dist, t, step_size, symmetrize: bool = False
    ):
        n_categories = pred_dist.size(-1)

        curr = torch.argmax(curr_dist, dim=-1).unsqueeze(-1)
        pred_probs_curr = torch.gather(pred_dist, -1, curr)

        # Setup batched time tensor and noise tensor
        ones = [1] * (len(pred_dist.shape) - 1)
        times = t.view(-1, *ones).clamp(min=self.eps, max=1.0 - self.eps)
        noise = torch.zeros_like(times)
        noise[times + step_size < 1.0] = self.cat_noise_level

        # Off-diagonal step probs
        mult = (1 + noise + (noise * (n_categories - 1) * times)) / (1 - times)
        first_term = step_size * mult * pred_dist
        second_term = step_size * noise * pred_probs_curr
        step_probs = (first_term + second_term).clamp(max=1.0)

        # On-diagonal step probs
        step_probs.scatter_(-1, curr, 0.0)
        diags = (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
        step_probs.scatter_(-1, curr, diags)

        # # Sample and convert back to one-hot so that all strategies represent data the same way
        samples = torch.distributions.Categorical(step_probs).sample()
        if symmetrize:
            samples = smolF.symmetrize_bonds(samples, is_one_hot=False)
        samples = smolF.one_hot_encode_tensor(samples, n_categories)

        return samples

    def _coord_velocity_step(self, curr_coords, pred_coords, t, step_size):
        ones = [1] * (len(pred_coords.shape) - 1)
        times = t.view(-1, *ones).clamp(min=self.eps, max=1.0 - self.eps)

        min_noise_level = 0.01
        noise_schedule = (
            self.coord_noise_level * torch.cos(0.5 * torch.pi * times) + min_noise_level
        )

        beta_t = torch.pow(times, self.corrector_sch_a) * torch.pow(
            1 - times, self.corrector_sch_b
        )
        beta_t = noise_schedule * beta_t
        alpha_t = beta_t + 1

        # Forward velocity toward prediction
        forward_vel = (pred_coords - curr_coords) * (
            1 / torch.clamp(1 - times, min=self.eps)
        )

        # Backward velocity toward zero (or mean coordinates)
        coord_prior = torch.zeros_like(curr_coords)  # or use dataset mean
        backward_vel = (curr_coords - coord_prior) * (
            1 / torch.clamp(times, min=self.eps)
        )

        coord_velocity = (alpha_t * forward_vel) - (beta_t * backward_vel)
        coords = curr_coords + (step_size * coord_velocity)

        return coords

    def _velocity_sample_step(self, curr_dist, pred_dist, t, step_size):
        n_categories = pred_dist.size(-1)

        ones = [1] * (len(pred_dist.shape) - 1)
        times = t.view(-1, *ones).clamp(min=self.eps, max=1.0 - self.eps)

        # min_noise_level = 0.01  # Always maintain some noise
        # noise_schedule = (
        #     self.cat_noise_level * torch.cos(0.5 * torch.pi * times) + min_noise_level
        # )

        beta_t = torch.pow(times, self.corrector_sch_a) * torch.pow(
            1 - times, self.corrector_sch_b
        )
        beta_t = self.cat_noise_level * beta_t
        alpha_t = beta_t + 1

        # Curr dist should be one-hot
        forward_vel = (pred_dist - curr_dist) * (
            1 / torch.clamp(1 - times, min=self.eps)
        )
        # Assume uniform dist for prior
        backward_vel = (curr_dist - (1 / n_categories)) * (
            1 / torch.clamp(times, min=self.eps)
        )

        prob_vel = (alpha_t * forward_vel) - (beta_t * backward_vel)
        step_dist = curr_dist + (step_size * prob_vel)
        step_dist = step_dist.clamp(min=0.0)
        # step_dist = step_dist / (step_dist.sum(dim=-1, keepdim=True) + self.eps)

        samples = torch.distributions.Categorical(step_dist).sample()
        return smolF.one_hot_encode_tensor(samples, n_categories)

    def _corrector_iter_step(self, curr_dist, pred_dist, t, step_size):
        n_categories = pred_dist.size(-1)

        ones = [1] * (len(pred_dist.shape) - 1)
        times = t.view(-1, *ones).clamp(min=self.eps, max=1.0 - self.eps)

        # Curr dist should be one-hot
        forward_vel = (pred_dist - curr_dist) * (
            1 / torch.clamp(1 - times, min=self.eps)
        )

        # NOTE Assumes uniform dist for prior
        backward_vel = (curr_dist - (1 / n_categories)) * (
            1 / torch.clamp(times, min=self.eps)
        )

        prob_vel = (self.corrector_iter_weight * forward_vel) - (
            self.corrector_iter_weight * backward_vel
        )
        step_dist = curr_dist + (step_size * prob_vel)
        step_dist = step_dist.clamp(min=0.0)

        samples = torch.distributions.Categorical(step_dist).sample()
        return smolF.one_hot_encode_tensor(samples, n_categories)

    def _check_cat_sampling_strategy(self, strategy, mask_index, name):
        if strategy not in ["linear", "mask", "uniform-sample", "velocity-sample"]:
            raise ValueError(f"{name} sampling strategy '{strategy}' is not supported.")

        if strategy == "mask" and mask_index is None:
            raise ValueError(
                f"{name}_mask_index must be provided if using the mask sampling strategy."
            )

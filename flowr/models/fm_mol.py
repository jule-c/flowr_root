import os
from collections import defaultdict
from typing import Optional

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from rdkit import Chem
from torch.optim.lr_scheduler import LinearLR, OneCycleLR
from torchmetrics import MetricCollection

import flowr.util.metrics as Metrics
from flowr.data.data_info import GeneralInfos as DataInfos
from flowr.models.integrator import Integrator
from flowr.models.losses import LossComputer
from flowr.models.mol_builder import MolBuilder
from flowr.models.semla import MolecularGenerator
from flowr.util.tokeniser import Vocabulary

_T = torch.Tensor
_BatchT = dict[str, _T]


def create_list_defaultdict():
    return defaultdict(list)


# *********************************************************************************************************************
# ******************************************** Lightning Flow Matching Models *****************************************
# *********************************************************************************************************************


class LigandCFM(pl.LightningModule):
    def __init__(
        self,
        gen: MolecularGenerator,
        vocab: Vocabulary,
        vocab_charges: Vocabulary,
        lr: float,
        integrator: Integrator,
        add_feats: Optional[int] = None,
        vocab_hybridization: Optional[Vocabulary] = None,
        vocab_aromatic: Optional[Vocabulary] = None,
        beta1: float = 0.9,
        beta2: float = 0.999,
        weight_decay: float = 1e-12,
        coord_scale: float = 1.0,
        type_strategy: str = "ce",
        bond_strategy: str = "ce",
        coord_loss_weight: float = 1.0,
        type_loss_weight: float = 1.0,
        bond_loss_weight: float = 1.0,
        charge_loss_weight: float = 1.0,
        hybridization_loss_weight: float = 1.0,
        distance_loss_weight_lig: float = None,
        use_ema: bool = True,
        self_condition: bool = False,
        remove_hs: bool = False,
        remove_aromaticity: bool = False,
        lr_schedule: str = "constant",
        lr_gamma: float = 0.998,
        sampling_strategy: str = "linear",
        warm_up_steps: Optional[int] = None,
        total_steps: Optional[int] = None,
        train_mols: Optional[list[str]] = None,
        type_mask_index: Optional[int] = None,
        bond_mask_index: Optional[int] = None,
        func_group_inpainting: bool = False,
        scaffold_inpainting: bool = False,
        fragment_inpainting: bool = False,
        core_inpainting: bool = False,
        linker_inpainting: bool = False,
        substructure_inpainting: bool = False,
        graph_inpainting: bool = False,
        corrector_iters: int = None,
        use_t_loss_weights: bool = False,
        pretrained_weights: bool = False,
        dataset_info: DataInfos = None,
        data_path: str = None,
        **kwargs,
    ):
        super().__init__()

        if type_strategy not in ["mse", "ce", "mask"]:
            raise ValueError(f"Unsupported type training strategy '{type_strategy}'")

        if bond_strategy not in ["ce", "mask"]:
            raise ValueError(f"Unsupported bond training strategy '{bond_strategy}'")

        if lr_schedule not in ["constant", "one-cycle", "exponential"]:
            raise ValueError(f"LR scheduler {lr_schedule} not supported.")

        if lr_schedule == "one-cycle" and total_steps is None:
            raise ValueError(
                "total_steps must be provided when using the one-cycle LR scheduler."
            )

        self.feature_keys = ["atomics", "bonds", "charges"]
        self.sc_feature_keys = ["atomics", "bonds"]
        if add_feats is not None:
            self.feature_keys.append("hybridization")

        self.gen = gen
        self.vocab = vocab
        self.vocab_charges = vocab_charges
        self.vocab_hybridization = vocab_hybridization
        self.add_feats = add_feats
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.coord_scale = coord_scale
        self.type_strategy = type_strategy
        self.bond_strategy = bond_strategy
        self.coord_loss_weight = coord_loss_weight
        self.type_loss_weight = type_loss_weight
        self.bond_loss_weight = bond_loss_weight
        self.charge_loss_weight = charge_loss_weight
        self.hybridization_loss_weight = hybridization_loss_weight
        self.distance_loss_weight_lig = distance_loss_weight_lig
        self.lr_schedule = lr_schedule
        self.lr_gamma = lr_gamma
        self.sampling_strategy = sampling_strategy
        self.warm_up_steps = warm_up_steps
        self.total_steps = total_steps
        self.type_mask_index = type_mask_index
        self.bond_mask_index = bond_mask_index
        self.func_group_inpainting = func_group_inpainting
        self.scaffold_inpainting = scaffold_inpainting
        self.graph_inpainting = graph_inpainting
        self.fragment_inpainting = fragment_inpainting
        self.core_inpainting = core_inpainting
        self.linker_inpainting = linker_inpainting
        self.substructure_inpainting = substructure_inpainting
        self.corrector_iters = corrector_iters
        self.use_t_loss_weights = use_t_loss_weights
        self.train_mols = train_mols
        self.pretrained_weights = pretrained_weights

        # Conditional mode
        if (
            self.func_group_inpainting
            or self.scaffold_inpainting
            or self.linker_inpainting
            or self.substructure_inpainting
            or self.fragment_inpainting
            or self.core_inpainting
        ):
            self.inpainting_mode = True
            if self.graph_inpainting:
                self.inpainting_mode_inf = "graph"
            else:
                self.inpainting_mode_inf = "fragment"
        else:
            self.inpainting_mode_inf = "graph" if self.graph_inpainting else None
            self.inpainting_mode = False

        # Self-conditioning mode
        self.self_condition = self_condition
        self.self_condition_mode = "stacking"
        self.self_condition_prob = 0.5
        self._inpaint_self_condition = True

        # Anything else passed into kwargs will also be saved
        hparams = {
            "lr": lr,
            "add_feats": add_feats,
            "coord_scale": coord_scale,
            "coord_loss_weight": coord_loss_weight,
            "type_loss_weight": type_loss_weight,
            "bond_loss_weight": bond_loss_weight,
            "charge_loss_weight": charge_loss_weight,
            "hybridization_loss_weight": hybridization_loss_weight,
            "distance_loss_weight_lig": distance_loss_weight_lig,
            "use_t_loss_weights": use_t_loss_weights,
            "type_strategy": type_strategy,
            "bond_strategy": bond_strategy,
            "self_condition": self_condition,
            "func_group_inpainting": func_group_inpainting,
            "scaffold_inpainting": scaffold_inpainting,
            "linker_inpainting": linker_inpainting,
            "substructure_inpainting": substructure_inpainting,
            "fragment_inpainting": fragment_inpainting,
            "core_inpainting": core_inpainting,
            "corrector_iters": corrector_iters,
            "remove_hs": remove_hs,
            "remove_aromaticity": remove_aromaticity,
            "lr_schedule": lr_schedule,
            "sampling_strategy": sampling_strategy,
            "use_ema": use_ema,
            "warm_up_steps": warm_up_steps,
            "data_path": data_path,
            **gen.hparams,
            **integrator.hparams,
            **kwargs,
        }
        self.save_hyperparameters(hparams)

        # Initialize loss computer
        self.loss_computer = LossComputer(
            coord_loss_weight=coord_loss_weight,
            type_loss_weight=type_loss_weight,
            bond_loss_weight=bond_loss_weight,
            charge_loss_weight=charge_loss_weight,
            hybridization_loss_weight=hybridization_loss_weight,
            distance_loss_weight_lig=distance_loss_weight_lig,
            type_strategy=type_strategy,
            bond_strategy=bond_strategy,
            use_t_loss_weights=use_t_loss_weights,
        )

        builder = MolBuilder(
            vocab,
            vocab_charges,
            vocab_hybridization=vocab_hybridization,
            vocab_aromatic=vocab_aromatic,
            save_dir=self.hparams.save_dir,
        )
        self.integrator = integrator
        self.builder = builder
        self.dataset_info = dataset_info
        self.train_mols = train_mols

        gen_mol_metrics = {
            "validity": Metrics.Validity(),
            "fc-validity": Metrics.Validity(connected=True),
            "pb-validity": Metrics.PoseBustersValidityMolecule(),
            "uniqueness": Metrics.Uniqueness(),
            "energy-validity": Metrics.EnergyValidity(),
            "opt-energy-validity": Metrics.EnergyValidity(optimise=True),
            "energy": Metrics.AverageEnergy(),
            "energy-per-atom": Metrics.AverageEnergy(per_atom=True),
            "strain": Metrics.AverageStrainEnergy(),
            "strain-per-atom": Metrics.AverageStrainEnergy(per_atom=True),
            "opt-rmsd": Metrics.AverageOptRmsd(),
        }
        self.gen_dist_metrics = None
        if self.dataset_info is not None and train_mols is not None:
            gen_dist_metrics = Metrics.DistributionDistance(
                dataset_info=self.dataset_info, train_mols=train_mols
            )
            self.gen_dist_metrics = MetricCollection(
                {"distribution-distance": gen_dist_metrics}, compute_groups=False
            )

        self.gen_mol_metrics = MetricCollection(gen_mol_metrics, compute_groups=False)
        if self.graph_inpainting is not None:
            print(f"Using graph inpainting mode: {self.graph_inpainting}")
            docking_metrics = {
                "dock-rmsd": Metrics.MolecularPairRMSD(fix_order=True),
                "dock-shape-tanimoto": Metrics.MolecularPairShapeTanimotoSim(
                    align=True
                ),
            }
            self.docking_metrics = MetricCollection(
                docking_metrics, compute_groups=False
            )

        if not pretrained_weights:
            self._init_params()

    def _get_times(self, times: list, lig_mask: _T, inpaint_mask: _T):
        """Get the times for the model."""
        # times is a list of tensors, each of shape [batch_size,]
        # We need to convert it to a list of tensors of shape [batch_size, num_atoms, 1]
        # If inpainting is used, we also need to inpaint the times

        n_atoms = lig_mask.size(1)
        ligand_times_cont = times[0].view(-1, 1, 1).expand(-1, n_atoms, -1)
        ligand_times_disc = times[1].view(-1, 1, 1).expand(-1, n_atoms, -1)
        # pocket_times = t[2].view(-1, 1, 1).expand(-1, pocket_coords.size(1), -1) # rigid pocket, not needed

        if inpaint_mask is not None and (self.inpainting_mode or self.graph_inpainting):
            # Check which molecules have full fragment masks (all 1s for valid atoms)
            valid_atoms_per_mol = lig_mask.bool().sum(
                dim=1
            )  # Number of valid atoms per molecule
            fragment_atoms_per_mol = (inpaint_mask.bool() & lig_mask.bool()).sum(
                dim=1
            )  # Number of fragment atoms per molecule
            full_fragment_molecules = (
                valid_atoms_per_mol == fragment_atoms_per_mol
            ) & (valid_atoms_per_mol > 0)

            # Only inpaint continuous times for molecules that don't have full fragment masks (fragment inpainting)
            partial_fragment_molecules = ~full_fragment_molecules
            partial_inpaint_mask = (
                inpaint_mask.bool() & partial_fragment_molecules.unsqueeze(1)
            )

            ligand_times_cont = self.builder._inpaint_times(
                ligand_times_cont.squeeze(), partial_inpaint_mask
            ).unsqueeze(-1)

            # Always inpaint discrete times when any inpaint mask is present
            ligand_times_disc = self.builder._inpaint_times(
                ligand_times_disc.squeeze(), inpaint_mask.bool()
            ).unsqueeze(-1)

        times = [ligand_times_cont, ligand_times_disc]
        return times

    def _prepare_self_condition_batch(
        self,
        lig_interp: dict,
        lig_data: dict,
        lig_prior: dict,
        times: list,
        training: bool = True,
    ) -> Optional[dict]:
        """Prepare self-conditioning batch based on the mode."""

        if not self.self_condition:
            return None

        # Initialize based on mode
        if self.self_condition_mode == "stacking":
            cond_batch = {
                "coords": lig_prior["coords"],
                "atomics": torch.zeros_like(lig_interp["atomics"]),
                "bonds": torch.zeros_like(lig_interp["bonds"]),
            }
            if self.add_feats:
                cond_batch["hybridization"] = torch.zeros_like(
                    lig_interp["hybridization"]
                )

        elif self.self_condition_mode == "residual":
            cond_batch = None
        else:
            raise ValueError(
                f"Unknown self-conditioning mode: {self.self_condition_mode}"
            )

        # Apply graph inpainting if needed (for stacking mode)
        if (
            self.self_condition_mode == "stacking"
            and self.graph_inpainting
            and self._inpaint_self_condition
        ):
            cond_batch = self.builder.inpaint_graph(
                lig_data,
                cond_batch,
                feature_keys=["coords", "atomics", "bonds"],
                overwrite_with_zeros=True,
            )

        # During training, randomly use self-conditioning
        if training and torch.rand(1).item() < self.self_condition_prob:
            with torch.no_grad():
                # Get prediction from current state
                cond_out = self(
                    lig_interp,
                    times,
                    cond_batch=(
                        cond_batch if self.self_condition_mode == "stacking" else None
                    ),
                    training=True,
                )

                if self.self_condition_mode == "stacking":
                    # Standard stacking mode
                    cond_batch = {
                        "coords": cond_out["coords"],
                        "atomics": F.softmax(cond_out["atomics"], dim=-1),
                        "bonds": F.softmax(cond_out["bonds"], dim=-1),
                    }
                    if self.add_feats:
                        cond_batch["hybridization"] = F.softmax(
                            cond_out["hybridization"], dim=-1
                        )

                elif self.self_condition_mode == "residual":
                    # For residual mode: pass predicted final state
                    cond_batch = {
                        "coords": cond_out["coords"],
                        "atomics": F.softmax(cond_out["atomics"], dim=-1),
                        "bonds": F.softmax(cond_out["bonds"], dim=-1),
                        "charges": F.softmax(cond_out["charges"], dim=-1),
                    }
                    if self.add_feats:
                        cond_batch["hybridization"] = F.softmax(
                            cond_out["hybridization"], dim=-1
                        )

        return cond_batch

    def forward(
        self,
        batch,
        t,
        cond_batch=None,
        training=False,
    ):
        """Predict molecular coordinates and atom types

        Args:
            batch (dict[str, Tensor]): Batched pointcloud data
            t (torch.Tensor): Interpolation times between 0 and 1, shape [batch_size]
            training (bool): Whether to run forward in training mode
            cond_batch (dict[str, Tensor]): Predictions from previous step, if we are using self conditioning

        Returns:
            (predicted coordinates, atom type logits (unnormalised probabilities))
            Both torch.Tensor, shapes [batch_size, num_atoms, 3] and [batch_size, num atoms, vocab_size]
        """

        coords = batch["coords"]
        atom_types = batch["atomics"]
        bonds = batch["bonds"]
        charges = batch["charges"]
        mask = batch["mask"]

        # Get times in the right format
        inpaint_mask = batch.get("fragment_mask", None)
        times = self._get_times(t, lig_mask=mask, inpaint_mask=inpaint_mask)

        extra_feats = (
            torch.argmax(batch["hybridization"], dim=-1) if self.add_feats else None
        )  # for now only hybridization

        if cond_batch is not None:
            out = self.gen(
                coords,
                torch.argmax(atom_types, dim=-1),
                torch.argmax(bonds, dim=-1),
                atom_charges=torch.argmax(charges, dim=-1),
                atom_mask=mask,
                times=times,
                extra_feats=extra_feats,
                cond_coords=cond_batch["coords"],
                cond_atomics=cond_batch["atomics"],
                cond_bonds=cond_batch["bonds"],
                inpaint_mask=inpaint_mask,
            )
        else:
            out = self.gen(
                coords,
                torch.argmax(atom_types, dim=-1),
                torch.argmax(bonds, dim=-1),
                atom_charges=torch.argmax(charges, dim=-1),
                atom_mask=mask,
                times=times,
                extra_feats=extra_feats,
                inpaint_mask=inpaint_mask,
            )
        out["times"] = times
        return out

    def training_step(self, batch, b_idx):
        # Input data
        prior, data, interpolated, times = batch
        # Times (bs, 2) -> (lig_times_cont, lig_times_disc)
        times = times.T

        cond_batch = self._prepare_self_condition_batch(
            interpolated, data, prior, times, training=True
        )

        predicted = self(interpolated, times, cond_batch=cond_batch, training=True)

        # import pdb

        # pdb.set_trace()
        # self.builder.tensors_to_xyz(
        #     prior=prior,
        #     interpolated=predicted,
        #     data=data,
        #     coord_scale=self.coord_scale,
        #     idx=1,
        #     save_dir=".",
        # )

        losses = self._loss(data, interpolated, predicted)
        loss = sum(list(losses.values()))

        for name, loss_val in losses.items():
            self.log(
                f"train-{name}",
                loss_val,
                prog_bar=True,
                on_step=True,
                logger=True,
                sync_dist=True,
            )

        self.log(
            "train-loss", loss, prog_bar=True, on_step=True, logger=True, sync_dist=True
        )

        return loss

    # def on_after_backward(self) -> None:
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(name)
    #     import pdb

    #     pdb.set_trace()

    def validation_step(self, batch, b_idx):
        # Input data
        prior, data, interpolated, times = batch

        # Times (bs, 2) -> (lig_times_cont, lig_times_disc)
        times = times.T

        # Predict
        cond_batch = self._prepare_self_condition_batch(
            interpolated, data, prior, times, training=False
        )
        predicted = self(interpolated, times, cond_batch=cond_batch, training=False)

        # Get the losses
        losses = self._loss(data, interpolated, predicted)
        for name, loss_val in losses.items():
            self.log(
                f"val-{name}",
                loss_val,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )

    def on_validation_epoch_end(self):
        # Skip generation during sanity checking to save time
        if self.trainer.sanity_checking:
            return

        # Get the prior data
        num_val_samples = 500
        dataloader = self.trainer.datamodule.val_dataloader(subset=num_val_samples)
        for batch in dataloader:
            prior, data, interpolated, times = batch
            prior = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in prior.items()
            }
            data = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in data.items()
            }
            interpolated = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in interpolated.items()
            }
            # Build starting times for the integrator
            lig_times_cont = torch.zeros(prior["coords"].size(0), device=self.device)
            lig_times_disc = torch.zeros(prior["coords"].size(0), device=self.device)
            prior_times = [lig_times_cont, lig_times_disc]

            # Generate
            gen_batch = self._generate(
                prior,
                steps=self.integrator.steps,
                times=prior_times,
                strategy=self.sampling_strategy,
                corr_iters=self.corrector_iters,
            )
            gen_mols = self._generate_mols(gen_batch)
            # if not self.trainer.sanity_checking:
            self.gen_mol_metrics.update(gen_mols)
            if self.gen_dist_metrics is not None:
                self.gen_dist_metrics.update(gen_mols)
            if self.graph_inpainting:
                true_mols = self._generate_mols(data)
                if self.hparams.remove_hs:
                    true_mols = [Chem.RemoveHs(mol) for mol in true_mols]
                self.docking_metrics.update(gen_mols, true_mols)

        # Compute metrics
        gen_dist_results = {}
        gen_metrics_results = self.gen_mol_metrics.compute()
        if self.gen_dist_metrics is not None:
            gen_dist_results = self.gen_dist_metrics.compute()

        if self.graph_inpainting:
            docking_metrics_results = self.docking_metrics.compute()
        else:
            docking_metrics_results = {}

        metrics = {
            **gen_metrics_results,
            **gen_dist_results,
            **docking_metrics_results,
        }

        for metric, value in metrics.items():
            progbar = True if metric == "fc-validity" else False
            if isinstance(value, dict):
                for k, v in value.items():
                    self.log(
                        f"val-{k}",
                        v.to(self.device),
                        on_epoch=True,
                        logger=True,
                        prog_bar=progbar,
                        sync_dist=True,
                    )
            else:
                self.log(
                    f"val-{metric}",
                    value.to(self.device),
                    on_epoch=True,
                    logger=True,
                    prog_bar=progbar,
                    sync_dist=True,
                )

        self.gen_mol_metrics.reset()
        if self.gen_dist_metrics is not None:
            self.gen_dist_metrics.reset()
        if self.graph_inpainting and self.docking_metrics is not None:
            self.docking_metrics.reset()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.gen.parameters(),
            lr=self.lr,
            amsgrad=True,
            foreach=True,
            weight_decay=1e-12,
        )

        if self.lr_schedule == "constant":
            warm_up_steps = 0 if self.warm_up_steps is None else self.warm_up_steps
            scheduler = LinearLR(opt, start_factor=1e-2, total_iters=warm_up_steps)

        # TODO could use warm_up_steps to shift peak of one cycle
        elif self.lr_schedule == "one-cycle":
            scheduler = OneCycleLR(
                opt, max_lr=self.lr, total_steps=self.total_steps, pct_start=0.3
            )
        elif self.lr_schedule == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=self.lr_gamma)
        else:
            raise ValueError(f"LR schedule {self.lr_schedule} is not supported.")

        if self.lr_schedule == "constant":
            config = {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
            return config
        else:
            scheduler = {
                "scheduler": scheduler,
                "interval": "epoch",
                # "frequency": self.hparams["lr_frequency"],
                # "monitor": self.validity,
                "strict": False,
            }
        return [opt], [scheduler]

    def _loss(self, data, interpolated, predicted):
        """Compute all losses using the dedicated loss computer."""
        return self.loss_computer.compute_losses(data, interpolated, predicted)

    def _build_trajectory(
        self,
        curr,
        predicted,
        iter: int = 0,
        step: int = 0,
    ):
        self.builder.write_xyz_file_from_batch(
            data=predicted,
            coord_scale=self.coord_scale,
            path=os.path.join(self.hparams.save_dir, f"traj_pred_mols_{iter}"),
            t=step,
        )

        self.builder.write_xyz_file_from_batch(
            data=curr,
            coord_scale=self.coord_scale,
            path=os.path.join(self.hparams.save_dir, f"traj_interp_{iter}"),
            t=step,
        )

    def _save_trajectory(
        self,
        predicted,
        iter: int = 0,
        remove_intermediate_files: bool = True,
    ):
        pred_mols = self._generate_mols(predicted)
        self.builder.write_trajectory_as_xyz(
            pred_mols=pred_mols,
            file_path=os.path.join(self.hparams.save_dir, f"traj_pred_mols_{iter}"),
            save_path=os.path.join(
                self.hparams.save_dir, "trajectories_pred_mols", f"traj_{iter}"
            ),
            remove_intermediate_files=remove_intermediate_files,
        )
        self.builder.write_trajectory_as_xyz(
            pred_mols=pred_mols,
            file_path=os.path.join(self.hparams.save_dir, f"traj_interp_{iter}"),
            save_path=os.path.join(
                self.hparams.save_dir, "trajectories_interp", f"traj_{iter}"
            ),
            remove_intermediate_files=remove_intermediate_files,
        )

    def _update_times(
        self,
        times,
        step_size: float,
    ):
        lig_times_cont = times[0] + step_size
        lig_times_disc = times[1] + step_size
        times = [
            lig_times_cont,
            lig_times_disc,
        ]
        return times

    def _get_predictions(self, out):
        coords = out["coords"]
        type_probs = F.softmax(out["atomics"], dim=-1)
        bond_probs = F.softmax(out["bonds"], dim=-1)
        charge_probs = F.softmax(out["charges"], dim=-1)
        hybridization_probs = (
            F.softmax(out["hybridization"], dim=-1) if "hybridization" in out else None
        )
        cond_batch = {
            "coords": coords,
            "atomics": type_probs,
            "bonds": bond_probs,
        }
        predicted = {
            "coords": coords,
            "atomics": type_probs,
            "bonds": bond_probs,
            "charges": charge_probs,
            "mask": out["mask"],
        }
        if hybridization_probs is not None:
            predicted["hybridization"] = hybridization_probs
        return predicted, cond_batch

    def _generate(
        self,
        prior: dict,
        steps: int,
        times: list,
        strategy: str = "linear",
        save_traj: bool = False,
        solver: str = "euler",
        corr_iters=None,
        corr_step_size=None,
        iter: int = 0,
    ):

        corr_iters = 0 if corr_iters is None else corr_iters

        if strategy == "linear":
            # time_points = np.linspace(0, 0.999, steps + 1).tolist()
            time_points = torch.linspace(0, 1, steps + 1)
        elif strategy == "log":
            # time_points = (1 - np.geomspace(0.01, 0.999, steps + 1)).tolist()
            # time_points.reverse()
            time_points = 1.0 - torch.logspace(-2, 0, steps + 1).flip(0)
            time_points = time_points - torch.min(time_points)
            time_points = time_points / torch.max(time_points)
        else:
            raise ValueError(f"Unknown ODE integration strategy '{strategy}'")

        curr = {
            k: (
                v.clone()
                if torch.is_tensor(v)
                else v.copy() if isinstance(v, list) else v
            )
            for k, v in prior.items()
        }
        assert "fragment_mask" in curr.keys(), "fragment_mask must be in prior"
        assert curr["fragment_mask"] is not None, "fragment_mask must not be None"

        cond_batch = {
            "coords": prior["coords"],
            "atomics": torch.zeros_like(prior["atomics"]),
            "bonds": torch.zeros_like(prior["bonds"]),
        }
        if self.graph_inpainting:
            cond_batch = self.builder.inpaint_graph(
                prior, cond_batch, feature_keys=self.sc_feature_keys
            )

        step_sizes = [t1 - t0 for t0, t1 in zip(time_points[:-1], time_points[1:])]
        with torch.no_grad():
            for i, step_size in enumerate(step_sizes):

                cond = cond_batch if self.self_condition else None
                # Run the model
                out = self(
                    curr,
                    times,
                    training=False,
                    cond_batch=cond,
                )
                predicted, cond_batch = self._get_predictions(out)
                # Euler step
                curr = self.integrator.step(curr, predicted, prior, times, step_size)
                # Update times for the next step
                times = self._update_times(
                    times,
                    step_size,
                )

                # Inpainting for the ligand if required
                if self.inpainting_mode:
                    curr = self.builder.inpaint_molecule(
                        data=prior,
                        prediction=curr,
                    )
                elif self.graph_inpainting:
                    curr = self.builder.inpaint_graph(
                        prior,
                        curr,
                        feature_keys=self.feature_keys,
                    )
                    cond_batch = self.builder.inpaint_graph(
                        prior,
                        cond_batch,
                        feature_keys=self.sc_feature_keys,
                    )

                # Save trajectory
                if save_traj:
                    self._build_trajectory(
                        curr,
                        predicted,
                        iter=iter,
                        step=i,
                    )

            # Corrector iterations at the end of sampling
            for _ in range(corr_iters):
                cond = cond_batch if self.self_condition else None
                out = self(
                    curr,
                    times,
                    training=False,
                    cond_batch=cond,
                )

                predicted, cond_batch = self._get_predictions(out)
                if self.graph_inpainting:
                    # Inpaint the predicted structure if graph inpainting is used
                    predicted = self.builder.inpaint_graph(
                        prior,
                        predicted,
                        feature_keys=self.feature_keys,
                    )
                    cond_batch = self.builder.inpaint_graph(
                        prior,
                        cond_batch,
                        feature_keys=self.sc_feature_keys,
                    )
                step_size = 1 / steps if corr_step_size is None else corr_step_size
                curr = self.integrator.corrector_iter(
                    curr, predicted, prior, times, step_size
                )
                # Inpainting for the ligand if required
                if self.inpainting_mode:
                    curr = self.builder.inpaint_molecule(
                        data=prior,
                        prediction=curr,
                    )
                elif self.graph_inpainting:
                    curr = self.builder.inpaint_graph(
                        prior,
                        curr,
                        feature_keys=self.feature_keys,
                    )
                    cond_batch = self.builder.inpaint_graph(
                        prior,
                        cond_batch,
                        feature_keys=self.sc_feature_keys,
                    )

        # Final corrector prediction
        eps = -1e-4
        times = self._update_times(
            times,
            eps,
        )
        cond = cond_batch if self.self_condition else None
        with torch.no_grad():
            out = self(
                curr,
                times,
                training=False,
                cond_batch=cond,
            )

        predicted, _ = self._get_predictions(out)
        if self.graph_inpainting:
            predicted = self.builder.inpaint_graph(
                prior,
                predicted,
                feature_keys=self.feature_keys,
            )

        # Move everything to CPU
        predicted = {
            k: v.cpu().detach() if torch.is_tensor(v) else v
            for k, v in predicted.items()
        }
        # Scale back coordinates if necessary
        predicted["coords"] = predicted["coords"] * self.coord_scale

        # import pdb

        # pdb.set_trace()
        # Save trajectory if required
        if save_traj:
            self._save_trajectory(
                predicted,
                iter=iter,
            )
        return predicted

    def _generate_mols(self, generated, scale=1.0, sanitise=True):
        coords = generated["coords"] * scale
        atom_dists = generated["atomics"]
        bond_dists = generated["bonds"]
        charge_dists = generated["charges"]
        masks = generated["mask"]

        mols = self.builder.mols_from_tensors(
            coords,
            atom_dists,
            masks,
            bond_dists=bond_dists,
            charge_dists=charge_dists,
            sanitise=sanitise,
        )
        return mols

    def _generate_ligs(self, generated, lig_mask, scale=1.0, sanitise=False):
        """
        Generate ligand mols from output tensors
        """
        coords = generated["coords"] * scale
        coords = self.builder.undo_zero_com_batch(
            coords, generated["mask"], [system.com for system in generated["complex"]]
        )
        atom_dists = generated["atomics"]
        bond_dists = generated["bonds"]
        charge_dists = generated["charges"]

        mols = self.builder.ligs_from_complex(
            coords,
            mask=lig_mask,
            atom_dists=atom_dists,
            bond_dists=bond_dists,
            charge_dists=charge_dists,
            sanitise=sanitise,
        )
        return mols

    def _generate_stabilities(self, generated):
        coords = generated["coords"]
        atom_dists = generated["atomics"]
        bond_dists = generated["bonds"]
        charge_dists = generated["charges"]
        masks = generated["mask"]
        stabilities = self.builder.mol_stabilities(
            coords, atom_dists, masks, bond_dists, charge_dists
        )
        return stabilities

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

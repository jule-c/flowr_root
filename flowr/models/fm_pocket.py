import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from rdkit import Chem
from tensordict import TensorDict
from torch.optim.lr_scheduler import LinearLR, OneCycleLR
from torchmetrics import MetricCollection

import flowr.util.metrics as Metrics
from flowr.constants import INPAINT_ENCODER as inpaint_mode_encoder
from flowr.data.data_info import GeneralInfos as DataInfos
from flowr.models.integrator import Integrator
from flowr.models.losses import LossComputer
from flowr.models.mol_builder import MolBuilder
from flowr.models.score_modifier import MaxGaussianModifier, MinGaussianModifier
from flowr.models.semla import MolecularGenerator
from flowr.util.device import get_device
from flowr.util.tokeniser import Vocabulary

_T = torch.Tensor
_BatchT = dict[str, _T]
OptFloat = Optional[float]


def create_list_defaultdict():
    return defaultdict(list)


# *********************************************************************************************************************
# ******************************************** Lightning Flow Matching Models *****************************************
# *********************************************************************************************************************


def apply_smc_guidance(
    predicted: Dict[str, torch.Tensor],
    prior: Dict[str, torch.Tensor],
    current: Dict[str, torch.Tensor],
    pocket_data: Dict[str, torch.Tensor],
    pocket_equis: torch.Tensor,
    pocket_invs: torch.Tensor,
    cond_batch: Dict[str, torch.Tensor] | None,
    value_key: str = "affinity",
    subvalue_key: str = "pic50",
    mu: float = 8.0,
    sigma=2.0,
    maximize: bool = True,
) -> tuple[
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor] | None,
]:

    predicted_values: TensorDict = predicted[value_key]
    predicted_values: torch.Tensor = predicted_values[subvalue_key].squeeze(-1)
    modifier = (
        MaxGaussianModifier(mu=mu, sigma=sigma)
        if maximize
        else MinGaussianModifier(mu=mu, sigma=sigma)
    )
    weights = modifier(predicted_values)
    assert weights.ndim == 1, "Probability weights must be 1-dimensional"

    weights_softmax = predicted_values.softmax(dim=0)
    # weights_combined = weights + weights_softmax
    weights_combined = weights_softmax
    selected_ids = torch.multinomial(
        weights_combined, num_samples=len(weights_combined), replacement=True
    )

    prior[value_key] = prior[value_key].to(selected_ids.device)
    prior["docking_score"] = prior["docking_score"].to(selected_ids.device)
    for val in prior.values():
        if hasattr(val, "to"):
            val.to(selected_ids.device)

    prior = {
        key: value[selected_ids] if len(value) > 0 else value
        for key, value in prior.items()
    }
    current = {
        key: value[selected_ids] if len(value) > 0 else value
        for key, value in current.items()
    }
    cond_batch = (
        {
            key: value[selected_ids] if len(value) > 0 else value
            for key, value in cond_batch.items()
        }
        if cond_batch is not None
        else None
    )
    predicted = {
        key: value[selected_ids] if len(value) > 0 else value
        for key, value in predicted.items()
    }
    pocket_data = {
        key: (
            value[selected_ids]
            if len(value) > 0 and isinstance(value, torch.Tensor)
            else value
        )
        for key, value in pocket_data.items()
    }
    pocket_data["complex"] = [pocket_data["complex"][i] for i in selected_ids.tolist()]
    pocket_equis = pocket_equis[selected_ids]
    pocket_invs = pocket_invs[selected_ids]

    out = (
        predicted,
        prior,
        current,
        pocket_data,
        pocket_equis,
        pocket_invs,
        cond_batch,
    )
    return out


def apply_selective_smc_guidance(
    predicted: tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    prior: Dict[str, torch.Tensor],
    current: Dict[str, torch.Tensor],
    pocket_data: tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    pocket_equis: tuple[torch.Tensor, torch.Tensor],
    pocket_invs: tuple[torch.Tensor, torch.Tensor],
    cond_batch: Dict[str, torch.Tensor] | None,
    value_key: str = "affinity",
    subvalue_key: str = "pic50",
    mu: float = 8.0,
    sigma=2.0,
    maximize: bool = True,
) -> tuple[
    tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    Dict[str, torch.Tensor] | None,
]:

    # unroll tuples specifically for target and untargeted pockets
    predicted_target, predicted_untarget = predicted
    pocket_data_target, pocket_data_untarget = pocket_data
    pocket_equis_target, pocket_equis_untarget = pocket_equis
    pocket_invs_target, pocket_invs_untarget = pocket_invs

    predicted_values_target: TensorDict = predicted_target[value_key]
    predicted_values_target: torch.Tensor = predicted_values_target[
        subvalue_key
    ].squeeze(-1)
    weights_target_softmax = predicted_values_target.softmax(dim=0)

    predicted_values_untarget: TensorDict = predicted_untarget[value_key]
    predicted_values_untarget: torch.Tensor = predicted_values_untarget[
        subvalue_key
    ].squeeze(-1)
    weights_untarget_softmax = (-1.0 * predicted_values_untarget).softmax(dim=0)

    weights_combined = weights_target_softmax + weights_untarget_softmax

    selected_ids = torch.multinomial(
        weights_combined, num_samples=len(weights_combined), replacement=True
    )

    prior[value_key] = prior[value_key].to(selected_ids.device)
    prior["docking_score"] = prior["docking_score"].to(selected_ids.device)
    for val in prior.values():
        if hasattr(val, "to"):
            val.to(selected_ids.device)

    prior = {
        key: (
            value[selected_ids]
            if len(value) > 0 and isinstance(value, torch.Tensor)
            else value
        )
        for key, value in prior.items()
    }
    prior["fragment_mode"] = [prior["fragment_mode"][i] for i in selected_ids.tolist()]
    current = {
        key: (
            value[selected_ids]
            if len(value) > 0 and isinstance(value, torch.Tensor)
            else value
        )
        for key, value in current.items()
    }
    cond_batch = (
        {
            key: (
                value[selected_ids]
                if len(value) > 0 and isinstance(value, torch.Tensor)
                else value
            )
            for key, value in cond_batch.items()
        }
        if cond_batch is not None
        else None
    )
    predicted_target = {
        key: (
            value[selected_ids]
            if len(value) > 0 and isinstance(value, torch.Tensor)
            else value
        )
        for key, value in predicted_target.items()
    }
    pocket_data_target = {
        key: (
            value[selected_ids]
            if len(value) > 0 and isinstance(value, torch.Tensor)
            else value
        )
        for key, value in pocket_data_target.items()
    }
    pocket_data_target["complex"] = [
        pocket_data_target["complex"][i] for i in selected_ids.tolist()
    ]
    pocket_equis_target = pocket_equis_target[selected_ids]
    pocket_invs_target = pocket_invs_target[selected_ids]

    predicted_untarget = deepcopy(predicted_target)

    pocket_data_untarget = {
        key: (
            value[selected_ids]
            if len(value) > 0 and isinstance(value, torch.Tensor)
            else value
        )
        for key, value in pocket_data_untarget.items()
    }
    pocket_data_untarget["complex"] = [
        pocket_data_untarget["complex"][i] for i in selected_ids.tolist()
    ]
    pocket_equis_untarget = pocket_equis_untarget[selected_ids]
    pocket_invs_untarget = pocket_invs_untarget[selected_ids]

    # put pack into tuples
    predicted = (predicted_target, predicted_untarget)
    pocket_data = (pocket_data_target, pocket_data_untarget)
    pocket_equis = (pocket_equis_target, pocket_equis_untarget)
    pocket_invs = (pocket_invs_target, pocket_invs_untarget)

    out = (
        predicted,
        prior,
        current,
        pocket_data,
        pocket_equis,
        pocket_invs,
        cond_batch,
    )
    return out


class LigandPocketCFM(pl.LightningModule):
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
        # Add confidence training parameters
        train_confidence: bool = False,
        plddt_confidence_loss_weight: OptFloat = None,
        confidence_gen_steps: int = 20,
        confidence_every_n: int = 2,
        beta1: float = 0.9,
        beta2: float = 0.999,
        weight_decay: float = 1e-12,
        coord_scale: float = 1.0,
        type_strategy: str = "ce",
        bond_strategy: str = "ce",
        coord_loss_weight: float = 1.0,
        type_loss_weight: float = 1.0,
        bond_loss_weight: float = 1.0,
        interaction_loss_weight: float = 1.0,
        charge_loss_weight: float = 1.0,
        hybridization_loss_weight: float = 1.0,
        distance_loss_weight_lig: float = None,
        distance_loss_weight_lig_pocket: float = None,
        smooth_distance_loss_weight_lig: OptFloat = None,
        smooth_distance_loss_weight_lig_pocket: OptFloat = None,
        affinity_loss_weight: float = None,
        docking_loss_weight: float = None,
        pocket_noise: str = "random",
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
        flow_interactions: bool = False,
        predict_interactions: bool = False,
        interaction_inpainting: bool = False,
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
        inpaint_self_condition: bool = False,
        **kwargs,
    ):
        super().__init__()

        if type_strategy not in ["mse", "ce", "mask"]:
            raise ValueError(f"Unsupported type training strategy '{type_strategy}'")

        if bond_strategy not in ["ce", "mask"]:
            raise ValueError(f"Unsupported bond training strategy '{bond_strategy}'")

        if lr_schedule not in ["constant", "one-cycle", "exponential", "cosine"]:
            raise ValueError(f"LR scheduler {lr_schedule} not supported.")

        if lr_schedule == "one-cycle" and total_steps is None:
            raise ValueError(
                "total_steps must be provided when using the one-cycle LR scheduler."
            )

        # INPAINTING AND SELF-CONDITIONING SETTINGS
        self.self_condition = self_condition
        self.self_condition_mode = "stacking"
        self.self_condition_prob = 0.5
        self._inpaint_self_condition = False
        if self.self_condition_mode == "residual":
            from flowr.models.self_conditioning import SelfConditioningResidualLayer

            self.residual_layer = SelfConditioningResidualLayer(
                n_atom_types=vocab.size(),
                n_bond_types=5,  # Adjust based on your bond vocabulary
                n_charge_types=vocab_charges.size(),
                n_extra_atom_feats=vocab_hybridization.size() if add_feats else None,
                rbf_dim=20,
                rbf_dmax=10.0,
            )

        self.feature_keys = ["atomics", "bonds", "charges"]
        self.sc_feature_keys = ["atomics", "bonds"]  # "charges"
        if add_feats is not None:
            self.feature_keys.append("hybridization")
            # self.sc_feature_keys.append("hybridization")

        # Save hyperparameters
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
        self.interaction_loss_weight = interaction_loss_weight
        self.charge_loss_weight = charge_loss_weight
        self.hybridization_loss_weight = hybridization_loss_weight
        self.distance_loss_weight_lig = distance_loss_weight_lig
        self.distance_loss_weight_lig_pocket = distance_loss_weight_lig_pocket
        self.smooth_distance_loss_weight_lig = smooth_distance_loss_weight_lig
        self.smooth_distance_loss_weight_lig_pocket = (
            smooth_distance_loss_weight_lig_pocket
        )
        self.affinity_loss_weight = affinity_loss_weight
        self.docking_loss_weight = docking_loss_weight
        self.self_condition = self_condition
        self.lr_schedule = lr_schedule
        self.lr_gamma = lr_gamma
        self.sampling_strategy = sampling_strategy
        self.warm_up_steps = warm_up_steps
        self.total_steps = total_steps
        self.type_mask_index = type_mask_index
        self.bond_mask_index = bond_mask_index
        self.pocket_noise = pocket_noise
        self.flow_interactions = flow_interactions
        self.predict_interactions = predict_interactions
        self.interaction_inpainting = interaction_inpainting
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

        # Confidence training parameters
        self.train_confidence = train_confidence
        self.plddt_confidence_loss_weight = plddt_confidence_loss_weight
        self.confidence_gen_steps = confidence_gen_steps
        self.confidence_every_n = confidence_every_n

        if (
            self.interaction_inpainting
            or self.func_group_inpainting
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

        # Anything else passed into kwargs will also be saved
        hparams = {
            "lr": lr,
            "add_feats": add_feats,
            "coord_scale": coord_scale,
            "coord_loss_weight": coord_loss_weight,
            "type_loss_weight": type_loss_weight,
            "bond_loss_weight": bond_loss_weight,
            "interaction_loss_weight": interaction_loss_weight,
            "charge_loss_weight": charge_loss_weight,
            "hybridization_loss_weight": hybridization_loss_weight,
            "distance_loss_weight_lig": distance_loss_weight_lig,
            "distance_loss_weight_lig_pocket": distance_loss_weight_lig_pocket,
            "smooth_distance_loss_weight_lig": smooth_distance_loss_weight_lig,
            "smooth_distance_loss_weight_lig_pocket": smooth_distance_loss_weight_lig_pocket,
            "plddt_confidence_loss_weight": plddt_confidence_loss_weight,
            "use_t_loss_weights": use_t_loss_weights,
            "affinity_loss_weight": affinity_loss_weight,
            "docking_loss_weight": docking_loss_weight,
            "type_strategy": type_strategy,
            "bond_strategy": bond_strategy,
            "self_condition": self_condition,
            "pocket_noise": pocket_noise,
            "flow_interactions": flow_interactions,
            "predict_interactions": predict_interactions,
            "interaction_inpainting": interaction_inpainting,
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
            distance_loss_weight_pocket=None,
            distance_loss_weight_lig_pocket=distance_loss_weight_lig_pocket,
            smooth_distance_loss_weight_lig=smooth_distance_loss_weight_lig,
            smooth_distance_loss_weight_pocket=None,
            smooth_distance_loss_weight_lig_pocket=smooth_distance_loss_weight_lig_pocket,
            plddt_confidence_loss_weight=plddt_confidence_loss_weight,
            affinity_loss_weight=affinity_loss_weight,
            docking_loss_weight=docking_loss_weight,
            type_strategy=type_strategy,
            bond_strategy=bond_strategy,
            use_t_loss_weights=use_t_loss_weights,
        )

        # Initialize confidence module if training confidence
        self.confidence_module = None
        if self.train_confidence:
            assert (
                self.plddt_confidence_loss_weight > 0
            ), "Confidence loss weight must be greater than 0 when training confidence module."
            from flowr.models.pocket import ConfidenceModule

            self.confidence_module = ConfidenceModule(
                d_inv=384,
                d_equi=128,
                d_message=gen.ligand_dec.d_message,
                n_layers=8,
                n_attn_heads=gen.ligand_dec.n_attn_heads,
                d_message_ff=gen.ligand_dec.d_message_ff,
                d_edge=gen.ligand_dec.d_edge,
                n_atom_types=gen.ligand_dec.n_atom_types,
                n_bond_types=gen.ligand_dec.n_bond_types,
                n_charge_types=gen.ligand_dec.n_charge_types,
                n_extra_atom_feats=gen.ligand_dec.n_extra_atom_feats,
                emb_size=gen.ligand_dec.emb_size,
                d_pocket_inv=gen.ligand_dec.d_pocket_inv,
                use_rbf=True,
                use_distances=True,
            )

        builder = MolBuilder(
            vocab,
            vocab_charges,
            vocab_hybridization=vocab_hybridization,
            vocab_aromatic=vocab_aromatic,
            pocket_noise=self.pocket_noise,
            save_dir=self.hparams.save_dir,
        )

        self.integrator = integrator
        self.builder = builder
        self.dataset_info = dataset_info

        gen_mol_metrics = {
            "validity": Metrics.Validity(),
            "fc-validity": Metrics.Validity(connected=True),
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
        if self.dataset_info is not None and self.train_mols is not None:
            gen_dist_metrics = Metrics.DistributionDistance(
                dataset_info=self.dataset_info, train_mols=self.train_mols
            )
            self.gen_dist_metrics = MetricCollection(
                {"distribution-distance": gen_dist_metrics}, compute_groups=False
            )

        self.posebusters_validity = MetricCollection(
            {"pb_validity": Metrics.PoseBustersValidity()}
        )

        # if self.train_mols is not None:
        #     print("Initialising novelty metric...")
        #     gen_mol_metrics["novelty"] = Metrics.Novelty(self.train_mols)
        #     print("Novelty metric complete.")

        self.gen_mol_metrics = MetricCollection(gen_mol_metrics, compute_groups=False)

        if self.graph_inpainting:
            docking_metrics = {
                "dock-rmsd": Metrics.MolecularPairRMSD(fix_order=False),
                "dock-shape-tanimoto": Metrics.MolecularPairShapeTanimotoSim(
                    align=False
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
        pocket_data: dict,
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

        # Apply inpainting if needed (for stacking mode)
        if self.self_condition_mode == "stacking" and self._inpaint_self_condition:
            if self.inpainting_mode:
                cond_batch = self.builder.inpaint_molecule(
                    lig_data,
                    cond_batch,
                    pocket_mask=pocket_data["mask"].bool(),
                    keep_interactions=self.flow_interactions,
                )
            elif self.graph_inpainting:
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
                    pocket_data,
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

                if (
                    self.self_condition_mode == "stacking"
                    and self._inpaint_self_condition
                ):
                    if self.inpainting_mode:
                        cond_batch = self.builder.inpaint_molecule(
                            lig_data,
                            cond_batch,
                            pocket_mask=pocket_data["mask"].bool(),
                            keep_interactions=self.flow_interactions,
                        )
                    elif self.graph_inpainting:
                        cond_batch = self.builder.inpaint_graph(
                            lig_data,
                            cond_batch,
                            feature_keys=["coords", "atomics", "bonds"],
                            overwrite_with_zeros=True,
                        )

        return cond_batch

    def forward(
        self,
        batch,
        pocket_batch,
        t,
        cond_batch=None,
        pocket_equis=None,
        pocket_invs=None,
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

        pocket_coords = pocket_batch["coords"]
        pocket_atoms = pocket_batch["atom_names"]
        pocket_bonds = pocket_batch["bonds"]
        pocket_charges = pocket_batch["charges"]
        pocket_res = pocket_batch["res_names"]
        pocket_mask = pocket_batch["mask"]

        extra_feats = (
            torch.argmax(batch["hybridization"], dim=-1) if self.add_feats else None
        )  # for now only hybridization

        interactions = batch["interactions"] if self.flow_interactions else None

        # Get times in the right format
        inpaint_mask = batch.get("fragment_mask", None)
        times = self._get_times(t, lig_mask=mask, inpaint_mask=inpaint_mask)

        # Encode inpainting mode
        inpaint_mode = (
            torch.stack(
                [
                    torch.tensor(inpaint_mode_encoder[mode]).to(get_device())
                    for mode in batch["fragment_mode"]
                ],
                dim=0,
            )
            if self.inpainting_mode
            else None
        )

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
                pocket_coords=pocket_coords,
                pocket_atom_names=pocket_atoms,
                pocket_atom_charges=torch.argmax(pocket_charges, dim=-1),
                pocket_bond_types=torch.argmax(pocket_bonds, dim=-1),
                pocket_res_types=pocket_res,
                pocket_atom_mask=pocket_mask,
                pocket_equis=pocket_equis,
                pocket_invs=pocket_invs,
                interactions=(
                    torch.argmax(interactions, dim=-1)
                    if interactions is not None
                    else None
                ),
                inpaint_mask=inpaint_mask,
                inpaint_mode=inpaint_mode,
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
                pocket_coords=pocket_coords,
                pocket_atom_names=pocket_atoms,
                pocket_atom_charges=torch.argmax(pocket_charges, dim=-1),
                pocket_bond_types=torch.argmax(pocket_bonds, dim=-1),
                pocket_res_types=pocket_res,
                pocket_atom_mask=pocket_mask,
                pocket_equis=pocket_equis,
                pocket_invs=pocket_invs,
                interactions=(
                    torch.argmax(interactions, dim=-1)
                    if interactions is not None
                    else None
                ),
                inpaint_mask=inpaint_mask,
                inpaint_mode=inpaint_mode,
            )
        out["times"] = times
        out["fragment_mode"] = batch.get("fragment_mode", None)
        return out

    def training_step(self, batch, b_idx):
        # Input data
        prior, data, interpolated, times = batch
        # Extract pocket data
        pocket_data = self.builder.extract_pocket_from_complex(data)
        pocket_data["interactions"] = data["interactions"]

        # Extract ligand data
        lig_prior = self.builder.extract_ligand_from_complex(prior)
        lig_interp = self.builder.extract_ligand_from_complex(interpolated)
        lig_interp["fragment_mask"] = data["fragment_mask"]
        lig_interp["fragment_mode"] = data["fragment_mode"]
        lig_interp["interactions"] = interpolated["interactions"]
        lig_data = self.builder.extract_ligand_from_complex(data)
        lig_data["pocket_mask"] = pocket_data["mask"]
        lig_data["interactions"] = data["interactions"]
        lig_data["fragment_mask"] = data["fragment_mask"]
        lig_data["fragment_mode"] = data["fragment_mode"]

        # Times (bs, 3) -> (lig_times_cont, lig_times_disc, pocket_times)
        times = times.T

        # If training with self conditioning, half the time generate a conditional batch by setting cond to zeros
        cond_batch = self._prepare_self_condition_batch(
            lig_interp, lig_data, lig_prior, pocket_data, times, training=True
        )

        predicted = self(
            lig_interp, pocket_data, times, cond_batch=cond_batch, training=True
        )

        # Confidence training: generate structures and predict confidence
        confidence_predictions = None
        if (
            self.train_confidence
            and self.confidence_module is not None
            and self.plddt_confidence_loss_weight > 0
        ):
            # Retrieve fragment mask for conditional generation
            lig_prior["fragment_mask"] = prior["fragment_mask"]

            # Build starting times for confidence generation
            lig_times_cont = torch.zeros(
                lig_prior["coords"].size(0), device=self.device
            )
            lig_times_disc = torch.zeros(
                lig_prior["coords"].size(0), device=self.device
            )
            pocket_times = torch.zeros(
                pocket_data["coords"].size(0), device=self.device
            )
            conf_times = [lig_times_cont, lig_times_disc, pocket_times]

            # Generate structures for confidence prediction
            generated_batch, pocket_equis, pocket_invs = self._generate_for_confidence(
                lig_prior,
                pocket_data,
                steps=self.confidence_gen_steps,
                times=conf_times,
            )

            # Predict confidence scores for generated structures
            confidence_predictions = self.confidence_module(
                coords=generated_batch["coords"],
                atom_types=torch.argmax(generated_batch["atomics"], dim=-1),
                bond_types=torch.argmax(generated_batch["bonds"], dim=-1),
                atom_charges=torch.argmax(generated_batch["charges"], dim=-1),
                extra_feats=(
                    torch.argmax(generated_batch["hybridization"], dim=-1)
                    if self.add_feats
                    else None
                ),
                atom_mask=generated_batch["mask"],
                pocket_coords=pocket_data["coords"],
                pocket_equis=pocket_equis,
                pocket_invs=pocket_invs,
                pocket_atom_mask=pocket_data["mask"],
            )

            # Add confidence predictions to the predicted dict for loss computation
            predicted["plddt"] = confidence_predictions

        # import pdb

        # pdb.set_trace()
        # self.builder.tensors_to_xyz(
        #     prior=prior,
        #     interpolated=interpolated,
        #     data=data,
        #     coord_scale=self.coord_scale,
        #     idx=1,
        #     save_dir="/hpfs/userws/cremej01/projects/tmp",
        # )

        losses = self._loss(lig_data, lig_interp, predicted, pocket_data=pocket_data)
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
        prior, data, _, _ = batch

        # Extract pocket data
        pocket_data = self.builder.extract_pocket_from_complex(data)
        pocket_data["interactions"] = prior["interactions"]
        pocket_data["complex"] = data["complex"]
        # Extract ligand data
        lig_prior = self.builder.extract_ligand_from_complex(prior)
        lig_prior["fragment_mask"] = prior["fragment_mask"]
        lig_prior["fragment_mode"] = prior["fragment_mode"]
        lig_prior["interactions"] = prior["interactions"]

        # Build starting times for the integrator
        lig_times_cont = torch.zeros(prior["coords"].size(0), device=self.device)
        lig_times_disc = torch.zeros(prior["coords"].size(0), device=self.device)
        pocket_times = torch.zeros(pocket_data["coords"].size(0), device=self.device)
        prior_times = [lig_times_cont, lig_times_disc, pocket_times]

        # Generate
        gen_batch = self._generate(
            lig_prior,
            pocket_data,
            steps=self.integrator.steps,
            times=prior_times,
            strategy=self.sampling_strategy,
            corr_iters=self.corrector_iters,
            final_corr_pred=True,
            final_inpaint=False,
        )
        gen_mols = self._generate_mols(gen_batch)

        if not self.trainer.sanity_checking:
            self.gen_mol_metrics.update(gen_mols)
            if self.gen_dist_metrics is not None:
                self.gen_dist_metrics.update(gen_mols)
            ref_pdbs_with_hs = self._retrieve_pdbs_with_hs(
                data,
                stage="val",
            )
            self.posebusters_validity.update(gen_mols, ref_pdbs_with_hs)
            if self.graph_inpainting and self.docking_metrics is not None:
                ref_mols = self.retrieve_ligs_with_hs(data)
                if self.hparams.remove_hs:
                    ref_mols = [Chem.RemoveHs(mol) for mol in ref_mols]
                self.docking_metrics.update(gen_mols, ref_mols)

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            gen_dist_results = {}

            gen_metrics_results = self.gen_mol_metrics.compute()
            if self.gen_dist_metrics is not None:
                gen_dist_results = self.gen_dist_metrics.compute()
            posebusters_results = self.posebusters_validity.compute()

            if self.graph_inpainting:
                docking_metrics_results = self.docking_metrics.compute()
            else:
                docking_metrics_results = {}

            metrics = {
                **gen_metrics_results,
                **gen_dist_results,
                **posebusters_results,
                **docking_metrics_results,
            }
            for metric, value in metrics.items():
                # Show main validity and individual critical metrics in progress bar
                progbar = metric in ["pb_validity"]
                if isinstance(value, dict):
                    for k, v in value.items():
                        self.log(
                            f"val-{k}",
                            v.to(self.device),
                            on_epoch=True,
                            logger=True,
                            prog_bar=False,
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
            self.posebusters_validity.reset()
            if self.graph_inpainting and self.docking_metrics is not None:
                self.docking_metrics.reset()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def predict_step(self, batch, batch_idx):
        # Input data
        prior, data, _, _ = batch

        # Extract pocket data
        pocket_data = self.builder.extract_pocket_from_complex(data)
        pocket_data["interactions"] = prior["interactions"]
        pocket_data["complex"] = data["complex"]
        # Extract ligand data
        lig_prior = self.builder.extract_ligand_from_complex(prior)
        lig_prior["fragment_mask"] = prior["fragment_mask"]
        lig_prior["fragment_mode"] = prior["fragment_mode"]
        lig_prior["interactions"] = prior["interactions"]

        # Starting times for the integrator
        lig_times_cont = torch.zeros(prior["coords"].size(0), device=self.device)
        lig_times_disc = torch.zeros(prior["coords"].size(0), device=self.device)
        pocket_times = torch.zeros(pocket_data["coords"].size(0), device=self.device)
        prior_times = [lig_times_cont, lig_times_disc, pocket_times]

        # Generate
        output = self._generate(
            lig_prior,
            pocket_data,
            steps=self.integrator.steps,
            times=prior_times,
            strategy=self.sampling_strategy,
            solver="euler",
            corr_iters=self.corrector_iters,
            save_traj=False,
            iter=batch_idx,
        )
        gen_ligs = self._generate_mols(output)

        # retrieve ground truth/native ligands and pdbs
        ref_ligs_with_hs = self.retrieve_ligs_with_hs(data)
        ref_pdbs_with_hs = self.retrieve_pdbs_with_hs(
            data, save_dir=Path(self.hparams.save_dir) / "ref_pdbs"
        )

        # group ligands by pdb in a list of lists as we are potentially sampling N ligands per target;
        # de-duplicate native ligands and pdbs as they are loaded N times
        ligs_by_pdb = defaultdict(create_list_defaultdict)
        ligs_by_pdb_with_hs = defaultdict(create_list_defaultdict)
        for gen_lig, ref_lig_with_hs, pdb_with_hs in zip(
            gen_ligs, ref_ligs_with_hs, ref_pdbs_with_hs
        ):
            ligs_by_pdb_with_hs[pdb_with_hs]["ref"] = ref_lig_with_hs
        gen_ligs_by_pdb = [v["gen"] for _, v in ligs_by_pdb.items()]
        ref_ligs_with_hs = [v["ref"] for v in ligs_by_pdb_with_hs.values()]
        ref_pdbs_with_hs = [pdb for pdb in ligs_by_pdb_with_hs]

        outputs = {
            "gen_ligs": gen_ligs_by_pdb,
            "ref_ligs_with_hs": ref_ligs_with_hs,
            "ref_pdbs_with_hs": ref_pdbs_with_hs,
        }
        return outputs

    def _loss(self, data, interpolated, predicted, pocket_data, times=None):
        """Compute all losses using the dedicated loss computer."""
        return self.loss_computer.compute_losses(
            data,
            interpolated,
            predicted,
            pocket_data,
            inpaint_mode=self.inpainting_mode,
        )

    def _build_trajectory(
        self,
        curr,
        predicted,
        pocket_data,
        iter: int = 0,
        step: int = 0,
    ):
        self.builder.write_xyz_file_from_batch(
            data=predicted,
            coord_scale=self.coord_scale,
            path=os.path.join(self.hparams.save_dir, f"traj_pred_mols_{iter}"),
            t=step,
        )
        predicted["res_names"] = torch.zeros(
            predicted["atomics"].size(0),
            predicted["atomics"].size(1),
            device=self.device,
        ).long()  # dummy res names
        pred_complex = self.builder.add_ligand_to_pocket(
            lig_data=predicted,
            pocket_data=pocket_data,
        )
        self.builder.write_xyz_file_from_batch(
            data=pred_complex,
            coord_scale=self.coord_scale,
            path=os.path.join(self.hparams.save_dir, f"traj_pred_{iter}"),
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
    ):
        pred_mols = self._generate_mols(predicted)
        self.builder.write_trajectory_as_xyz(
            pred_mols=pred_mols,
            file_path=os.path.join(self.hparams.save_dir, f"traj_pred_mols_{iter}"),
            save_path=os.path.join(
                self.hparams.save_dir, "trajectories_pred_mols", f"traj_{iter}"
            ),
            remove_intermediate_files=True,
        )
        self.builder.write_trajectory_as_xyz(
            pred_mols=pred_mols,
            file_path=os.path.join(self.hparams.save_dir, f"traj_pred_{iter}"),
            save_path=os.path.join(
                self.hparams.save_dir, "trajectories_pred", f"traj_{iter}"
            ),
            remove_intermediate_files=True,
        )
        self.builder.write_trajectory_as_xyz(
            pred_mols=pred_mols,
            file_path=os.path.join(self.hparams.save_dir, f"traj_interp_{iter}"),
            save_path=os.path.join(
                self.hparams.save_dir, "trajectories_interp", f"traj_{iter}"
            ),
            remove_intermediate_files=True,
        )

    def _update_times(
        self,
        times,
        step_size: float,
    ):
        lig_times_cont = times[0] + step_size
        lig_times_disc = times[1] + step_size
        pocket_times = times[2] + step_size
        times = [
            lig_times_cont,
            lig_times_disc,
            pocket_times,
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
        if self.predict_interactions or self.flow_interactions:
            interaction_probs = F.softmax(out["interactions"], dim=-1)

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
        if "affinity" in out:
            predicted["affinity"] = out["affinity"]
        if "docking" in out:
            predicted["docking"] = out["docking"]
        if hybridization_probs is not None:
            predicted["hybridization"] = hybridization_probs
        if self.predict_interactions or self.flow_interactions:
            predicted["interactions"] = interaction_probs
        return predicted, cond_batch

    def _generate_for_confidence(
        self,
        prior: dict,
        pocket_data: dict,
        steps: int,
        times: list,
    ):
        """Generate structures for confidence training (lighter version of _generate)"""

        time_points = torch.linspace(0, 1, steps + 1, device=self.device)
        step_sizes = [t1 - t0 for t0, t1 in zip(time_points[:-1], time_points[1:])]

        curr = {
            k: (
                v.clone()
                if torch.is_tensor(v)
                else v.copy() if isinstance(v, list) else v
            )
            for k, v in prior.items()
        }

        cond_batch = {
            "coords": prior["coords"],
            "atomics": torch.zeros_like(prior["atomics"]),
            "bonds": torch.zeros_like(prior["bonds"]),
        }

        with torch.no_grad():
            # Use pre-computed pocket encodings if available
            pocket_equis, pocket_invs = self.gen.get_pocket_encoding(
                pocket_data["coords"],
                pocket_data["atom_names"],
                pocket_atom_charges=torch.argmax(pocket_data["charges"], dim=-1),
                pocket_bond_types=torch.argmax(pocket_data["bonds"], dim=-1),
                pocket_res_types=pocket_data["res_names"],
                pocket_atom_mask=pocket_data["mask"],
            )

            for i, step_size in enumerate(step_sizes):
                cond = cond_batch if self.self_condition else None

                # Run the model
                out = self(
                    curr,
                    pocket_data,
                    times,
                    cond_batch=cond,
                    pocket_equis=pocket_equis,
                    pocket_invs=pocket_invs,
                    training=False,
                )
                predicted, cond_batch = self._get_predictions(out)

                # Euler step
                curr = self.integrator.step(curr, predicted, prior, times, step_size)

                # Update times for the next step
                times = self._update_times(times, step_size)

                # Apply inpainting if needed
                if self.inpainting_mode and self.inpainting_mode_inf == "fragment":
                    curr = self.builder.inpaint_molecule(
                        data=prior,
                        prediction=curr,
                        pocket_mask=pocket_data["mask"].bool(),
                        keep_interactions=self.flow_interactions,
                    )
                elif self.graph_inpainting:
                    curr = self.builder.inpaint_graph(
                        data=prior,
                        prediction=curr,
                        keep_interactions=self.flow_interactions,
                    )
                    cond_batch = self.builder.inpaint_graph(
                        data=prior,
                        prediction=cond_batch,
                        feature_keys=["coords", "atomics", "bonds"],
                        overwrite_with_zeros=True,
                    )

        # Final corrector prediction
        eps = -1e-4
        times = self._update_times(
            times,
            eps,
        )
        with torch.no_grad():
            cond = cond_batch if self.self_condition else None
            out = self(
                curr,
                pocket_data,
                times,
                training=False,
                cond_batch=cond,
                pocket_equis=pocket_equis,
                pocket_invs=pocket_invs,
            )
        predicted, _ = self._get_predictions(out)

        return predicted, pocket_equis, pocket_invs

    def _generate(
        self,
        prior: dict,
        pocket_data: dict,
        steps: int,
        times: list,
        strategy: str = "linear",
        save_traj: bool = False,
        solver: str = "euler",
        corr_iters=None,
        corr_step_size=None,
        iter: int = 0,
        apply_guidance: bool = False,
        guidance_window_start: float = 0.0,
        guidance_window_end: float = 1.0,
        value_key: str = "affinity",
        subvalue_key: str = "pic50",
        mu: float = 8.0,
        sigma: float = 2.0,
        maximize: bool = True,
        coord_noise_level: float = 0.2,
        final_inpaint: bool = False,
        final_corr_pred: bool = True,
    ):

        self.integrator.use_sde_simulation = (
            self.integrator.use_sde_simulation or apply_guidance
        )
        if self.integrator.use_sde_simulation:
            self.integrator.coord_noise_level = coord_noise_level

        corr_iters = 0 if corr_iters is None else corr_iters

        if strategy == "linear":
            # time_points = np.linspace(0, 0.999, steps + 1).tolist()
            time_points = torch.linspace(0, 1, steps + 1)
        elif strategy == "log":
            # time_points = (1 - np.geomspace(0.01, 0.999, steps + 1)).tolist()
            # time_points.reverse()
            time_points = 1.0 - torch.logspace(-2, 0, steps + 1).flip(0)
            time_points = time_points - torch.amin(time_points)
            time_points = time_points / torch.amax(time_points)
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

        cond_batch = {
            "coords": prior["coords"],
            "atomics": torch.zeros_like(prior["atomics"]),
            "bonds": torch.zeros_like(prior["bonds"]),
        }
        if self._inpaint_self_condition:
            if self.inpainting_mode:
                cond_batch = self.builder.inpaint_molecule(
                    prior,
                    cond_batch,
                    pocket_mask=pocket_data["mask"].bool(),
                    keep_interactions=self.flow_interactions,
                )
            elif self.graph_inpainting:
                cond_batch = self.builder.inpaint_graph(
                    prior,
                    cond_batch,
                    feature_keys=["coords", "atomics", "bonds"],
                    overwrite_with_zeros=True,
                )

        step_sizes = [t1 - t0 for t0, t1 in zip(time_points[:-1], time_points[1:])]
        with torch.no_grad():
            # Generate pocket encodings only once at inference (NOTE: only for rigid SBDD)
            pocket_equis, pocket_invs = self.gen.get_pocket_encoding(
                pocket_data["coords"],
                pocket_data["atom_names"],
                pocket_atom_charges=torch.argmax(pocket_data["charges"], dim=-1),
                pocket_bond_types=torch.argmax(pocket_data["bonds"], dim=-1),
                pocket_res_types=pocket_data["res_names"],
                pocket_atom_mask=pocket_data["mask"],
            )
            for i, step_size in enumerate(step_sizes):
                cond = cond_batch if self.self_condition else None
                # Run the model
                out = self(
                    curr,
                    pocket_data,
                    times,
                    cond_batch=cond,
                    pocket_equis=pocket_equis,
                    pocket_invs=pocket_invs,
                    training=False,
                )
                predicted, cond_batch = self._get_predictions(out)

                # Integrate the ODE
                if solver == "midpoint":
                    if self.graph_inpainting:
                        raise ValueError(
                            "Midpoint solver needs to be updated for graph inpainting."
                        )
                    # Euler step
                    curr = self.integrator.step(
                        curr, predicted, prior, times, step_size * 0.5
                    )
                    times = self._update_times(
                        times,
                        step_size * 0.5,
                    )
                    out = self(
                        curr,
                        pocket_data,
                        times,
                        training=False,
                        cond_batch=cond,
                        pocket_equis=pocket_equis,
                        pocket_invs=pocket_invs,
                    )
                    predicted, cond_batch = self._get_predictions(out)
                    # Euler step
                    curr = self.integrator.step(
                        curr, predicted, prior, times, step_size
                    )
                    times = self._update_times(
                        times,
                        step_size * (-0.5),
                    )
                else:
                    # Euler step
                    curr = self.integrator.step(
                        curr, predicted, prior, times, step_size
                    )

                # Inpainting for the ligand if required
                if self.inpainting_mode and self.inpainting_mode_inf == "fragment":
                    curr = self.builder.inpaint_molecule(
                        data=prior,
                        prediction=curr,
                        pocket_mask=pocket_data["mask"].bool(),
                        keep_interactions=self.flow_interactions,
                    )
                elif self.graph_inpainting:
                    curr = self.builder.inpaint_graph(
                        prior,
                        curr,
                        feature_keys=self.feature_keys,
                    )
                # Self-conditioning inpainting
                if self._inpaint_self_condition:
                    if self.inpainting_mode:
                        cond_batch = self.builder.inpaint_molecule(
                            prior,
                            cond_batch,
                            pocket_mask=pocket_data["mask"].bool(),
                            keep_interactions=self.flow_interactions,
                        )
                    elif self.graph_inpainting:
                        cond_batch = self.builder.inpaint_graph(
                            prior,
                            cond_batch,
                            feature_keys=["coords", "atomics", "bonds"],
                            overwrite_with_zeros=True,
                        )

                # Apply SMC guidance
                if (
                    apply_guidance
                    and guidance_window_start <= times[0][0]
                    and times[0][0] <= guidance_window_end
                ):
                    (
                        predicted,
                        prior,
                        curr,
                        pocket_data,
                        pocket_equis,
                        pocket_invs,
                        cond_batch,
                    ) = apply_smc_guidance(
                        predicted=predicted,
                        prior=prior,
                        current=curr,
                        pocket_data=pocket_data,
                        pocket_equis=pocket_equis,
                        pocket_invs=pocket_invs,
                        cond_batch=cond_batch,
                        value_key=value_key,
                        subvalue_key=subvalue_key,
                        mu=mu,
                        sigma=sigma,
                        maximize=maximize,
                    )

                # Update times for the next step
                times = self._update_times(
                    times,
                    step_size,
                )

                # Build trajectory
                if save_traj:
                    self._build_trajectory(
                        curr,
                        predicted,
                        pocket_data,
                        iter=iter,
                        step=i,
                    )

            # Corrector iterations at the end of sampling
            for _ in range(corr_iters):
                cond = cond_batch if self.self_condition else None
                out = self(
                    curr,
                    pocket_data,
                    times,
                    training=False,
                    cond_batch=cond,
                    pocket_equis=pocket_equis,
                    pocket_invs=pocket_invs,
                )

                predicted, cond_batch = self._get_predictions(out)
                step_size = 1 / steps if corr_step_size is None else corr_step_size
                curr = self.integrator.corrector_iter(
                    curr, predicted, prior, times, step_size
                )
                if self.inpainting_mode and self.inpainting_mode_inf == "fragment":
                    curr = self.builder.inpaint_molecule(
                        data=prior,
                        prediction=curr,
                        pocket_mask=pocket_data["mask"].bool(),
                        keep_interactions=self.flow_interactions,
                    )
                elif self.graph_inpainting:
                    curr = self.builder.inpaint_graph(
                        prior,
                        curr,
                        feature_keys=self.feature_keys,
                    )
                # Self-conditioning inpainting
                if self._inpaint_self_condition:
                    if self.inpainting_mode:
                        cond_batch = self.builder.inpaint_molecule(
                            prior,
                            cond_batch,
                            pocket_mask=pocket_data["mask"].bool(),
                            keep_interactions=self.flow_interactions,
                        )
                    elif self.graph_inpainting:
                        cond_batch = self.builder.inpaint_graph(
                            prior,
                            cond_batch,
                            feature_keys=["coords", "atomics", "bonds"],
                            overwrite_with_zeros=True,
                        )

        if final_corr_pred:
            # Final corrector prediction
            eps = -1e-4
            times = self._update_times(
                times,
                eps,
            )
            with torch.no_grad():
                cond = cond_batch if self.self_condition else None
                out = self(
                    curr,
                    pocket_data,
                    times,
                    training=False,
                    cond_batch=cond,
                    pocket_equis=pocket_equis,
                    pocket_invs=pocket_invs,
                )
            predicted, _ = self._get_predictions(out)

            if final_inpaint:
                # Inpainting for the ligand if required
                if self.inpainting_mode and self.inpainting_mode_inf == "fragment":
                    predicted = self.builder.inpaint_molecule(
                        data=prior,
                        prediction=predicted,
                        pocket_mask=pocket_data["mask"].bool(),
                        keep_interactions=self.flow_interactions,
                    )
                elif self.graph_inpainting:
                    predicted = self.builder.inpaint_graph(
                        prior,
                        predicted,
                        feature_keys=self.feature_keys,
                    )
            # Add to trajectory
            if save_traj:
                self._build_trajectory(
                    curr,
                    predicted,
                    pocket_data,
                    iter=iter,
                    step=i + 1,
                )
        else:
            predicted = curr

        # Move everything to CPU
        predicted = {
            k: v.cpu().detach() if torch.is_tensor(v) else v
            for k, v in predicted.items()
        }
        # Scale back coordinates if necessary
        predicted["coords"] = predicted["coords"] * self.coord_scale
        # Undo zero COM alignment if necessary
        if "complex" in pocket_data:
            predicted["coords"] = self.builder.undo_zero_com_batch(
                predicted["coords"],
                predicted["mask"],
                com_list=[system.com for system in pocket_data["complex"]],
            )

        # Save trajectory
        if save_traj:
            self._save_trajectory(
                predicted,
                iter=iter,
            )

        return predicted

    def _generate_selective(
        self,
        prior: dict,
        pocket_data_target: dict,
        pocket_data_untarget: dict,
        steps: int,
        times: list,
        strategy: str = "linear",
        save_traj: bool = False,
        solver: str = "euler",
        corr_iters=None,
        corr_step_size=None,
        iter: int = 0,
        apply_guidance: bool = False,
        guidance_window_start: float = 0.0,
        guidance_window_end: float = 1.0,
        value_key: str = "affinity",
        subvalue_key: str = "pic50",
        mu: float = 8.0,
        sigma: float = 2.0,
        maximize: bool = True,
        coord_noise_level: float = 0.2,
    ):

        assert apply_guidance

        self.integrator.use_sde_simulation = (
            self.integrator.use_sde_simulation or apply_guidance
        )
        if self.integrator.use_sde_simulation:
            self.integrator.coord_noise_level = coord_noise_level

        corr_iters = 0 if corr_iters is None else corr_iters

        if strategy == "linear":
            # time_points = np.linspace(0, 0.999, steps + 1).tolist()
            time_points = torch.linspace(0, 1, steps + 1)
        elif strategy == "log":
            # time_points = (1 - np.geomspace(0.01, 0.999, steps + 1)).tolist()
            # time_points.reverse()
            time_points = 1.0 - torch.logspace(-2, 0, steps + 1).flip(0)
            time_points = time_points - torch.amin(time_points)
            time_points = time_points / torch.amax(time_points)
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

        cond_batch = {
            "coords": prior["coords"],
            "atomics": torch.zeros_like(prior["atomics"]),
            "bonds": torch.zeros_like(prior["bonds"]),
        }

        if self.graph_inpainting and self._inpaint_self_condition:
            cond_batch["coords"] = torch.zeros_like(prior["coords"])

        step_sizes = [t1 - t0 for t0, t1 in zip(time_points[:-1], time_points[1:])]
        with torch.no_grad():

            pocket_equis_target, pocket_invs_target = self.gen.get_pocket_encoding(
                pocket_data_target["coords"],
                pocket_data_target["atom_names"],
                pocket_atom_charges=torch.argmax(pocket_data_target["charges"], dim=-1),
                pocket_bond_types=torch.argmax(pocket_data_target["bonds"], dim=-1),
                pocket_res_types=pocket_data_target["res_names"],
                pocket_atom_mask=pocket_data_target["mask"],
            )

            pocket_equis_untarget, pocket_invs_untarget = self.gen.get_pocket_encoding(
                pocket_data_untarget["coords"],
                pocket_data_untarget["atom_names"],
                pocket_atom_charges=torch.argmax(
                    pocket_data_untarget["charges"], dim=-1
                ),
                pocket_bond_types=torch.argmax(pocket_data_untarget["bonds"], dim=-1),
                pocket_res_types=pocket_data_untarget["res_names"],
                pocket_atom_mask=pocket_data_untarget["mask"],
            )

            for i, step_size in enumerate(step_sizes):
                cond = cond_batch if self.self_condition else None

                # Run the model on the selected target pocket
                out_target = self(
                    curr,
                    pocket_data_target,
                    times,
                    cond_batch=cond,
                    pocket_equis=pocket_equis_target,
                    pocket_invs=pocket_invs_target,
                    training=False,
                )
                predicted_target, cond_batch = self._get_predictions(out_target)

                # Run the model on the selected untarget pocket
                out_untarget = self(
                    curr,
                    pocket_data_untarget,
                    times,
                    cond_batch=cond_batch,
                    pocket_equis=pocket_equis_untarget,
                    pocket_invs=pocket_invs_untarget,
                    training=False,
                )
                predicted_untarget, _ = self._get_predictions(out_untarget)

                # Integrate the ODE using Euler
                curr = self.integrator.step(
                    curr, predicted_target, prior, times, step_size
                )

                # put into tuples
                predicted = (predicted_target, predicted_untarget)
                pocket_data = (pocket_data_target, pocket_data_untarget)
                pocket_equis = (pocket_equis_target, pocket_equis_untarget)
                pocket_invs = (pocket_invs_target, pocket_invs_untarget)

                # import pdb; pdb.set_trace()
                # guidance
                if (
                    apply_guidance
                    and guidance_window_start <= times[0][0]
                    and times[0][0] <= guidance_window_end
                ):
                    (
                        predicted,
                        prior,
                        curr,
                        pocket_data,
                        pocket_equis,
                        pocket_invs,
                        cond_batch,
                    ) = apply_selective_smc_guidance(
                        predicted=predicted,
                        prior=prior,
                        current=curr,
                        pocket_data=pocket_data,
                        pocket_equis=pocket_equis,
                        pocket_invs=pocket_invs,
                        cond_batch=cond_batch,
                        value_key=value_key,
                        subvalue_key=subvalue_key,
                        mu=mu,
                        sigma=sigma,
                        maximize=maximize,
                    )

                    # unroll
                    predicted_target, predicted_untarget = predicted
                    pocket_data_target, pocket_data_untarget = pocket_data
                    pocket_equis_target, pocket_equis_untarget = pocket_equis
                    pocket_invs_target, pocket_invs_untarget = pocket_invs

                # Update times for the next step
                times = self._update_times(
                    times,
                    step_size,
                )

                # Inpainting for the ligand if required
                if self.inpainting_mode and self.inpainting_mode_inf == "fragment":
                    curr = self.builder.inpaint_molecule(
                        data=prior,
                        prediction=curr,
                        pocket_mask=pocket_data_target["mask"].bool(),
                        keep_interactions=self.flow_interactions,
                    )
                elif self.graph_inpainting:
                    curr = self.builder.inpaint_graph(
                        prior,
                        curr,
                        feature_keys=self.feature_keys,
                    )
                    if self._inpaint_self_condition:
                        cond_batch = self.builder.inpaint_graph(
                            prior,
                            cond_batch,
                            feature_keys=["coords", "atomics", "bonds"],
                            overwrite_with_zeros=True,
                        )

                # Save trajectory
                if save_traj:
                    self._build_trajectory(
                        curr,
                        predicted[0],
                        pocket_data[0],
                        iter=iter,
                        step=i,
                    )

            # Corrector iterations at the end of sampling
            for _ in range(corr_iters):
                cond = cond_batch if self.self_condition else None
                out_target = self(
                    curr,
                    pocket_data_target,
                    times,
                    training=False,
                    cond_batch=cond,
                    pocket_equis=pocket_equis_target,
                    pocket_invs=pocket_invs_target,
                )

                predicted_target, cond_batch = self._get_predictions(out_target)
                if self.graph_inpainting:
                    # Inpaint the predicted structure if graph inpainting is used
                    predicted_target = self.builder.inpaint_graph(
                        prior,
                        predicted_target,
                        feature_keys=self.feature_keys,
                    )
                    if self._inpaint_self_condition:
                        cond_batch = self.builder.inpaint_graph(
                            prior,
                            cond_batch,
                            feature_keys=["coords", "atomics", "bonds"],
                            overwrite_with_zeros=True,
                        )
                step_size = 1 / steps if corr_step_size is None else corr_step_size
                curr = self.integrator.corrector_iter(
                    curr, predicted_target, prior, times, step_size
                )
                if self.inpainting_mode and self.inpainting_mode_inf == "fragment":
                    curr = self.builder.inpaint_molecule(
                        data=prior,
                        prediction=curr,
                        pocket_mask=pocket_data_target["mask"].bool(),
                        keep_interactions=self.flow_interactions,
                    )
                elif self.graph_inpainting:
                    curr = self.builder.inpaint_graph(
                        prior,
                        curr,
                        feature_keys=self.feature_keys,
                    )
                    if self._inpaint_self_condition:
                        cond_batch = self.builder.inpaint_graph(
                            prior,
                            cond_batch,
                            feature_keys=["coords", "atomics", "bonds"],
                            overwrite_with_zeros=True,
                        )

        # Final corrector prediction
        eps = -1e-4
        times = self._update_times(
            times,
            eps,
        )
        with torch.no_grad():
            cond = cond_batch if self.self_condition else None
            out_target = self(
                curr,
                pocket_data_target,
                times,
                training=False,
                cond_batch=cond,
                pocket_equis=pocket_equis_target,
                pocket_invs=pocket_invs_target,
            )

            # Get affinity predictions
            out_non_target = self(
                out_target,
                pocket_data_untarget,
                times,
                training=False,
                cond_batch=cond,
                pocket_equis=pocket_equis_untarget,
                pocket_invs=pocket_invs_untarget,
            )

        predicted_target, _ = self._get_predictions(out_target)
        predicted_untarget, _ = self._get_predictions(out_non_target)

        if self.graph_inpainting:
            predicted_target = self.builder.inpaint_graph(
                prior,
                predicted_target,
                feature_keys=self.feature_keys,
            )

        # Move everything to CPU
        predicted_target = {
            k: v.cpu().detach() if torch.is_tensor(v) else v
            for k, v in predicted_target.items()
        }
        # Scale back coordinates if necessary
        predicted_target["coords"] = predicted_target["coords"] * self.coord_scale

        # Move everything to CPU
        predicted_untarget = {
            k: v.cpu().detach() if torch.is_tensor(v) else v
            for k, v in predicted_untarget.items()
        }

        # Undo zero COM alignment if necessary
        if "complex" in pocket_data_target:
            predicted_target["coords"] = self.builder.undo_zero_com_batch(
                predicted_target["coords"],
                predicted_target["mask"],
                com_list=[system.com for system in pocket_data_target["complex"]],
            )

        # append tensor affinity dict
        affinity_untarget = predicted_untarget.get("affinity", None)
        if affinity_untarget is not None:
            affinity_untarget = {
                key + "_untarget": val for key, val in affinity_untarget.items()
            }

        if "affinity" in predicted_target.keys() and affinity_untarget is not None:
            predicted_target["affinity"].update(affinity_untarget)

        if save_traj:
            self._save_trajectory(
                predicted_target,
                iter=iter,
            )

        return predicted_target

    def _predict_affinity(self, ligand_prior, ligand_data, pocket_data, times):
        """
        Predict the binding affinity of a batch of protein-ligand complexes.

        Args:
            ligand_prior: Prior ligand state (noisy/starting state)
            ligand_data: Target ligand state (clean/final state)
            pocket_data: Pocket information
            times: Time points for the flow

        Returns:
            Dictionary containing predicted properties including affinity
        """
        with torch.no_grad():
            # First pass: Generate conditional batch from prior (similar to training)
            # Initialize cond_batch with zeros for stacking mode
            cond_batch = {
                "coords": ligand_prior["coords"],
                "atomics": torch.zeros_like(ligand_prior["atomics"]),
                "bonds": torch.zeros_like(ligand_prior["bonds"]),
            }
            if self.add_feats:
                cond_batch["hybridization"] = torch.zeros_like(
                    ligand_prior["hybridization"]
                )

            # Run prediction on prior to get conditional batch
            out_prior = self(
                ligand_data,
                pocket_data,
                times,
                cond_batch=cond_batch if self.self_condition else None,
                training=False,
            )

            # Prepare conditional batch from prior prediction
            cond_batch = {
                "coords": out_prior["coords"],
                "atomics": F.softmax(out_prior["atomics"], dim=-1),
                "bonds": F.softmax(out_prior["bonds"], dim=-1),
            }
            if self.add_feats:
                cond_batch["hybridization"] = F.softmax(
                    out_prior["hybridization"], dim=-1
                )

            # Second pass: Predict on actual ligand data using conditional batch from prior
            out = self(
                ligand_data,
                pocket_data,
                times,
                cond_batch=cond_batch if self.self_condition else None,
                training=False,
            )
            predicted, _ = self._get_predictions(out)

        # Move everything to CPU
        predicted = {
            k: v.cpu().detach() if torch.is_tensor(v) else v
            for k, v in predicted.items()
        }

        # Scale back coordinates if necessary
        predicted["coords"] = predicted["coords"] * self.coord_scale

        # Undo zero COM alignment if necessary
        if "complex" in pocket_data:
            predicted["coords"] = self.builder.undo_zero_com_batch(
                predicted["coords"],
                predicted["mask"],
                com_list=[system.com for system in pocket_data["complex"]],
            )

        return predicted

    def _generate_mols(self, generated, scale=1.0, sanitise=True, add_hs=False):
        coords = generated["coords"] * scale
        atom_dists = generated["atomics"]
        bond_dists = generated["bonds"]
        charge_dists = generated["charges"]
        hybridization_dists = generated.get("hybridization", None)
        masks = generated["mask"]

        mols = self.builder.mols_from_tensors(
            coords,
            atom_dists,
            masks,
            bond_dists=bond_dists,
            charge_dists=charge_dists,
            hybridization_dists=hybridization_dists,
            sanitise=sanitise,
            add_hs=add_hs,
        )

        # affinity: TensorDict | None = generated.get("affinity", None)
        # docking: TensorDict | None = generated.get("docking", None)
        # for data in [affinity, docking]:
        #     if data is not None:
        #         mols = self.builder.add_properties_from_tensor_dict(mols, data)
        return mols

    def _generate_ligs(self, generated, lig_mask, scale=1.0, sanitise=False):
        """
        Generate ligand mols from output tensors
        """
        generated = {
            k: v.cpu().detach() if torch.is_tensor(v) else v
            for k, v in generated.items()
        }
        coords = generated["coords"] * scale
        coords = self.builder.undo_zero_com_batch(
            coords, generated["mask"], [system.com for system in generated["complex"]]
        )
        atom_dists = generated["atomics"]
        bond_dists = generated["bonds"]
        charge_dists = generated["charges"]
        hybridization_dists = generated.get("hybridization", None)

        mols = self.builder.ligs_from_complex(
            coords,
            mask=lig_mask,
            atom_dists=atom_dists,
            bond_dists=bond_dists,
            charge_dists=charge_dists,
            hybridization_dists=hybridization_dists,
            sanitise=sanitise,
        )
        return mols

    def retrieve_ligs_with_hs(self, data, save_idx=None):
        """
        Retrieve native ligand mols with hydrogens from the PocketComplex data
        NOTE: depending on the data, the ligand may come without hydrogens (e.g., CrossDocked2020)
        """
        systems = data["complex"] if save_idx is None else [data["complex"][save_idx]]
        ligs = [system.ligand.orig_mol.to_rdkit() for system in systems]
        if save_idx is not None:
            return ligs[0]
        return ligs

    def _retrieve_pdbs(self, data, coords=None, iter="", stage="ref_val"):
        """
        Generate PDB files from pocket data.
        If (predicted) coords are provided, the PDB files will be generated using these coordinates,
        otherwise the (CoM aligned and potentially normalized) coordinates from the data will be used.
        """
        systems = data["complex"]
        pdb_path = Path(self.hparams.data_path) / f"{stage}_pdbs"
        pdb_path.mkdir(parents=True, exist_ok=True)
        pdb_files = [
            str(pdb_path / (system.metadata["system_id"] + f"{iter}.pdb"))
            for system in systems
        ]
        if coords is None:
            mask = data["mask"]
            pocket_mask = data["pocket_mask"]
            # When no (predicted) coords are provided, use the coords from the data and undo zero COM alignment
            # NOTE: Predicted coordinates are always scaled back to the original coordinates and are not zero COM aligned anymore
            coords = data["coords"] * self.coord_scale
            coords = self.builder.undo_zero_com_batch(
                coords, mask, [system.com for system in systems]
            )
        else:
            pocket_mask = data["mask"]
        _ = [
            system.holo.set_coords(coord[mask.bool()].cpu().numpy()).write_pdb(
                pdb_file, include_bonds=True
            )
            for system, coord, mask, pdb_file in zip(
                systems, coords, pocket_mask, pdb_files
            )
        ]
        return pdb_files

    def retrieve_pdbs(self, data, save_dir, save_idx=None, coords=None, iter=""):
        """
        Retrieve PDB files from the PocketComplex data.
        """
        systems = data[
            "complex"
        ]  # if save_idx is None else [data["complex"][save_idx]]
        pdb_path = Path(save_dir)
        pdb_path.mkdir(parents=True, exist_ok=True)
        pdb_files = [
            str(pdb_path / (system.metadata["system_id"] + ".pdb"))
            for system in systems
        ]
        if coords is None:
            pdb_files = [
                str(pdb_path / (system.metadata["system_id"] + ".pdb"))
                for system in systems
            ]
            mask = data["mask"]
            pocket_mask = data["pocket_mask"]
            # When no (predicted) coords are provided, use the coords from the data and undo zero COM alignment
            # NOTE: Predicted coordinates are always scaled back to the original coordinates and are not zero COM aligned anymore
            coords = data["coords"] * self.coord_scale
            coords = self.builder.undo_zero_com_batch(
                coords, mask, [system.com for system in systems]
            )
        else:
            pdb_files = [
                str(pdb_path / (system.metadata["system_id"] + f"_{iter}{i}.pdb"))
                for i, system in enumerate(systems)
            ]
            pocket_mask = data["mask"]
        _ = [
            system.holo.set_coords(coord[mask.bool()].cpu().numpy()).write_pdb(
                pdb_file, include_bonds=True
            )
            for system, coord, mask, pdb_file in zip(
                systems, coords, pocket_mask, pdb_files
            )
        ]
        if save_idx is not None:
            return pdb_files[0]
        return pdb_files

    def _retrieve_pdbs_with_hs(self, data, stage="ref_val", pocket_type="holo"):
        """
        Retrieve PDB files with hydrogens from the PocketComplex data considering CoM alignment
        """
        systems = data["complex"]
        pdb_path = Path(self.hparams.save_dir) / f"{stage}_pdbs"
        pdb_path.mkdir(parents=True, exist_ok=True)

        pdb_files = []

        for system in systems:
            pdb_file = str(pdb_path / (system.metadata["system_id"] + "_with_hs.pdb"))
            pdb_files.append(pdb_file)

            # Check if file already exists to avoid unnecessary computation
            if Path(pdb_file).exists():
                continue

            # Only write if file doesn't exist
            if pocket_type == "apo":
                system.apo.orig_pocket.write_pdb(pdb_file, include_bonds=True)
            elif pocket_type == "holo":
                system.holo.orig_pocket.write_pdb(pdb_file, include_bonds=True)
            else:
                raise ValueError(f"Unknown pocket type '{pocket_type}'")

        return pdb_files

    def retrieve_pdbs_with_hs(self, data, save_dir, save_idx=None, pocket_type="holo"):
        """
        Generate PDB files with hydrogens from the PocketComplex data considering CoM alignment
        """
        systems = data["complex"] if save_idx is None else [data["complex"][save_idx]]
        pdb_path = Path(save_dir)
        pdb_path.mkdir(parents=True, exist_ok=True)
        pdb_files = [
            str(pdb_path / (system.metadata["system_id"] + "_with_hs.pdb"))
            for system in systems
        ]
        if pocket_type == "apo":
            for system, pdb_file in zip(systems, pdb_files):
                system.apo.orig_pocket.write_pdb(pdb_file, include_bonds=True)
        elif pocket_type == "holo":
            for system, pdb_file in zip(systems, pdb_files):
                system.holo.orig_pocket.write_pdb(pdb_file, include_bonds=True)
        else:
            raise ValueError(f"Unknown pocket type '{pocket_type}'")

        if save_idx is not None:
            return pdb_files[0]
        return pdb_files

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

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers for the model."""

        # Get all model parameters
        params = list(self.gen.parameters())

        # Add confidence module parameters if training confidence
        if self.train_confidence and self.confidence_module is not None:
            params.extend(list(self.confidence_module.parameters()))

        # Initialize optimizer and learning rate scheduler
        opt = torch.optim.AdamW(
            params,
            lr=self.lr,
            amsgrad=True,
            foreach=True,
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay,
        )

        if self.lr_schedule == "constant":
            warm_up_steps = 0 if self.warm_up_steps is None else self.warm_up_steps
            scheduler = LinearLR(opt, start_factor=0.05, total_iters=warm_up_steps)

        # TODO could use warm_up_steps to shift peak of one cycle
        elif self.lr_schedule == "one-cycle":
            scheduler = OneCycleLR(
                opt, max_lr=self.lr, total_steps=self.total_steps, pct_start=0.3
            )
        elif self.lr_schedule == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=self.lr_gamma)

        elif self.lr_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.total_steps, eta_min=self.lr * 0.05
            )
            # Combine with warmup
            if self.warm_up_steps and self.warm_up_steps > 0:
                warmup_scheduler = LinearLR(
                    opt, start_factor=0.05, total_iters=self.warm_up_steps
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    opt,
                    schedulers=[warmup_scheduler, scheduler],
                    milestones=[self.warm_up_steps],
                )
        else:
            raise ValueError(f"LR schedule {self.lr_schedule} is not supported.")

        if self.lr_schedule == "constant" or self.lr_schedule == "cosine":
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

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

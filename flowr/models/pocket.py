import copy
from typing import Optional

import torch
from tensordict import TensorDict

import flowr.util.functional as smolF
from flowr.models.pocket_util import (
    MLP,
    EquisSphc,
    GatedEquivariantBlock,
    RadialBasisEmbedding,
    RadialBasisEmbeddingPL,
    SemlaLayer,
    TimeFourierEncoding,
    _CoordNorm,
    _InvariantEmbedding,
    _PairwiseMessages,
)
from flowr.models.semla import BondRefine

# *****************************************************************************
# ****************************** Pocket Encoder *******************************
# *****************************************************************************


class PocketEncoder(torch.nn.Module):
    def __init__(
        self,
        d_equi,
        d_inv,
        d_message,
        n_layers,
        n_attn_heads,
        d_message_ff,
        d_edge,
        n_atom_names,
        n_bond_types,
        n_res_types,
        n_charge_types=7,
        emb_size=64,
        fixed_equi=False,
        use_rbf=False,
        use_distances=False,
        use_crossproducts=False,
        eps=1e-6,
    ):
        super().__init__()

        if fixed_equi and d_equi != 1:
            raise ValueError(f"If fixed_equi is True d_equi must be 1, got {d_equi}")

        self.d_equi = d_equi
        self.d_inv = d_inv
        self.d_message = d_message
        self.n_layers = n_layers
        self.n_attn_heads = n_attn_heads
        self.d_message_ff = d_message_ff
        self.d_edge = d_edge
        self.emb_size = emb_size
        self.fixed_equi = fixed_equi
        self.use_rbf = use_rbf
        self.use_distances = use_distances
        self.use_crossproducts = use_crossproducts
        self.eps = eps

        # ***Embedding and encoding modules *** #
        self.inv_emb = _InvariantEmbedding(
            d_inv,
            n_atom_names,
            n_bond_types,
            emb_size,
            n_charge_types=n_charge_types,
            n_res_types=n_res_types,
        )
        # *** Distances and edges *** #
        if self.use_rbf:
            self.rbf_embed = RadialBasisEmbedding(
                d_edge=emb_size, num_rbf=32, cutoff=5.0, rbf_type="center"
            )
            emb_size *= 2
        self.bond_emb = _PairwiseMessages(
            d_equi, d_inv, d_inv, d_message, d_edge, d_message_ff, emb_size
        )

        if not fixed_equi:
            self.coord_emb = torch.nn.Linear(1, d_equi, bias=False)

        # Create a stack of encoder layers
        layer = SemlaLayer(
            d_equi,
            d_inv,
            d_message,
            n_attn_heads,
            d_message_ff,
            d_self_edge_in=d_edge,
            fixed_equi=fixed_equi,
            use_distances=self.use_distances,
            use_crossproducts=self.use_crossproducts,
            zero_com=False,
            eps=eps,
        )

        layers = self._get_clones(layer, n_layers)
        self.layers = torch.nn.ModuleList(layers)

    @property
    def hparams(self):
        return {
            "d_equi": self.d_equi,
            "d_inv": self.d_inv,
            "d_message": self.d_message,
            "n_layers": self.n_layers,
            "n_attn_heads": self.n_attn_heads,
            "d_message_ff": self.d_message_ff,
            "d_edge": self.d_edge,
            "emb_size": self.emb_size,
            "fixed_equi": self.fixed_equi,
            "use_rbf": self.use_rbf,
            "use_distances": self.use_distances,
            "use_crossproducts": self.use_crossproducts,
            "eps": self.eps,
        }

    def forward(
        self, coords, atom_names, atom_charges, res_types, bond_types, atom_mask=None
    ):
        """Encode the protein pocket into a learnable representation

        Args:
            coords (torch.Tensor): Coordinate tensor, shape [B, N, 3]
            atom_names (torch.Tensor): Atom name indices, shape [B, N]
            atom_charges (torch.Tensor): Atom charge indices, shape [B, N]
            residue_types (torch.Tensor): Residue type indices for each atom, shape [B, N]
            bond_types (torch.Tensor): Bond type indicies for each pair, shape [B, N, N]
            atom_mask (torch.Tensor): Mask for atoms, shape [B, N], 1 for real atom, 0 otherwise

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Equivariant and invariant features, [B, N, 3, d_equi] and [B, N, d_inv]
        """

        atom_mask = torch.ones_like(coords[..., 0]) if atom_mask is None else atom_mask
        adj_matrix = smolF.adj_from_node_mask(atom_mask, self_connect=True)

        rbf_embeds = self.rbf_embed(coords, mask=atom_mask) if self.use_rbf else None

        coords = coords.unsqueeze(-1)
        equis = coords if self.fixed_equi else self.coord_emb(coords)

        invs, edges = self.inv_emb(
            atom_names,
            bond_types,
            atom_mask,
            atom_charges=atom_charges,
            res_types=res_types,
        )
        if self.use_rbf:
            edges = torch.cat((edges, rbf_embeds), dim=-1)
        edges = self.bond_emb(equis, invs, equis, invs, edge_feats=edges)
        edges = edges * adj_matrix.unsqueeze(-1)

        for layer in self.layers:
            equis, invs, _, _, _ = layer(
                equis,
                invs,
                edges,
                adj_matrix,
                atom_mask,
            )

        return equis, invs

    def _get_clones(self, module, n):
        return [copy.deepcopy(module) for _ in range(n)]

    def freeze_bottom_layers(self, n_layers_to_train: int):
        """
        Freeze all but the top n_layers_to_train layers of the PocketEncoder.
        """
        n_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            for name, param in layer.named_parameters():
                if i < n_layers - n_layers_to_train:
                    param.requires_grad = False
                else:
                    param.requires_grad = True


# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# ****************************** Ligand Decoder *******************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************


class LigandDecoder(torch.nn.Module):
    """Class for generating ligands

    By default no pocket conditioning is used, to allow pocket conditioning set d_pocket_inv to the size of the pocket
    invariant feature vectors. d_equi must be the same for both pocket and ligand.
    """

    def __init__(
        self,
        # Core architecture parameters
        d_equi: int,
        d_inv: int,
        d_message: int,
        n_layers: int,
        n_attn_heads: int,
        d_message_ff: int,
        d_edge: int,
        emb_size: int = 64,
        # Data dimensions
        n_atom_types: int = 15,
        n_bond_types: int = 5,
        n_charge_types: int = 8,
        n_extra_atom_feats: Optional[int] = None,
        # Feature flags
        predict_interactions: bool = False,
        flow_interactions: bool = False,
        predict_affinity: bool = False,
        predict_docking_score: bool = False,
        # Model features
        use_rbf: bool = False,
        use_distances: bool = False,
        use_crossproducts: bool = False,
        use_lig_pocket_rbf: bool = False,
        use_fourier_time_embed: bool = False,
        use_sphcs: bool = False,
        use_inpaint_mode_embed: bool = False,
        # Training features
        self_cond: bool = False,
        intermediate_coord_updates: bool = True,
        coord_update_every_n: Optional[int] = None,
        coord_skip_connect: bool = True,
        graph_inpainting: bool = False,
        # Pocket conditioning
        d_pocket_inv: Optional[int] = None,
        n_interaction_types: Optional[int] = None,
        # Numerical stability
        eps: float = 1e-6,
    ):
        super().__init__()

        # Main model args
        self.d_equi = d_equi
        self.d_inv = d_inv
        self.d_message = d_message
        self.n_layers = n_layers
        self.n_attn_heads = n_attn_heads
        self.d_message_ff = d_message_ff
        self.d_edge = d_edge
        self.emb_size = emb_size
        self.d_pocket_inv = d_pocket_inv
        self.self_cond = self_cond
        self.eps = eps
        self.flow_interactions = flow_interactions
        self.predict_interactions = predict_interactions
        self.use_sphcs = use_sphcs
        self.use_lig_pocket_rbf = use_lig_pocket_rbf
        self.use_fourier_time_embed = use_fourier_time_embed
        self.use_rbf = use_rbf
        self.use_distances = use_distances
        self.use_crossproducts = use_crossproducts
        self.use_inpaint_mode_embed = use_inpaint_mode_embed
        self.intermediate_coord_updates = intermediate_coord_updates
        self.coord_update_every_n = coord_update_every_n
        self.interactions = n_interaction_types is not None
        self.coord_skip_connect = coord_skip_connect
        self.graph_inpainting = graph_inpainting
        self.predict_affinity = predict_affinity
        self.predict_docking_score = predict_docking_score

        # Add inpainting mode embedding
        if self.use_inpaint_mode_embed:
            # Embedding for hierarchical inpaint mode encoding
            # First index: 0-3 for main category, Second index: 0-8 for subcategory
            self.inpaint_mode_embed_main = torch.nn.Embedding(4, emb_size // 2)
            self.inpaint_mode_embed_sub = torch.nn.Embedding(9, emb_size // 2)
            self.inpaint_mode_proj = torch.nn.Linear(emb_size, d_inv)
        else:
            self.inpaint_mode_embed_main = None
            self.inpaint_mode_embed_sub = None
            self.inpaint_mode_proj = None

        # Size args
        self.n_atom_types = n_atom_types
        self.n_bond_types = n_bond_types
        self.n_charge_types = n_charge_types
        self.n_extra_atom_feats = n_extra_atom_feats

        if d_pocket_inv is None and n_interaction_types is not None:
            raise ValueError(
                "Pocket conditioning is required for interaction encoding and prediction."
            )

        coord_proj_feats = 2 if self_cond else 1
        if self.use_sphcs:
            self.equis_sphcs = EquisSphc(
                l_min=0,
                l_max=3,
                r_cut=5,
                eps=1e-8,
                return_sphc_distance_matrix=True,
                p=6,
                kappa=1.0,
            )
            self.embed_sphcs = torch.nn.Linear(
                self.equis_sphcs.out_dim, d_equi, bias=False
            )
            # coord_proj_feats += self.equis_sphcs.out_dim
        d_cond_edge_in = (
            d_edge + 1
            if n_interaction_types is not None
            else d_edge if self.use_lig_pocket_rbf else None
        )
        d_cond_edge_out = (
            d_edge
            if n_interaction_types is not None
            or self.predict_affinity
            or self.predict_docking_score
            else None
        )

        # *** Invariant embedding and encoding modules ***
        self.inv_emb = _InvariantEmbedding(
            d_inv,
            n_atom_types,
            n_bond_types,
            emb_size,
            n_charge_types=n_charge_types,
            n_time_feats=emb_size,
            n_extra_feats=n_extra_atom_feats,
            self_cond=self_cond,
            max_size=512,
            use_fourier_time_embed=use_fourier_time_embed,
        )

        # *** Bond / edges embedding *** #
        if self.use_rbf:
            self.rbf_embed = RadialBasisEmbedding(
                d_edge=emb_size,
                num_rbf=32,
                cutoff=5.0,
                learnable_cutoff=False,
                eps=1e-6,
                rbf_type="center",
            )
            emb_size *= 2
        self.bond_emb = _PairwiseMessages(
            d_equi, d_inv, d_inv, d_message, d_edge, d_message_ff, emb_size
        )

        # *** Time embedding *** #
        self.time_fourier = (
            TimeFourierEncoding(posenc_dim=d_equi, max_len=200, random_permute=False)
            if use_fourier_time_embed
            else None
        )
        self.time_emb_cont = (
            MLP(d_equi, d_equi) if use_fourier_time_embed else MLP(1, d_equi)
        )

        # *** Coordinate embedding *** #
        self.coord_emb = torch.nn.Linear(coord_proj_feats, d_equi, bias=False)

        # *** Layer stack ***
        # Create n_layers - 1 encoder layers with conditional coord updates
        layers = []
        for i in range(n_layers - 1):
            # Determine if this layer should have intermediate coord updates
            if intermediate_coord_updates:
                if coord_update_every_n is not None:
                    # Enable coord updates every Nth layer (symmetric distribution)
                    layer_updates = i % coord_update_every_n == coord_update_every_n - 1
                else:
                    # Enable coord updates for all layers
                    layer_updates = True
            else:
                # Disable all intermediate coord updates
                layer_updates = False

            enc_layer = SemlaLayer(
                d_equi,
                d_inv,
                d_message,
                n_attn_heads,
                d_message_ff,
                d_inv_cond=d_pocket_inv,
                d_self_edge_in=d_edge,
                d_cond_edge_in=d_cond_edge_in,
                use_distances=self.use_distances,
                use_crossproducts=self.use_crossproducts,
                intermediate_coord_updates=layer_updates,
                zero_com=False,
                eps=eps,
            )
            layers.append(enc_layer)

        # Create one final layer which also produces edge feature outputs
        # The final decoder layer does not have intermediate coord updates
        dec_layer = SemlaLayer(
            d_equi,
            d_inv,
            d_message,
            n_attn_heads,
            d_message_ff,
            d_inv_cond=d_pocket_inv,
            d_self_edge_in=d_edge,
            d_self_edge_out=d_edge,
            d_cond_edge_in=d_cond_edge_in,
            d_cond_edge_out=d_cond_edge_out,
            use_distances=self.use_distances,
            use_crossproducts=self.use_crossproducts,
            intermediate_coord_updates=False,
            zero_com=False,
            eps=eps,
        )
        layers.append(dec_layer)

        self.layers = torch.nn.ModuleList(layers)

        # *** Final norms and projections ***

        self.final_coord_norm = _CoordNorm(d_equi, zero_com=False, eps=eps)
        self.final_inv_norm = torch.nn.LayerNorm(d_inv)
        self.final_bond_norm = torch.nn.LayerNorm(d_edge)

        self.coord_out_proj = torch.nn.Linear(d_equi, 1, bias=False)
        self.atom_type_proj = torch.nn.Linear(d_inv, n_atom_types)
        self.atom_charge_proj = torch.nn.Linear(d_inv, n_charge_types)
        if n_extra_atom_feats is not None:
            self.extra_feats_proj = torch.nn.Linear(d_inv, n_extra_atom_feats)

        self.bond_refine = BondRefine(
            d_inv, d_message, d_edge, d_ff=d_inv, norm_feats=False
        )
        self.bond_proj = torch.nn.Linear(d_edge, n_bond_types)

        # *** Modules for interactions ***
        self.interaction_emb = (
            torch.nn.Embedding(n_interaction_types, d_edge)
            if n_interaction_types is not None and flow_interactions
            else None
        )
        self.radial_basis_embed = (
            RadialBasisEmbeddingPL(
                d_edge=d_cond_edge_in,
                num_rbf=32,
                cutoff=5.0,
                learnable_cutoff=False,
                eps=1e-6,
                rbf_type="center",
            )
            if self.use_lig_pocket_rbf
            else None
        )
        self.interaction_refine = (
            _PairwiseMessages(
                d_equi,
                d_inv,
                d_pocket_inv,
                d_message,
                d_edge,
                d_message_ff,
                d_edge=d_edge,
                include_distances=self.use_distances,
            )
            if n_interaction_types is not None
            or self.predict_affinity
            or self.predict_docking_score
            else None
        )
        self.interaction_proj = (
            torch.nn.Linear(d_edge, n_interaction_types)
            if n_interaction_types is not None
            else None
        )

        if self.predict_affinity or self.predict_docking_score:
            assert (
                self.use_lig_pocket_rbf
            ), "Ligand-pocket RBF must be used for affinity/docking score prediction."

            # Gated equivariant block for processing ligand and pocket features
            self.lig_gate = GatedEquivariantBlock(
                d_inv=d_inv,
                d_equi=d_equi,
                d_out=d_inv,
                return_vector=False,  # Only return scalar features
            )
            self.pocket_gate = GatedEquivariantBlock(
                d_inv=d_pocket_inv,
                d_equi=d_equi,
                d_out=d_inv,
                return_vector=False,  # Only return scalar features
            )
            # Simple pooling layers
            self.lig_pool = torch.nn.Sequential(
                torch.nn.Linear(2 * d_inv, d_inv // 2),
                torch.nn.SiLU(),
                torch.nn.Linear(d_inv // 2, d_inv // 4),
            )
            # Pre-norm layer for pocket features
            self.pocket_prenorm = torch.nn.LayerNorm(d_inv)
            self.pocket_pool = torch.nn.Sequential(
                torch.nn.Linear(d_inv, d_inv // 2),
                torch.nn.SiLU(),
                torch.nn.Linear(d_inv // 2, d_inv // 4),
            )
            # Interaction pooling
            self.interaction_pool = torch.nn.Sequential(
                torch.nn.Linear(d_edge, d_inv // 2),
                torch.nn.SiLU(),
                torch.nn.Linear(d_inv // 2, d_inv // 4),
            )
            # Combined feature dimension: ligand + pocket + interaction
            combined_dim = 3 * (d_inv // 4)

        if predict_affinity:
            # Individual affinity prediction heads
            self.pic50_head = torch.nn.Sequential(
                torch.nn.Linear(combined_dim, d_inv // 2),
                torch.nn.SiLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(d_inv // 2, 1),
                torch.nn.ReLU(),
            )
            self.pkd_head = torch.nn.Sequential(
                torch.nn.Linear(combined_dim, d_inv // 2),
                torch.nn.SiLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(d_inv // 2, 1),
                torch.nn.ReLU(),
            )
            self.pki_head = torch.nn.Sequential(
                torch.nn.Linear(combined_dim, d_inv // 2),
                torch.nn.SiLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(d_inv // 2, 1),
                torch.nn.ReLU(),
            )
            self.pec50_head = torch.nn.Sequential(
                torch.nn.Linear(combined_dim, d_inv // 2),
                torch.nn.SiLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(d_inv // 2, 1),
                torch.nn.ReLU(),
            )

        if predict_docking_score:
            # Docking score prediction heads
            self.vina_head = torch.nn.Sequential(
                torch.nn.Linear(combined_dim, d_inv // 2),
                torch.nn.SiLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(d_inv // 2, 1),
            )
            self.gnina_head = torch.nn.Sequential(
                torch.nn.Linear(combined_dim, d_inv // 2),
                torch.nn.SiLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(d_inv // 2, 1),
            )

    @property
    def hparams(self):
        return {
            "d_equi": self.d_equi,
            "d_inv": self.d_inv,
            "d_message": self.d_message,
            "n_layers": self.n_layers,
            "n_attn_heads": self.n_attn_heads,
            "d_message_ff": self.d_message_ff,
            "d_edge": self.d_edge,
            "emb_size": self.emb_size,
            "self_cond": self.self_cond,
            "eps": self.eps,
            "interactions": self.interactions,
            "flow_interactions": self.flow_interactions,
            "predict_interactions": self.predict_interactions,
            "coord_skip_connect": self.coord_skip_connect,
            "coord_update_every_n": self.coord_update_every_n,
            "use_rbf": self.use_rbf,
            "use_lig_pocket_rbf": self.use_lig_pocket_rbf,
            "use_distances": self.use_distances,
            "use_crossproducts": self.use_crossproducts,
            "use_fourier_time_embed": self.use_fourier_time_embed,
            "use_sphcs": self.use_sphcs,
            "predict_affinity": self.predict_affinity,
            "predict_docking_score": self.predict_docking_score,
        }

    def forward(
        self,
        coords,
        atom_types,
        bond_types,
        atom_charges,
        atom_mask,
        times=None,
        extra_feats=None,
        cond_coords=None,
        cond_atomics=None,
        cond_bonds=None,
        pocket_coords=None,
        pocket_equis=None,
        pocket_invs=None,
        pocket_atom_mask=None,
        interactions=None,
        inpaint_mask=None,
        inpaint_mode=None,
    ):
        """Generate ligand atom types, coords, charges and bonds

        Args:
            coords (torch.Tensor): Coordinate tensor, shape [B, N, 3]
            atom_types (torch.Tensor): Atom name indices, shape [B, N]
            bond_types (torch.Tensor): Bond type indicies for each pair, shape [B, N, N]
            atom_charges (torch.Tensor): Atom charge indices, shape [B, N]
            atom_mask (torch.Tensor): Mask for atoms, shape [B, N], 1 for real atom, 0 otherwise
            times (list[torch.Tensor]): Time steps for each atom, list of time tensors of shape [B, N, n_times]
            extra_feats (torch.Tensor): Additional atom features, shape [B, N, n_extra_atom_feats]
            cond_coords (torch.Tensor): Self conditioning coords, shape [B, N, 3]
            cond_atomics (torch.Tensor): Self conditioning atom types, shape [B, N, n_atom_types]
            cond_bonds (torch.Tensor): Self conditioning bond types, shape [B, N, N, n_bond_types]
            pocket_coords (torch.Tensor): Original pocket coords, shape [B, N_p, 3]
            pocket_equis (torch.Tensor): Equivariant encoded pocket features, shape [B, N_p, d_equi]
            pocket_invs (torch.Tensor): Invariant encoded pocket features, shape [B, N_p, d_pocket_inv]
            pocket_atom_mask (torch.Tensor): Mask for pocket atom, shape [B, N_p], 1 for real, 0 otherwise
            interactions (torch.Tensor): Interaction types between pocket and ligand, shape [B, N, N_p]
            inpaint_mask (torch.Tensor): Fragment mask for atoms, shape [B, N], 1 for fixed atom, 0 for variable (learnable) atoms

        Returns:
            (predicted coordinates, atom type logits, bond logits, atom charge logits)
            All torch.Tensor, shapes:
                Coordinates: [B, N, 3],
                Type logits: [B, N, n_atom_types],
                Bond logits: [B, N, N, n_bond_types],
                Charge logits: [B, N, n_charge_types]
                Interaction logits: [B, N, N_p, n_interaction_types] if interactions are provided
        """

        if (cond_atomics is not None or cond_bonds is not None) and not self.self_cond:
            raise ValueError(
                "Conditional inputs were provided but the model was initialised with self_cond as False."
            )

        if (cond_atomics is None or cond_bonds is None) and self.self_cond:
            raise ValueError(
                "Conditional inputs must be provided if using self conditioning."
            )

        if (
            pocket_invs is not None or pocket_equis is not None
        ) and self.d_pocket_inv is None:
            raise ValueError(
                "Pocket cond inputs were provided but the model was not initialised for pocket cond."
            )

        if (
            pocket_invs is None or pocket_equis is None
        ) and self.d_pocket_inv is not None:
            raise ValueError(
                "Pocket cond inputs must be provided if using pocket conditioning."
            )

        if not self.interactions and interactions is not None:
            raise ValueError(
                "Interactions were provided but the model was not initialised for interactions."
            )

        atom_mask = torch.ones_like(coords[..., 0]) if atom_mask is None else atom_mask
        adj_matrix = smolF.adj_from_node_mask(atom_mask, self_connect=True)

        # Work out adj matrix between pocket and ligand, if required
        cond_adj_matrix = None
        if self.d_pocket_inv is not None:
            cond_adj_matrix = atom_mask.float().unsqueeze(
                2
            ) * pocket_atom_mask.float().unsqueeze(1)
            cond_adj_matrix = cond_adj_matrix.long()

        # Embed interaction types, if required
        interaction_feats = None
        if self.use_lig_pocket_rbf:
            interaction_feats = self.radial_basis_embed(
                coords,
                pocket_coords,
                ligand_mask=atom_mask,
                pocket_mask=pocket_atom_mask,
            )

        # Embed distances
        rbf_embeds = self.rbf_embed(coords, mask=atom_mask) if self.use_rbf else None

        # Project coords to d_equi
        if self.self_cond:
            _coords = torch.cat(
                (coords.unsqueeze(-1), cond_coords.unsqueeze(-1)), dim=-1
            )
        else:
            _coords = coords.unsqueeze(-1)

        # Embed coordinates
        ## Embed continuous time and add to equivariant features
        if self.use_fourier_time_embed:
            times_cont = self.time_fourier(times[0])  # [B, N, d_equi]
            times_cont = self.time_emb_cont(times_cont)
        else:
            times_cont = self.time_emb_cont(times[0])  # [B, N, d_equi]
        equis = self.coord_emb(_coords) + times_cont.unsqueeze(2).expand(
            -1, -1, 3, -1
        )  # [batch_size, N, 3, d_equi]

        if self.use_sphcs:
            B, N = coords.shape[:2]
            equis_sphc, sphc_dst = self.equis_sphcs(coords, mask=atom_mask)
            equis_sphc = self.embed_sphcs(equis_sphc)
            equis_sphc = equis_sphc.unsqueeze(2).expand(B, N, 3, -1)
            # _coords = torch.cat((_coords, equis_sphc), dim=-1)
            equis = equis + equis_sphc

        # Embed invariant features
        invs, edges = self.inv_emb(
            atom_types,
            bond_types,
            atom_mask=atom_mask,
            atom_charges=atom_charges,
            cond_types=cond_atomics,
            cond_bonds=cond_bonds,
            times=times[1],
            extra_feats=extra_feats,
        )

        # Add inpainting mode embedding if provided
        if self.use_inpaint_mode_embed:
            # inpaint_mode shape: [batch_size, 2]
            # Embed main category (first index) and subcategory (second index)
            main_embed = self.inpaint_mode_embed_main(
                inpaint_mode[:, 0]
            )  # [B, emb_size//2]
            sub_embed = self.inpaint_mode_embed_sub(
                inpaint_mode[:, 1]
            )  # [B, emb_size//2]

            # Combine embeddings
            inpaint_embed = torch.cat([main_embed, sub_embed], dim=-1)  # [B, emb_size]
            inpaint_embed = self.inpaint_mode_proj(inpaint_embed)  # [B, d_inv]

            # Broadcast to all atoms and add to invariant features
            inpaint_embed = inpaint_embed.unsqueeze(1).expand(
                -1, invs.size(1), -1
            )  # [B, N, d_inv]
            invs = invs + inpaint_embed * atom_mask.unsqueeze(-1)

        if self.use_rbf:
            edges = torch.cat((edges, rbf_embeds), dim=-1)
        edges = self.bond_emb(equis, invs, equis, invs, edge_feats=edges)
        edges = edges * adj_matrix.unsqueeze(-1)

        # Iterate over Semla layers
        for layer in self.layers:
            equis, invs, coords, _edges, _interaction_feats = layer(
                equis,
                invs,
                edges,
                adj_matrix,
                atom_mask,
                coords=coords,
                cond_coords=pocket_coords,
                cond_equis=pocket_equis,
                cond_invs=pocket_invs,
                cond_edges=interaction_feats,
                cond_node_mask=pocket_atom_mask,
                cond_adj_matrix=cond_adj_matrix,
            )
            edges = _edges if _edges is not None else edges
            interaction_feats = (
                (_interaction_feats)
                if _interaction_feats is not None
                else interaction_feats
            )

        if self.interactions or self.predict_affinity or self.predict_docking_score:
            # Pass interactions through refinement layer and project to logits, if required
            interaction_feats = self.interaction_refine(
                equis,
                invs,
                pocket_equis,
                pocket_invs,
                interaction_feats,
            )
            if self.interactions:
                interaction_logits = self.interaction_proj(interaction_feats)

        # Project coords back to one equivariant feature
        equis_norm = self.final_coord_norm(equis, atom_mask)
        out_coords = self.coord_out_proj(equis_norm).squeeze(-1)
        if self.coord_skip_connect:
            out_coords = coords + out_coords * atom_mask.unsqueeze(-1)

        # Project invariant features to atom and charge logits
        invs_norm = self.final_inv_norm(invs)
        atom_type_logits = self.atom_type_proj(invs_norm) * atom_mask.unsqueeze(-1)
        charge_logits = self.atom_charge_proj(invs_norm) * atom_mask.unsqueeze(-1)
        extra_feats_logits = (
            self.extra_feats_proj(invs_norm) * atom_mask.unsqueeze(-1)
            if extra_feats is not None
            else None
        )

        # Pass bonds through refinement layer and project to logits
        edge_norm = self.final_bond_norm(edges)
        edge_out = self.bond_refine(out_coords, invs_norm, atom_mask, edge_norm)
        bond_logits = self.bond_proj(
            0.5 * (edge_out + edge_out.transpose(1, 2))
        ) * adj_matrix.unsqueeze(-1)
        # bond_logits = self.bond_proj(
        #     edge_out + edge_out.transpose(1, 2)
        # ) * adj_matrix.unsqueeze(-1)

        affinity_pred, docking_pred = None, None
        if self.predict_affinity or self.predict_docking_score:
            assert (
                interaction_feats is not None
            ), "Interaction features must be provide for affinity prediction."
            # Process ligand features through gated equivariant block
            lig_feats = self.lig_gate(invs_norm, equis_norm, node_mask=atom_mask).sum(
                dim=1
            )  # [B, d_inv]
            lig_norm = lig_feats / 100.0
            lig_mean = lig_feats / (
                atom_mask.unsqueeze(-1).sum(dim=1) + 1e-6
            )  # [B, d_inv]
            lig_feats = self.lig_pool(
                torch.cat([lig_norm, lig_mean], dim=-1)
            )  # [B, d_inv//4]

            # Pocket pooling: mean over atoms
            pocket_feats = self.pocket_prenorm(
                self.pocket_gate(pocket_invs, pocket_equis, node_mask=pocket_atom_mask)
            ).sum(
                dim=1
            )  # [B, d_inv]
            pocket_feats = pocket_feats / (
                pocket_atom_mask.unsqueeze(-1).sum(dim=1) + 1e-6
            )  # [B, d_inv]
            pocket_feats = self.pocket_pool(pocket_feats)  # [B, d_inv//4]

            # Interaction pooling: mean over ligand-pocket pairs
            interaction_mask = atom_mask.unsqueeze(2) * pocket_atom_mask.unsqueeze(
                1
            )  # [B, N, N_p]
            interaction_feats = (
                interaction_feats * interaction_mask.unsqueeze(-1)
            ).sum(dim=(1, 2)) / (
                interaction_mask.unsqueeze(-1).sum(dim=(1, 2)) + 1e-6
            )  # [B, d_edge]
            interaction_feats = self.interaction_pool(
                interaction_feats
            )  # [B, d_inv//4]

            # Combine all features
            combined_features = torch.cat(
                [lig_feats, pocket_feats, interaction_feats], dim=-1
            )  # [B, 3*d_inv//4]

            # Predict individual affinity values
            if self.predict_affinity:
                affinity_pred = TensorDict(
                    {
                        "pic50": self.pic50_head(combined_features),  # [B, 1]
                        "pkd": self.pkd_head(combined_features),  # [B, 1]
                        "pki": self.pki_head(combined_features),  # [B, 1]
                        "pec50": self.pec50_head(combined_features),  # [B, 1]
                    }
                )

            # Predict docking scores
            if self.predict_docking_score:
                docking_pred = TensorDict(
                    {
                        "vina_score": self.vina_head(combined_features),  # [B, 1]
                        "gnina_score": self.gnina_head(combined_features),  # [B, 1]
                    }
                )

        out = {
            "coords": out_coords,
            "atomics": atom_type_logits,
            "bonds": bond_logits,
            "charges": charge_logits,
            "interactions": interaction_logits if self.interactions else None,
            "mask": atom_mask,
            "affinity": affinity_pred,
            "docking_score": docking_pred,
        }
        if extra_feats_logits is not None:
            out["hybridization"] = extra_feats_logits
        return out

    def _get_clones(self, module, n):
        return [copy.deepcopy(module) for _ in range(n)]


# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# ****************************** Ligand Generator *****************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************


class LigandGenerator(torch.nn.Module):
    """Main entry point class for generating ligands.

    This class allows both unconditional and pocket-conditioned models to be created. The pocket-conditioned model
    can be created by passing in a PocketEncoder object with the pocket_enc argument, this will automatically setup
    the ligand decoder to use condition attention in addition to self attention. If pocket_enc is None the ligand
    decoder is setup as an unconditional generator.
    """

    def __init__(
        self,
        # Core architecture parameters
        d_equi: int,
        d_inv: int,
        d_message: int,
        n_layers: int,
        n_attn_heads: int,
        d_message_ff: int,
        d_edge: int,
        emb_size: int = 64,
        # Data dimensions
        n_atom_types: int = 15,
        n_bond_types: int = 5,
        n_charge_types: int = 8,
        n_extra_atom_feats: Optional[int] = None,
        # Pocket conditioning
        pocket_enc: Optional[PocketEncoder] = None,
        n_interaction_types: Optional[int] = None,
        # Feature flags
        flow_interactions: bool = False,
        predict_interactions: bool = False,
        predict_affinity: bool = False,
        predict_docking_score: bool = False,
        # Model features
        use_rbf: bool = False,
        use_sphcs: bool = False,
        use_distances: bool = False,
        use_crossproducts: bool = False,
        use_lig_pocket_rbf: bool = False,
        use_fourier_time_embed: bool = False,
        # Training features
        self_cond: bool = False,
        coord_skip_connect: bool = True,
        coord_update_every_n: Optional[int] = None,
        graph_inpainting: Optional[str] = None,
        use_inpaint_mode_embed: bool = False,
        # Numerical stability
        eps: float = 1e-6,
    ):
        super().__init__()

        duplicate_pocket_equi = False
        if pocket_enc is not None:
            duplicate_pocket_equi = pocket_enc.d_equi == 1
            if not duplicate_pocket_equi and pocket_enc.d_equi != d_equi:
                raise ValueError(
                    "d_equi must be either the same for the pocket and ligand or 1 for the pocket."
                )

        d_pocket_inv = pocket_enc.d_inv if pocket_enc is not None else None

        self.d_equi = d_equi
        self.duplicate_pocket_equi = duplicate_pocket_equi
        self.graph_inpainting = graph_inpainting

        ligand_dec = LigandDecoder(
            d_equi=d_equi,
            d_inv=d_inv,
            d_message=d_message,
            n_layers=n_layers,
            n_attn_heads=n_attn_heads,
            d_message_ff=d_message_ff,
            d_edge=d_edge,
            emb_size=emb_size,
            n_atom_types=n_atom_types,
            n_bond_types=n_bond_types,
            n_charge_types=n_charge_types,
            n_extra_atom_feats=n_extra_atom_feats,
            d_pocket_inv=d_pocket_inv,
            n_interaction_types=n_interaction_types,
            flow_interactions=flow_interactions,
            predict_interactions=predict_interactions,
            predict_affinity=predict_affinity,
            predict_docking_score=predict_docking_score,
            use_rbf=use_rbf,
            use_sphcs=use_sphcs,
            use_distances=use_distances,
            use_crossproducts=use_crossproducts,
            use_lig_pocket_rbf=use_lig_pocket_rbf,
            use_fourier_time_embed=use_fourier_time_embed,
            self_cond=self_cond,
            coord_skip_connect=coord_skip_connect,
            coord_update_every_n=coord_update_every_n,
            graph_inpainting=graph_inpainting,
            use_inpaint_mode_embed=False,
            eps=eps,
        )

        self.pocket_enc = pocket_enc
        self.ligand_dec = ligand_dec

    @property
    def hparams(self):
        hparams = self.ligand_dec.hparams
        if self.pocket_enc is not None:
            pocket_hparams = {
                f"pocket-{name}": val for name, val in self.pocket_enc.hparams.items()
            }
            hparams = {**hparams, **pocket_hparams}

        return hparams

    def forward(
        self,
        coords,
        atom_types,
        bond_types,
        atom_charges,
        atom_mask,
        times,
        extra_feats=None,
        cond_coords=None,
        cond_atomics=None,
        cond_bonds=None,
        pocket_coords=None,
        pocket_atom_names=None,
        pocket_atom_charges=None,
        pocket_res_types=None,
        pocket_bond_types=None,
        pocket_atom_mask=None,
        pocket_equis=None,
        pocket_invs=None,
        interactions=None,
        inpaint_mask=None,
        inpaint_mode=None,
    ):

        if self.pocket_enc is not None:
            if pocket_equis is None and pocket_invs is None:
                pocket_equis, pocket_invs = self.get_pocket_encoding(
                    pocket_coords,
                    pocket_atom_names,
                    pocket_atom_charges,
                    pocket_res_types,
                    pocket_bond_types,
                    pocket_atom_mask=pocket_atom_mask,
                )

        decoder_out = self.decode(
            coords,
            atom_types,
            bond_types,
            atom_charges=atom_charges,
            atom_mask=atom_mask,
            times=times,
            extra_feats=extra_feats,
            cond_coords=cond_coords,
            cond_atomics=cond_atomics,
            cond_bonds=cond_bonds,
            pocket_coords=pocket_coords,
            pocket_equis=pocket_equis,
            pocket_invs=pocket_invs,
            pocket_atom_mask=pocket_atom_mask,
            interactions=interactions,
            inpaint_mask=inpaint_mask,
            inpaint_mode=inpaint_mode,
        )

        return decoder_out

    def get_pocket_encoding(
        self,
        pocket_coords,
        pocket_atom_names,
        pocket_atom_charges,
        pocket_res_types,
        pocket_bond_types,
        pocket_atom_mask=None,
    ):
        if None in [
            pocket_coords,
            pocket_atom_names,
            pocket_atom_charges,
            pocket_res_types,
            pocket_bond_types,
        ]:
            raise ValueError(
                "All pocket inputs must be provided if the model is created with pocket cond."
            )

        if self.pocket_enc is None:
            raise ValueError(
                "Cannot call encode on a model initialised without a pocket encoder."
            )

        pocket_equis, pocket_invs = self.pocket_enc(
            pocket_coords,
            pocket_atom_names,
            pocket_atom_charges,
            pocket_res_types,
            pocket_bond_types,
            atom_mask=pocket_atom_mask,
        )

        if self.duplicate_pocket_equi:
            pocket_equis = pocket_equis.expand(-1, -1, -1, self.d_equi)

        return pocket_equis, pocket_invs

    def decode(
        self,
        coords,
        atom_types,
        bond_types,
        atom_charges,
        atom_mask,
        times,
        extra_feats=None,
        cond_coords=None,
        cond_atomics=None,
        cond_bonds=None,
        pocket_coords=None,
        pocket_equis=None,
        pocket_invs=None,
        pocket_atom_mask=None,
        interactions=None,
        inpaint_mask=None,
        inpaint_mode=None,
    ):

        if self.pocket_enc is not None and pocket_invs is None:
            raise ValueError(
                "The model was initialised with pocket conditioning but pocket_invs was not provided."
            )

        if self.pocket_enc is not None and pocket_equis is None:
            raise ValueError(
                "The model was initialised with pocket conditioning but pocket_invs was not provided."
            )

        decoder_out = self.ligand_dec(
            coords,
            atom_types,
            bond_types,
            atom_charges=atom_charges,
            atom_mask=atom_mask,
            times=times,
            extra_feats=extra_feats,
            cond_coords=cond_coords,
            cond_atomics=cond_atomics,
            cond_bonds=cond_bonds,
            pocket_coords=pocket_coords,
            pocket_equis=pocket_equis,
            pocket_invs=pocket_invs,
            pocket_atom_mask=pocket_atom_mask,
            interactions=interactions,
            inpaint_mask=inpaint_mask,
            inpaint_mode=inpaint_mode,
        )

        return decoder_out


# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# ****************************** Confidence Module ****************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************


class ConfidenceHead(torch.nn.Module):
    """Class for predicting confidence scores (pLDDT) for ligand atoms.

    Similar to LigandDecoder but outputs per-atom confidence scores instead of molecular properties.
    Supports pocket conditioning via cross-attention. No time inputs needed as this runs on predicted structures.
    """

    def __init__(
        self,
        d_equi,
        d_inv,
        d_message,
        n_layers,
        n_attn_heads,
        d_message_ff,
        d_edge,
        n_atom_types,
        n_bond_types,
        n_charge_types,
        n_extra_atom_feats=None,
        n_output_bins=50,
        emb_size=64,
        d_pocket_inv=None,
        use_rbf=False,
        use_distances=False,
        use_crossproducts=False,
        eps=1e-6,
    ):
        super().__init__()

        self.d_equi = d_equi
        self.d_inv = d_inv
        self.d_message = d_message
        self.n_layers = n_layers
        self.n_attn_heads = n_attn_heads
        self.d_message_ff = d_message_ff
        self.d_edge = d_edge
        self.emb_size = emb_size
        self.d_pocket_inv = d_pocket_inv
        self.eps = eps
        self.use_rbf = use_rbf
        self.n_output_bins = n_output_bins

        coord_proj_feats = 1

        # *** Embedding and encoding modules ***
        self.inv_emb = _InvariantEmbedding(
            d_inv,
            n_atom_types,
            n_bond_types,
            emb_size,
            n_charge_types=n_charge_types,
            n_extra_feats=n_extra_atom_feats,
            self_cond=False,
            max_size=512,
        )
        # *** Bond / edge embeddings for ligand ***
        if self.use_rbf:
            self.rbf_embed = RadialBasisEmbedding(
                d_edge=emb_size,
                num_rbf=32,
                cutoff=5.0,
                learnable_cutoff=False,
                eps=1e-6,
                rbf_type="center",
            )
            emb_size *= 2
        self.bond_emb = _PairwiseMessages(
            d_equi, d_inv, d_inv, d_message, d_edge, d_message_ff, emb_size
        )

        # *** Coordinates embedding ***
        self.coord_emb = torch.nn.Linear(coord_proj_feats, d_equi, bias=False)

        # *** RBF embedding for ligand-pocket interactions ***
        self.radial_basis_embed = RadialBasisEmbeddingPL(
            d_edge=d_edge,
            num_rbf=32,
            cutoff=5.0,
            learnable_cutoff=False,
            eps=1e-6,
            rbf_type="center",
        )

        # *** Layer stack ***
        enc_layer = SemlaLayer(
            d_equi,
            d_inv,
            d_message,
            n_attn_heads,
            d_message_ff,
            d_inv_cond=d_pocket_inv,
            d_self_edge_in=d_edge,
            d_cond_edge_in=d_edge,
            d_cond_edge_out=d_edge,
            zero_com=False,
            use_distances=use_distances,
            use_crossproducts=use_crossproducts,
            eps=eps,
        )
        layers = self._get_clones(enc_layer, n_layers)
        self.layers = torch.nn.ModuleList(layers)

        # *** Final norms ***
        self.final_coord_norm = _CoordNorm(d_equi, zero_com=False, eps=eps)
        self.final_inv_norm = torch.nn.LayerNorm(d_inv)

        # *** Gated equivariant block to combine scalar and directional information *** #
        self.equi_gate = GatedEquivariantBlock(
            d_inv=d_inv,
            d_equi=d_equi,
            d_out=d_inv,
            return_vector=False,  # Only return scalar features
        )
        # Ligand pool
        self.lig_proj = torch.nn.Sequential(
            torch.nn.Linear(d_inv, d_inv // 2),
            torch.nn.SiLU(),
            torch.nn.Linear(d_inv // 2, d_inv // 2),
        )
        # Interaction pooling
        self.interaction_proj = torch.nn.Sequential(
            torch.nn.Linear(d_edge, d_inv // 2),
            torch.nn.SiLU(),
            torch.nn.Linear(d_inv // 2, d_inv // 2),
        )
        # *** Final confidence head with uncertainty estimation *** #
        self.confidence_head = torch.nn.Sequential(
            torch.nn.Linear(d_inv, d_inv),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(d_inv, d_inv // 2),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(d_inv // 2, n_output_bins),
        )

    @property
    def hparams(self):
        return {
            "d_equi": self.d_equi,
            "d_inv": self.d_inv,
            "d_message": self.d_message,
            "n_layers": self.n_layers,
            "n_attn_heads": self.n_attn_heads,
            "d_message_ff": self.d_message_ff,
            "d_edge": self.d_edge,
            "emb_size": self.emb_size,
            "eps": self.eps,
            "use_rbf": self.use_rbf,
            "use_distances": self.use_distances,
            "use_crossproducts": self.use_crossproducts,
        }

    def forward(
        self,
        coords,
        atom_types,
        bond_types,
        atom_charges,
        atom_mask,
        extra_feats=None,
        pocket_coords=None,
        pocket_equis=None,
        pocket_invs=None,
        pocket_atom_mask=None,
    ):
        """Predict confidence scores (pLDDT) for ligand atoms

        Args:
            coords (torch.Tensor): Coordinate tensor, shape [B, N, 3]
            atom_types (torch.Tensor): Atom name indices, shape [B, N]
            bond_types (torch.Tensor): Bond type indices for each pair, shape [B, N, N]
            atom_charges (torch.Tensor): Atom charge indices, shape [B, N]
            atom_mask (torch.Tensor): Mask for atoms, shape [B, N], 1 for real atom, 0 otherwise
            extra_feats (torch.Tensor): Additional atom features, shape [B, N, n_extra_atom_feats]
            pocket_coords (torch.Tensor): Original pocket coords, shape [B, N_p, 3]
            pocket_equis (torch.Tensor): Equivariant encoded pocket features, shape [B, N_p, 3, d_equi]
            pocket_invs (torch.Tensor): Invariant encoded pocket features, shape [B, N_p, d_pocket_inv]
            pocket_atom_mask (torch.Tensor): Mask for pocket atoms, shape [B, N_p], 1 for real, 0 otherwise

        Returns:
            dict: Dictionary containing confidence predictions
                plddt: Per-atom confidence scores [B, N, 1]
                mask: Atom mask [B, N]
        """

        if pocket_invs is None or pocket_equis is None:
            raise ValueError(
                "Pocket cond inputs must be provided for using pocket conditioning."
            )

        atom_mask = torch.ones_like(coords[..., 0]) if atom_mask is None else atom_mask
        adj_matrix = smolF.adj_from_node_mask(atom_mask, self_connect=True)

        # Work out adj matrix between pocket and ligand, if required
        cond_adj_matrix = atom_mask.float().unsqueeze(
            2
        ) * pocket_atom_mask.float().unsqueeze(1)
        cond_adj_matrix = cond_adj_matrix.long()

        # Embed interaction types, if required
        interaction_feats = self.radial_basis_embed(
            coords,
            pocket_coords,
            ligand_mask=atom_mask,
            pocket_mask=pocket_atom_mask,
        )

        # Embed distances
        rbf_embeds = self.rbf_embed(coords, mask=atom_mask) if self.use_rbf else None

        # Project coords to d_equi
        coords = coords.unsqueeze(-1)
        equis = self.coord_emb(coords)  # [batch_size, N, 3, d_equi]

        # Embed invariant features (no time inputs)
        invs, edges = self.inv_emb(
            atom_types,
            bond_types,
            atom_mask=atom_mask,
            atom_charges=atom_charges,
            times=None,  # No time needed for confidence prediction
            extra_feats=extra_feats,
        )

        if rbf_embeds is not None:
            edges = torch.cat((edges, rbf_embeds), dim=-1)

        edges = self.bond_emb(equis, invs, equis, invs, edge_feats=edges)
        edges = edges * adj_matrix.unsqueeze(-1)

        # Iterate over Semla layers
        for layer in self.layers:
            equis, invs, _, _, interaction_out = layer(
                equis,
                invs,
                edges,
                adj_matrix,
                atom_mask,
                cond_equis=pocket_equis,
                cond_invs=pocket_invs,
                cond_edges=interaction_feats,
                cond_adj_matrix=cond_adj_matrix,
            )

        # Normalize features
        equis_norm = self.final_coord_norm(equis, atom_mask)
        invs_norm = self.final_inv_norm(invs)

        # 1. Generate gates from invariant features
        lig_feats = self.equi_gate(invs_norm, equis_norm, atom_mask)  # [B, N, d_inv]
        lig_feats = self.lig_proj(lig_feats)

        # 2. Interaction pooling: mean over ligand-pocket pairs
        interaction_mask = atom_mask.unsqueeze(2) * pocket_atom_mask.unsqueeze(
            1
        )  # [B, N, N_p]
        interaction_feats = (interaction_out * interaction_mask.unsqueeze(-1)).sum(
            dim=(2)
        ) / (
            interaction_mask.unsqueeze(-1).sum(dim=(2)) + 1e-6
        )  # [B, N, d_edge]
        interaction_feats = self.interaction_proj(
            interaction_feats
        )  # [B, N, d_inv // 2]
        combined_feats = torch.cat(
            [lig_feats, interaction_feats], dim=-1
        )  # [B, N, d_inv]

        # 3. Predict confidence scores
        plddt_scores = self.confidence_head(combined_feats)  # [B, N, n_output_bins]

        return plddt_scores

    def _get_clones(self, module, n):
        return [copy.deepcopy(module) for _ in range(n)]


class ConfidenceModule(torch.nn.Module):
    """Main entry point class for predicting ligand confidence scores.

    This class expects pre-computed pocket encodings and does not run any pocket encoding itself.
    It only requires pocket_equis and pocket_invs as inputs.
    """

    def __init__(
        self,
        d_inv,
        d_equi,
        d_message,
        n_layers,
        n_attn_heads,
        d_message_ff,
        d_edge,
        n_atom_types,
        n_bond_types,
        n_charge_types,
        d_pocket_inv,
        n_extra_atom_feats=None,
        n_output_bins=50,  # Default number of output bins for confidence scores
        emb_size=64,
        use_rbf=False,
        use_distances=False,
        use_crossproducts=False,
        eps=1e-6,
    ):
        super().__init__()

        self.d_equi = d_equi
        confidence_head = ConfidenceHead(
            d_equi,
            d_inv,
            d_message,
            n_layers,
            n_attn_heads,
            d_message_ff,
            d_edge,
            n_atom_types,
            n_bond_types,
            n_charge_types,
            d_pocket_inv=d_pocket_inv,
            n_extra_atom_feats=n_extra_atom_feats,
            n_output_bins=n_output_bins,
            emb_size=emb_size,
            use_rbf=use_rbf,
            use_distances=use_distances,
            use_crossproducts=use_crossproducts,
            eps=eps,
        )

        self.confidence_head = confidence_head

    @property
    def hparams(self):
        return self.confidence_head.hparams

    def forward(
        self,
        coords,
        atom_types,
        bond_types,
        atom_charges,
        atom_mask,
        extra_feats=None,
        pocket_coords=None,
        pocket_equis=None,
        pocket_invs=None,
        pocket_atom_mask=None,
    ):
        """Predict confidence scores for ligand atoms

        Args:
            coords (torch.Tensor): Ligand coordinates [B, N, 3]
            atom_types (torch.Tensor): Ligand atom types [B, N]
            bond_types (torch.Tensor): Ligand bond types [B, N, N]
            atom_charges (torch.Tensor): Ligand atom charges [B, N]
            atom_mask (torch.Tensor): Ligand atom mask [B, N]
            extra_feats (torch.Tensor): Extra ligand features [B, N, n_extra]
            pocket_coords (torch.Tensor): Pocket coordinates [B, N_p, 3] (for RBF if enabled)
            pocket_equis (torch.Tensor): Pre-computed pocket equivariant features [B, N_p, 3, d_equi]
            pocket_invs (torch.Tensor): Pre-computed pocket invariant features [B, N_p, d_inv]
            pocket_atom_mask (torch.Tensor): Pocket atom mask [B, N_p]

        Returns:
            dict: Dictionary containing confidence predictions
        """

        if pocket_equis is None or pocket_invs is None:
            raise ValueError(
                "Pre-computed pocket encodings (pocket_equis, pocket_invs) must be provided when using pocket conditioning."
            )

        confidence_out = self.confidence_head(
            coords,
            atom_types,
            bond_types,
            atom_charges=atom_charges,
            atom_mask=atom_mask,
            extra_feats=extra_feats,
            pocket_coords=pocket_coords,
            pocket_equis=pocket_equis,
            pocket_invs=pocket_invs,
            pocket_atom_mask=pocket_atom_mask,
        )

        return confidence_out

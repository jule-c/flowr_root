import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from glob import glob
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from rdkit import Chem
from tensordict import TensorDict

import flowr.util.functional as smolF
import flowr.util.metrics as Metrics
import flowr.util.rdkit as smolRD


class MolBuilder:
    def __init__(
        self,
        vocab,
        vocab_charges,
        vocab_hybridization=None,
        vocab_aromatic=None,
        pocket_noise=None,
        save_dir=None,
        n_workers=12,
    ):
        self.vocab = vocab
        self.vocab_charges = vocab_charges
        self.vocab_hybridization = vocab_hybridization
        self.vocab_aromatic = vocab_aromatic
        self.pocket_noise = pocket_noise
        self.save_dir = save_dir
        self.n_workers = n_workers
        self._executor = None

    def shutdown(self):
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    def _startup(self):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(self.n_workers)

    def mols_from_smiles(self, smiles, explicit_hs=False):
        self._startup()
        futures = [
            self._executor.submit(smolRD.mol_from_smiles, smi, explicit_hs)
            for smi in smiles
        ]
        mols = [future.result() for future in futures]
        self.shutdown()
        return mols

    def mols_from_tensors(
        self,
        coords,
        atom_dists,
        mask,
        bond_dists=None,
        charge_dists=None,
        hybridization_dists=None,
        aromaticity_dists=None,
        sanitise=True,
        add_hs=False,
    ):
        extracted = self._extract_mols(
            coords,
            atom_dists,
            mask,
            bond_dists=bond_dists,
            charge_dists=charge_dists,
            hybridization_dists=hybridization_dists,
            aromaticity_dists=aromaticity_dists,
        )

        self._startup()
        build_fn = partial(self._mol_from_tensors, sanitise=sanitise, add_hs=add_hs)
        futures = [self._executor.submit(build_fn, *items) for items in extracted]
        mols = [future.result() for future in futures]
        self.shutdown()

        return mols

    def ligs_from_complex(
        self,
        coords,
        mask,
        atom_dists,
        bond_dists=None,
        charge_dists=None,
        hybridization_dists=None,
        aromaticity_dists=None,
        sanitise=True,
        add_hs=False,
    ):
        extracted = self._extract_pocket_or_lig(
            coords,
            mask,
            atom_dists,
            bond_dists=bond_dists,
            charge_dists=charge_dists,
            hybridization_dists=hybridization_dists,
            aromaticity_dists=aromaticity_dists,
        )

        self._startup()
        build_fn = partial(self._mol_from_tensors, sanitise=sanitise, add_hs=add_hs)
        futures = [self._executor.submit(build_fn, *items) for items in extracted]
        mols = [future.result() for future in futures]
        self.shutdown()

        return mols

    # def pdb_from_tensors(coords, pdb_file):

    def pockets_from_complex(
        self,
        coords,
        pocket_mask,
        atom_dists,
        bond_dists,
        charge_dists=None,
        sanitise=False,
        add_hs=False,
    ):

        extracted = self._extract_pocket_or_lig(
            coords,
            pocket_mask,
            atom_dists=atom_dists,
            bond_dists=bond_dists,
            charge_dists=charge_dists,
        )

        self._startup()
        build_fn = partial(self._pocket_from_tensors, sanitise=sanitise, add_hs=add_hs)
        futures = [self._executor.submit(build_fn, *items) for items in extracted]
        pdbs = [future.result() for future in futures]
        self.shutdown()

        return pdbs

    def _pocket_from_tensors(
        self,
        coords,
        atom_dists,
        bond_dists=None,
        charge_dists=None,
        sanitise=False,
        add_hs=False,
    ):
        tokens = self._mol_extract_atomics(atom_dists)
        bonds = self._mol_extract_bonds(bond_dists) if bond_dists is not None else None
        charges = (
            self._mol_extract_charges(charge_dists)
            if charge_dists is not None
            else None
        )
        mol = smolRD.mol_from_atoms(
            coords.numpy(),
            tokens,
            bonds=bonds,
            charges=charges,
            repeated_sanitise=sanitise,
            add_hs=add_hs,
        )
        return mol

    def _mol_from_tensors(
        self,
        coords,
        atom_dists,
        bond_dists=None,
        charge_dists=None,
        hybridization_dists=None,
        aromaticity_dists=None,
        sanitise=True,
        add_hs=False,
    ):
        tokens = self._mol_extract_atomics(atom_dists)
        bonds = self._mol_extract_bonds(bond_dists) if bond_dists is not None else None
        charges = (
            self._mol_extract_charges(charge_dists)
            if charge_dists is not None
            else None
        )
        hybridization = (
            self._mol_extract_hybridization(hybridization_dists)
            if hybridization_dists is not None
            else None
        )
        aromaticity = (
            self._mol_extract_aromaticity(aromaticity_dists)
            if aromaticity_dists is not None
            else None
        )
        return smolRD.mol_from_atoms(
            coords.numpy(),
            tokens,
            bonds=bonds,
            charges=charges,
            hybridization=hybridization,
            aromaticity=aromaticity,
            sanitise=sanitise,
            add_hs=add_hs,
        )

    def mol_stabilities(self, coords, atom_dists, mask, bond_dists, charge_dists):
        extracted = self._extract_mols(
            coords, atom_dists, mask, bond_dists=bond_dists, charge_dists=charge_dists
        )
        mol_atom_stabilities = [self.atom_stabilities(*items) for items in extracted]
        return mol_atom_stabilities

    def atom_stabilities(self, coords, atom_dists, bond_dists, charge_dists):
        n_atoms = coords.shape[0]

        atomics = self._mol_extract_atomics(atom_dists)
        bonds = self._mol_extract_bonds(bond_dists)
        charges = self._mol_extract_charges(charge_dists).tolist()

        # Recreate the adj to ensure it is symmetric
        bond_indices = torch.tensor(bonds[:, :2])
        bond_types = torch.tensor(bonds[:, 2])
        adj = smolF.adj_from_edges(bond_indices, bond_types, n_atoms, symmetric=True)

        adj[adj == 4] = 1.5
        valencies = adj.sum(dim=-1).long()

        stabilities = []
        for i in range(n_atoms):
            atom_type = atomics[i]
            charge = charges[i]
            valence = valencies[i].item()

            if atom_type not in Metrics.ALLOWED_VALENCIES:
                stabilities.append(False)
                continue

            allowed = Metrics.ALLOWED_VALENCIES[atom_type]
            atom_stable = Metrics._is_valid_valence(valence, allowed, charge)
            stabilities.append(atom_stable)

        return stabilities

    # Separate each molecule from the batch
    def _extract_mols(
        self,
        coords,
        atom_dists,
        mask,
        bond_dists=None,
        charge_dists=None,
        hybridization_dists=None,
        aromaticity_dists=None,
    ):
        coords_list = []
        atom_dists_list = []
        bond_dists_list = []
        charge_dists_list = []
        hybridization_dists_list = []
        aromaticity_dists_list = []

        n_atoms = mask.sum(dim=1)
        for idx in range(coords.size(0)):
            mol_atoms = n_atoms[idx]
            mol_coords = coords[idx, :mol_atoms, :].cpu()
            mol_token_dists = atom_dists[idx, :mol_atoms, :].cpu()

            coords_list.append(mol_coords)
            atom_dists_list.append(mol_token_dists)

            if bond_dists is not None:
                mol_bond_dists = bond_dists[idx, :mol_atoms, :mol_atoms, :].cpu()
                bond_dists_list.append(mol_bond_dists)
            else:
                bond_dists_list.append(None)

            if charge_dists is not None:
                mol_charge_dists = charge_dists[idx, :mol_atoms, :].cpu()
                charge_dists_list.append(mol_charge_dists)
            else:
                charge_dists_list.append(None)
            if hybridization_dists is not None:
                mol_hybridization_dists = hybridization_dists[idx, :mol_atoms, :].cpu()
                hybridization_dists_list.append(mol_hybridization_dists)
            else:
                hybridization_dists_list.append(None)
            if aromaticity_dists is not None:
                mol_aromaticity_dists = aromaticity_dists[idx, :mol_atoms, :].cpu()
                aromaticity_dists_list.append(mol_aromaticity_dists)
            else:
                aromaticity_dists_list.append(None)

        zipped = zip(
            coords_list,
            atom_dists_list,
            bond_dists_list,
            charge_dists_list,
            hybridization_dists_list,
            aromaticity_dists_list,
        )
        return zipped

    def _extract_pocket_or_lig(
        self,
        coords,
        mask,
        atom_dists=None,
        bond_dists=None,
        charge_dists=None,
        hybridization_dists=None,
        aromaticity_dists=None,
    ):
        """
        Extract the ligand or pocket from the complex data
        Specify the mask to extract the ligand or pocket
        """
        coords_list = []
        atom_dists_list = []
        bond_dists_list = []
        charge_dists_list = []
        hybridization_dists_list = []
        aromaticity_dists_list = []

        for idx in range(coords.size(0)):
            mol_coords = coords[idx][mask[idx].bool()].cpu()
            coords_list.append(mol_coords)
            if atom_dists is not None:
                mol_token_dists = atom_dists[idx][mask[idx]].cpu()
                atom_dists_list.append(mol_token_dists)

            if bond_dists is not None:
                present_indices = mask[idx].nonzero(as_tuple=True)[0]
                mol_bond_dists = bond_dists[idx][
                    present_indices[:, None], present_indices
                ].cpu()
                bond_dists_list.append(mol_bond_dists)
            else:
                bond_dists_list.append(None)

            if charge_dists is not None:
                mol_charge_dists = charge_dists[idx][mask[idx].bool()].cpu()
                charge_dists_list.append(mol_charge_dists)
            else:
                charge_dists_list.append(None)
            if hybridization_dists is not None:
                mol_hybridization_dists = hybridization_dists[idx][
                    mask[idx].bool()
                ].cpu()
                hybridization_dists_list.append(mol_hybridization_dists)
            else:
                hybridization_dists_list.append(None)
            if aromaticity_dists is not None:
                mol_aromaticity_dists = aromaticity_dists[idx][mask[idx].bool()].cpu()
                aromaticity_dists_list.append(mol_aromaticity_dists)
            else:
                aromaticity_dists_list.append(None)

        zipped = zip(
            coords_list,
            atom_dists_list,
            bond_dists_list,
            charge_dists_list,
            hybridization_dists_list,
            aromaticity_dists_list,
        )
        return zipped

    # Take index with highest probability and convert to token
    def _mol_extract_atomics(self, atom_dists):
        vocab_indices = torch.argmax(atom_dists, dim=1).tolist()
        tokens = self.vocab.tokens_from_indices(vocab_indices)
        return tokens

    # Convert to atomic number bond list format
    def _mol_extract_bonds(self, bond_dists):
        bond_types = torch.argmax(bond_dists, dim=-1)
        bonds = smolF.bonds_from_adj(bond_types)
        return bonds.long().numpy()

    # Convert index from model to actual atom charge
    def _mol_extract_charges(self, charge_dists):
        charge_types = torch.argmax(charge_dists, dim=-1).tolist()
        charges = self.vocab_charges.tokens_from_indices(charge_types)
        return np.array(charges)

    def _mol_extract_hybridization(self, hybridization_dists):
        hybridization_types = torch.argmax(hybridization_dists, dim=-1).tolist()
        hybridization_types = self.vocab_hybridization.tokens_from_indices(
            hybridization_types
        )
        return hybridization_types

    def _mol_extract_aromaticity(self, aromaticity_dists):
        aromaticity_types = torch.argmax(aromaticity_dists, dim=-1).tolist()
        aromaticity_types = self.vocab_aromatic.tokens_from_indices(aromaticity_types)
        return aromaticity_types

    def write_xyz_file(self, coords, atom_types, filename):
        out = f"{len(coords)}\n\n"
        assert len(coords) == len(atom_types)
        for i in range(len(coords)):
            out += f"{atom_types[i]} {coords[i, 0]:.3f} {coords[i, 1]:.3f} {coords[i, 2]:.3f}\n"
        with open(filename, "w") as f:
            f.write(out)

    def tensors_to_xyz(
        self,
        prior=None,
        interpolated=None,
        data=None,
        predicted=None,
        coord_scale=1.0,
        idx=0,
        save_dir=".",
    ):
        """
        Write the coordinates and atom types of the ligand and pocket atoms to an xyz file.
        Can be used to debug the model while training or inference to see how the ligand and pocket atoms are placed.

        idx: index of the molecule in the batch to write to file
        """
        if data is not None:
            mask = data["mask"]
            coords = data["coords"]
            atom_dists = data["atomics"]
            extracted = list(
                self._extract_mols(
                    coords,
                    atom_dists,
                    mask,
                )
            )
            coords = extracted[idx][0] * coord_scale
            atom_types = self._mol_extract_atomics(extracted[idx][1])
            self.write_xyz_file(
                coords=coords,
                atom_types=atom_types,
                filename=os.path.join(save_dir, "ref.xyz"),
            )

        if interpolated is not None:
            mask = interpolated["mask"]
            coords = interpolated["coords"]
            atom_dists = interpolated["atomics"]
            extracted = list(
                self._extract_mols(
                    coords,
                    atom_dists,
                    mask,
                )
            )
            coords = extracted[idx][0] * coord_scale
            atom_types = self._mol_extract_atomics(extracted[idx][1])
            self.write_xyz_file(
                coords=coords,
                atom_types=atom_types,
                filename=os.path.join(save_dir, "interpolated.xyz"),
            )

        if prior is not None:
            mask = prior["mask"]
            coords = prior["coords"]
            atom_dists = prior["atomics"]
            extracted = list(
                self._extract_mols(
                    coords,
                    atom_dists,
                    mask,
                )
            )
            coords = extracted[idx][0] * coord_scale
            atom_types = self._mol_extract_atomics(extracted[idx][1])
            self.write_xyz_file(
                coords=coords,
                atom_types=atom_types,
                filename=os.path.join(save_dir, "prior.xyz"),
            )

        if predicted is not None:
            mask = predicted["mask"]
            coords = predicted["coords"]
            atom_dists = predicted["atomics"]
            extracted = list(
                self._extract_mols(
                    coords,
                    atom_dists,
                    mask,
                )
            )
            coords = extracted[idx][0] * coord_scale
            atom_types = self._mol_extract_atomics(extracted[idx][1])
            self.write_xyz_file(
                coords=coords,
                atom_types=atom_types,
                filename=os.path.join(save_dir, "predicted.xyz"),
            )

    def write_xyz_file_from_batch(
        self,
        data,
        coord_scale=1.0,
        path=".",
        t=0,
    ):
        if not os.path.exists(path):
            os.makedirs(path)

        mask = data["mask"]
        coords = data["coords"]
        atom_dists = data["atomics"]
        extracted = list(
            self._extract_mols(
                coords,
                atom_dists,
                mask,
            )
        )
        bs = len(coords)
        for idx in range(bs):
            save_dir = os.path.join(path, f"graph_{idx}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            coords = extracted[idx][0] * coord_scale
            atom_types = self._mol_extract_atomics(extracted[idx][1])
            self.write_xyz_file(
                coords=coords,
                atom_types=atom_types,
                filename=os.path.join(save_dir, f"latent_{t}.xyz"),
            )

    def write_trajectory_as_xyz(
        self,
        pred_mols,
        file_path,
        save_path,
        remove_intermediate_files=True,
    ):

        def get_key(fp):
            filename = os.path.splitext(os.path.basename(fp))[0]
            int_part = filename.split("_")[-1]
            return int(int_part)

        save_path = Path(save_path)
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=False)
        for i in range(len(pred_mols)):
            if smolRD.mol_is_valid(pred_mols[i], connected=True):
                files = sorted(
                    glob(os.path.join(file_path, f"graph_{i}/latent_*.xyz")),
                    key=get_key,
                )
                traj_path = save_path / f"trajectory_{i}.xyz"
                if traj_path.is_file():
                    traj_path.unlink()

                for j, file in enumerate(files):
                    with open(file, "r") as f:
                        lines = f.readlines()

                    with open(traj_path, "a") as file:
                        for line in lines:
                            file.write(line)
                        if (
                            j == len(files) - 1
                        ):  ####write the last timestep 10x for better visibility
                            for _ in range(10):
                                for line in lines:
                                    file.write(line)

        if remove_intermediate_files:
            shutil.rmtree(file_path)

    def add_ligand_to_pocket(
        self,
        lig_data,
        pocket_data,
    ):
        """
        Optimized version that combines ligand and pocket data more efficiently.
        """
        pocket_coords = pocket_data["coords"]
        pocket_atomics = pocket_data["atomics"]
        pocket_charges = pocket_data["charges"]
        pocket_res_names = pocket_data["res_names"]
        pocket_hybridization = pocket_data.get("hybridization", None)
        pocket_input_mask = pocket_data["mask"].bool()

        lig_coords = lig_data["coords"]
        lig_atomics = lig_data["atomics"]
        lig_charges = lig_data["charges"]
        lig_res_names = lig_data["res_names"]
        lig_hybridization = lig_data.get("hybridization", None)
        lig_input_mask = lig_data["mask"].bool()

        batch_size = lig_coords.size(0)
        device = lig_coords.device

        # Calculate the number of atoms for each molecule in the batch
        pocket_n_atoms = pocket_input_mask.sum(dim=1)
        lig_n_atoms = lig_input_mask.sum(dim=1)
        total_n_atoms = pocket_n_atoms + lig_n_atoms
        max_atoms = total_n_atoms.max().item()

        # Pre-allocate output tensors
        combined_coords = torch.zeros(
            batch_size, max_atoms, 3, device=device, dtype=lig_coords.dtype
        )
        combined_atomics = torch.zeros(
            batch_size,
            max_atoms,
            lig_atomics.size(-1),
            device=device,
            dtype=lig_atomics.dtype,
        )
        combined_charges = torch.zeros(
            batch_size,
            max_atoms,
            pocket_charges.size(-1),
            device=device,
            dtype=pocket_charges.dtype,
        )
        combined_res_names = torch.zeros(
            batch_size,
            max_atoms,
            device=device,
            dtype=pocket_res_names.dtype,
        )
        if pocket_hybridization is not None:
            combined_hybridization = torch.zeros(
                batch_size,
                max_atoms,
                pocket_hybridization.size(-1),
                device=device,
                dtype=pocket_hybridization.dtype,
            )
        else:
            combined_hybridization = None
        combined_mask = torch.zeros(
            batch_size, max_atoms, device=device, dtype=torch.int
        )
        ligand_mask = torch.zeros(batch_size, max_atoms, device=device, dtype=torch.int)
        pocket_mask = torch.zeros(batch_size, max_atoms, device=device, dtype=torch.int)

        # Vectorized copying using advanced indexing
        for i in range(batch_size):
            n_pocket = pocket_n_atoms[i].item()
            n_lig = lig_n_atoms[i].item()
            n_total = n_pocket + n_lig

            # Copy pocket data
            if n_pocket > 0:
                combined_coords[i, :n_pocket] = pocket_coords[i, pocket_input_mask[i]]
                combined_atomics[i, :n_pocket] = pocket_atomics[i, pocket_input_mask[i]]
                combined_charges[i, :n_pocket] = pocket_charges[i, pocket_input_mask[i]]
                combined_res_names[i, :n_pocket] = pocket_res_names[
                    i, pocket_input_mask[i]
                ]

            # Copy ligand data
            if n_lig > 0:
                combined_coords[i, n_pocket:n_total] = lig_coords[i, lig_input_mask[i]]
                combined_atomics[i, n_pocket:n_total] = lig_atomics[
                    i, lig_input_mask[i]
                ]
                combined_charges[i, n_pocket:n_total] = lig_charges[
                    i, lig_input_mask[i]
                ]
                combined_res_names[i, n_pocket:n_total] = lig_res_names[
                    i, lig_input_mask[i]
                ]

            # Copy hybridization if available
            if pocket_hybridization is not None:
                combined_hybridization[i, :n_pocket] = pocket_hybridization[
                    i, pocket_input_mask[i]
                ]
            if lig_hybridization is not None:
                combined_hybridization[i, n_pocket:n_total] = lig_hybridization[
                    i, lig_input_mask[i]
                ]

            # Set mask
            combined_mask[i, :n_total] = 1
            ligand_mask[i, n_pocket:n_total] = 1
            pocket_mask[i, :n_pocket] = 1

        out = {
            "coords": combined_coords,
            "atomics": combined_atomics,
            "res_names": combined_res_names,
            "mask": combined_mask,
            "lig_mask": ligand_mask,
            "pocket_mask": pocket_mask,
        }
        if pocket_charges is not None:
            out["charges"] = combined_charges
        if pocket_hybridization is not None:
            out["hybridization"] = combined_hybridization

        return out

    def extract_ligand_from_complex(self, data):
        """Fully vectorized version of extract_ligand_from_complex."""
        coords = data["coords"]
        atomics = data["atomics"]
        bonds = data["bonds"]
        lig_mask = data["lig_mask"].bool()

        batch_size = coords.shape[0]
        max_lig_atoms = lig_mask.sum(dim=1).max().item()

        # Extract coordinates and atomics efficiently
        lig_coords_list = [coords[i, lig_mask[i]] for i in range(batch_size)]
        lig_atomics_list = [atomics[i, lig_mask[i]] for i in range(batch_size)]

        # Vectorized bond extraction
        lig_bonds = torch.zeros(
            batch_size,
            max_lig_atoms,
            max_lig_atoms,
            bonds.shape[-1],
            device=bonds.device,
            dtype=bonds.dtype,
        )

        for i in range(batch_size):
            mask_i = lig_mask[i]
            num_atoms = mask_i.sum().item()
            if num_atoms > 0:
                lig_indices = mask_i.nonzero(as_tuple=True)[0]
                lig_bonds[i, :num_atoms, :num_atoms] = bonds[i][
                    lig_indices[:, None], lig_indices
                ]

        # Pad tensors efficiently
        lig_coords = smolF.pad_tensors(lig_coords_list)
        lig_atomics = smolF.pad_tensors(lig_atomics_list)
        atom_mask = smolF.pad_tensors(
            [torch.ones(coords.size(0), dtype=torch.int) for coords in lig_coords_list]
        ).to(lig_mask.device)

        out = {
            "coords": lig_coords,
            "atomics": lig_atomics,
            "bonds": lig_bonds,
            "mask": atom_mask,
        }

        # Handle optional fields efficiently
        for field in ["charges", "hybridization", "res_names"]:
            if field in data:
                field_list = [data[field][i, lig_mask[i]] for i in range(batch_size)]
                out[field] = smolF.pad_tensors(field_list)

        # Copy scalar fields directly
        for field in ["affinity", "docking_score"]:
            if field in data:
                out[field] = data[field]

        return out

    def extract_pocket_from_complex(self, data):
        """Extract the pocket from the complex data - optimized version."""
        coords = data["coords"]
        atomics = data["atomics"]
        bonds = data["bonds"]
        charges = data["charges"]
        hybridization = data.get("hybridization", None)
        pocket_mask = data["pocket_mask"].bool()

        batch_size = coords.shape[0]
        max_pocket_atoms = pocket_mask.sum(dim=1).max().item()

        # Extract coordinates and features efficiently
        pocket_coords_list = [coords[i, pocket_mask[i]] for i in range(batch_size)]
        pocket_atomics_list = [atomics[i, pocket_mask[i]] for i in range(batch_size)]
        pocket_charges_list = [charges[i, pocket_mask[i]] for i in range(batch_size)]
        pocket_hybridization_list = (
            [hybridization[i, pocket_mask[i]] for i in range(batch_size)]
            if hybridization is not None
            else None
        )

        # Extract string-based features (atom names and residue names)
        pocket_atoms_list = [
            data["atom_names"][i][pocket_mask[i]] for i in range(batch_size)
        ]
        pocket_res_names_list = [
            data["res_names"][i][pocket_mask[i]] for i in range(batch_size)
        ]

        # Vectorized bond extraction
        pocket_bonds = torch.zeros(
            batch_size,
            max_pocket_atoms,
            max_pocket_atoms,
            bonds.shape[-1],
            device=bonds.device,
            dtype=bonds.dtype,
        )

        for i in range(batch_size):
            mask_i = pocket_mask[i]
            num_atoms = mask_i.sum().item()
            if num_atoms > 0:
                pocket_indices = mask_i.nonzero(as_tuple=True)[0]
                pocket_bonds[i, :num_atoms, :num_atoms] = bonds[i][
                    pocket_indices[:, None], pocket_indices
                ]

        # Pad tensors efficiently
        pocket_coords = smolF.pad_tensors(pocket_coords_list)
        pocket_atomics = smolF.pad_tensors(pocket_atomics_list)
        pocket_charges = smolF.pad_tensors(pocket_charges_list)
        pocket_hybridization = (
            smolF.pad_tensors(pocket_hybridization_list)
            if hybridization is not None
            else None
        )
        pocket_atoms = smolF.pad_tensors(pocket_atoms_list)
        pocket_res_names = smolF.pad_tensors(pocket_res_names_list)

        # Create atom mask - fix the bug from ligand extraction
        atom_mask = smolF.pad_tensors(
            [torch.ones(len(coords), dtype=torch.int) for coords in pocket_coords_list]
        ).to(pocket_mask.device)

        out = {
            "coords": pocket_coords,
            "atomics": pocket_atomics,
            "atom_names": pocket_atoms,
            "res_names": pocket_res_names,
            "bonds": pocket_bonds,
            "charges": pocket_charges,
            "mask": atom_mask,
        }
        if pocket_hybridization is not None:
            out["hybridization"] = pocket_hybridization

        return out

    def _overwrite_pocket_data(
        self, coords, atomics, bonds, prior, charges=None, pocket_noise="random"
    ):
        """
        Overwrite the coordinates, atomics, and bonds with the prior data for the pocket.
        Args:
            coords (torch.Tensor): Coordinates of the molecule. Shape: (B, N, 3)
            atomics (torch.Tensor): Atomic features of the molecule. Shape: (B, N, D)
            bonds (torch.Tensor): Bond features of the molecule. Shape: (B, N, N, D)
            prior (dict): Prior data containing the original pocket information.
            charges (torch.Tensor, optional): Charges of the molecule. Shape: (B, N, 1)
            pocket_noise (str): Type of noise to apply to the pocket.
        Returns:
            tuple: Updated coordinates, atomics, bonds, and charges (if provided).
        """
        pocket_mask = prior["pocket_mask"].bool()
        if self.pocket_noise == "fix":
            coords = torch.where(pocket_mask.unsqueeze(-1), prior["coords"], coords)
        atomics = torch.where(pocket_mask.unsqueeze(-1), prior["atomics"], atomics)
        pocket_pair_mask = pocket_mask.unsqueeze(1) & pocket_mask.unsqueeze(2)
        bonds = torch.where(pocket_pair_mask.unsqueeze(-1), prior["bonds"], bonds)
        if charges is not None:
            charges = torch.where(pocket_mask.unsqueeze(-1), prior["charges"], charges)
            return coords, atomics, bonds, charges
        return coords, atomics, bonds

    def overwrite_pocket_data(self, data, prior):
        """
        Overwrite the coordinates, atomics, charges, and hybridization with the prior data for the pocket.

        Args:
            data (dict): Dictionary containing complex data with keys:
                - coords (torch.Tensor): Coordinates of the molecule. Shape: (B, N, 3)
                - atomics (torch.Tensor): Atomic features of the molecule. Shape: (B, N, D)
                - bonds (torch.Tensor, optional): Bond features of the molecule. Shape: (B, N, N, D)
                - charges (torch.Tensor, optional): Charges of the molecule. Shape: (B, N, C)
                - hybridization (torch.Tensor, optional): Hybridization features. Shape: (B, N, H)
                - mask (torch.Tensor): Atom mask. Shape: (B, N)
                - lig_mask (torch.Tensor): Ligand mask. Shape: (B, N)
                - pocket_mask (torch.Tensor): Pocket mask. Shape: (B, N)
                - res_names (torch.Tensor, optional): Residue names. Shape: (B, N)
            prior (dict): Prior data containing the original pocket information with same structure.

        Returns:
            dict: Updated data dictionary with pocket data overwritten from prior.
        """
        # Create a copy of the input data to avoid modifying the original
        updated_data = data.copy()

        # Get the pocket mask
        pocket_mask = prior["pocket_mask"].bool()

        # Overwrite coordinates based on pocket_noise setting
        if self.pocket_noise == "fix":
            updated_data["coords"] = torch.where(
                pocket_mask.unsqueeze(-1), prior["coords"], data["coords"]
            )

        # Always overwrite atomics for pocket atoms
        updated_data["atomics"] = torch.where(
            pocket_mask.unsqueeze(-1), prior["atomics"], data["atomics"]
        )

        # Overwrite charges if present in both data and prior
        if "charges" in data and data["charges"] is not None and "charges" in prior:
            updated_data["charges"] = torch.where(
                pocket_mask.unsqueeze(-1), prior["charges"], data["charges"]
            )

        # Overwrite hybridization if present in both data and prior
        if (
            "hybridization" in data
            and data["hybridization"] is not None
            and "hybridization" in prior
        ):
            updated_data["hybridization"] = torch.where(
                pocket_mask.unsqueeze(-1), prior["hybridization"], data["hybridization"]
            )

        # Overwrite bonds if present in both data and prior
        if "bonds" in data and data["bonds"] is not None and "bonds" in prior:
            pocket_pair_mask = pocket_mask.unsqueeze(1) & pocket_mask.unsqueeze(2)
            updated_data["bonds"] = torch.where(
                pocket_pair_mask.unsqueeze(-1), prior["bonds"], data["bonds"]
            )

        return updated_data

    def _inpaint_times(self, times, mask=None):
        times = (
            times.masked_fill(mask, 1.0)
            if mask is not None
            else torch.ones_like(times).to(times.device)
        )
        return times

    def inpaint_molecule(
        self,
        data: dict,
        prediction: dict,
        pocket_mask: Optional[torch.Tensor] = None,
        keep_interactions: Optional[bool] = False,
        symmetrize: bool = True,
    ) -> dict:
        """
        Vectorized inpainting based on the interactions.
        Returns:
        dict: Updated prediction dictionary with inpainted coordinates, atomics, and bonds.
        """
        # Unpack prior
        coords = data["coords"]  # (B, N_l, C)
        atomics = data["atomics"]  # (B, N_l, d)
        charges = data.get("charges", None)  # (B, N_l, c)
        hybridization = data.get("hybridization", None)  # (B, N_l, h)
        bonds = data.get("bonds", None)  # (B, N_l, N_l, n_bonds)
        fragment_mask = data["fragment_mask"].bool()  # (B, N_l)
        fragment_mode = data["fragment_mode"]

        # Unpack predicted
        pred_coords = prediction["coords"]  # (B, N_l, C)
        pred_atomics = prediction["atomics"]  # (B, N_l, d)
        pred_charges = prediction.get("charges", None)  # (B, N_l, c)
        pred_hybridization = prediction.get("hybridization", None)  # (B, N_l, h)
        pred_bonds = prediction.get("bonds", None)  # (B, N_l, N_l, n_bonds)
        lig_mask = data["mask"].bool()  # (B, N_l)

        # Inpainting mask
        inpaint_mask = fragment_mask & lig_mask  # (B, N_l)

        # Overwrite coordinates and atomics where interactions are present
        pred_coords[inpaint_mask, :] = coords[inpaint_mask, :]
        pred_atomics[inpaint_mask, :] = atomics[inpaint_mask, :]
        if pred_charges is not None:
            pred_charges[inpaint_mask, :] = charges[inpaint_mask, :]
        if pred_hybridization is not None:
            pred_hybridization[inpaint_mask, :] = hybridization[inpaint_mask, :]

        # ========== BOND INPAINTING WITH SYMMETRIZATION ==========
        if bonds is not None and pred_bonds is not None:
            inpaint_mask_pair = inpaint_mask.unsqueeze(2) & inpaint_mask.unsqueeze(
                1
            )  # (B, N, N)
            # bond_mask = torch.argmax(bonds, dim=-1) != 0  # (B, N, N)
            # fixed_mask = inpaint_mask_pair & bond_mask  # (B, N, N)
            # pred_bonds[inpaint_mask_pair] = bonds[inpaint_mask_pair]
            pred_bonds = torch.where(inpaint_mask_pair.unsqueeze(-1), bonds, pred_bonds)
            if symmetrize:
                pred_bonds = smolF.symmetrize_bonds(pred_bonds, is_one_hot=True)

        out = {
            "coords": pred_coords,
            "atomics": pred_atomics,
            "charges": pred_charges,
            "hybridization": pred_hybridization,
            "bonds": pred_bonds,
            "mask": data["mask"],
            "fragment_mask": fragment_mask,
            "fragment_mode": fragment_mode,
        }
        if keep_interactions and "interactions" in prediction:
            out["interactions"] = prediction["interactions"]

        return out

    def inpaint_graph(
        self,
        data: dict,
        prediction: dict,
        pocket_mask: Optional[torch.Tensor] = None,
        keep_interactions: Optional[bool] = False,
        feature_keys: Optional[list] = None,
        overwrite_with_zeros: Optional[bool] = False,
    ) -> dict:
        """
        Vectorized inpainting based on the interactions.
        Returns:
        dict: Updated prediction dictionary with inpainted atomics, and bonds.

        Stable version incase mixed denovo/conditional is performed in the batch
        """

        if feature_keys is None:
            feature_keys = ["atomics", "bonds", "charges", "hybridization"]

        out = {}
        assert "fragment_mask" in data, "Fragment mask is required for inpainting."
        fragment_mask = data["fragment_mask"].bool()  # (B, N_l)
        fragment_mode = data["fragment_mode"]  # (B,)
        lig_mask = data["mask"].bool()  # (B, N_l)
        out["fragment_mask"] = fragment_mask
        out["fragment_mode"] = fragment_mode
        out["mask"] = lig_mask

        # Check which molecules have full fragment masks (all 1s for valid atoms)
        valid_atoms_per_mol = lig_mask.sum(dim=1)  # Number of valid atoms per molecule
        fragment_atoms_per_mol = (fragment_mask & lig_mask).sum(
            dim=1
        )  # Number of fragment atoms per molecule
        full_fragment_molecules = (valid_atoms_per_mol == fragment_atoms_per_mol) & (
            valid_atoms_per_mol > 0
        )
        inpaint_mask = fragment_mask & lig_mask  # (B, N_l)
        if full_fragment_molecules.any():
            for feature in feature_keys:
                source = data[feature] if feature in data else None
                target = prediction[feature] if feature in prediction else None
                if feature != "bonds":
                    if target is not None:
                        # Create a mask that only affects molecules with full fragment masks
                        full_inpaint_mask = (
                            inpaint_mask & full_fragment_molecules.unsqueeze(1)
                        )
                        if overwrite_with_zeros:
                            target[full_inpaint_mask, :] = 0
                        elif source is not None:
                            target[full_inpaint_mask, :] = source[full_inpaint_mask, :]
                else:
                    if target is not None:
                        full_inpaint_mask_pair = (
                            inpaint_mask & full_fragment_molecules.unsqueeze(1)
                        ).unsqueeze(2) & (
                            inpaint_mask & full_fragment_molecules.unsqueeze(1)
                        ).unsqueeze(
                            1
                        )
                        if overwrite_with_zeros:
                            target[full_inpaint_mask_pair] = 0
                        elif source is not None:
                            target[full_inpaint_mask_pair] = source[
                                full_inpaint_mask_pair
                            ]
                out[feature] = target

        if keep_interactions and "interactions" in prediction:
            out["interactions"] = prediction["interactions"]

        # overwrite prediction dictionary with updated values
        prediction.update(out)
        # use update to make sure pre-existing key/values are present
        return prediction

    # def inpaint_graph(
    #     self,
    #     data: dict,
    #     prediction: dict,
    #     pocket_mask: Optional[torch.Tensor] = None,
    #     keep_interactions: Optional[bool] = False,
    #     feature_keys: Optional[list] = None,
    # ) -> dict:
    #     """
    #     Vectorized inpainting based on the interactions.
    #     Returns:
    #     dict: Updated prediction dictionary with inpainted atomics, and bonds.

    #     Stable version incase mixed denovo/conditional is performed in the batch
    #     """

    #     if feature_keys is None:
    #         feature_keys = ["atomics", "bonds", "charges", "hybridization"]

    #     out = {}
    #     assert "fragment_mask" in data, "Fragment mask is required for inpainting."
    #     fragment_mask = data["fragment_mask"].bool()  # (B, N_l)
    #     lig_mask = data["mask"].bool()  # (B, N_l)
    #     out["fragment_mask"] = fragment_mask
    #     out["mask"] = lig_mask
    #     inpaint_mask = fragment_mask & lig_mask  # (B, N_l)
    #     has_fragments = fragment_mask.sum(dim=1) > 0
    #     if has_fragments.any():
    #         for feature in feature_keys:
    #             source = data[feature] if feature in data else None
    #             target = prediction[feature] if feature in prediction else None
    #             if feature != "bonds":
    #                 if target is not None and source is not None:
    #                     target[inpaint_mask, :] = source[inpaint_mask, :]
    #             else:
    #                 if target is not None and source is not None:
    #                     inpaint_mask_pair = inpaint_mask.unsqueeze(
    #                         2
    #                     ) & inpaint_mask.unsqueeze(1)
    #                     target[inpaint_mask_pair] = source[inpaint_mask_pair]
    #             out[feature] = target

    #     if keep_interactions and "interactions" in prediction:
    #         out["interactions"] = prediction["interactions"]

    #     # overwrite prediction dictionary with updated values
    #     prediction.update(out)
    #     # use update to make sure pre-existing key/values are present
    #     return prediction

    def inpaint_interactions(
        self,
        data: dict,
        prediction: dict,
        pocket_mask: torch.Tensor,
        keep_interactions: bool = False,
    ) -> dict:
        """
        Vectorized inpainting based on the interactions.
        Returns:
        dict: Updated prediction dictionary with inpainted coordinates, atomics, and bonds.
        """
        # Unpack prior
        coords = data["coords"]  # (B, N_l, C)
        atomics = data["atomics"]  # (B, N_l, d)
        charges = data.get("charges", None)  # (B, N_l, c)
        hybridization = data.get("hybridization", None)  # (B, N_l, h)
        bonds = data.get("bonds", None)  # (B, N_l, N_l, n_bonds)
        interactions = data["interactions"]  # (B, N_l, N_p, n_interactions)

        # Unpack predicted
        pred_coords = prediction["coords"]  # (B, N_l, C)
        pred_atomics = prediction["atomics"]  # (B, N_l, d)
        pred_charges = prediction.get("charges", None)  # (B, N_l, c)
        pred_hybridization = prediction.get("hybridization", None)  # (B, N_l, h)
        pred_bonds = prediction.get("bonds", None)  # (B, N_l, N_l, n_bonds)
        lig_mask = data["mask"].bool()  # (B, N_l)

        # Interaction mask
        inpaint_mask = interactions[..., 1:].sum(dim=(2, 3)) > 0
        inpaint_mask = inpaint_mask & lig_mask

        # Overwrite coordinates and atomics where interactions are present
        pred_coords[inpaint_mask] = coords[inpaint_mask]
        pred_atomics[inpaint_mask] = atomics[inpaint_mask]
        if pred_charges is not None:
            pred_charges[inpaint_mask] = charges[inpaint_mask]
        if pred_hybridization is not None:
            pred_hybridization[inpaint_mask] = hybridization[inpaint_mask]

        # Overwrite bonds with a pairwise fixed mask:
        if bonds is not None and pred_bonds is not None:
            inpaint_mask = inpaint_mask.unsqueeze(2) & inpaint_mask.unsqueeze(1)
            bond_mask = torch.argmax(bonds, dim=-1) != 0
            fixed_mask = inpaint_mask & bond_mask
            pred_bonds[fixed_mask] = bonds[fixed_mask]
            # pred_bonds = torch.where(fixed_mask, bonds, pred_bonds)

        out = {
            "coords": pred_coords,
            "atomics": pred_atomics,
            "charges": pred_charges,
            "hybridization": pred_hybridization,
            "bonds": pred_bonds,
            "mask": data["mask"],
        }
        if keep_interactions and "interactions" in prediction:
            out["interactions"] = prediction["interactions"]

        return out

    def undo_zero_com(self, coords, com):
        return coords + com

    def undo_zero_com_batch(self, coords, node_mask, com_list):
        shifted_coords = torch.zeros_like(coords)
        for i in range(coords.size(0)):
            shifted = self.undo_zero_com(coords[i], com_list[i]) * node_mask[
                i
            ].unsqueeze(-1)
            shifted_coords[i] = shifted
        return shifted_coords

    def add_properties_from_tensor_dict(
        self, mols: List[Chem.Mol | None], properties: TensorDict
    ):
        properties = properties.to("cpu")
        keys = properties.keys()
        for i, mol in enumerate(mols):
            if isinstance(mol, Chem.Mol):
                for key in keys:
                    v = np.round(properties[key][i].squeeze().item(), 4).item()
                    mol.SetProp(key, str(v))
        return mols

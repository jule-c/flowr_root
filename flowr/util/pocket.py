from __future__ import annotations

import copy
import pickle
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import MDAnalysis as mda
import numpy as np
import prolif as plf
import torch
from biotite.structure import AtomArray, BondList
from rdkit import Chem
from tensordict import TensorDict
from tqdm import tqdm

import flowr.util.functional as smolF
from flowr.util.molrepr import GeometricMol, GeometricMolBatch
from flowr.util.rdkit import PeriodicTable
from flowr.util.tokeniser import (
    pocket_atom_name_encoder,
    pocket_res_name_encoder,
)

_T = torch.Tensor

PICKLE_PROTOCOL = 4


INTERACTION_PARAMETERS = {
    "HBAcceptor": {
        "distance": 3.7,
        "donor": "[$([O,S,#7;+0]),$([N+1])]-[H]",
    },
    "HBDonor": {"distance": 3.7},
    "CationPi": {"distance": 5.5},
    "PiCation": {"distance": 5.5},
    "Anionic": {"distance": 5},
    "Cationic": {"distance": 5},
}

# All possible prolif interactions types where PiStacking includes both edge-to-face and face-to-face
# Users must use a subset of this list when specifying interactions
PROLIF_INTERACTIONS = [
    # "Hydrophobic",
    # "VdWContact",
    # "MetalAcceptor",
    # "MetalDonor",
    "Cationic",
    "Anionic",
    "XBAcceptor",
    "XBDonor",
    "CationPi",
    "PiCation",
    "PiStacking",
    "HBAcceptor",
    "HBDonor",
]

GROUP_METAL_MAP = {
    "MetalAcceptor": "metal-interaction",
    "MetalDonor": "metal-interaction",
}

GROUP_IONIC_MAP = {"Cationic": "ionic-interaction", "Anionic": "ionic-interaction"}

GROUP_HALOGEN_MAP = {"XBAcceptor": "halogen-bond", "XBDonor": "halogen-bond"}

GROUP_PI_MAP = {"CationPi": "pi-cation", "PiCation": "pi-cation"}

GROUP_HBOND_MAP = {"HBAcceptor": "hydrogen-bond", "HBDonor": "hydrogen-bond"}


PT = PeriodicTable()


# **********************
# *** Util functions ***
# **********************


def _check_type(obj, obj_type, name="object"):
    if not isinstance(obj, obj_type):
        raise TypeError(
            f"{name} must be an instance of {obj_type} or one of its subclasses, got {type(obj)}"
        )


def _check_dict_key(map, key, dict_name="dictionary"):
    if key not in map:
        raise RuntimeError(f"{dict_name} must contain key {key}")


def _check_interaction_types_exist(interaction_types):
    for int_type in interaction_types:
        if int_type not in PROLIF_INTERACTIONS:
            raise ValueError(f"Interaction type {int_type} is not recognised.")


# Check that all interactions in check_interactions exist in interaction_profile
def _check_interaction_profile(check_interactions, interaction_profile):
    for int_type in check_interactions:
        if int_type not in interaction_profile:
            raise ValueError(
                f"Interaction type {int_type} not found in current interaction profile."
            )


# ************************
# *** Module functions ***
# ************************


def calc_system_metrics(system: PocketComplex) -> dict:
    metrics = {}

    # TODO calculate some holo and ligand metrics
    if system.apo is None or system.holo is None:
        return metrics

    backbone_mask = struc.filter_peptide_backbone(system.holo.atoms)
    apo_backbone = system.apo.atoms[backbone_mask]
    holo_backbone = system.holo.atoms[backbone_mask]

    pocket_rmsd = struc.rmsd(system.apo.atoms, system.holo.atoms)
    backbone_rmsd = struc.rmsd(apo_backbone, holo_backbone)

    metrics["pocket_rmsd"] = pocket_rmsd
    metrics["backbone_rmsd"] = backbone_rmsd

    return metrics


# ****************************
# *** Binding Interactions ***
# ****************************


class BindingInteractions:
    """A class for extracting and representing interactions between a protein pocket and a ligand.

    Interactions are extracted using Prolif and stored in bit vectors with one vector for each possible pair of atoms
    between the protein and the ligand.
    """

    def __init__(self, interaction_types: list[str], interaction_arr: np.ndarray):
        if len(interaction_arr.shape) != 3:
            raise ValueError(
                f"interaction_arr must have 3 dimensions, got shape {interaction_arr.shape}"
            )

        n_ints = len(interaction_types)
        arr_dim = interaction_arr.shape[-1]
        if n_ints != arr_dim:
            raise ValueError(
                f"Number of interaction types must match final dim of arr, got {n_ints} and {arr_dim}"
            )

        self.interaction_types = interaction_types
        self.interaction_arr = interaction_arr

    @property
    def array(self) -> np.ndarray:
        return self.interaction_arr.copy()

    # *** Conversion functions ***

    @staticmethod
    def from_system(
        system: PocketComplex, interaction_types: list[str] = None
    ) -> BindingInteractions:
        if interaction_types is not None:
            _check_interaction_types_exist(interaction_types)

        # By default look for all interaction types
        interaction_types = (
            PROLIF_INTERACTIONS if interaction_types is None else interaction_types
        )
        interaction_arr = BindingInteractions.interaction_array(
            system, interaction_types
        )
        return BindingInteractions(interaction_types, interaction_arr)

    @staticmethod
    def from_bytes(data: bytes, interaction_types: list[str] = None):
        obj = pickle.loads(data)

        _check_dict_key(obj, "interaction-types")
        _check_dict_key(obj, "interaction-arr")

        int_types = obj["interaction-types"]
        int_arr = obj["interaction-arr"]

        interactions = BindingInteractions(int_types, int_arr)
        if interaction_types is not None:
            return interactions.subset(interaction_types)
        return interactions

    def to_bytes(self):
        data_dict = {
            "interaction-types": self.interaction_types,
            "interaction-arr": self.interaction_arr,
        }
        return pickle.dumps(data_dict, protocol=PICKLE_PROTOCOL)

    # *** Useful functions ***

    def subset(self, interaction_types: list[str]) -> BindingInteractions:
        _check_interaction_profile(interaction_types, self.interaction_types)
        int_type_mask = np.array(
            [int_type in interaction_types for int_type in self.interaction_types]
        )
        interaction_arr = self.interaction_arr[:, :, int_type_mask]
        subset_interactions = BindingInteractions(interaction_types, interaction_arr)
        return subset_interactions

    def group_metal_interactions(self) -> BindingInteractions:
        return self._group_interactions_from_map(GROUP_METAL_MAP)

    def group_ionic_interactions(self) -> BindingInteractions:
        return self._group_interactions_from_map(GROUP_IONIC_MAP)

    def group_halogen_interactions(self) -> BindingInteractions:
        return self._group_interactions_from_map(GROUP_HALOGEN_MAP)

    def group_pi_cation_interactions(self) -> BindingInteractions:
        return self._group_interactions_from_map(GROUP_PI_MAP)

    def group_hbond_interactions(self) -> BindingInteractions:
        return self._group_interactions_from_map(GROUP_HBOND_MAP)

    def group_interactions(self, include_hbond: bool = True) -> BindingInteractions:
        interaction_obj = self.copy()

        if all(
            [int_type in self.interaction_types for int_type in GROUP_METAL_MAP.keys()]
        ):
            interaction_obj = interaction_obj.group_metal_interactions()

        if all(
            [int_type in self.interaction_types for int_type in GROUP_IONIC_MAP.keys()]
        ):
            interaction_obj = interaction_obj.group_ionic_interactions()

        if all(
            [
                int_type in self.interaction_types
                for int_type in GROUP_HALOGEN_MAP.keys()
            ]
        ):
            interaction_obj = interaction_obj.group_halogen_interactions()

        if all(
            [int_type in self.interaction_types for int_type in GROUP_PI_MAP.keys()]
        ):
            interaction_obj = interaction_obj.group_pi_cation_interactions()

        if include_hbond and all(
            [int_type in self.interaction_types for int_type in GROUP_HBOND_MAP.keys()]
        ):
            interaction_obj = interaction_obj.group_hbond_interactions()

        return interaction_obj

    def _group_interactions_from_map(self, group_map):
        _check_interaction_profile(list(group_map.keys()), self.interaction_types)

        # Get the list of new interaction types by either looking up in the group map or using the current type
        grouped_interaction_types = [
            group_map.get(int_type, int_type) for int_type in self.interaction_types
        ]
        grouped_interaction_types = list(set(grouped_interaction_types))
        interaction_idx_map = {
            int_type: idx for idx, int_type in enumerate(grouped_interaction_types)
        }

        arr_shape = (*self.interaction_arr.shape[:-1], len(grouped_interaction_types))
        grouped_arr = np.zeros(arr_shape, dtype=np.int8)

        # Add the old interactions onto the new array in their grouped locations
        for int_idx, int_type in enumerate(self.interaction_types):
            grouped_int_type = group_map.get(int_type, int_type)
            grouped_int_idx = interaction_idx_map[grouped_int_type]
            curr_interactions = self.interaction_arr[:, :, int_idx]
            grouped_arr[:, :, grouped_int_idx] += curr_interactions

        # Ensure that all bits are either zero or one
        grouped_arr = np.minimum(grouped_arr, 1)
        interactions = BindingInteractions(grouped_interaction_types, grouped_arr)
        return interactions

    def copy(self):
        interaction_types = self.interaction_types[:]
        interaction_arr = np.copy(self.interaction_arr)
        return BindingInteractions(interaction_types, interaction_arr)

    # *** Prolif interaction functions ***

    @staticmethod
    def interaction_array(
        system: PocketComplex,
        interaction_types: list[str] = None,
        return_fp: bool = False,
    ) -> Union[np.ndarray, tuple[np.ndarray, plf.Fingerprint]]:
        """Build an atom-level interaction array, array shape [n_atoms_ligand, n_atoms_pocket, n_int_types]"""

        if interaction_types is not None:
            _check_interaction_types_exist(interaction_types)

        interaction_types = (
            PROLIF_INTERACTIONS if interaction_types is None else interaction_types
        )
        plf_interaction_map = {
            int_type: idx for idx, int_type in enumerate(interaction_types)
        }

        plf_fp = BindingInteractions.interaction_fp(system, interaction_types)
        atom_interactions = BindingInteractions._interactions_from_ifp(plf_fp.ifp[0])

        # Create a bit vector for each protein-ligand atom pair and set the bits for the given interactions
        arr_shape = (len(system.holo), len(system.ligand), len(interaction_types))
        interaction_arr = np.zeros(arr_shape, dtype=np.int8)

        for int_type, p_idx, l_idx in atom_interactions:
            int_idx = plf_interaction_map[int_type]
            interaction_arr[p_idx, l_idx, int_idx] = 1

        if return_fp:
            return interaction_arr, plf_fp

        return interaction_arr

    @staticmethod
    def interaction_fp(
        system: PocketComplex, interaction_types: list[str] = None
    ) -> plf.Fingerprint:
        """Run the prolif interaction algorithm and retrun the prolif IFP object"""

        if interaction_types is not None:
            _check_interaction_types_exist(interaction_types)

        interaction_types = (
            PROLIF_INTERACTIONS if interaction_types is None else interaction_types
        )
        holo_mol, ligand_mol = BindingInteractions.prepare_prolif_mols(system)
        plf_fp = plf.Fingerprint(
            interactions=interaction_types,
            parameters=INTERACTION_PARAMETERS,
            count=True,
        )
        plf_fp.run_from_iterable([ligand_mol], holo_mol, residues="all", progress=False)
        return plf_fp

    @staticmethod
    def prepare_prolif_mols(
        system: PocketComplex, lig_resname="LIG", lig_resnumber=-1
    ) -> tuple[plf.Molecule, plf.Molecule]:
        """Create prolif molecules for pocket and ligand."""

        holo_mol = system.holo.to_prolif()
        ligand_rdkit = system.ligand.to_rdkit()
        ligand_mol = plf.Molecule.from_rdkit(
            ligand_rdkit, resname=lig_resname, resnumber=lig_resnumber
        )
        return holo_mol, ligand_mol

    @staticmethod
    def _interactions_from_ifp(ifp):
        """Loops through the interactions obtained from Prolif to provide them in list format.

        Interactions are returned as a list of tuples [(<interaction_type>, <protein_atom_idx>, <ligand_atom_idx>)]

        Note that for interactions which involve mutliple atoms in ligand or protein (eg. Pi interactions), all
        possible pairings of atoms will be listed. So for pi stacking interactions, for example, 36 interactions will
        be provided for two interacting 6-member rings.
        """

        interaction_list = []
        for _, res_interactions in ifp.items():
            for int_type, interactions in res_interactions.items():
                for interaction in interactions:
                    l_atom_idxs = interaction["parent_indices"]["ligand"]
                    p_atom_idxs = interaction["parent_indices"]["protein"]
                    int_tuples = [
                        (int_type, p_idx, l_idx)
                        for p_idx in p_atom_idxs
                        for l_idx in l_atom_idxs
                    ]
                    interaction_list.extend(int_tuples)

        return interaction_list


# **********************************
# *** Pocket and Complex Classes ***
# **********************************


# TODO implement own version of AtomArray and BondArray for small molecules
# Use these for Smol molecule implementations


class ProteinPocket:
    def __init__(
        self,
        atoms: AtomArray,
        bonds: BondList,
        mol: GeometricMol = None,
        orig_pocket: Optional[ProteinPocket] = None,
        str_id: Optional[str] = None,
    ):
        self._check_atom_array(atoms)

        if "charge" not in atoms.get_annotation_categories():
            atoms.add_annotation("charge", np.float32)

        self.atoms = atoms
        self.bonds = bonds
        self.orig_pocket = orig_pocket
        self.str_id = str_id
        self._mol = mol

    @property
    def seq_length(self) -> int:
        return len(self.atoms)

    @property
    def res_names(self):
        try:
            res_names = torch.tensor(
                [pocket_res_name_encoder[res] for res in self.atoms.res_name]
            ).long()
        except Exception:
            print("Error encoding residue names. Trying fallback, replacing with UNK.")
            res_names = []
            for res in self.atoms.res_name:
                try:
                    res_names.append(pocket_res_name_encoder[res])
                except KeyError:
                    print(f"Replacing {res} with UNK")
                    res_names.append(pocket_res_name_encoder["UNK"])
            res_names = torch.tensor(res_names).long()
        return res_names

    # @property
    # def atom_names(self):
    #     def fix_atom_name(atom: str) -> str:
    #         # Some PDB files have atom names like 1HG1, 2HG1 etc. which does not match with the Plinder data the model normally is trained on! The format is HG11, HG12 etc.
    #         if atom and atom[0].isdigit():
    #             pos = 0
    #             while pos < len(atom) and atom[pos].isdigit():
    #                 pos += 1
    #             return atom[pos:] + atom[:pos]
    #         return atom

    #     atom_names = torch.tensor(
    #         [
    #             pocket_atom_name_encoder[fix_atom_name(atom)]
    #             for atom in self.atoms.atom_name
    #         ]
    #     ).long()  # .replace("'", ""))]
    #     return atom_names

    @property
    def atom_names(self):
        atom_names = torch.tensor(
            [pocket_atom_name_encoder[atom] for atom in self.atoms.element]
        ).long()
        return atom_names

    @property
    def atom_symbols(self) -> list[str]:
        return self.atoms.element.tolist()

    @property
    def n_residues(self) -> int:
        res_ids = set(self.atoms.res_id)
        return len(res_ids)

    def __len__(self) -> int:
        return self.seq_length

    # *** Subset functions ***

    def select_atoms(
        self,
        mask: np.ndarray,
        str_id: Optional[str] = None,
        keep_orig_pocket: bool = False,
    ) -> ProteinPocket:
        """Select atoms in the pocket using a binary np mask. True means keep the atom. Returns a copy."""

        # Numpy will throw an error if the size of the mask doesn't match the atoms so don't handle this explicitly
        atom_struc = self.atoms.copy()
        bonds = self.bonds.copy()
        atom_struc.bonds = bonds
        atom_subset = atom_struc[mask]

        bond_subset = atom_subset.bonds
        atom_subset.bonds = None

        str_id = str_id if str_id is not None else self.str_id
        if keep_orig_pocket:
            atom_struc.bonds = None
            orig_pocket = ProteinPocket(atom_struc, bonds, str_id=str_id)
            pocket = ProteinPocket(
                atom_subset, bond_subset, str_id=str_id, orig_pocket=orig_pocket
            )
        else:
            pocket = ProteinPocket(atom_subset, bond_subset, str_id=str_id)
        return pocket

    def remove_hs(
        self, str_id: Optional[str] = None, keep_orig_pocket: bool = False
    ) -> ProteinPocket:
        """Returns a copy of the object with hydrogens removed"""

        atom_struc = self.atoms.copy()
        bonds = self.bonds.copy()
        atom_struc.bonds = bonds
        atoms_no_hs = atom_struc[atom_struc.element != "H"]

        bonds_no_hs = atoms_no_hs.bonds
        atoms_no_hs.bonds = None

        str_id = str_id if str_id is not None else self.str_id
        if keep_orig_pocket:
            atom_struc.bonds = None
            orig_pocket = ProteinPocket(atom_struc, bonds, str_id)
            pocket = ProteinPocket(
                atoms_no_hs, bonds_no_hs, str_id=str_id, orig_pocket=orig_pocket
            )
        else:
            pocket = ProteinPocket(atoms_no_hs, bonds_no_hs, str_id=str_id)
        return pocket

    # *** Conversion functions ***

    @staticmethod
    def from_pocket_atoms(
        atoms: AtomArray, infer_res_bonds: bool = False
    ) -> ProteinPocket:
        # Will either infer bonds or bonds will be taken from the atoms (bonds on atoms could be None)
        if infer_res_bonds:
            bonds = struc.connect_via_residue_names(atoms, inter_residue=True)
        else:
            bonds = atoms.bonds

        return ProteinPocket(atoms, bonds)

    @staticmethod
    def from_protein(
        structure: AtomArray,
        chain_id: int,
        res_ids: list[int],
        infer_res_bonds: bool = False,
    ) -> ProteinPocket:
        chain = structure[structure.chain_id == chain_id]
        pocket = chain[np.isin(chain.res_id, res_ids)]
        return ProteinPocket.from_pocket_atoms(pocket, infer_res_bonds=infer_res_bonds)

    @staticmethod
    def from_bytes(data: bytes, remove_hs: bool = False) -> ProteinPocket:
        obj = pickle.loads(data)

        _check_dict_key(obj, "atoms")
        _check_dict_key(obj, "bonds")
        _check_dict_key(obj, "str_id")

        atoms = obj["atoms"]
        bonds = obj["bonds"]
        str_id = obj["str_id"]

        pocket = ProteinPocket(atoms, bonds, str_id=str_id)
        if remove_hs:
            pocket = pocket.remove_hs()
        return pocket

    def to_geometric_mol(self) -> GeometricMol:
        """Convert pocket to Smol GeometricMol format"""

        # atoms = ["Se" if el == "SE" else el for el in self.atoms.element.tolist()]
        atoms = [atom.capitalize() for atom in self.atoms.element.tolist()]
        atomics = torch.tensor([PT.atomic_from_symbol(atom) for atom in atoms])
        coords = torch.tensor(self.atoms.coord)
        charges = torch.tensor(self.atoms.charge)

        bonds = self.bonds.as_array().astype(np.int32)
        bond_types = torch.tensor(
            [b if b < 4 else 4 for b in bonds[:, 2]]
        ).long()  # TODO remove this when we have a better way to handle bond types.
        # biotite provides much more sophisticated bond types, like aromatic single, double etc.
        # For now, we just map all aromatic bond types to 4 (aromatic)
        bond_indices = torch.tensor(bonds[:, :2].astype(np.int32)).long()

        mol = GeometricMol(coords, atomics, bond_indices, bond_types, charges=charges)
        return mol

    @property
    def mol(self) -> GeometricMol:
        if self._mol is None:
            self._mol = self.to_geometric_mol()
        return self._mol

    def to_bytes(self) -> bytes:
        dict_repr = {"atoms": self.atoms, "bonds": self.bonds, "str_id": self.str_id}
        byte_obj = pickle.dumps(dict_repr, protocol=PICKLE_PROTOCOL)
        return byte_obj

    def to_prolif(self) -> plf.Molecule:
        # Create a copy of the holo with chain id always only char
        pocket_copy = ProteinPocket(self.atoms, self.bonds)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Write to pdb but don't include bonds since this doesn't seem to be working for pdb files
            write_path = Path(tmp_dir) / "pocket.pdb"
            pocket_copy.write_pdb(write_path, include_bonds=True)

            try:
                # Create prolif molecule by first reading using MDAnalysis
                holo_mda = mda.Universe(str(write_path.resolve()), guess_bonds=True)
                holo_mol = plf.Molecule.from_mda(holo_mda)
            except Exception as e:
                print(f"Error reading PDB file {write_path}: {e}")
                print("Trying to read using RDKit instead of MDAnalysis...")
                protein_mol = Chem.MolFromPDBFile(
                    str(write_path.resolve()), removeHs=False, proximityBonding=True
                )
                holo_mol = plf.Molecule.from_rdkit(protein_mol)

        return holo_mol

    # *** IO functions ***

    def set_coords(self, coords: np.ndarray) -> None:
        atoms = self.atoms.copy()
        bonds = self.bonds.copy()
        atoms.coord = coords
        return ProteinPocket(atoms, bonds, str_id=self.str_id)

    def write_cif(
        self, filepath: Union[str, Path], include_bonds: bool = False
    ) -> None:
        if include_bonds:
            self.atoms.bonds = self.bonds
        else:
            self.atoms.bonds = None

        cif_file = pdbx.CIFFile()
        pdbx.set_structure(cif_file, self.atoms, include_bonds=include_bonds)
        cif_file.write(Path(filepath))

        if include_bonds:
            self.atoms.bonds = None

    def write_pdb(
        self, filepath: Union[str, Path], include_bonds: bool = False
    ) -> None:
        """Note that all chains must have a one character chain id to be a valid pdb file"""
        if len(self.atoms.chain_id[0]) > 1:
            self.atoms.chain_id = np.array([id[-1] for id in self.atoms.chain_id])

        if include_bonds:
            assert (
                self.bonds is not None
            ), "Bonds must be provided to include them in the pdb file as CONECT entries"
            self.atoms.bonds = self.bonds
        else:
            self.atoms.bonds = None

        pdb_file = pdb.PDBFile()
        pdb.set_structure(pdb_file, self.atoms)
        pdb_file.write(Path(filepath))

        if include_bonds:
            self.atoms.bonds = None

    # *** Other helper functions ***

    def copy(self) -> ProteinPocket:
        """Creates a deep copy of this object"""

        atom_copy = self.atoms.copy()
        bond_copy = self.bonds.copy()
        orig_pocket = self.orig_pocket.copy() if self.orig_pocket is not None else None
        mol_copy = self.mol._copy_with()
        str_id_copy = self.str_id[:] if self.str_id is not None else None

        pocket_copy = ProteinPocket(
            atom_copy, bond_copy, mol_copy, str_id=str_id_copy, orig_pocket=orig_pocket
        )
        return pocket_copy

    def _copy_with(
        self,
        atoms: AtomArray = None,
        bonds: BondList = None,
        mol: GeometricMol = None,
        orig_pocket: ProteinPocket = None,
        com: Optional[torch.Tensor] = None,
    ) -> ProteinPocket:
        atoms = self.atoms.copy() if atoms is None else atoms
        bonds = self.bonds.copy() if bonds is None else bonds
        mol = self.mol._copy_with() if mol is None else mol
        orig_pocket = (
            self.orig_pocket.copy()
            if orig_pocket is None and self.orig_pocket is not None
            else orig_pocket
        )

        pocket_copy = ProteinPocket(
            atoms, bonds, mol, str_id=self.str_id, orig_pocket=orig_pocket
        )
        return pocket_copy

    def _check_atom_array(self, atoms: AtomArray) -> None:
        annotations = atoms.get_annotation_categories()

        # coord doesn't exist in annotations but should always be in atom array
        # so no need to check for coords

        # Check required annotations are provided
        _check_dict_key(annotations, "res_name", "atom array")
        _check_dict_key(annotations, "element", "atom array")


class PocketComplex:
    def __init__(
        self,
        ligand: GeometricMol,
        holo: Optional[ProteinPocket] = None,
        apo: Optional[ProteinPocket] = None,
        interactions: Optional[np.ndarray] = None,
        metadata: Optional[dict] = None,
        fragment_mask: Optional[torch.Tensor] = None,
        fragment_mode: Optional[str] = None,
        com: Optional[torch.Tensor] = None,
    ):
        assert (
            holo is not None or apo is not None
        ), "Either holo or apo must be provided"

        # Make sure ligand is a GeometricMol
        _check_type(ligand, GeometricMol, "ligand")
        if holo is not None:
            _check_type(holo, ProteinPocket, "holo")
        if apo is not None:
            _check_type(apo, ProteinPocket, "apo")

        metadata = {} if metadata is None else metadata

        if apo is not None and holo is not None:
            PocketComplex._check_holo_apo_match(holo, apo)
        if interactions is not None and holo is not None:
            PocketComplex._check_interactions(interactions, holo, ligand)

        self.ligand = ligand
        self.holo = holo
        self.apo = apo
        self.interactions = interactions
        self.metadata = metadata
        self.fragment_mask = fragment_mask
        self.fragment_mode = fragment_mode
        self.com = com

    def __len__(self) -> int:
        return self.seq_length

    @property
    def seq_length(self) -> int:
        pocket_len = len(self.holo) if self.holo is not None else len(self.apo)
        return pocket_len + len(self.ligand)

    @property
    def system_id(self) -> str:
        return self.metadata.get("system_id")

    @property
    def str_id(self) -> str:
        return self.ligand.str_id

    @property
    def split(self) -> str:
        return self.metadata.get("split")

    @property
    def is_covalent(self) -> bool:
        return self.metadata.get("is_covalent")

    @property
    def apo_type(self) -> str:
        return self.metadata.get("apo_type")

    def to_bytes(self) -> bytes:
        dict_repr = {
            "holo": self.holo.to_bytes(),
            "ligand": self.ligand.to_bytes(),
            "metadata": self.metadata,
        }

        if self.interactions is not None:
            dict_repr["interactions"] = pickle.dumps(
                self.interactions, protocol=PICKLE_PROTOCOL
            )

        if self.apo is not None:
            dict_repr["apo"] = self.apo.to_bytes()

        byte_obj = pickle.dumps(dict_repr, protocol=PICKLE_PROTOCOL)
        return byte_obj

    @staticmethod
    def from_bytes(
        data: bytes,
        remove_hs: bool = False,
        remove_aromaticity: bool = False,
        skip_non_valid: bool = False,
    ) -> PocketComplex:
        obj = pickle.loads(data)

        _check_dict_key(obj, "holo")
        _check_dict_key(obj, "ligand")
        _check_dict_key(obj, "metadata")

        holo = ProteinPocket.from_bytes(obj["holo"], remove_hs=False)
        ligand = GeometricMol.from_bytes(
            obj["ligand"],
            remove_hs=False,
            keep_orig_data=True,
            skip_non_valid=skip_non_valid,
        )
        if ligand is None:
            return
        apo = (
            ProteinPocket.from_bytes(obj["apo"]) if obj.get("apo") is not None else None
        )

        if obj.get("interactions") is not None:
            try:
                interactions = BindingInteractions.from_bytes(
                    obj["interactions"], interaction_types=PROLIF_INTERACTIONS
                )
                interactions_arr = interactions.array
            except Exception:
                interactions_arr = pickle.loads(obj["interactions"])
                assert isinstance(
                    interactions_arr, np.ndarray
                ), "Interactions must be a numpy array if not provided as bytes"
        else:
            interactions_arr = None

        system = PocketComplex(
            ligand=ligand,
            holo=holo,
            apo=apo,
            interactions=interactions_arr,
            metadata=obj["metadata"],
        )
        system = system.remove_hs(
            include_ligand=remove_hs, remove_aromaticity=remove_aromaticity
        )

        return system

    def remove_hs(
        self,
        include_ligand: Optional[bool] = False,
        remove_aromaticity: Optional[bool] = False,
    ) -> PocketComplex:
        """Remove hydrogen atoms, by default Hs are only removed from the protein."""

        if include_ligand:
            ligand_mask = np.array([bool(a != 1) for a in self.ligand._atomics])
            ligand = self.ligand.remove_hs(remove_aromaticity=remove_aromaticity)
        else:
            ligand_mask = None
            ligand = self.ligand

        holo_subset = None
        if self.holo is not None:
            protein_atom_mask = self.holo.atoms.element != "H"
            holo_subset = self.holo.select_atoms(
                protein_atom_mask, keep_orig_pocket=True
            )

        apo_subset = None
        if self.apo is not None:
            protein_atom_mask = self.apo.atoms.element != "H"
            apo_subset = self.apo.select_atoms(protein_atom_mask, keep_orig_pocket=True)

        interactions_subset = None
        if self.interactions is not None:
            interactions_subset = self.interactions[protein_atom_mask]
            if ligand_mask is not None:
                interactions_subset = interactions_subset[:, ligand_mask, :]

        subset = PocketComplex(
            ligand=ligand,
            holo=holo_subset,
            apo=apo_subset,
            interactions=interactions_subset,
            metadata=self.metadata,
        )
        return subset

    def store_metrics_(self):
        self.metadata["metrics"] = calc_system_metrics(self)

    @staticmethod
    def _check_holo_apo_match(holo: ProteinPocket, apo: ProteinPocket):

        # Check sizes match
        if len(holo) != len(apo):
            raise ValueError(
                f"Apo and holo must have the same number of atoms, got {len(apo)} and {len(holo)}"
            )

        # Check atom names match
        if not (holo.atoms.atom_name == apo.atoms.atom_name).all():
            raise ValueError("All apo and holo atom names must match.")

        # Check bonds match
        if not (holo.bonds.as_array() == apo.bonds.as_array()).all():
            raise ValueError("All apo and holo bonds must match.")

    @staticmethod
    def _check_interactions(interactions_arr, holo, ligand):

        int_shape = tuple(interactions_arr.shape)

        if int_shape[0] != len(holo):
            err = f"Dim 0 of interactions must match the length of the holo pocket, got {int_shape[0]} and {len(holo)}"
            raise ValueError(err)

        if int_shape[1] != len(ligand):
            err = f"Dim 1 of interactions must match the length of the ligand, got {int_shape[0]} and {len(ligand)}"
            raise ValueError(err)

    ######## *** PocketComplex state functions *** ########

    def coords(self, state="holo") -> _T:
        if state == "holo":
            return torch.cat((self.holo.mol.coords, self.ligand.coords), dim=0).float()
        elif state == "apo":
            return torch.cat((self.apo.mol.coords, self.ligand.coords), dim=0).float()
        else:
            raise ValueError(f"Unknown state {state}")

    def atomics(self, state="holo") -> _T:
        if state == "holo":
            return torch.cat(
                (self.holo.mol.atomics, self.ligand.atomics), dim=0
            ).float()
        elif state == "apo":
            return torch.cat((self.apo.mol.atomics, self.ligand.atomics), dim=0).float()
        else:
            raise ValueError(f"Unknown state {state}")

    def charges(self, state="holo") -> _T:
        if state == "holo":
            return torch.cat(
                (self.holo.mol.charges, self.ligand.charges), dim=0
            ).float()
        elif state == "apo":
            return torch.cat((self.apo.mol.charges, self.ligand.charges), dim=0).float()
        else:
            raise ValueError(f"Unknown state {state}")

    def hybridization(self, state="holo") -> _T:
        ligand_hybridization = self.ligand.hybridization
        if ligand_hybridization is None or len(ligand_hybridization.size()) == 1:
            # Either not provided or untransformed, so return empty tensor
            return
        n_hybrid = ligand_hybridization.shape[1]
        if state == "holo":
            pocket_hybridization = torch.zeros(self.holo.seq_length, n_hybrid).long()
            return torch.cat((pocket_hybridization, ligand_hybridization), dim=0).long()
        elif state == "apo":
            pocket_hybridization = torch.zeros(self.apo.seq_length, n_hybrid).long()
            return torch.cat((pocket_hybridization, ligand_hybridization), dim=0).long()
        else:
            raise ValueError(f"Unknown state {state}")

    def _validate_affinity_value(self, key: str) -> _T:
        """Helper function to validate and convert affinity values to tensors."""
        if key in self.metadata and self.metadata[key] is not None:
            value = self.metadata[key]
            if isinstance(value, str):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    return torch.tensor(float("nan")).float()
            if not np.isfinite(value):
                return torch.tensor(float("nan")).float()
            return torch.tensor(value).float()
        return torch.tensor(float("nan")).float()

    def pic50(self) -> _T:
        return self._validate_affinity_value("pic50")

    def pki(self) -> _T:
        return self._validate_affinity_value("pki")

    def pkd(self) -> _T:
        return self._validate_affinity_value("pkd")

    def pec50(self) -> _T:
        return self._validate_affinity_value("pec50")

    def kiba_score(self) -> _T:
        if "kiba_score" in self.metadata and self.metadata["kiba_score"] is not None:
            return torch.tensor(self.metadata["kiba_score"]).float()
        return torch.tensor(float("nan")).float()

    def gnina_score(self) -> _T:
        if "gnina_score" in self.metadata and self.metadata["gnina_score"] is not None:
            return torch.tensor(self.metadata["gnina_score"]).float()
        return torch.tensor(float("nan")).float()

    def vina_score(self) -> _T:
        if "vina_score" in self.metadata and self.metadata["vina_score"] is not None:
            return torch.tensor(self.metadata["vina_score"]).float()
        return torch.tensor(float("nan")).float()

    def glide_score(self) -> _T:
        if "glide_score" in self.metadata and self.metadata["glide_score"] is not None:
            return torch.tensor(self.metadata["glide_score"]).float()
        return torch.tensor(float("nan")).float()

    def bond_indices(self, state="holo") -> _T:
        if state == "holo":
            return torch.cat(
                (
                    self.holo.mol.bond_indices,
                    self.ligand.bond_indices + self.holo.mol.seq_length,
                ),
                dim=0,
            ).long()
        elif state == "apo":
            return torch.cat(
                (
                    self.apo.mol.bond_indices,
                    self.ligand.bond_indices + self.apo.mol.seq_length,
                ),
                dim=0,
            ).long()
        else:
            raise ValueError(f"Unknown state {state}")

    def bond_types(self, state="holo") -> _T:
        if state == "holo":
            return torch.cat(
                (self.holo.mol.bond_types, self.ligand.bond_types), dim=0
            ).long()
        elif state == "apo":
            return torch.cat(
                (self.apo.mol.bond_types, self.ligand.bond_types), dim=0
            ).long()
        else:
            raise ValueError(f"Unknown state {state}")

    def atom_names(self, state="holo") -> _T:
        lig_atom_names = torch.ones(self.ligand.seq_length).long()
        if state == "holo":
            return torch.cat((self.holo.atom_names, lig_atom_names), dim=0).long()
        elif state == "apo":
            return torch.cat((self.apo.atom_names, lig_atom_names), dim=0)
        else:
            raise ValueError(f"Unknown state {state}")

    def res_names(self, state="holo") -> _T:
        lig_res_names = torch.ones(self.ligand.seq_length).long()
        if state == "holo":
            return torch.cat((self.holo.res_names, lig_res_names), dim=0).long()
        elif state == "apo":
            return torch.cat((self.apo.res_names, lig_res_names), dim=0).long()
        else:
            raise ValueError(f"Unknown state {state}")

    def atoms_list(self, state="holo") -> list[str]:
        if state == "holo":
            return self.holo.atoms
        elif state == "apo":
            return self.apo.atoms
        else:
            raise ValueError(f"Unknown state {state}")

    def bonds_list(self, state="holo") -> list[int]:
        if state == "holo":
            return self.holo.bonds
        elif state == "apo":
            return self.apo.bonds
        else:
            raise ValueError(f"Unknown state {state}")

    def lig_mask(self, state="holo") -> _T:
        if state == "holo":
            return torch.cat(
                (torch.zeros(self.holo.seq_length), torch.ones(self.ligand.seq_length)),
                dim=0,
            ).long()
        elif state == "apo":
            return torch.cat(
                (torch.zeros(self.apo.seq_length), torch.ones(self.ligand.seq_length)),
                dim=0,
            ).long()
        else:
            raise ValueError(f"Unknown state {state}")

    def pocket_mask(self, state="holo") -> _T:
        if state == "holo":
            return torch.cat(
                (torch.ones(self.holo.seq_length), torch.zeros(self.ligand.seq_length)),
                dim=0,
            ).long()
        elif state == "apo":
            return torch.cat(
                (torch.ones(self.apo.seq_length), torch.zeros(self.ligand.seq_length)),
                dim=0,
            ).long()
        else:
            raise ValueError(f"Unknown state {state}")

    ######## *** PocketComplex geometric functions *** ########

    def rotate(self):
        from scipy.spatial.transform import Rotation

        ligand = self.ligand
        holo_mol = self.holo.mol
        R = Rotation.random()
        lig_coords = torch.tensor(R.apply(ligand.coords), dtype=torch.float32)
        holo_coords = torch.tensor(R.apply(holo_mol.coords), dtype=torch.float32)
        holo_mol = holo_mol._copy_with(coords=holo_coords)
        holo = self.holo._copy_with(mol=holo_mol)
        ligand = ligand._copy_with(coords=lig_coords)
        return PocketComplex(
            ligand,
            holo=holo,
            interactions=self.interactions,
            metadata=self.metadata,
            com=self.com,
        )

    def move_holo_and_lig_to_holo_com(self):
        """
        Move holo and ligand structure to holo center of mass (COM).
        ONLY needed for holo conditioned flow matching. Align ligand to holo COM.
        """

        ligand = self.ligand
        holo_mol = self.holo.mol
        holo_com = holo_mol.com.unsqueeze(0)
        shifted_holo = holo_mol.coords - holo_com
        shifted_lig = ligand.coords - holo_com
        holo_mol = holo_mol._copy_with(coords=shifted_holo)
        holo = self.holo._copy_with(mol=holo_mol, com=holo_com)
        ligand = ligand._copy_with(coords=shifted_lig, com=holo_com)
        return PocketComplex(
            ligand,
            holo=holo,
            interactions=self.interactions,
            metadata=self.metadata,
            com=holo_com,
        )

    def move_holo_and_lig_to_holo_lig_com(self):
        """
        Move holo and ligand structure to center of mass (COM) of holo and ligand.
        ONLY needed for full random complex flow matching! Align both holo and ligand to common COM.
        """
        holo_mol = self.holo.mol
        ligand = self.ligand
        common_com = self.coords(state="holo").sum(dim=0) / (self.seq_length)
        shifted_holo = holo_mol.coords - common_com
        shifted_lig = ligand.coords - common_com
        holo_mol = holo_mol._copy_with(coords=shifted_holo)
        holo = self.holo._copy_with(mol=holo_mol, com=common_com)
        ligand = ligand._copy_with(coords=shifted_lig, com=common_com)
        return PocketComplex(
            ligand,
            holo=holo,
            interactions=self.interactions,
            metadata=self.metadata,
            com=common_com,
        )

    def move_apo_and_holo_and_lig_to_apo_com(self):
        """
        Move apo, holo and ligand structure to apo center of mass (COM).
        ONLY needed for apo-holo flow matching. Align holo and ligand to apo COM.
        """

        ligand = self.ligand
        apo_mol, holo_mol = self.apo.mol, self.holo.mol
        apo_com = apo_mol.com.unsqueeze(0)
        shifted_apo = apo_mol.coords - apo_com
        shifted_holo = holo_mol.coords - apo_com
        shifted_lig = ligand.coords - apo_com
        apo_mol = apo_mol._copy_with(coords=shifted_apo)
        holo_mol = holo_mol._copy_with(coords=shifted_holo)
        apo = self.apo._copy_with(mol=apo_mol)
        holo = self.holo._copy_with(mol=holo_mol, com=apo_com)
        ligand = ligand._copy_with(coords=shifted_lig, com=apo_com)
        return PocketComplex(
            ligand,
            holo=holo,
            apo=apo,
            interactions=self.interactions,
            metadata=self.metadata,
            com=apo_com,
        )

    def move_apo_and_lig_to_apo_com(self):
        """
        Move (random/interpolated) pocket and ligand structure to pocket center of mass (COM).
        ONLY needed for pocket conditioned flow matching. Align ligand to pocket COM.
        """
        ligand = self.ligand
        apo_mol = self.apo.mol
        apo_com = apo_mol.com.unsqueeze(0)
        shifted_apo = apo_mol.coords - apo_com
        shifted_lig = ligand.coords - apo_com
        apo_mol = apo_mol._copy_with(coords=shifted_apo)
        apo = self.apo._copy_with(mol=apo_mol)
        ligand = ligand._copy_with(coords=shifted_lig)
        return PocketComplex(
            ligand,
            apo=apo,
            interactions=self.interactions,
            metadata=self.metadata,
            com=apo_com,
        )

    def move_apo_and_lig_to_apo_lig_com(self):
        """
        Move (random/interpolated) pocket and ligand structure to center of mass (COM) of (random/interpolated) and ligand.
        ONLY needed for full random complex flow matching! Align both ligand and pocket to common COM.
        """
        ligand = self.ligand
        apo_mol = self.apo.mol
        common_com = self.coords(state="apo").sum(dim=0) / (self.seq_length)
        shifted_apo = apo_mol.coords - common_com
        shifted_lig = ligand.coords - common_com
        apo_mol = apo_mol._copy_with(coords=shifted_apo)
        apo = self.apo._copy_with(mol=apo_mol)
        ligand = ligand._copy_with(coords=shifted_lig)
        return PocketComplex(
            ligand,
            apo=apo,
            interactions=self.interactions,
            metadata=self.metadata,
            com=common_com,
        )

    def move_lig_to_apo_com(self):
        """
        Move ligand structure to apo center of mass (COM).
        ONLY needed for apo conditioned flow matching. Align ligand to apo COM.
        """

        ligand = self.ligand
        apo_coords = torch.tensor(self.apo.atoms.coord)
        apo_mol = self.apo.mol._copy_with(coords=apo_coords)
        apo_com = apo_mol.com.unsqueeze(0)
        shifted_lig = ligand.coords - apo_com
        ligand = ligand._copy_with(coords=shifted_lig)
        return PocketComplex(
            ligand,
            apo=self.apo,
            interactions=self.interactions,
            metadata=self.metadata,
            com=apo_com,
        )

    # *** Interface Methods ***

    def write_complex_pdb(
        self, filepath: Union[str, Path], obabel=False, pymol=True
    ) -> None:
        """Write the complex to a pdb file"""
        with tempfile.NamedTemporaryFile(suffix=".pdb") as tmp:
            pocket_pdb = tmp.name
            self.holo.write_pdb(pocket_pdb, include_bonds=True)
            with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
                lig_sdf = tmp.name
                self.ligand.write_sdf(lig_sdf)
                lig_pdb = Path(lig_sdf.replace(".sdf", ".pdb"))
                if obabel:
                    import os

                    cmd = f"obabel {lig_sdf} -O {lig_pdb}"
                    os.system(cmd)
                    cmd = f"obabel {pocket_pdb} {lig_pdb} -O {filepath}"
                    os.system(cmd)
                elif pymol:
                    from pymol import cmd

                    cmd.reinitialize()
                    cmd.load(pocket_pdb, "protein")
                    cmd.load(lig_sdf, "ligand")
                    cmd.save(filepath, "protein or ligand")
                    cmd.reinitialize()

    def copy(self) -> PocketComplex:
        """Creates a deep copy of this object"""
        assert (
            self.holo is not None or self.apo is not None
        ), "Cannot copy complex without holo and apo"
        assert self.ligand is not None, "Cannot copy complex without ligand"

        ligand_copy = self.ligand.copy()
        holo_copy = self.holo.copy() if self.holo is not None else None
        apo_copy = self.apo.copy() if self.apo is not None else None

        interactions_copy = (
            np.copy(self.interactions) if self.interactions is not None else None
        )

        metadata_copy = copy.deepcopy(self.metadata)

        return PocketComplex(
            ligand_copy,
            holo=holo_copy,
            apo=apo_copy,
            interactions=interactions_copy,
            metadata=metadata_copy,
        )

    def _copy_with(
        self,
        ligand: GeometricMol = None,
        holo: ProteinPocket = None,
        apo: ProteinPocket = None,
        interactions: np.ndarray = None,
    ) -> PocketComplex:
        ligand_copy = self.ligand.copy() if ligand is None else ligand
        holo_copy = self.holo.copy() if holo is None and self.holo is not None else holo
        apo_copy = self.apo.copy() if apo is None and self.apo is not None else apo

        if interactions is None:
            interactions_copy = (
                np.copy(self.interactions) if self.interactions is not None else None
            )
        else:
            interactions_copy = interactions
        metadata_copy = copy.deepcopy(self.metadata)
        com_copy = self.com.clone() if self.com is not None else None

        return PocketComplex(
            ligand_copy,
            holo=holo_copy,
            apo=apo_copy,
            interactions=interactions_copy,
            metadata=metadata_copy,
            com=com_copy,
        )

    def update_mols_(
        self,
        ligand: GeometricMol = None,
        holo_mol: GeometricMol = None,
        apo_mol: GeometricMol = None,
    ):
        if ligand is not None:
            self.ligand = ligand
        if holo_mol is not None:
            self.holo.mol = holo_mol
        if apo_mol is not None:
            self.apo.mol = apo_mol


# **********************************
# ***  PocketComplexBatch class  ***
# **********************************


class PocketComplexBatch(Sequence):
    def __init__(self, systems: list[PocketComplex]):
        for system in systems:
            _check_type(system, PocketComplex, "system")

        self._systems = systems
        self._device = None

    @property
    def system_id(self) -> list[str]:
        return [system.system_id for system in self._systems]

    @property
    def retrieve_ligands(self) -> GeometricMolBatch:
        ligands = [system.ligand.copy() for system in self._systems]
        return GeometricMolBatch.from_list(ligands)

    def coords(self, state="apo") -> _T:
        return smolF.pad_tensors(
            [system.coords(state=state) for system in self._systems]
        )

    def atomics(self, state="apo") -> _T:
        return smolF.pad_tensors(
            [system.atomics(state=state) for system in self._systems]
        )

    def charges(self, state="apo") -> _T:
        return smolF.pad_tensors(
            [system.charges(state=state) for system in self._systems]
        )

    def hybridization(self, state="apo") -> _T:
        if self._systems[0].hybridization(state=state) is None:
            return
        return smolF.pad_tensors(
            [system.hybridization(state=state) for system in self._systems]
        ).float()

    def affinity(self) -> TensorDict:
        """Returns all affinity values for the batch as a TensorDict.
        Note:
            This method returns all available affinity values for each system in the batch.
            If a value is not available for a system, it will return NaN for that affinity type.
        Returns:
            TensorDict: A TensorDict containing all affinity values with batch dimension.
        """
        # Directly stack individual values - much more efficient
        affinity_batch = TensorDict(
            {
                "pic50": torch.stack([system.pic50() for system in self._systems]),
                "pkd": torch.stack([system.pkd() for system in self._systems]),
                "pki": torch.stack([system.pki() for system in self._systems]),
                "pec50": torch.stack([system.pec50() for system in self._systems]),
            },
            batch_size=[self.batch_size],
        )

        return affinity_batch

    def docking_score(self) -> TensorDict:
        """Returns all docking scores for the batch as a TensorDict.
        Note:
            This method returns all available docking scores for each system in the batch.
            If a value is not available for a system, it will return NaN for that docking score type.
        Returns:
            TensorDict: A TensorDict containing all docking scores with batch dimension.
        """
        # Directly stack individual values - much more efficient
        docking_batch = TensorDict(
            {
                "vina_score": torch.stack(
                    [system.vina_score() for system in self._systems]
                ),
                "gnina_score": torch.stack(
                    [system.gnina_score() for system in self._systems]
                ),
                "glide_score": torch.stack(
                    [system.glide_score() for system in self._systems]
                ),
            },
            batch_size=[self.batch_size],
        )

        return docking_batch

    def adjacency(self, state="apo") -> _T:
        n_atoms = max(self.seq_length)
        adjs = [
            smolF.adj_from_edges(
                system.bond_indices(state=state),
                system.bond_types(state=state),
                n_atoms,
                symmetric=True,
            )
            for system in self._systems
        ]
        return torch.stack(adjs)

    def interactions(self, state="apo") -> _T:
        if self._systems[0].interactions is not None:
            n_interaction = self._systems[0].interactions.shape[-1]
            if state == "apo":
                n_pocket = max(
                    [self.system.apo.seq_length for self.system in self._systems]
                )
            else:
                n_pocket = max(
                    [self.system.holo.seq_length for self.system in self._systems]
                )
            n_ligand = max(
                [self.system.ligand.seq_length for self.system in self._systems]
            )
            interactions = []
            for i, system in enumerate(self._systems):
                interaction = torch.zeros(n_pocket, n_ligand, n_interaction)
                interaction[
                    : system.interactions.shape[0], : system.interactions.shape[1], :
                ] = system.interactions
                interactions.append(interaction)
            return (
                torch.stack(interactions).permute(0, 2, 1, 3).long()
            )  # (batch, n_lig, n_pocket)

        else:
            return []

    def atom_names(self, state="apo") -> _T:
        return smolF.pad_tensors(
            [system.atom_names(state=state) for system in self._systems]
        )

    def res_names(self, state="apo") -> _T:
        return smolF.pad_tensors(
            [system.res_names(state=state) for system in self._systems]
        )

    def atoms_list(self, state="apo") -> list[list[str]]:
        return [system.atoms_list(state=state) for system in self._systems]

    def bonds_list(self, state="apo") -> list[list[int]]:
        return [system.bonds_list(state=state) for system in self._systems]

    def lig_mask(self, state="apo") -> _T:
        return smolF.pad_tensors(
            [system.lig_mask(state=state) for system in self._systems]
        ).long()

    def pocket_mask(self, state="apo") -> _T:
        return smolF.pad_tensors(
            [system.pocket_mask(state=state) for system in self._systems]
        ).long()

    def fragment_mask(self) -> _T:
        if self._systems[0].fragment_mask is None:
            return []
        return smolF.pad_tensors(
            [system.fragment_mask for system in self._systems]
        ).long()

    def fragment_mode(self) -> Optional[str]:
        if self._systems[0].fragment_mode is None:
            return []
        return [system.fragment_mode for system in self._systems]

    @property
    def mask(self) -> _T:
        return smolF.pad_tensors(
            [torch.ones(system.seq_length) for system in self._systems]
        ).long()

    @property
    def seq_length(self) -> list[int]:
        return [system.seq_length for system in self._systems]

    @property
    def batch_size(self) -> int:
        return len(self._systems)

    def remove(self, indices: list[int]):
        self._systems.pop(indices)

    def append(self, other: PocketComplexBatch):
        systems = self._systems + other._systems
        return systems

    def split(self, n_chunks):
        chunk_size = self.batch_size // n_chunks
        remainder = self.batch_size % n_chunks
        chunks = []
        start = 0
        for i in range(n_chunks):
            chunk_end = start + chunk_size + (1 if i < remainder else 0)
            chunks.append(self._systems[start:chunk_end])
            start = chunk_end
        return chunks

    # *** Sequence methods ***

    def __len__(self) -> int:
        return self.batch_size

    def __getitem__(self, item: int) -> PocketComplex:
        return self._systems[item]

    # *** Helper methods ***
    def remove_hs(self, include_ligand: Optional[bool] = False):
        subset_systems = [
            system.remove_hs(include_ligand=include_ligand) for system in self._systems
        ]
        return PocketComplexBatch(subset_systems)

    # *** IO methods ***

    def to_rdkit(self, vocab, kekulize: bool = False) -> list[Chem.Mol]:
        return [
            system.ligand.to_rdkit(vocab, kekulize=kekulize) for system in self._systems
        ]

    def to_bytes(self) -> bytes:
        system_bytes = [system.to_bytes() for system in self._systems]
        return pickle.dumps(system_bytes)

    def to_sdf(
        self, vocab, save_path: Union[str, Path], kekulize: bool = False
    ) -> None:
        ligands = defaultdict(list)
        for system in self._systems:
            pdb_file = Path(save_path) / (system.metadata["system_id"] + "_with_hs.pdb")
            if not pdb_file.exists():
                coords = system.holo.orig_pocket.mol.coords
                system.holo.orig_pocket.set_coords(coords.cpu().numpy()).write_pdb(
                    pdb_file, include_bonds=True
                )
            lig = system.ligand.to_rdkit(vocab=vocab, kekulize=kekulize)
            ligands[system.system_id].append(lig)
        for target, ligs in ligands.items():
            lig_path = Path(save_path, f"{target}.sdf")
            if not lig_path.exists():
                writer = Chem.SDWriter(lig_path)
                for lig in ligs:
                    writer.write(lig)
                writer.close()
            else:
                # load existing sdf file and append new ligands
                suppl = Chem.SDMolSupplier(str(lig_path))
                writer = Chem.SDWriter(lig_path)
                for lig in suppl:
                    writer.write(lig)
                for lig in ligs:
                    writer.write(lig)
                writer.close()

    @staticmethod
    def from_bytes(
        data: bytes,
        remove_hs: bool = False,
        remove_aromaticity: bool = False,
        skip_non_valid: bool = False,
    ) -> PocketComplexBatch:
        systems = [
            PocketComplex.from_bytes(
                system,
                remove_hs=remove_hs,
                remove_aromaticity=remove_aromaticity,
                skip_non_valid=skip_non_valid,
            )
            for system in tqdm(pickle.loads(data), desc="Loading complexes", ascii=True)
        ]
        full_len = len(systems)
        systems = [system for system in systems if system is not None]
        print(f"Failed to load {full_len-len(systems)} complexes.")
        return PocketComplexBatch(systems)

    @staticmethod
    def from_list(systems: list[PocketComplex]) -> PocketComplexBatch:
        return PocketComplexBatch(systems)

    @staticmethod
    def from_sdf(
        pdb_file: Union[str, Path],
        sdf_path: Union[str, Path],
        remove_hs: bool,
        protonate_pocket: bool,
        cut_pocket: bool,
        pocket_cutoff: float,
        compute_interactions: bool = True,
        pocket_type: str = "holo",
        n_workers: int = None,
    ) -> PocketComplexBatch:
        import multiprocessing as mp
        from functools import partial

        from flowr.data.preprocess_pdbs import process_complex

        sdf_files = (
            Path(sdf_path).glob("*.sdf")
            if Path(sdf_path).is_dir()
            else [Path(sdf_path)]
        )

        # Collect all molecules from all SDF files
        all_mols = []
        for sdf in sdf_files:
            suppl = Chem.SDMolSupplier(str(sdf), removeHs=False)
            for mol in suppl:
                if mol is not None:
                    all_mols.append(mol)

        if not compute_interactions:
            # Single-threaded processing when interactions are not computed
            systems = []
            for mol in tqdm(all_mols, desc="Processing molecules"):
                system = process_complex(
                    pdb_path=pdb_file,
                    ligand_mol=mol,
                    add_bonds_to_protein=True,
                    add_hs_to_protein=protonate_pocket,
                    pocket_cutoff=pocket_cutoff,
                    cut_pocket=cut_pocket,
                    pocket_type=pocket_type,
                    compute_interactions=False,
                )
                if system is not None:
                    system = system.remove_hs(include_ligand=remove_hs)
                    systems.append(system)
        else:
            # Multiprocessing when interactions need to be computed
            if n_workers is None:
                n_workers = min(mp.cpu_count(), len(all_mols))
            # Create partial function with fixed arguments
            process_func = partial(
                _process_single_complex,
                pdb_path=pdb_file,
                add_bonds_to_protein=True,
                add_hs_to_protein=protonate_pocket,
                pocket_cutoff=pocket_cutoff,
                cut_pocket=cut_pocket,
                pocket_type=pocket_type,
                compute_interactions=True,
                remove_hs=remove_hs,
            )

            # Process molecules in parallel
            with mp.Pool(n_workers) as pool:
                results = list(
                    tqdm(
                        pool.imap(process_func, all_mols),
                        total=len(all_mols),
                        desc=f"Processing molecules with {n_workers} workers",
                    )
                )

            # Filter out None results
            systems = [system for system in results if system is not None]

        print(
            f"Processed {len(systems)} complexes from SDF files. Failed: {len(all_mols) - len(systems)}"
        )
        return PocketComplexBatch(systems)

    def to_list(self):
        return self._systems

    def to_dict(self, state="apo"):
        return {
            "coords": self.coords(state=state),
            "atomics": self.atomics(state=state),
            "charges": self.charges(state=state),
            "bonds": self.bonds(state=state),
            "atom_names": self.atom_names(state=state),
            "res_names": self.res_names(state=state),
            "lig_mask": self.lig_mask(state=state),
            "pocket_mask": self.pocket_mask(state=state),
            "mask": self.mask,
        }

    # *** Other methods ***

    def _copy(self) -> PocketComplexBatch:
        return PocketComplexBatch(self._systems.copy())

    @staticmethod
    def from_batches(batches: list[PocketComplexBatch]) -> PocketComplexBatch:
        all_systems = [system for batch in batches for system in batch]
        return PocketComplexBatch(all_systems)


def _process_single_complex(
    ligand_mol,
    pdb_path,
    add_bonds_to_protein,
    add_hs_to_protein,
    pocket_cutoff,
    cut_pocket,
    pocket_type,
    compute_interactions,
    remove_hs,
):
    """Helper function for multiprocessing complex creation."""
    try:
        from flowr.data.preprocess_pdbs import process_complex

        system = process_complex(
            pdb_path=pdb_path,
            ligand_mol=ligand_mol,
            add_bonds_to_protein=add_bonds_to_protein,
            add_hs_to_protein=add_hs_to_protein,
            pocket_cutoff=pocket_cutoff,
            cut_pocket=cut_pocket,
            pocket_type=pocket_type,
            compute_interactions=compute_interactions,
        )

        if system is not None:
            return system.remove_hs(include_ligand=remove_hs)
        return None

    except Exception as e:
        print(f"Error processing ligand: {e}")
        return None

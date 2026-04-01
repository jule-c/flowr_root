import torch
from rdkit import Chem, RDLogger

# Add these imports at the top of interpolate.py
from rdkit.Geometry import Point3D

import flowr.util.functional as smolF
from flowr.constants import ATOM_ENCODER as atom_encoder

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

bond_dict = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


class BuildMolecule:
    def __init__(
        self,
        molecule,
        remove_hs=False,
        kekulize=False,
    ):
        self.positions = molecule["coords"]
        self.atom_types = molecule["atomics"].long()
        self.charges = molecule["charges"]
        self.bond_indices = molecule["bond_indices"]
        self.bond_types = molecule["bond_types"]
        self.device = molecule["device"]
        self.smiles = molecule["id"]

        if len(molecule["bond_types"].shape) == 1:
            n_atoms = len(self.positions)
            adj = smolF.adj_from_edges(
                self.bond_indices,
                self.bond_types,
                n_atoms,
                symmetric=True,
            )
            self.adj = adj.long()

        self.rdkit_mol = self.build_rdkit_molecule()
        if remove_hs:
            self.rdkit_mol = Chem.RemoveHs(self.rdkit_mol)
            if kekulize:
                Chem.Kekulize(self.rdkit_mol, clearAromaticFlags=True)
            self.positions = torch.from_numpy(
                self.rdkit_mol.GetConformer().GetPositions()
            )
            self.atom_types = torch.tensor(
                [a.GetAtomicNum() for a in self.rdkit_mol.GetAtoms()]
            )
            self.charges = torch.tensor(
                [a.GetFormalCharge() for a in self.rdkit_mol.GetAtoms()]
            )
            adj = torch.from_numpy(
                Chem.rdmolops.GetAdjacencyMatrix(self.rdkit_mol, useBO=True)
            )
            bond_indices = adj.nonzero().contiguous().T
            bond_indices = bond_indices[:, bond_indices[0] < bond_indices[1]]
            bond_types = adj[bond_indices[0], bond_indices[1]]
            bond_types[bond_types == 1.5] = 4
            self.bond_indices = bond_indices.T
            self.bond_types = bond_types.long()

        self.molecule = {
            "coords": self.positions,
            "atomics": self.atom_types,
            "charges": self.charges,
            "bond_types": self.bond_types,
            "bond_indices": self.bond_indices,
            "device": self.device,
            "id": self.smiles,
        }

    def build_rdkit_molecule(self):

        mol = Chem.RWMol()

        for atom, charge in zip(self.atom_types, self.charges):
            if atom == -1:
                continue
            try:
                a = Chem.Atom(atom.item())
            except Exception:
                raise ("Failed to retrieve atom type")
            if charge.item() != 0:
                a.SetFormalCharge(charge.item())
            mol.AddAtom(a)

        edge_types = torch.triu(self.adj, diagonal=1)
        edge_types[edge_types == -1] = 0
        all_bonds = torch.nonzero(edge_types)
        for i, bond in enumerate(all_bonds):
            if bond[0].item() != bond[1].item():
                mol.AddBond(
                    bond[0].item(),
                    bond[1].item(),
                    bond_dict[edge_types[bond[0], bond[1]].item()],
                )

        try:
            mol = mol.GetMol()
        except Chem.KekulizeException:
            print("Can't kekulize molecule")
            return

        # Set coordinates
        positions = self.positions.double()
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(
                i,
                Point3D(
                    positions[i][0].item(),
                    positions[i][1].item(),
                    positions[i][2].item(),
                ),
            )
        mol.AddConformer(conf)
        return mol


class Molecule:
    def __init__(
        self,
        rdkit_mol,
        device="cpu",
    ):
        self.rdkit_mol = rdkit_mol
        if rdkit_mol is not None:
            positions = (
                torch.tensor(rdkit_mol.GetConformers()[0].GetPositions())
                .float()
                .to(device)
            )
            atom_types = (
                torch.tensor(
                    [atom_encoder[a.GetSymbol()] for a in rdkit_mol.GetAtoms()]
                )
                .long()
                .to(device)
            )
            charges = (
                torch.tensor([a.GetFormalCharge() for a in rdkit_mol.GetAtoms()])
                .long()
                .to(device)
            )

            self.positions = positions
            self.atom_types = atom_types
            self.charges = charges

            adj = torch.from_numpy(
                Chem.rdmolops.GetAdjacencyMatrix(rdkit_mol, useBO=True)
            ).to(device)
            self.bond_types = adj.long()

            self.num_nodes = len(atom_types)

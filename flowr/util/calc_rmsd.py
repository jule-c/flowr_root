import copy

import numpy as np
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1
from Bio.SVDSuperimposer import SVDSuperimposer

from flowr.util.write_pdb import PDBio

AA = [
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
]


def align_rmsd(
    ref_pdb: str, pred_pdb: str, atom_types=["CA", "N", "C", "O"]
) -> SVDSuperimposer:
    """
    Aligns a model structure onto a native structure
    Using the atom types listed in `atom_types`.
    """

    p = PDBParser(QUIET=True)
    native = p.get_structure("native", ref_pdb)
    model = p.get_structure("model", pred_pdb)

    native_seq = "".join(
        [
            protein_letters_3to1[r.resname]
            for r in native[0].get_residues()
            if r.resname in AA
        ]
    )
    model_seq = "".join(
        [
            protein_letters_3to1[r.resname]
            for r in model[0].get_residues()
            if r.resname in AA
        ]
    )

    assert len(model_seq) == len(
        native_seq
    ), "The sequences should be of identical length."

    native_coords = [
        a.coord
        for a in native[0].get_atoms()
        if a.parent.resname in AA and a.name in atom_types
    ]
    model_coords = [
        a.coord
        for a in model[0].get_atoms()
        if a.parent.resname in AA and a.name in atom_types
    ]

    si = SVDSuperimposer()
    si.set(np.array(native_coords), np.array(model_coords))
    si.run()  # Run the SVD alignment

    return si.get_rms()


def fit_rms(ref_c, c):
    # move geometric center to the origin
    ref_trans = np.average(ref_c, axis=0)
    ref_c = ref_c - ref_trans
    c_trans = np.average(c, axis=0)
    c = c - c_trans

    # covariance matrix
    C = np.dot(c.T, ref_c)

    # Singular Value Decomposition
    (r1, s, r2) = np.linalg.svd(C)

    # compute sign (remove mirroring)
    if np.linalg.det(C) < 0:
        r2[2, :] *= -1.0
    U = np.dot(r1, r2)
    return (c_trans, U, ref_trans)


class RMSDcalculator:
    def __init__(self, atoms1, atoms2, name=None):
        xyz1 = self.get_xyz(atoms1, name=name)
        xyz2 = self.get_xyz(atoms2, name=name)
        self.set_rmsd(xyz1, xyz2)

    def get_xyz(self, atoms, name=None):
        xyz = []
        for atom in atoms:
            if name:
                if atom.name != name:
                    continue
            xyz.append([atom.x, atom.y, atom.z])
        return np.array(xyz)

    def set_rmsd(self, c1, c2):
        self.rmsd = 0.0
        self.c_trans, self.U, self.ref_trans = fit_rms(c1, c2)
        new_c2 = np.dot(c2 - self.c_trans, self.U) + self.ref_trans
        self.rmsd = np.sqrt(np.average(np.sum((c1 - new_c2) ** 2, axis=1)))

    def get_aligned_coord(self, atoms, name=None):
        new_c2 = copy.deepcopy(atoms)
        for atom in new_c2:
            atom.x, atom.y, atom.z = (
                np.dot(np.array([atom.x, atom.y, atom.z]) - self.c_trans, self.U)
                + self.ref_trans
            )
        return new_c2


if __name__ == "__main__":
    import os

    main_folder = "."
    pdb_ref = "1y3q.pdb"
    pdb_pred = "1y3q.pdb"
    pdbf1 = os.path.join(main_folder, pdb_ref)
    pdbf2 = os.path.join(main_folder, pdb_pred)
    pdb1 = PDBio(pdbf1)
    pdb2 = PDBio(pdbf2)
    atoms1 = pdb1.get_atoms(to_dict=False)
    atoms2 = pdb2.get_atoms(to_dict=False)

    RMSDcalculator = RMSDcalculator(atoms1, atoms2, name="CA")
    rmsd = RMSDcalculator.rmsd
    new_atoms = RMSDcalculator.get_aligned_coord(atoms2)
    pdb2.write_pdb(os.path.join(main_folder, f"aligned_{pdb_pred}"), new_atoms)
    print("RMSD : %8.3f" % rmsd)
    print("New structure file: ", os.path.join(main_folder, f"aligned_{pdb_pred}"))

    rmsd_align = align_rmsd(
        os.path.join(main_folder, pdb_ref), os.path.join(main_folder, pdb_pred)
    )
    print(f"RMSD after alignment with BioPython: {rmsd_align} angstroms")

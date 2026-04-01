import sys

from flowr.util.read_pdb import PDB


class PDBio(PDB):
    template = "{head:6s}{serial:5d} {name:<4}{altLoc:1s}{resName:3s} {chainID:1s}{resSeq:4d}{iCode:1s}   {x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{tempFactor:6.2f}          {element:>2s}{charge:2s}\n"

    def write_pdb(self, filename, chainID=None, atoms=None, MODEL=[None, 1]):
        if not atoms:
            atoms = self.atoms
        f = open(filename, "w")
        for atom in atoms:
            atom.head = "ATOM"
            if chainID:
                if atom.chainID != chainID:
                    continue
            if atom.MODEL not in MODEL:
                continue
            atom_info = atom.__dict__.copy()
            if len(atom.name) < 4 and len(atom.element) == 1:
                atom_info["name"] = " " + atom.name
            f.write(self.template.format(**atom_info))
        f.write("TER")
        f.close()


if __name__ == "__main__":
    pdb = PDBio(sys.argv[1])
    filename = "./test.pdb"
    chainID = "A"
    pdb.write_pdb(filename, chainID)

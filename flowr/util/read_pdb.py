class Atom:
    def __init__(self, line):
        self.serial = int(line[6:11])
        self.name = line[11:16].strip()
        self.altLoc = line[16:17].strip()
        self.resName = line[17:20]
        self.chainID = line[21:22]
        self.resSeq = int(line[22:26])
        self.iCode = line[26:27].strip()
        self.x = float(line[30:38])
        self.y = float(line[38:46])
        self.z = float(line[46:54])
        self.occupancy = line[54:60].strip()
        self.tempFactor = line[60:66].strip()
        self.element = line[76:78].strip()
        self.charge = line[78:80].strip()
        if self.occupancy:
            self.occupancy = float(self.occupancy)
        if self.tempFactor:
            self.tempFactor = float(self.tempFactor)

    def __getitem__(self, key):
        return self.__dict__[key]


class PDB:
    def __init__(self, file):
        self.file = file
        self.atoms = []
        self.parse()

    def parse(self):
        MODEL = None
        f = open(self.file, "r")
        for line in f.readlines():
            if line.startswith("MODEL"):
                MODEL = int(line.split()[1])
            if line.startswith("ATOM"):
                atom = Atom(line)
                atom.MODEL = MODEL
                self.atoms.append(atom)
        f.close()

    def get_atoms(self, to_dict=True):
        """Return a list of all atoms.

        If to_dict is True, each atom is represented as a dictionary.
        Otherwise, a list of Atom objects is returned."""
        if to_dict:
            return [x.__dict__ for x in self.atoms]
        else:
            return self.atoms

    def get_model(self, model_num, to_dict=True):
        """Return all atoms where MODEL == model_num"""
        model_atoms = [x for x in self.atoms if x.MODEL == model_num]
        if to_dict:
            return [atom.__dict__ for atom in model_atoms]
        else:
            return model_atoms


if __name__ == "__main__":
    import sys

    pdb = PDB(sys.argv[1])
    atoms = pdb.get_atoms(to_dict=False)

    # print the name of each atom
    for atom in atoms:
        print(atom.serial, atom.name, atom.x, atom.y, atom.z, atom.element)

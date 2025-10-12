
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Structure import Structure

def load_biopython_structure(pdb_path: str) -> Structure:
    """
    Load a structure from a PDB file using Biopython's PDBParser.
    
    Args:
        pdb_path (str): The file path to the PDB file.
        
    Returns:
        Bio.PDB.Structure.Structure: The structure parsed from the PDB file.
    """
    parser = PDBParser()
    structure = parser.get_structure("PDB_structure", pdb_path)
    return structure

def save_biopython_structure(structure: Structure, pdb_path: str):
    """
    Save a structure to a PDB file using Biopython's PDBIO.
    
    Args:
        structure (Bio.PDB.Structure.Structure): The structure to save.
        pdb_path (str): The file path to save the PDB file to.
    """
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_path)

def ids_scriptly_increasing(structure: Structure) -> bool:
    """
    Check if the IDs of residues in a structure are strictly increasing.
    
    Args:
        structure (Bio.PDB.Structure.Structure): The structure to check.
        
    Returns:
        bool: True if the IDs are strictly increasing, False otherwise.
    """
    ids = []
    
    for model in structure:
        for chain in model:
            for i, residue in enumerate(chain):
                ids.append((chain.id, residue.id, i))

    if ids != sorted(ids):
        return False
    return True

def reorder_ids(structure: Structure) -> Structure:
    """
    Reorder the IDs of residues in a structure to be strictly increasing.
    
    Args:
        structure (Bio.PDB.Structure.Structure): The structure to reorder.
        
    Returns:
        Bio.PDB.Structure.Structure: The structure with reordered IDs.
    """
    for model in structure:
        for chain in model:
            chain_id = chain.id
            # Extract residues from the chain
            residues = [residue for residue in chain]
            # Sort residues based on their id[1]
            sorted_residues = sorted(residues, key=lambda residue: residue.id[1])
            # Clear the chain of residues before adding them back in sorted order
            # Iterate over a copy of the list of residues to safely remove them
            for residue in residues:
                chain.detach_child(residue.id)
            for residue in sorted_residues:
                # Prevent adding a residue that is already present
                if not chain.has_id(residue.id):
                    chain.add(residue)
    return structure


def remove_connect_lines(pdb_path):
    """
    Remove CONECT lines from a PDB file.
    """

    with open(pdb_path, "r") as f:
        lines = f.readlines()
    with open(pdb_path, "w") as f:
        for line in lines:
            if line.startswith("CONECT"):
                continue
            f.write(line)


if __name__ == "__main__":
    from posecheck.utils.constants import EXAMPLE_PDB_PATH
    # import temp path 
    import tempfile

    structure = load_biopython_structure(EXAMPLE_PDB_PATH)

    print(ids_scriptly_increasing(structure))

    reordered_structure = reorder_ids(structure)

    print(ids_scriptly_increasing(reordered_structure))

    with tempfile.NamedTemporaryFile(suffix=".pdb") as temp:
        print(temp.name)
        save_biopython_structure(reordered_structure, temp.name)
        print(load_biopython_structure(temp.name))


    
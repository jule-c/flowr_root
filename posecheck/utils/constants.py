import os

# get directory of this file
DIR = os.path.dirname(os.path.abspath(__file__))
# get directory of project and data
DATA_DIR = os.path.join(DIR, "../../data/")
PROJECT_DIR = os.path.join(DIR, "../../")

CROSSOCKED_PDB_DIR = os.path.join(DATA_DIR, "crossdocked/")
CROSSOCKED_PDBQT_DIR = os.path.join(DATA_DIR, "crossdocked_pdbqt/")

PROTEINS_PATH = os.path.join(DATA_DIR, "proteins.pt")

# Example paths

EXAMPLE_PDB_PATH = os.path.join(DATA_DIR, "examples/1a2g.pdb")
EXAMPLE_LIGAND_PATH = os.path.join(DATA_DIR, "examples/1a2g_ligand.sdf")

# Constants collected from the codebase
FORCEFIELD = "uff"
ADD_COORDS = True
ROUND_DIGITS = 2

# PATHS
# REDUCE_PATH = os.path.join(PROJECT_DIR, "bin/reduce")
REDUCE_PATH = "reduce"
SPLIT_PATH = os.path.join(PROJECT_DIR, "data/crossdocked_split_by_name.pkl")

# Docking params
DOCKING_METHOD = "smina"  # TODO add support for vina and qvina2
# SMINA_PATH = '/home/cch57/projects/poses_benchmark/smina/smina.static'
SMINA_PATH = os.path.join(PROJECT_DIR, "bin/smina.static")
EXHAUSTIVENESS = 8
BOX_SIZE = 25

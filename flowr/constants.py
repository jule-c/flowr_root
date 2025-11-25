## **** GLOBAL ATOM/ELEMENT ENCODER! **** ##
CORE_ATOMS = [
    "H",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Si",
    "P",
    "S",
    "Cl",
    "Se",
    "Br",
    "I",
    "Bi",
]
ATOM_ENCODER = {atom: idx for idx, atom in enumerate(CORE_ATOMS)}

INPAINT_ENCODER = {
    "de_novo": [0, 0],
    "scaffold": [1, 1],
    "func_group": [1, 2],
    "interaction": [1, 3],
    "linker": [1, 4],
    "core": [1, 5],
    "fragment": [2, 6],
    "substructure": [2, 7],
    "graph": [3, 8],
}

AFFINITY_PROP_NAMES = [
    "pic50",
    "pkd",
    "pki",
    "pec50",
]

## **** GLOBAL LIGAND SIZE **** ##
MAX_LIGAND_SIZE = 250

# Declarations to be used in scripts
QM9_COORDS_STD_DEV = 1.723299503326416
GEOM_COORDS_STD_DEV = 2.407038688659668
COMBINED_COORDS_STD_DEV = 2.407038688659668
SPINDR_COORDS_STD_DEV = 3.5152788162231445  # 2.8421707153320312
HIQBIND_COORDS_STD_DEV = SPINDR_COORDS_STD_DEV
SAIR_COORDS_STD_DEV = SPINDR_COORDS_STD_DEV
CROSSDOCKED_COORDS_STD_DEV = SPINDR_COORDS_STD_DEV
KINODATA_COORDS_STD_DEV = SPINDR_COORDS_STD_DEV
BINDINGMOAD_COORDS_STD_DEV = SPINDR_COORDS_STD_DEV
BINDINGNET_COORDS_STD_DEV = SPINDR_COORDS_STD_DEV

# Dataloading bucket limits
QM9_BUCKET_LIMITS = [12, 16, 18, 20, 22, 24, 30]
GEOM_DRUGS_BUCKET_LIMITS = [24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 96, 192]
COMBINED_BUCKET_LIMITS = [24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 96, 192]

GEOM_DRUGS_BUCKET_LIMITS_noHs = [10, 13, 15, 18, 20, 23, 26, 29, 32, 35, 40, 45, 200]
ENAMINE_BUCKET_LIMITS = [10, 13, 15, 18, 20, 23, 26, 29, 32, 35, 40, 200]
LMDB_BUCKET_LIMITS = [24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 96, 192]
SPINDR_BUCKET_LIMITS = [
    100,
    140,
    180,
    200,
    220,
    230,
    240,
    250,
    260,
    280,
    320,
    360,
    400,
    440,
    1000,
]
SAIR_BUCKET_LIMITS = [
    180,
    220,
    230,
    240,
    250,
    260,
    280,
    320,
    360,
    400,
    440,
    460,
    480,
    500,
    550,
    600,
    1000,
]
HIQBIND_BUCKET_LIMITS = [
    120,
    180,
    200,
    220,
    230,
    240,
    250,
    260,
    280,
    300,
    320,
    340,
    360,
    400,
    440,
    600,
    800,
    1200,
]
KINODATA_BUCKET_LIMITS = [
    120,
    140,
    160,
    180,
    200,
    220,
    240,
    260,
    280,
    320,
    600,
]
CROSSDOCKED_BUCKET_LIMITS = [
    160,
    180,
    200,
    220,
    240,
    260,
    280,
    320,
    400,
    500,
    600,
    700,
    1500,
]
BINDINGMOAD_BUCKET_LIMITS = [
    120,
    140,
    160,
    180,
    200,
    220,
    240,
    260,
    280,
    320,
    600,
]

BINDINGNET_BUCKET_LIMITS = [
    140,
    180,
    200,
    220,
    230,
    240,
    250,
    260,
    280,
    320,
    360,
    380,
    400,
    420,
    440,
    460,
    500,
    550,
    600,
    650,
    700,
    1000,
]

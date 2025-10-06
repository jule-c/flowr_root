import os
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pandas as pd
from rdkit import Chem

from flowr.util.molrepr import GeometricMol
from flowr.util.pocket import ProteinPocket
from flowr.util.schrodinger import SchrodingerJob

# Number of heavy atoms in each amino acid, not including the backbone OH group
AMINO_ACID_SIZES = {
    "ARG": 11,
    "HIS": 10,
    "LYS": 9,
    "ASP": 8,
    "GLU": 9,
    "SER": 6,
    "THR": 7,
    "ASN": 8,
    "GLN": 9,
    "CYS": 6,
    "SEC": 6,
    "GLY": 4,
    "PRO": 7,
    "ALA": 5,
    "VAL": 7,
    "ILE": 8,
    "LEU": 8,
    "MET": 8,
    "PHE": 11,
    "TYR": 12,
    "TRP": 14,
}


# *************************************
# ***** Plinder exception classes *****
# *************************************


class EmptyHoloPocket(Exception):
    """Used when there are no residues to load for the holo pocket"""

    pass


class UnknownPocketResidues(Exception):
    """Used when unnatural amino acids appear in a pocket"""

    pass


class LigandNotFound(Exception):
    """USed when we cannot find an appropriate ligand within a file"""

    pass


# **************************************
# ***** System filtering functions *****
# **************************************


def setup_plinder_env(data_path, pl_release, pl_version):
    """Setup env vars for plinder"""

    plinder_path = str(Path(data_path).resolve())
    version = f"{pl_release}/{pl_version}"
    os.environ["PLINDER_LOCAL_DIR"] = plinder_path

    import plinder.core

    cfg = plinder.core.get_config()
    cfg.data.plinder_dir = f"{plinder_path}/{version}"
    cfg.data.plinder_mount = str(Path(plinder_path).parent)


def filter_multi_interactions(df):
    """Return df with multi ligand and multi protein interactions removed"""

    return df[
        (df["system_num_interacting_protein_chains"] == 1)
        & (df["system_num_ligand_chains"] == 1)
    ]


def filter_mol_type(df, keep_covalent=True):
    """Returns df with non small molecule ligand types removed"""

    filter = (
        df["ligand_is_fragment"]
        | df["ligand_is_ion"]
        | df["ligand_is_oligo"]
        | df["ligand_is_cofactor"]
        | df["ligand_is_other"]
        | df["ligand_is_artifact"]
    )

    if not keep_covalent:
        filter = filter | df["ligand_is_covalent"]

    return df[~filter]


def filter_systems(data_path, pl_release, pl_version, keep_covalent=True):
    data_path = Path(data_path)
    version = f"{pl_release}/{pl_version}"

    # Read core index files
    splits_df = pd.read_parquet(data_path / f"{version}/splits/split.parquet")
    index_df = pd.read_parquet(
        data_path / f"{version}/index/annotation_table_nonredundant.parquet"
    )

    # Keep single protein, single ligand systems with reasonable molecule types
    single_pl_df = filter_multi_interactions(index_df)
    drug_like_df = filter_mol_type(single_pl_df, keep_covalent=keep_covalent)

    # Only keep systems which appear in splits file and are not removed
    splits_df = splits_df[splits_df["split"].isin(["train", "val", "test"])]
    systems_df = pd.merge(splits_df, drug_like_df, how="inner", on="system_id")

    systems = systems_df["system_id"].tolist()
    splits = systems_df["split"].tolist()

    # Check that there are no duplicates in the system ids
    if len(set(systems)) != len(systems):
        raise RuntimeError("Found duplicate system ids in the index file.")

    return systems, splits


# ***************************************
# ***** Amino acid helper functions *****
# ***************************************


def is_natural(structure):
    """Check if this structure only contains natural amino acids"""

    _, residues = struc.get_residues(structure)
    natural_res = [res in AMINO_ACID_SIZES for res in residues]
    return all(natural_res)


def remove_unknown_residues(structure):
    allowed = set(AMINO_ACID_SIZES.keys())
    res_ids, res_names = struc.get_residues(structure)
    keep_ids = [res_id for res_id, res in zip(res_ids, res_names) if res in allowed]
    remaining_structure = structure[np.isin(structure.res_id, keep_ids)]
    return remaining_structure


def check_residue_sizes(struct):
    res_ids, res_names = struc.get_residues(struct)
    for res_id, res_name in zip(res_ids, res_names):
        res_struct = struct[struct.res_id == res_id]
        heavy_atoms = res_struct[res_struct.element != "H"]
        exp_n_atoms = AMINO_ACID_SIZES[res_name]
        if len(heavy_atoms) != exp_n_atoms:
            print(
                f"Residue {res_id} ({res_name}) has {len(heavy_atoms)} atoms -- expected {exp_n_atoms}"
            )


# *************************************************************************
# **** Functions for loading structures or converting representations *****
# *************************************************************************


def load_structure(
    struct_path,
    chain_id=None,
    model=1,
    include_hs=False,
    include_hetero=False,
    include_charge=True,
    include_bonds=True,
    file_type="cif",
):
    if file_type == "cif":
        file = pdbx.CIFFile.read(struct_path)
        read_fn = pdbx.get_structure
    elif file_type == "pdb":
        file = pdb.PDBFile.read(struct_path)
        read_fn = pdb.get_structure
    else:
        raise ValueError(
            f"Unknown file type {file_type}, must be either 'cif' or 'pdb'"
        )

    # Sometimes reading charges fails so try to read first but fall back to without charge if not
    try:
        extra = ["charge"] if include_charge else []
        structure = read_fn(
            file, model=model, extra_fields=extra, include_bonds=include_bonds
        )
    except:
        structure = read_fn(file, model=model, include_bonds=include_bonds)

    if chain_id is not None:
        structure = structure[structure.chain_id == chain_id]

    if not include_hs:
        structure = structure[structure.element != "H"]

    if not include_hetero:
        structure = structure[~structure.hetero]

    return structure


def load_holo_structure(system, include_hs=False, include_hetero=False):
    """Load selected chain of holo structure"""

    system_id_parts = [part for part in system.system_id.split("__")]
    holo_structure = load_structure(
        system.receptor_cif,
        chain_id=system_id_parts[2],
        include_hs=include_hs,
        include_hetero=include_hetero,
        include_charge=True,
        include_bonds=True,
        file_type="cif",
    )

    # If the bonds could not be read from the file, then infer them
    if holo_structure.bonds is None:
        bonds = struc.connect_via_residue_names(holo_structure, inter_residue=True)
        holo_structure.bonds = bonds

    return holo_structure


def load_complex_structure(system, include_hs=False, include_hetero=False):
    """Load the holo structure chain with load_holo_structure, then load the ligand chain and combine"""

    system_id_parts = [part for part in system.system_id.split("__")]
    holo_structure = load_holo_structure(
        system, include_hs=include_hs, include_hetero=include_hetero
    )

    # Load just the ligand chain from the complex structure, hetero must be set to True since ligand is hetero
    ligand_structure = load_structure(
        system.system_cif,
        chain_id=system_id_parts[-1],
        include_hs=include_hs,
        include_hetero=True,
        include_charge=True,
        include_bonds=True,
        file_type="cif",
    )

    if not include_hs:
        ligand_structure = ligand_structure[ligand_structure.element != "H"]

    # Set the res name to LIG so that we have the same for all ligands
    ligand_structure.res_name = ["LIG"] * len(ligand_structure)

    complex_structure = holo_structure + ligand_structure
    return complex_structure


def holo_pocket_from_structure(system, holo_structure, ligand_idx=0):
    """Just keep holo pocket using plinder residue ids"""

    system_id_parts = [part for part in system.system_id.split("_") if part != ""]
    chain_id = system_id_parts[2]
    holo_residue_ids = list(
        system.system["ligands"][ligand_idx]["neighboring_residues"][chain_id]
    )

    if len(holo_residue_ids) < 3:
        print(
            f"WARNING -- system {system.system_id} has only {len(holo_residue_ids)} residues in pocket."
        )

    if len(holo_residue_ids) == 0:
        raise EmptyHoloPocket(f"No pocket residues found for system {system.system_id}")

    holo_pocket_atoms = holo_structure[np.isin(holo_structure.res_id, holo_residue_ids)]
    holo_pocket = ProteinPocket.from_pocket_atoms(holo_pocket_atoms)

    # Throw an error if there are unknown amino acids in the pocket since we cannot recover from this
    if not is_natural(holo_pocket_atoms):
        raise UnknownPocketResidues(
            f"Found unknown amino acids in holo pocket for system {system.system_id}"
        )

    return holo_pocket


def load_ligand(system):
    system_id_parts = [part for part in system.system_id.split("_") if part != ""]
    ligand_id = system_id_parts[-1]
    sdf_path = system.ligands[ligand_id]
    mol = Chem.MolFromMolFile(sdf_path)
    return GeometricMol.from_rdkit(mol)


def load_ligand_mae(mae_path, ligand_chain_id, sanitise=True, remove_hs=False):
    suppl = Chem.MaeMolSupplier(Path(mae_path), sanitize=sanitise, removeHs=remove_hs)
    mols = [mol for mol in suppl if mol is not None]
    if len(mols) == 0:
        raise LigandNotFound(f"Could not find read molecules from {mae_path}")

    chain_mol_map = Chem.rdmolops.SplitMolByPDBChainId(mols[0])
    if ligand_chain_id not in chain_mol_map:
        raise LigandNotFound(
            f"No molecule with ligand chain id {ligand_chain_id} were found."
        )

    mol = chain_mol_map[ligand_chain_id]
    return GeometricMol.from_rdkit(mol)


# TODO expand this include other useful info from system file
def load_system_metadata(system, lidand_idx=0):
    metadata = {
        "system_id": system.system_id,
        "is_covalent": system.system["ligands"][lidand_idx]["is_covalent"],
    }
    return metadata


# ********************************************
# ***** Functions for plinder related IO *****
# ********************************************


def delete_dir(dir_path: Path, rm_rempty_dir: bool = True):
    """Delete all contents of folder at <dir_path> and the folder itself"""

    for path in dir_path.iterdir():
        if path.is_dir():
            delete_dir(path)
        else:
            path.unlink()

    if rm_rempty_dir:
        dir_path.rmdir()


def delete_group_folders(data_path, pdb_code):
    """Delete all folders from group <pdb_code> directly under <path>"""

    for path in data_path.iterdir():
        if path.is_dir() and path.stem[1:3] == pdb_code:
            delete_dir(path, rm_rempty_dir=True)


def cleanup_group(pdb_code, data_path, pl_release, pl_iteration):
    """Delete zipped grouped files and all systems which are part of this group"""

    version = f"{pl_release}/{pl_iteration}"
    plinder_path = str(Path(data_path).resolve())
    data_path = Path(f"{plinder_path}/{version}")

    # Delete files and system folders under /systems
    delete_group_folders(data_path / "systems", pdb_code)
    (data_path / f"systems/{pdb_code}_done").unlink(missing_ok=True)
    (data_path / f"systems/{pdb_code}.zip").unlink(missing_ok=True)

    # Delete files and system folders under /linked_structures
    # delete_group_folders((data_path / "linked_structures") / "apo", pdb_code)
    # delete_group_folders((data_path / "linked_structures") / "pred", pdb_code)
    # (data_path / f"linked_structures/{pdb_code}_done").unlink(missing_ok=True)
    # (data_path / f"linked_structures/{pdb_code}.zip").unlink(missing_ok=True)

    # Delete zipped file in entries
    (data_path / f"entries/{pdb_code}.zip").unlink(missing_ok=True)


# ********************************************
# ***** Functions for plinder processing *****
# ********************************************


def run_prep_wizard(system_id, complex_structure, tmp_path, config_path):
    schrodinger = SchrodingerJob(config_path)

    print(f"Running prep wizard function with system {system_id}")

    tmp_folder = Path(tmp_path) / system_id
    if tmp_folder.exists() and tmp_folder.is_dir():
        delete_dir(tmp_folder, rm_rempty_dir=True)

    tmp_folder.mkdir(exist_ok=False, parents=True)

    complex_path = tmp_folder / "complex.pdb"
    mae_path = tmp_folder / "complex.mae"
    output_path = tmp_folder / "output.pdb"
    prepwizard_log = tmp_folder / "prepwizard.log"
    mae_log = tmp_folder / "mae.log"

    # Split out the complex back into ligand and protein
    lig_structure = complex_structure[complex_structure.res_name == "LIG"]
    holo_structure = complex_structure[complex_structure.res_name != "LIG"]

    # To work with pdb files we need to set a one char chain id
    lig_structure.chain_id = ["L"] * len(lig_structure)
    lig_structure.res_name = ["UNK"] * len(lig_structure)
    holo_structure.chain_id = ["A"] * len(holo_structure)

    # Save whole complex to pdb file, do try to infer bonds with the ligand
    complex_structure = holo_structure + lig_structure
    complex_smol = ProteinPocket.from_pocket_atoms(
        complex_structure, infer_res_bonds=False
    )
    complex_smol.to_pdb_file(complex_path, include_bonds=True)

    print("Running prep wizard...")

    # Run schrodinger tools
    output = schrodinger.prepwizard(complex_path, mae_path, prepwizard_log)
    if output is None:
        print("Prep wizard successful")
    else:
        print("Prep wizard unsuccessful...")
        print("Return code:", output.returncode)
        print("Stdout:")
        print(output.stdout)
        print()
        print("stderr")
        print(output.stderr)
        print()

    schrodinger.mae2pdb(mae_path, output_path, mae_log)

    # Load the output structure and split back into holo protein and ligand
    output_structure = load_structure(
        output_path,
        include_hs=True,
        include_hetero=True,
        include_charge=True,
        include_bonds=False,
        file_type="pdb",
    )

    # Since we cannot get bond types from pdb files we load the holo through the pdb and infer bonds
    holo_structure = output_structure[output_structure.chain_id != "L"]
    bonds = struc.connect_via_residue_names(holo_structure, inter_residue=True)
    holo_structure.bonds = bonds

    # Set the chain id of the holo back to what it should be
    holo_chain_id = system_id.split("__")[2]
    holo_structure.chain_id = [holo_chain_id] * len(holo_structure)

    # Load the ligand using RDKit from the intermediate MAE file and into smol format
    ligand_smol = load_ligand_mae(mae_path, "L")

    return holo_structure, ligand_smol


# TODO check this
def run_prep_wizard_protein(system_id, protein_structure, tmp_path, config_path):
    schrodinger = SchrodingerJob(config_path)

    print(f"Running prep wizard function for apo with system {system_id}")

    tmp_folder = Path(tmp_path) / f"{system_id}_pocket"
    if tmp_folder.exists() and tmp_folder.is_dir():
        delete_dir(tmp_folder, rm_rempty_dir=True)

    tmp_folder.mkdir(exist_ok=False, parents=True)

    protein_path = tmp_folder / "protein.pdb"
    mae_path = tmp_folder / "protein.mae"
    output_path = tmp_folder / "output.pdb"
    prepwizard_log = tmp_folder / "prepwizard.log"
    mae_log = tmp_folder / "mae.log"

    # Save protein structure to pdb file
    protein_smol = ProteinPocket.from_pocket_atoms(
        protein_structure, infer_res_bonds=True
    )
    protein_smol.to_pdb_file(protein_path, include_bonds=True)

    print("Running prep wizard...")

    # Run schrodinger tools
    output = schrodinger.prepwizard(protein_path, mae_path, prepwizard_log)
    if output is None:
        print("Prep wizard successful")
    else:
        print("Prep wizard unsuccessful...")
        print("Return code:", output.returncode)
        print("Stdout:")
        print(output.stdout)
        print()
        print("stderr")
        print(output.stderr)
        print()

    schrodinger.mae2pdb(mae_path, output_path, mae_log)

    # Load the output structure from the schrodinger saved file
    output_structure = load_structure(
        output_path,
        include_hs=True,
        include_hetero=False,
        include_charge=True,
        include_bonds=False,
        file_type="pdb",
    )

    # Since we cannot get bond types from pdb files we load the holo through the pdb and infer bonds
    protein_structure = output_structure[output_structure.chain_id != "L"]
    bonds = struc.connect_via_residue_names(protein_structure, inter_residue=True)
    protein_structure.bonds = bonds

    # Set the chain id of the holo back to what it should be
    chain_id = system_id.split("__")[2]
    protein_structure.chain_id = [chain_id] * len(protein_structure)

    return protein_structure


def group_systems(system_ids, splits):
    pdb_codes = [sys_id[1:3] for sys_id in system_ids]
    single_char_codes = set([pdb_code[0] for pdb_code in pdb_codes])

    grouped_systems = {char_code: {} for char_code in single_char_codes}
    grouped_splits = {char_code: {} for char_code in single_char_codes}

    for pdb_code in pdb_codes:
        grouped_systems[pdb_code[0]][pdb_code] = []
        grouped_splits[pdb_code[0]][pdb_code] = []

    for system_id, split in zip(system_ids, splits):
        group_code = system_id[1]
        grouped_systems[group_code][system_id[1:3]].append(system_id)
        grouped_splits[group_code][system_id[1:3]].append(split)

    return grouped_systems, grouped_splits

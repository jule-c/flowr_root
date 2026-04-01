import re
import subprocess
from pathlib import Path

import pandas as pd
from rdkit import Chem

import flowr.util.rdkit as smolRD


def check_ligand_atom_types(complex_data, core_atoms_set):
    """
    Check if all ligand atom types in the complex are in CORE_ATOMS.

    Args:
        complex_data: The complex data loaded from LMDB
        core_atoms_set: Set of allowed core atoms

    Returns:
        bool: True if all ligand atom types are valid, False otherwise
    """

    # Extract ligand from complex
    ligand = complex_data.ligand

    # Get atomic numbers and convert to symbols
    atomic_nums = ligand.atomics.tolist()

    # Convert atomic numbers to element symbols
    # Assuming you have a periodic table mapping - you may need to adjust this
    symbols = [smolRD.PT.symbol_from_atomic(a) for a in atomic_nums]

    # Check if all symbols are in core atoms
    return all(symbol in core_atoms_set for symbol in symbols)


def check_ligand_size(complex_data, max_ligand_size):
    """
    Check if ligand size is within allowed limits.

    Args:
        complex_data: The complex data loaded from LMDB
        max_ligand_size: Maximum allowed ligand size

    Returns:
        bool: True if ligand size is valid, False otherwise
    """
    # Extract ligand from complex
    ligand = complex_data.ligand

    # Get ligand size
    ligand_size = ligand.seq_length

    # Check if ligand size is within allowed limits
    return ligand_size <= max_ligand_size


def fix_pdb(
    pdb_file: str,
    remove_hetero: bool = False,
    add_hs: bool = False,
    out_file: str = None,
):
    from openmm.app import PDBFile
    from pdbfixer import PDBFixer

    out_file = out_file if out_file is not None else pdb_file

    fixer = PDBFixer(filename=pdb_file)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    if remove_hetero:
        fixer.removeHeterogens(True)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    if add_hs:
        fixer.addMissingHydrogens(7.4)
    PDBFile.writeFile(
        fixer.topology,
        fixer.positions,
        open(out_file, "w"),
    )


def extract_gnina_score(protein_path: str, ligand_path: str) -> float:
    """
    Extract GNINA docking score using gnina command line tool.

    Args:
        protein_path (str): Path to protein PDB file
        ligand_path (str): Path to ligand SDF file

    Returns:
        float: GNINA affinity score in kcal/mol
    """
    # Run gnina command
    cmd = [
        "gnina",
        "-r",
        protein_path,
        "-l",
        ligand_path,
        "--score_only",
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300  # 5 minute timeout
    )

    if result.returncode != 0:
        raise RuntimeError(f"GNINA failed with return code {result.returncode}")

    # Extract affinity score from output
    output_text = result.stdout
    affinity_match = re.search(r"Affinity:\s+([-\d\.]+)\s+\(kcal/mol\)", output_text)

    if affinity_match:
        affinity_score = float(affinity_match.group(1))
        return affinity_score
    else:
        raise ValueError("Could not extract affinity score from GNINA output")


def _convert_to_molar(value: float, unit: str) -> float:
    """
    Convert concentration value to molar based on unit.

    Args:
        value: Concentration value
        unit: Unit string (M, mM, uM, nM, pM, fM)

    Returns:
        Concentration in molar, or None if unit not recognized
    """
    if pd.isna(unit):
        return None

    unit = unit.strip().upper()

    unit_conversions = {
        "M": 1.0,
        "MM": 1e-3,
        "MMOL/L": 1e-3,
        "UM": 1e-6,
        "UMOL/L": 1e-6,
        "μM": 1e-6,
        "NM": 1e-9,
        "NMOL/L": 1e-9,
        "PM": 1e-12,
        "PMOL/L": 1e-12,
        "FM": 1e-15,
        "FMOL/L": 1e-15,
    }

    if unit in unit_conversions:
        return value * unit_conversions[unit]
    else:
        print(f"Warning: Unknown unit '{unit}', assuming nM")
        return value * 1e-9  # Default to nM if unit not recognized


def extract_affinity_data_from_mol(mol: Chem.Mol) -> dict:
    """
    Extract affinity data from GatorAffinity dataset molecule properties.

    Args:
        mol: RDKit molecule object with GatorAffinity affinity properties

    Returns:
        dict: Dictionary with affinity data - can be empty if no data found
    """
    raise NotImplementedError(
        "This function is a placeholder and needs to be implemented."
    )
    affinity = {}

    # Get all property names from the molecule
    prop_names = mol.GetPropNames()

    # Extract GatorAffinity-specific properties
    if "pIC50" in prop_names:
        pic50 = float(mol.GetProp("pIC50"))
        affinity["pic50"] = pic50
    if "pKi" in prop_names:
        pki = float(mol.GetProp("pKi"))
        affinity["pki"] = pki
    if "pKd" in prop_names:
        pkd = float(mol.GetProp("pKd"))
        affinity["pkd"] = pkd
    if "pEC50" in prop_names:
        pec50 = float(mol.GetProp("pEC50"))
        affinity["pec50"] = pec50

    return affinity


def extract_affinity_data_from_csv(index_df: pd.DataFrame, system_id: str) -> dict:
    """
    Extract affinity data from BindingNet dataset index DataFrame.
    Args:
        index_df: DataFrame containing the dataset index with affinity data
        system_id: System identifier to look up in the DataFrame
    Returns:
        dict: Dictionary with affinity data - can be empty if no data found
    """

    raise NotImplementedError(
        "This function is a placeholder and needs to be implemented."
    )

    affinity = {}

    # Locate the row corresponding to the system_id
    row = index_df[index_df["system_id"] == system_id]

    if row.empty:
        print(f"Warning: No entry found for system_id {system_id} in index DataFrame.")
        return affinity

    row = row.iloc[0]  # Get the first matching row

    # Extract affinity values and convert to molar
    for measure in ["IC50", "Ki", "Kd", "EC50"]:
        value_col = f"{measure}_value"
        unit_col = f"{measure}_unit"

        if pd.notna(row[value_col]) and pd.notna(row[unit_col]):
            value = float(row[value_col])
            unit = str(row[unit_col])
            molar_value = _convert_to_molar(value, unit)

            if molar_value is not None:
                p_measure = -pd.np.log10(molar_value)
                affinity[f"p{measure.lower()}"] = p_measure

    return affinity


def calculate_docking_score(
    sdf_file: Path,
    pdb_file: Path,
    system_id: str,
    calc_vina_score: bool = False,
    calc_gnina_score: bool = True,
    num_workers: int = 1,
):
    """
    Calculate docking scores using Vina and GNINA.
    """
    gnina_score = None

    if calc_gnina_score:
        try:
            gnina_score = extract_gnina_score(
                protein_path=str(pdb_file), ligand_path=str(sdf_file)
            )
        except Exception as e:
            try:
                print(
                    f"Error extracting GNINA score for {system_id}: {e}. Trying to fix PDB file."
                )
                # Create temporary file in the same directory as the original PDB
                pdb_dir = Path(pdb_file).parent
                fixed_pdb_file = pdb_dir / f"temp_fixed_{system_id}.pdb"

                try:
                    fix_pdb(str(pdb_file), out_file=str(fixed_pdb_file))
                    gnina_score = extract_gnina_score(
                        protein_path=str(fixed_pdb_file),
                        ligand_path=str(sdf_file),
                    )
                    if gnina_score is not None:
                        print(
                            f"Successfully extracted GNINA score after fixing PDB for {system_id}: {gnina_score}"
                        )
                finally:
                    # Ensure cleanup happens even if an exception occurs
                    if fixed_pdb_file.exists():
                        fixed_pdb_file.unlink()

            except Exception as e:
                print(f"Error extracting GNINA score after fixing PDB: {e}")

    return {
        "gnina_score": gnina_score,
        "system_id": system_id,
    }

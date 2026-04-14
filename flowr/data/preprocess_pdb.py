import argparse
import os
import tempfile
from io import StringIO
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import requests
from rdkit import Chem
from rdkit.Chem import AllChem

# =============================================================================
# Biotite-based preprocessing (recommended)
# =============================================================================


def transform_pdb_biotite(
    out_dir: str,
    pdb_id: str,
    ligand_id: Optional[str] = None,
    chain_id: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Fetch a PDB structure, split into protein and ligand, and write output files.

    Uses biotite for structure parsing (CIF format) and the biotite-RDKit interface
    for ligand conversion with correct bond orders from the Chemical Component
    Dictionary.

    Args:
        out_dir: Output directory for protein PDB and ligand SDF files.
        pdb_id: 4-character PDB ID (e.g. '4ERW').
        ligand_id: 3-letter ligand residue name (e.g. 'STU'). If None, the
            largest non-standard residue is auto-detected.
        chain_id: Chain ID to select a specific ligand copy. If None, the first
            chain containing the ligand is used.

    Returns:
        Tuple of (protein_pdb_path, ligand_sdf_path).

    Raises:
        ValueError: If no ligand is found or the structure cannot be parsed.
    """
    import biotite.database.rcsb as rcsb
    import biotite.sequence as seq
    import biotite.structure.info as info
    import biotite.structure.io.pdb as pdb_io
    import biotite.structure.io.pdbx as pdbx
    from biotite.interface import rdkit as biotite_rdkit

    # --- Fetch and parse structure ---
    with tempfile.TemporaryDirectory() as tmpdir:
        cif_path = rcsb.fetch(pdb_id, "cif", target_path=tmpdir)
        cif_file = pdbx.CIFFile.read(cif_path)
        atoms = pdbx.get_structure(cif_file, model=1, include_bonds=True)

    # --- Identify residue types ---
    aa_names = {
        seq.ProteinSequence.convert_letter_1to3(c)
        for c in seq.ProteinSequence().get_alphabet()
    }
    all_res = set(np.unique(atoms.res_name).tolist())
    non_standard = [r for r in all_res if r not in aa_names]
    print(f"Non-standard residues: {non_standard}")

    # --- Determine ligand residue name ---
    ligand_id = _resolve_ligand_id(ligand_id, atoms, non_standard, pdb_id)

    # --- Select ligand atoms (single chain copy) ---
    ligand_atoms, chain_id = _select_ligand_atoms(atoms, ligand_id, chain_id)

    # --- Build protein mask (everything except the ligand) ---
    protein_atoms = _extract_protein(atoms, ligand_id)

    print(
        f"Protein: {len(protein_atoms)} atoms | "
        f"Ligand {ligand_id} (chain {chain_id}): {len(ligand_atoms)} atoms"
    )

    # --- Convert ligand to RDKit mol with correct bond orders ---
    rd_mol = _ligand_to_rdkit(ligand_atoms, ligand_id, info, biotite_rdkit)

    # --- Write outputs ---
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pdb_file_path = str(out_path / f"{pdb_id}.pdb")
    sdf_file_path = str(out_path / f"{pdb_id}_{ligand_id}_ligand.sdf")

    pdb_out = pdb_io.PDBFile()
    pdb_io.set_structure(pdb_out, protein_atoms)
    pdb_out.write(pdb_file_path)
    print(f"Wrote protein: {pdb_file_path}")

    writer = Chem.SDWriter(sdf_file_path)
    rd_mol.SetProp("_Name", f"{pdb_id}_{ligand_id}")
    writer.write(rd_mol)
    writer.close()
    print(f"Wrote ligand:  {sdf_file_path}")

    return pdb_file_path, sdf_file_path


_SOLVENT_IONS = {
    # Water
    "HOH",
    "WAT",
    "H2O",
    # Common ions
    "NA",
    "CL",
    "K",
    "MG",
    "CA",
    "ZN",
    "FE",
    "MN",
    "CU",
    "NI",
    "CO",
    # Common anions
    "SO4",
    "PO4",
    "NO3",
    "SCN",
    "BR",
    # Capping groups
    "ACE",
    "NME",
    # Cryoprotectants / crystallization additives
    "EDO",
    "GOL",
    "DMS",
    "PEG",
    "MPD",
    "BME",
    "DTT",
    "1PE",
    "P6G",
    "PGE",
    "2PE",
    "PE4",  # PEG variants
    # Common solvents
    "EOH",
    "MOH",
    "IPA",
    "FMT",
    "ACT",
    # Buffer molecules
    "TRS",
    "EPE",
    "MES",
    "CIT",
    "IMD",
    # Common modified amino acids (protein modifications, not ligands)
    "MSE",
    "OCS",
    "SEP",
    "TPO",
    "PTR",
    "CSO",
    "KCX",
    "LLP",
    "HYP",
    "PCA",
}


def _resolve_ligand_id(ligand_id, atoms, non_standard, pdb_id):
    """Auto-detect or validate the ligand residue name."""
    all_res = set(np.unique(atoms.res_name))

    if ligand_id is not None:
        if ligand_id not in all_res:
            raise ValueError(
                f"Ligand '{ligand_id}' not found in {pdb_id}. "
                f"Available non-standard residues: {non_standard}"
            )
        return ligand_id

    candidates = [r for r in non_standard if r not in _SOLVENT_IONS]
    if not candidates:
        raise ValueError(
            f"No ligand residues found in {pdb_id}. "
            f"Non-standard residues: {non_standard}"
        )
    # Pick the candidate with the most heavy atoms per single residue instance
    best, best_count = candidates[0], 0
    for cand in candidates:
        cand_heavy = (atoms.res_name == cand) & (atoms.element != "H")
        if not np.any(cand_heavy):
            continue
        # Count atoms in ONE instance (not all copies across chains)
        cand_atoms = atoms[cand_heavy]
        first_res_id = cand_atoms.res_id[0]
        first_chain = cand_atoms.chain_id[0]
        instance_count = int(
            np.sum(
                (cand_atoms.res_id == first_res_id)
                & (cand_atoms.chain_id == first_chain)
            )
        )
        if instance_count > best_count:
            best, best_count = cand, instance_count
    print(f"Auto-detected ligand: {best} ({best_count} heavy atoms)")
    return best


def _select_ligand_atoms(atoms, ligand_id, chain_id):
    """Select ligand atoms for a single chain copy. Returns (ligand_atoms, chain_id)."""
    lig_mask = atoms.res_name == ligand_id

    if chain_id is not None:
        lig_mask &= atoms.chain_id == chain_id
    else:
        chains_with_lig = np.unique(atoms.chain_id[lig_mask]).tolist()
        if len(chains_with_lig) == 0:
            raise ValueError(f"No atoms found for ligand {ligand_id}")
        chain_id = chains_with_lig[0]
        lig_mask &= atoms.chain_id == chain_id
        if len(chains_with_lig) > 1:
            print(
                f"Ligand {ligand_id} found in chains {chains_with_lig}, "
                f"using chain {chain_id}"
            )

    ligand_atoms = atoms[lig_mask]
    if len(ligand_atoms) == 0:
        raise ValueError(f"No atoms for ligand {ligand_id} in chain {chain_id}")
    return ligand_atoms, chain_id


def _extract_protein(atoms, ligand_id):
    """Extract protein atoms (everything except the ligand), fixing water chain IDs."""
    protein_atoms = atoms[atoms.res_name != ligand_id]

    water_mask = np.isin(protein_atoms.res_name, ["HOH", "WAT", "H2O"])
    empty_chain = water_mask & (protein_atoms.chain_id == "")
    if np.any(empty_chain):
        protein_atoms.chain_id[empty_chain] = "W"

    return protein_atoms


def _ligand_to_rdkit(ligand_atoms, res_name, info_module, biotite_rdkit):
    """
    Convert a biotite ligand AtomArray to an RDKit molecule with correct bond orders.

    Tries three strategies in order:
      1. Biotite CCD reference (ideal bond orders from Chemical Component Dictionary)
      2. RCSB ideal SDF template
      3. RCSB SMILES template
    Falls back to biotite's raw conversion if all template methods fail.
    """
    # Strategy 1: biotite CCD reference → to_mol gives correct bond orders
    try:
        ref_atoms = info_module.residue(res_name)
        if ref_atoms is not None and len(ref_atoms) > 0:
            rd_mol = biotite_rdkit.to_mol(ligand_atoms)
            Chem.SanitizeMol(rd_mol)
            print(f"Bond orders assigned via biotite CCD reference for {res_name}")
            return rd_mol
    except Exception as e:
        print(f"Biotite CCD reference failed for {res_name}: {e}")

    # Strategy 2: RCSB ideal SDF template
    try:
        raw_mol = biotite_rdkit.to_mol(ligand_atoms)
        ideal_sdf = get_ideal_ligand_sdf(res_name)
        if ideal_sdf:
            template = Chem.MolFromMolBlock(ideal_sdf, sanitize=False, removeHs=False)
            if template is not None:
                raw_no_h = Chem.RemoveHs(raw_mol, sanitize=False)
                tmpl_no_h = Chem.RemoveHs(template, sanitize=False)
                rd_mol = AllChem.AssignBondOrdersFromTemplate(tmpl_no_h, raw_no_h)
                Chem.SanitizeMol(rd_mol)
                rd_mol = Chem.AddHs(rd_mol, addCoords=True)
                print(f"Bond orders assigned via RCSB ideal SDF for {res_name}")
                return rd_mol
    except Exception as e:
        print(f"RCSB ideal SDF template failed for {res_name}: {e}")

    # Strategy 3: RCSB SMILES template
    try:
        smiles = get_ligand_smiles(res_name)
        if smiles:
            template = AllChem.MolFromSmiles(smiles)
            if template is not None:
                raw_mol = biotite_rdkit.to_mol(ligand_atoms)
                raw_no_h = Chem.RemoveHs(raw_mol, sanitize=False)
                rd_mol = AllChem.AssignBondOrdersFromTemplate(template, raw_no_h)
                Chem.SanitizeMol(rd_mol)
                rd_mol = Chem.AddHs(rd_mol, addCoords=True)
                print(f"Bond orders assigned via RCSB SMILES for {res_name}")
                return rd_mol
    except Exception as e:
        print(f"RCSB SMILES template failed for {res_name}: {e}")

    # Fallback: raw biotite conversion (bond orders may be wrong)
    print(
        f"Warning: Using raw biotite conversion for {res_name} — bond orders may be incorrect"
    )
    rd_mol = biotite_rdkit.to_mol(ligand_atoms)
    try:
        Chem.SanitizeMol(rd_mol)
    except Exception:
        pass
    return rd_mol


# =============================================================================
# ProDy-based preprocessing (legacy)
# =============================================================================


def get_pdb_components(pdb_id, ligand_id=None):
    from prody import parsePDB

    pdb = parsePDB(pdb_id)
    ligand = pdb.select("not protein and not water")

    if ligand_id is not None:
        # Select everything EXCEPT the specified ligand
        protein_complex = pdb.select(f"not resname {ligand_id}")
    else:
        # If no ligand specified, just exclude first ligand resname
        res_name_list = list(set(ligand.getResnames())) if ligand else []
        if res_name_list:
            protein_complex = pdb.select(f"not resname {res_name_list[0]}")
        else:
            protein_complex = pdb.select("all")

    return protein_complex, ligand


def get_ideal_ligand_sdf(res_name):
    """
    Download the ideal ligand structure from PDB's Chemical Component Dictionary.
    This provides correct bond orders and protonation states.
    """
    # Try to get the ideal SDF from RCSB
    url = f"https://files.rcsb.org/ligands/view/{res_name}_ideal.sdf"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text

    # Fallback to model SDF
    url = f"https://files.rcsb.org/ligands/view/{res_name}_model.sdf"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text

    return None


def get_ligand_smiles(res_name):
    """
    Get the canonical SMILES from RCSB's Chemical Component Dictionary.
    """
    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{res_name}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        # Try to get the canonical SMILES
        descriptors = data.get("rcsb_chem_comp_descriptor", {})
        smiles = descriptors.get("smiles")
        if smiles:
            return smiles
        # Fallback to SMILES_stereo
        smiles_stereo = descriptors.get("smiles_stereo")
        if smiles_stereo:
            return smiles_stereo
    return None


def transform_pdb(main_path, pdb_name, ligand_id, chain_id=None):
    """
    Read Ligand Expo data, split pdb into protein and ligands,
    write protein pdb, write ligand sdf files
    :param pdb_name: id from the pdb, doesn't need to have an extension
    :param chain_id: optional chain ID to select specific ligand copy
    :return:
    """
    protein_complex, ligand = get_pdb_components(pdb_name, ligand_id)
    pdb_file = write_pdb(main_path, protein_complex, pdb_name)
    res_name_list = list(set(ligand.getResnames()))
    if ligand_id is not None:
        for res in res_name_list:
            if ligand_id in res:
                new_mol = process_ligand(ligand, res, chain_id)
                sdf_file = write_sdf(main_path, new_mol, pdb_name, res)
    else:
        new_mol = process_ligand(ligand, res_name_list[0], chain_id)
        sdf_file = write_sdf(main_path, new_mol, pdb_name, res_name_list[0])
    return pdb_file, sdf_file


def process_ligand(ligand, res_name, chain_id=None):
    """
    Process ligand with correct bond orders and protonation states.
    """
    from prody import writePDBStream

    output = StringIO()
    if chain_id:
        sub_mol = ligand.select(f"resname {res_name} and chain {chain_id}")
    else:
        # Get first chain only
        ligand_subset = ligand.select(f"resname {res_name}")
        chains = list(set(ligand_subset.getChids()))
        sub_mol = ligand.select(f"resname {res_name} and chain {chains[0]}")

    writePDBStream(output, sub_mol)
    pdb_string = output.getvalue()
    rd_mol = AllChem.MolFromPDBBlock(pdb_string, removeHs=False)

    if rd_mol is None:
        raise ValueError(f"Could not parse ligand {res_name} from PDB block")

    # Method 1: Try to get template from RCSB's ideal SDF
    try:
        ideal_sdf = get_ideal_ligand_sdf(res_name)
        if ideal_sdf:
            template = Chem.MolFromMolBlock(ideal_sdf, removeHs=False)
            if template is not None:
                # Remove hydrogens for matching, then assign bond orders
                template_no_h = Chem.RemoveHs(template)
                rd_mol_no_h = Chem.RemoveHs(rd_mol)
                new_mol = AllChem.AssignBondOrdersFromTemplate(
                    template_no_h, rd_mol_no_h
                )
                # Add hydrogens back with correct 3D coordinates
                new_mol = Chem.AddHs(new_mol, addCoords=True)
                print(
                    f"Successfully assigned bond orders from ideal SDF for {res_name}"
                )
                return new_mol
    except Exception as e:
        print(f"Failed to use ideal SDF template: {e}")

    # Method 2: Try SMILES from RCSB
    try:
        smiles = get_ligand_smiles(res_name)
        if smiles:
            template = AllChem.MolFromSmiles(smiles)
            if template is not None:
                rd_mol_no_h = Chem.RemoveHs(rd_mol)
                new_mol = AllChem.AssignBondOrdersFromTemplate(template, rd_mol_no_h)
                new_mol = Chem.AddHs(new_mol, addCoords=True)
                print(f"Successfully assigned bond orders from SMILES for {res_name}")
                return new_mol
    except Exception as e:
        print(f"Failed to use SMILES template: {e}")

    # Method 3: Fallback - use RDKit's perception
    print(
        f"Warning: Using RDKit bond perception for {res_name} - results may be incorrect"
    )
    try:
        Chem.SanitizeMol(rd_mol)
    except Exception:
        pass

    return rd_mol


def write_pdb(main_path, protein, pdb_name):
    from prody import writePDB

    output_pdb_name = os.path.join(main_path, f"{pdb_name}.pdb")
    writePDB(f"{output_pdb_name}", protein)
    print(f"wrote {output_pdb_name}")
    return output_pdb_name


def write_sdf(main_path, new_mol, pdb_name, res_name):
    """
    Write an RDKit molecule to an SD file
    :param new_mol:
    :param pdb_name:
    :param res_name:
    :return:
    """
    outfile_name = os.path.join(main_path, f"{pdb_name}_{res_name}_ligand.sdf")
    writer = Chem.SDWriter(f"{outfile_name}")
    writer.write(new_mol)
    writer.close()
    print(f"wrote {outfile_name}")
    return outfile_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch a PDB structure, split into protein + ligand, "
        "and write .pdb / .sdf files."
    )
    parser.add_argument(
        "--main-path",
        type=Path,
        required=True,
        help="Output directory for protein PDB and ligand SDF files.",
    )
    parser.add_argument(
        "--pdb-id",
        type=str,
        required=True,
        help="The PDB ID of the Protein. E.g., 4ERW",
    )
    parser.add_argument(
        "--ligand-id",
        type=str,
        default=None,
        help="Ligand residue name (e.g. STU). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--chain-id",
        type=str,
        default=None,
        help="Chain ID to select a specific ligand copy (e.g. 'A').",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="biotite",
        choices=["biotite", "prody"],
        help="Structure parsing backend (default: biotite).",
    )
    args = parser.parse_args()

    pdb_sdf_dir = args.main_path
    pdb_sdf_dir.mkdir(exist_ok=True, parents=True)

    if args.backend == "biotite":
        pdb_file, sdf_file = transform_pdb_biotite(
            str(args.main_path), args.pdb_id, args.ligand_id, args.chain_id
        )
    else:
        if args.ligand_id is None:
            parser.error("--ligand-id is required when using the prody backend")
        pdb_file, sdf_file = transform_pdb(
            args.main_path, args.pdb_id, args.ligand_id, args.chain_id
        )

    print(f"pdb file: {pdb_file}")
    print(f"sdf file: {sdf_file}")

import argparse
import os
from io import StringIO
from pathlib import Path

import pypdb
from prody import parsePDB, writePDB, writePDBStream
from rdkit import Chem
from rdkit.Chem import AllChem


def get_pdb_components(pdb_id):
    pdb = parsePDB(pdb_id)
    protein = pdb.select("protein")
    ligand = pdb.select("not protein and not water")
    return protein, ligand


def process_ligand(ligand, res_name):
    output = StringIO()
    sub_mol = ligand.select(f"resname {res_name}")
    chem_desc = pypdb.describe_chemical(f"{res_name}")
    for item in chem_desc.get("pdbx_chem_comp_descriptor", []):
        if item.get("type") == "SMILES":
            sub_smiles = item.get("descriptor")
            break
    template = AllChem.MolFromSmiles(sub_smiles)
    writePDBStream(output, sub_mol)
    pdb_string = output.getvalue()
    rd_mol = AllChem.MolFromPDBBlock(pdb_string)
    try:
        new_mol = AllChem.AssignBondOrdersFromTemplate(template, rd_mol)
        return new_mol
    except Exception:
        return rd_mol


def write_pdb(main_path, protein, pdb_name):
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
    print(f"wrote {outfile_name}")
    return outfile_name


def transform_pdb(main_path, pdb_name, ligand_id):
    """
    Read Ligand Expo data, split pdb into protein and ligands,
    write protein pdb, write ligand sdf files
    :param pdb_name: id from the pdb, doesn't need to have an extension
    :return:
    """
    protein, ligand = get_pdb_components(pdb_name)
    pdb_file = write_pdb(main_path, protein, pdb_name)
    res_name_list = list(set(ligand.getResnames()))
    if ligand_id is not None:
        for res in res_name_list:
            if ligand_id in res:
                new_mol = process_ligand(ligand, res)
                sdf_file = write_sdf(main_path, new_mol, pdb_name, res)
    else:
        new_mol = process_ligand(ligand, res_name_list[0])
        sdf_file = write_sdf(main_path, new_mol, pdb_name, res_name_list[0])
    return pdb_file, sdf_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-path", type=Path)
    parser.add_argument(
        "--pdb-id", type=str, help="The PDB ID of the Protein. E.g., 4ERW"
    )
    parser.add_argument(
        "--ligand-id",
        type=str,
        help="The ligand identifier, which is taken from the residue name list. Fpr 4ERW, this corresponds to STU.",
    )
    args = parser.parse_args()

    pdb_sdf_dir = args.main_path
    pdb_sdf_dir.mkdir(exist_ok=True, parents=True)

    pdb_file, sdf_file = transform_pdb(args.main_path, args.pdb_id, args.ligand_id)
    print(f"pdb file: {pdb_file}")
    print(f"sdf file: {sdf_file}")

import argparse
import os

from rdkit import Chem
from rdkit.Chem import PandasTools

import useful_rdkit_utils as uru


def write_sdf_file(sdf_path, molecules, extract_mol=False):
    w = Chem.SDWriter(str(sdf_path))
    for m in molecules:
        if extract_mol:
            if m.rdkit_mol is not None:
                w.write(m.rdkit_mol)
        else:
            if m is not None:
                w.write(m)
    w.close()


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Aggregating evaluation dictionaries')
    parser.add_argument('--sdf-path', type=str, help='Path to unfiltered molecules as sdf file')
    parser.add_argument("--out-path", type=str, help='Path to store the filtered molecules as sdf file')
    parser.add_argument("--keep-duplicates", default=False, action="store_true")
    parser.add_argument("--docking-mode", default=None, type=str)
    args = parser.parse_args()
    return args


def walters_filter(args):
    df_0 = PandasTools.LoadSDF(args.sdf_path)

    # remove fragments
    df_0["num_frags"] = df_0.ROMol.apply(uru.count_fragments)
    df_1 = df_0.query("num_frags == 1").copy()
    print(f"Fragments removed: {len(df_0) - len(df_1)}")

    if not args.keep_duplicates:
        # remove InChi duplicates
        df_1["inchi"] = df_1.ROMol.apply(Chem.MolToInchiKey)
        df_2 = df_1.drop_duplicates(subset="inchi").copy()
        print(f"Duplicates removed: {len(df_1) - len(df_2)}")
    else:
        df_2 = df_1.copy()

    # remove odd ring systems
    ring_system_lookup = uru.RingSystemLookup.default()
    df_2["ring_systems"] = [list(ring_system_lookup.process_mol(x)) for x in df_2.ROMol]
    df_2[["min_ring", "min_freq"]] = [
        list(uru.get_min_ring_frequency(x)) for x in df_2.ring_systems
    ]
    df_3 = df_2.query("min_freq >= 100").copy()
    print(f"Odd ring systems removed: {len(df_2) - len(df_3)}")
    print(
        f"Percentage of odd ring systems overall: {100 * (len(df_2) - len(df_3)) / len(df_2)}"
    )

    # remove reactive or odd functional groups
    reos = uru.REOS()
    reos.set_active_rule_sets(["Dundee"])
    df_2[["rule_set", "reos"]] = [list(reos.process_mol(x)) for x in df_2.ROMol]
    df_3[["rule_set", "reos"]] = [list(reos.process_mol(x)) for x in df_3.ROMol]
    df_fgroups = df_2.query("reos == 'ok'").copy()
    df_4 = df_3.query("reos == 'ok'").copy()
    print(f"Wrong functional groups removed: {len(df_3) - len(df_4)}")
    print(
        f"Percentage of wrong functional groups overall: {100 * (len(df_2) - len(df_fgroups)) / len(df_2)}"
    )

    if args.docking_mode is not None:
        processed = args.docking_mode
        # sort by docking scores and save
        df_4[args.docking_mode] = df_4[args.docking_mode].astype(float)
        df_final = df_4.sort_values(args.docking_mode, ascending=True).reset_index(
            drop=True
        )
    else:
        processed = "sampled"
        df_final = df_4.copy()
    sdf_path = os.path.join(args.out_path, f"all_{processed}_mols_walters_filter.sdf")
    mols = list(df_final["ROMol"])
    write_sdf_file(sdf_path, mols)

    print(f"Filtering done. {len(df_4)} / {len(df_2)} unique molecules remaining!")
    print(f"Filtered molecules saved as sdf at {sdf_path}")


if __name__ == "__main__":
    args = get_args()
    walters_filter(args)

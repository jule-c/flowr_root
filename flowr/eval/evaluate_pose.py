from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdShapeHelpers


def _heavy_atom_indices(m: Chem.Mol) -> List[int]:
    """Return indices of heavy atoms (non‑hydrogen)."""
    return [i for i, a in enumerate(m.GetAtoms()) if a.GetAtomicNum() > 1]


def rmsd(gen: Chem.Mol, ref: Chem.Mol, align: bool = True) -> float:
    """
    Heavy‑atom RMSD after optimal alignment (Kabsch).
    Coordinates are NOT modified.
    """
    ref_ids = _heavy_atom_indices(ref)
    gen_ids = _heavy_atom_indices(gen)
    if len(ref_ids) != len(gen_ids):
        raise ValueError(
            "Generated and reference ligand must have the same "
            "number of heavy atoms for RMSD calculation."
        )
    # RDKit aligns gen -> ref internally and returns RMSD
    if align:
        rmsd_value = AllChem.GetBestRMS(ref, gen, ref_ids, gen_ids)
    else:
        rmsd_value = AllChem.CalcRMS(ref, gen, ref_ids, gen_ids)
    return rmsd_value


def shape_tanimoto_similarity(
    gen: Chem.Mol, ref: Chem.Mol, align: bool = True
) -> float:
    """
    Ultrafast shape‑based similarity (after alignment).
    """
    # Needs a conformer on both mols
    if gen.GetNumConformers() == 0 or ref.GetNumConformers() == 0:
        raise ValueError("Both molecules need 3‑D coordinates for shape Tanimoto.")
    if align:
        # Align by crippen O3A (chemistry sensitive) to ensure correct orientation
        AllChem.AssignBondOrdersFromTemplate(ref, gen)  # best‑effort, ignore failure
        o3a = rdMolAlign.GetO3A(gen, ref)
        o3a.Align()
    return 1.0 - rdShapeHelpers.ShapeTanimotoDist(gen, ref)


def _load_pocket_atoms(pocket_pdb_path: Path):
    """
    Return numpy array of pocket heavy‑atom coordinates.
    Only very light dependency on RDKit. For more elaborate needs use MDAnalysis.
    """
    pocket = Chem.MolFromPDBFile(str(pocket_pdb_path), removeHs=True)
    if pocket is None:
        raise IOError(f"Could not read pocket PDB: {pocket_pdb_path}")
    conf = pocket.GetConformer()
    coords = np.array(
        [list(conf.GetAtomPosition(i)) for i in _heavy_atom_indices(pocket)]
    )
    return coords


def clashes(gen: Chem.Mol, pocket_coords: np.ndarray, clash_dist: float = 2.0) -> int:
    """
    Count ligand heavy atoms closer than `clash_dist` Å to any protein heavy atom.
    """
    gconf = gen.GetConformer()
    gcoords = np.array(
        [list(gconf.GetAtomPosition(i)) for i in _heavy_atom_indices(gen)]
    )
    # broadcast distances
    dists = np.linalg.norm(gcoords[:, None, :] - pocket_coords[None, :, :], axis=-1)
    return int((dists < clash_dist).any(axis=1).sum())


def ifp_similarity(gen: Chem.Mol, ref: Chem.Mol, pocket_pdb_path: Path) -> float | None:
    """
    ProLIF IFP Tanimoto similarity, or None if ProLIF unavailable.
    """
    try:
        import prolif as plf
    except ModuleNotFoundError:
        return None  # Metric skipped

    # Build Protein object
    prot = plf.prolif_io.PDBFile.read(str(pocket_pdb_path)).to_rdkit()
    factory = plf.fingerprints.InteractionFingerprint(prot)
    vec_gen = factory.generate(gen)
    vec_ref = factory.generate(ref)
    return plf.metrics.tanimoto(vec_gen, vec_ref)


# --------------------------------------------------------------------------- #
# Main API                                                                     #
# --------------------------------------------------------------------------- #


def evaluate_poses(
    gen_mols: Iterable[Chem.Mol],
    ref_mol: Chem.Mol,
    pocket_pdb: str | Path,
    thresholds: Iterable[float] = (2.0, 3.0, 5.0),
) -> Dict[str, Any]:
    """
    Evaluate a collection of generated poses.

    Returns
    -------
    dict with keys:
        • 'per_pose' – pandas.DataFrame with all raw metrics
        • 'summary'  – dict of aggregated numbers
    """
    pocket_pdb = Path(pocket_pdb)
    pocket_coords = _load_pocket_atoms(pocket_pdb)

    records = []
    for i, gen in enumerate(gen_mols, 1):
        r = {
            "pose_id": i,
            "rmsd": rmsd(gen, ref_mol),
            "shape": shape_tanimoto_similarity(gen, ref_mol),
            "clashes": clashes(gen, pocket_coords),
        }
        ifp = ifp_similarity(gen, ref_mol, pocket_pdb)
        if ifp is not None:
            r["ifp"] = ifp
        records.append(r)

    df = pd.DataFrame.from_records(records).set_index("pose_id")

    # ------- aggregated stats --------------------------------------------- #
    summary = {
        "n_poses": len(df),
        "rmsd_mean": df["rmsd"].mean(),
        "rmsd_median": df["rmsd"].median(),
        "rmsd_std": df["rmsd"].std(ddof=0),
    }

    for t in thresholds:
        summary[f"frac_rmsd_lt_{t:.0f}A"] = (df["rmsd"] < t).mean()

    summary["shape_mean"] = df["shape"].mean()
    if "ifp" in df:
        summary["ifp_mean"] = df["ifp"].mean()
    summary["clash_mean"] = df["clashes"].mean()

    return {"per_pose": df, "summary": summary}


# --------------------------------------------------------------------------- #
# CLI (optional)                                                              #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Evaluate generated ligand poses.")
    ap.add_argument("ref_sdf", help="Reference ligand SDF (1 conformer)")
    ap.add_argument("gen_sdf", help="SDF of generated poses (multi‑mol OK)")
    ap.add_argument("pocket_pdb", help="Pocket PDB structure")
    ap.add_argument("-o", "--out", default="pose_eval.json", help="Output json file")
    args = ap.parse_args()

    ref = Chem.SDMolSupplier(args.ref_sdf, removeHs=False)[0]
    gens = Chem.SDMolSupplier(args.gen_sdf, removeHs=False)

    result = evaluate_poses(gens, ref, args.pocket_pdb)
    Path(args.out).write_text(json.dumps(result["summary"], indent=2))
    print(result["summary"])

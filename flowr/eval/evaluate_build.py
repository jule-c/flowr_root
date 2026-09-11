"""Build-outcome census over a generation run, for reading a validity A/B.

`evaluate_validity` in `flowr.util.metrics` computes the two numbers that matter but has no
caller, and generation itself prints only one rate -- and that one is `mol_is_valid(...,
connected=True)`, i.e. fully-connected validity alone. That single number cannot separate
the two failure modes it aggregates, which is exactly the distinction needed to read a
decode repair: a repair that escapes an over-valence by DELETING a bond turns a build
failure into a DISCONNECTED molecule, which lifts plain validity while leaving
fully-connected validity flat. Reported as one rate, that reads as an improvement.

So this splits the population four ways, and the four are exhaustive:

    failed        the builder returned None -- no molecule at all
    disconnected  a molecule that sanitises but is more than one fragment
    empty         a molecule that sanitises but has zero atoms
    fc_valid      sanitises AND is exactly one connected fragment

    validity     = (disconnected + empty + fc_valid) / total
    fc_validity  = fc_valid / total
    validity - fc_validity  is the disconnection share, the term to watch

Run generation WITHOUT `--filter_valid_unique` before using this: with the filter on, the
saved `gen_ligs` has already been reduced to the fully-connected survivors and every rate
below reads 1.0.

    python -m flowr.eval.evaluate_build RUN_DIR [RUN_DIR ...] [--json out.json]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
from rdkit import Chem, RDLogger

import flowr.util.rdkit as smolRD

RDLogger.DisableLog("rdApp.*")


def _flatten(ligs: Any) -> list:
    """`gen_ligs` is a flat list from the PDB path and a list-of-lists from the LMDB path."""
    out = []
    for item in ligs if isinstance(ligs, (list, tuple)) else [ligs]:
        if isinstance(item, (list, tuple)):
            out.extend(_flatten(item))
        else:
            out.append(item)
    return out


def census(mols: Iterable) -> dict[str, int]:
    """Count the four mutually exclusive build outcomes."""
    counts = {"failed": 0, "empty": 0, "disconnected": 0, "fc_valid": 0}
    for mol in mols:
        if mol is None or not smolRD.mol_is_valid(mol, connected=False):
            counts["failed"] += 1
        elif mol.GetNumAtoms() == 0:
            counts["empty"] += 1
        elif len(Chem.GetMolFrags(mol)) != 1:
            counts["disconnected"] += 1
        else:
            counts["fc_valid"] += 1
    return counts


def rates(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {"total": 0, "validity": 0.0, "fc_validity": 0.0, "disconnection": 0.0}
    built = counts["empty"] + counts["disconnected"] + counts["fc_valid"]
    return {
        "total": total,
        "validity": built / total,
        "fc_validity": counts["fc_valid"] / total,
        "disconnection": (built - counts["fc_valid"]) / total,
    }


def evaluate_run(run_dir: Path) -> dict[str, Any]:
    """Census every `samples*.pt` under one run directory."""
    files = sorted(run_dir.glob("samples*.pt")) if run_dir.is_dir() else [run_dir]
    if not files:
        raise FileNotFoundError(f"no samples*.pt under {run_dir}")
    mols: list = []
    for path in files:
        payload = torch.load(path, weights_only=False)
        mols.extend(_flatten(payload.get("gen_ligs", [])))
    counts = census(mols)
    return {
        "run": str(run_dir),
        "files": [f.name for f in files],
        **counts,
        **rates(counts),
    }


def _row(result: dict[str, Any]) -> str:
    return (
        f"{Path(result['run']).name[:34]:<34} {result['total']:>7} "
        f"{result['failed']:>7} {result['disconnected']:>7} {result['fc_valid']:>8} "
        f"{result['validity']:>9.4f} {result['fc_validity']:>12.4f}"
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dirs", nargs="+", help="generation --save_dir (or a samples*.pt)")
    ap.add_argument("--json", dest="json_out", default=None, help="also write JSON here")
    args = ap.parse_args()

    results = [evaluate_run(Path(d)) for d in args.run_dirs]

    header = (
        f"{'run':<34} {'total':>7} {'failed':>7} {'disconn':>7} {'fc_valid':>8} "
        f"{'validity':>9} {'fc_validity':>12}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(_row(result))

    if len(results) == 2:
        a, b = results
        print()
        print(f"delta ({Path(b['run']).name} - {Path(a['run']).name}):")
        print(f"  validity     {b['validity'] - a['validity']:+.4f}")
        print(f"  fc_validity  {b['fc_validity'] - a['fc_validity']:+.4f}")
        print(f"  failed       {b['failed'] - a['failed']:+d}")
        print(f"  disconnected {b['disconnected'] - a['disconnected']:+d}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json_out}")

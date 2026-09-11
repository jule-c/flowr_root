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
    sampled = 0
    prefiltered = None
    repair: dict | None = None
    for path in files:
        payload = torch.load(path, weights_only=False)
        mols.extend(_flatten(payload.get("gen_ligs", [])))
        sampled += int(payload.get("n_sampled") or 0)
        if payload.get("prefiltered") is not None:
            prefiltered = bool(payload["prefiltered"]) or bool(prefiltered)
        if payload.get("repair_stats"):
            repair = payload["repair_stats"]
    counts = census(mols)
    result = {
        "run": str(run_dir),
        "files": [f.name for f in files],
        **counts,
        **rates(counts),
        "n_sampled": sampled,
        "prefiltered": prefiltered,
        "repair_stats": repair,
    }
    # `sanitize_list` keeps only `mol_is_valid(..., connected=True)`, and it runs whether or
    # not --filter_valid_unique was passed. So the saved population is ALWAYS fully-connected
    # valid and the census above reads 100% on every run. The only honest rate is the YIELD:
    # survivors over what the model actually produced, which `n_sampled` preserves.
    if sampled:
        result["yield"] = counts["fc_valid"] / sampled
        result["lost"] = sampled - counts["fc_valid"]
    return result


def _row(result: dict[str, Any]) -> str:
    sampled = result.get("n_sampled") or 0
    return (
        f"{Path(result['run']).name[:30]:<30} {sampled:>9} {result['fc_valid']:>9} "
        f"{result.get('lost', 0):>6} "
        f"{(result.get('yield') or 0.0):>8.4f} {result['disconnected']:>8}"
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dirs", nargs="+", help="generation --save_dir (or a samples*.pt)")
    ap.add_argument("--json", dest="json_out", default=None, help="also write JSON here")
    args = ap.parse_args()

    results = [evaluate_run(Path(d)) for d in args.run_dirs]

    header = (
        f"{'run':<30} {'n_sampled':>9} {'fc_valid':>9} {'lost':>6} {'yield':>8} {'disconn':>8}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(_row(result))

    if len(results) == 2:
        a, b = results
        print()
        print(f"delta ({Path(b['run']).name} - {Path(a['run']).name}):")
        print(f"  yield        {(b.get('yield') or 0) - (a.get('yield') or 0):+.4f}")
        print(f"  fc_valid     {b['fc_valid'] - a['fc_valid']:+d}")
        print(f"  lost         {(b.get('lost') or 0) - (a.get('lost') or 0):+d}")
        if (b.get("n_sampled") or 0) != (a.get("n_sampled") or 0):
            print("  WARNING: the arms did not sample the same number of molecules, so the")
            print("           counts are not directly comparable -- compare yield instead.")

    for result in results:
        if result.get("repair_stats"):
            r = result["repair_stats"]
            print(
                f"\n{Path(result['run']).name}: valence repair "
                f"{r['repaired']}/{r['attempted']} repaired "
                f"({r['repaired_ok']} connected, {r['repaired_disconnected']} disconnected), "
                f"{r['unrepaired']} unrepairable, {r['rejected']} rejected, "
                f"bond deletions {r['edits_bond_deletions']}, "
                f"caps {r['cap_states']}/{r['cap_edits']}"
            )
    if any(r.get("prefiltered") for r in results):
        print(
            "\nNOTE: at least one run used --filter_valid_unique, so its population was "
            "filtered in-loop and its yield understates what the model produced."
        )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json_out}")

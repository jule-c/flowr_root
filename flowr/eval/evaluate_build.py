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
    python -m flowr.eval.evaluate_build RUN_DIR ... --posebusters [--pb-limit 500]

`--posebusters` adds the quality axis. Build success alone cannot say whether a change that
delivers MORE molecules delivers WORSE ones, and that is the question any change to the
decode or the sampler has to answer. It runs the `dock` config against the pocket the run
itself wrote out, on CPU, so it is deliberately separate from generation.
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
        "disconnection": counts["disconnected"] / total,
    }


def posebusters_validity(mols: list, pdb_file: str, limit: int | None = None) -> dict:
    """Fraction of molecules passing the full PoseBusters `dock` battery.

    Subsample with `limit` when the point is to compare arms rather than to certify a run:
    the battery is the slow part by orders of magnitude, and a few hundred molecules already
    separates rates that differ by more than a couple of points.
    """
    from flowr.util.metrics import evaluate_pb_validity

    usable = [m for m in mols if m is not None]
    if limit is not None and len(usable) > limit:
        step = len(usable) / limit  # even stride, not the first N -- order is batch order
        usable = [usable[int(i * step)] for i in range(limit)]
    if not usable:
        return {"pb_n": 0, "pb_validity": None}
    flags = evaluate_pb_validity(usable, pdb_file=pdb_file, return_list=True)
    return {"pb_n": len(flags), "pb_validity": float(sum(flags)) / len(flags)}


def _pocket_pdb(run_dir: Path, payload: dict) -> str | None:
    """The pocket this run actually generated into, for the PoseBusters protein term."""
    ref = payload.get("ref_pdb")
    if isinstance(ref, str) and Path(ref).is_file():
        return ref
    hits = sorted(run_dir.glob("ref_pdbs/*.pdb")) + sorted(run_dir.glob("*.pdb"))
    return str(hits[0]) if hits else None


def evaluate_run(run_dir: Path, posebusters: bool = False, pb_limit=None) -> dict[str, Any]:
    """Census every `samples*.pt` under one run directory."""
    files = sorted(run_dir.glob("samples*.pt")) if run_dir.is_dir() else [run_dir]
    if not files:
        raise FileNotFoundError(f"no samples*.pt under {run_dir}")
    mols: list = []
    sampled = 0
    survived = 0
    have_survived = False
    prefiltered = None
    repair: dict | None = None
    for path in files:
        payload = torch.load(path, weights_only=False)
        mols.extend(_flatten(payload.get("gen_ligs", [])))
        sampled += int(payload.get("n_sampled") or 0)
        if payload.get("n_fc_valid") is not None:
            survived += int(payload["n_fc_valid"])
            have_survived = True
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
    # The numerator is the run's own untruncated survivor count when it recorded one; the
    # delivered list is capped at --sample_n_molecules_per_target and would understate.
    numerator = survived if have_survived else counts["fc_valid"]
    if sampled:
        result["n_fc_valid"] = numerator
        result["yield"] = numerator / sampled
        result["lost"] = sampled - numerator
    else:
        # NOT zero -- unknown. Only `generate_from_pdb` records `n_sampled`; for every other
        # entrypoint the denominator simply is not in the file, and reporting 0.0000 there
        # would be indistinguishable from a run that lost everything.
        result["yield"] = None
        result["lost"] = None
    if posebusters:
        pdb = _pocket_pdb(run_dir, payload)
        if pdb is None:
            result.update({"pb_n": 0, "pb_validity": None, "pb_note": "no pocket pdb found"})
        else:
            result.update(posebusters_validity(mols, pdb, pb_limit))
    return result


def _row(result: dict[str, Any]) -> str:
    sampled = result.get("n_sampled") or 0
    fc = result.get("n_fc_valid", result["fc_valid"])
    y = result.get("yield")
    lost = result.get("lost")
    row = (
        f"{Path(result['run']).name[:30]:<30} "
        f"{(sampled if sampled else '?'):>9} {fc:>9} "
        f"{(lost if lost is not None else '?'):>6} "
        f"{(f'{y:.4f}' if y is not None else '?'):>8} {result['disconnected']:>8}"
    )
    if "pb_validity" in result:
        pb = result["pb_validity"]
        row += f" {(f'{pb:.4f}' if pb is not None else '?'):>11}"
    return row


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dirs", nargs="+", help="generation --save_dir (or a samples*.pt)")
    ap.add_argument("--json", dest="json_out", default=None, help="also write JSON here")
    ap.add_argument("--posebusters", action="store_true",
                    help="also run the PoseBusters dock battery (slow, CPU)")
    ap.add_argument("--pb-limit", type=int, default=None,
                    help="evenly subsample this many molecules per run for PoseBusters")
    args = ap.parse_args()

    results = [
        evaluate_run(Path(d), posebusters=args.posebusters, pb_limit=args.pb_limit)
        for d in args.run_dirs
    ]

    header = (
        f"{'run':<30} {'n_sampled':>9} {'fc_valid':>9} {'lost':>6} {'yield':>8} {'disconn':>8}"
        + (f" {'pb_validity':>11}" if any("pb_validity" in r for r in results) else "")
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(_row(result))

    if len(results) == 2:
        a, b = results
        print()
        print(f"delta ({Path(b['run']).name} - {Path(a['run']).name}):")
        if a.get("yield") is None or b.get("yield") is None:
            print("  yield        UNKNOWN -- at least one run did not record `n_sampled`,")
            print("               so there is no denominator and no delta to report.")
        else:
            print(f"  yield        {b['yield'] - a['yield']:+.4f}")
            print(f"  fc_valid     {b['n_fc_valid'] - a['n_fc_valid']:+d}")
            print(f"  lost         {b['lost'] - a['lost']:+d}")
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

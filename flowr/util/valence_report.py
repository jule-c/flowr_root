"""How much valence RDKit will accept on an atom, and which decoded atoms exceed it.

`MolBuilder` turns the three discrete heads into a molecule by argmaxing them
INDEPENDENTLY (`_mol_extract_atomics` / `_mol_extract_charges` / `_mol_extract_bonds`).
Independent argmaxes of three heads that are not independent is exactly how a chemically
impossible atom arises, and RDKit then refuses to sanitise it
(`AtomValenceException` -> `mol_from_atoms` returns None). This module answers the two
questions needed to do anything about that: what the limit IS, and which atoms break it.

Everything here is pure and side-effect free: it reads decoded arrays and returns numbers.
It constructs no molecule the caller can see -- the only RDKit work is the one-off probe
below, on throw-away scratch molecules.

THE LIMITS ARE PROBED OUT OF THE SANITISER, NOT READ OFF THE PERIODIC TABLE
--------------------------------------------------------------------------
RDKit rewrote its valence model in 2022.09 and the widely-copied
`max(GetValenceList(element)) + formal_charge` rule now disagrees with the sanitiser that
actually raises `AtomValenceException`. Measured in this repo's venv (RDKit 2026.03.6)
over this exact vocabulary, that rule differs from the probe on 54 of the 98
(element, charge) pairs -- so a table derived from it would describe different chemistry
from the function it exists to predict. `_probe_max_valence` asks the sanitiser directly,
which also keeps the table correct across RDKit upgrades for free.

NOT PORTED FROM THE DONOR BRANCH
--------------------------------
The donor's counterfactual-census half (`DecodedGraph`, `counterfactual_labels`,
`triple_bin` and friends) is deliberately absent: it depends on a build-failure census and
a valence loss that this repo does not have. What remains is the part `valence_repair`
needs, and it has no `models` dependency, so `util` never imports `models`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

from flowr.constants import CORE_ATOMS

# Valence contributed by one bond of each class, indexed by the bond head's class id:
# none / single / double / triple / aromatic. An aromatic bond contributes 1.5 (RDKit's own
# `Bond::getValenceContrib`), so an aromatic ring carbon reads 1.5 + 1.5 + 1 = 4 -- the same
# arithmetic the sanitiser does.
VOCAB_BOND_ORDERS = (0.0, 1.0, 2.0, 3.0, 1.5)

# Highest bond count probed when deriving the table. Every real limit is <= 6 (S/Se reach 6,
# so do P(-1), Si(-2), Cl(+1), I(+1), Bi(-1)), so an atom that still sanitises with 8 single
# bonds has no enforceable limit at all and is given the sentinel below.
_VALENCE_PROBE_CEILING = 8

# "No limit": large and FINITE (never `inf`, which would make `inf * 0` a NaN in any
# arithmetic a caller layers on top). Nothing a molecule can reach comes near it, so an
# atom carrying it is never reported as over-valent.
UNLIMITED_VALENCE = 1.0e4

# Charge vocabulary is ["<PAD>"] + [0, 1, 2, 3, -1, -2, -3] (`scriptutil._build_vocab_charges`).
_VOCAB_CHARGES = (0, 1, 2, 3, -1, -2, -3)


def _probe_max_valence(symbol: str, charge: int) -> float:
    """Largest number of single bonds RDKit will accept on ``symbol`` at ``charge``.

    Asks the SANITISER, not the periodic table: RDKit rewrote its valence model in 2022.09
    and the widely-copied ``max(GetValenceList(Z)) -/+ formal_charge`` rule now disagrees
    with it for many (element, charge) pairs (e.g. N(+2), P(-2), I(-2), S(-1)). Deriving the
    limit from the very function that raises `AtomValenceException` makes this table a
    surrogate for the real failure by construction, and keeps it correct across RDKit
    upgrades. The probe reproduces `mol_from_atoms`'s build exactly: heavy atoms only, no
    explicit hydrogens, `UpdatePropertyCache(strict=False)` then `SanitizeMol`.

    ACCEPTANCE IS NOT MONOTONE, so the whole range is scanned and the LARGEST accepted bond
    count is returned. An earlier version returned on the FIRST rejection, which silently
    assumes the accept-set is downward-closed. It is not: RDKit rejects a BARE ``H(+2)`` but
    accepts it with 1..8 bonds, and ``H(-3)`` accepts 1 and 2 but not 0. The early return
    therefore handed back ``-1.0`` for all four such cells, and a negative limit makes
    ``valence > limit`` true even at ZERO bonds -- an irreducible false positive. Those four
    holes at ``n = 0`` are simply not representable by an excess-only check; they are all
    hydrogen at a charge that does not exist, and hydrogen is stripped from the ligand here.

    Returns ``UNLIMITED_VALENCE`` when even ``_VALENCE_PROBE_CEILING`` bonds are accepted,
    and ``-1.0`` only if NO bond count at all is accepted (no such cell exists in the
    shipped vocabulary; `test_no_table_entry_is_negative` is the alarm if one appears).
    """
    from rdkit import Chem, rdBase

    best = -1.0
    with rdBase.BlockLogs():
        for n_bonds in range(_VALENCE_PROBE_CEILING + 1):
            editable = Chem.RWMol()
            atom = Chem.Atom(symbol)
            atom.SetFormalCharge(charge)
            centre = editable.AddAtom(atom)
            for _ in range(n_bonds):
                editable.AddBond(
                    centre, editable.AddAtom(Chem.Atom("F")), Chem.BondType.SINGLE
                )
            mol = editable.GetMol()
            try:
                for probe_atom in mol.GetAtoms():
                    probe_atom.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(mol)
            except Exception:
                continue
            best = float(n_bonds)
    return UNLIMITED_VALENCE if best == float(_VALENCE_PROBE_CEILING) else best


@lru_cache(maxsize=None)
def valence_limit_rows() -> tuple[tuple[float, ...], ...]:
    """The full ``[1 + len(CORE_ATOMS) + 1, 1 + len(_VOCAB_CHARGES)]`` limit table.

    Laid out to match the decode's own class indices: row 0 is the atom head's ``<PAD>``
    class, rows 1..14 are `CORE_ATOMS` in vocabulary order, and the trailing row is the
    ``<NOATOM>`` virtual-node class that `_build_vocab(virtual_nodes=True)` appends. Column
    0 is the charge head's ``<PAD>`` class. Every sentinel row and the sentinel column are
    ``UNLIMITED_VALENCE``: they are decode sentinels, not chemistry, and an atom that
    argmaxes to one dies at `mol_from_atoms` for a different reason entirely.

    Cached: the RDKit probe costs ~0.17 s in this venv (RDKit 2026.03.6) and is paid once
    per process, on first use -- never at import. It is that slow because it runs all
    ``_VALENCE_PROBE_CEILING + 1`` sanitisations for every cell rather than stopping at the
    first rejection, since RDKit's accept-set is not downward-closed (see
    `_probe_max_valence`).
    """
    n_charges = 1 + len(_VOCAB_CHARGES)
    unlimited = (UNLIMITED_VALENCE,) * n_charges
    rows = [unlimited]  # atom class 0 = "<PAD>" -- a decode sentinel, not an element
    for symbol in CORE_ATOMS:
        # charge class 0 = "<PAD>": also a sentinel, never a violation.
        rows.append(
            (UNLIMITED_VALENCE,)
            + tuple(_probe_max_valence(symbol, charge) for charge in _VOCAB_CHARGES)
        )
    rows.append(unlimited)  # trailing "<NOATOM>" virtual-node class
    return tuple(rows)


@lru_cache(maxsize=None)
def valence_limits() -> dict[tuple[str, int], float]:
    """`(element, formal charge) -> max explicit valence`, for the real elements only.

    The sentinel row/column of `valence_limit_rows` is dropped here on purpose: this is a
    lookup keyed by CHEMISTRY, and `max_valence` already answers "unlimited" for anything
    that is not in it.
    """
    rows = valence_limit_rows()
    return {
        (symbol, charge): float(rows[1 + atom_index][1 + charge_index])
        for atom_index, symbol in enumerate(CORE_ATOMS)
        for charge_index, charge in enumerate(_VOCAB_CHARGES)
    }


def max_valence(element: str, charge: int) -> float:
    """Max explicit valence for `element` at `charge`, or "unlimited" when off-vocabulary.

    Off-vocabulary is deliberately permissive: a token that is not an element (`<PAD>`,
    `<NOATOM>`) or a charge outside the head's vocabulary cannot reach the sanitiser as a
    valence error in the first place -- the build dies earlier, on the unknown token -- so
    reporting it as over-valent would INVENT a violation RDKit never raised.
    """
    try:
        key = (str(element), int(charge))
    except (TypeError, ValueError):
        return UNLIMITED_VALENCE
    return valence_limits().get(key, UNLIMITED_VALENCE)


def bond_order(bond_type: int) -> float:
    """Valence contributed by one bond of class `bond_type`.

    Classes outside the vocabulary (e.g. the categorical `mask` token, index 5) contribute
    nothing: such a molecule fails the build on the unknown bond type before any
    sanitisation, so it can never be a valence failure and must not be given a fabricated
    valence here.

    Public because `valence_repair` needs the SAME arithmetic to decide what counts as a
    bond demotion; a second copy there could disagree with the checker it has to satisfy.
    """
    orders = VOCAB_BOND_ORDERS
    if 0 <= bond_type < len(orders):
        return float(orders[bond_type])
    return 0.0


def atom_valences(n_atoms: int, bonds) -> list[float]:
    """Explicit valence of every atom, summed over its incident bonds.

    Mirrors `mol_from_atoms`'s own bond loop so the number is the one RDKit will compute:
    self-bonds are skipped (that loop skips `start == end`) and each UNORDERED pair is
    counted once. `_mol_extract_bonds` already emits the lower triangle only, but a caller
    holding a symmetric list must not read every valence doubled.
    """
    valences = [0.0] * int(n_atoms)
    if bonds is None:
        return valences
    seen: set[tuple[int, int]] = set()
    for row in np.asarray(bonds, dtype=np.int64).reshape(-1, 3).tolist():
        start, end, bond_type = int(row[0]), int(row[1]), int(row[2])
        if start == end:
            continue
        pair = (min(start, end), max(start, end))
        if pair in seen:
            continue
        seen.add(pair)
        order = bond_order(bond_type)
        if order == 0.0:
            continue
        for index in pair:
            if 0 <= index < len(valences):
                valences[index] += order
    return valences


@dataclass(frozen=True)
class ValenceViolation:
    """One over-valent atom: where it is, what it is, and by how much it offends."""

    index: int
    element: str
    charge: int
    valence: float
    limit: float


def valence_violations(tokens: Sequence[str], charges, bonds) -> list[ValenceViolation]:
    """Every over-valent atom of a decoded graph, in atom order.

    EXCESS ONLY, matching the build: `mol_from_atoms` is used with `add_hs=False` and never
    sets `SetNoImplicit` / `SetNumExplicitHs`, so RDKit fills a valence DEFICIT with
    implicit hydrogens and a deficit can never raise `AtomValenceException`.
    """
    valences = atom_valences(len(tokens), bonds)
    charge_list = list(np.asarray(charges).tolist()) if charges is not None else None
    out: list[ValenceViolation] = []
    for index, element in enumerate(tokens):
        charge = 0 if charge_list is None else charge_list[index]
        try:
            charge = int(charge)
        except (TypeError, ValueError):
            # A `<PAD>` charge stringifies the whole numpy array; such a molecule dies on
            # the unknown charge token, never as a valence error.
            continue
        limit = max_valence(element, charge)
        if valences[index] > limit:
            out.append(
                ValenceViolation(
                    index=index,
                    element=str(element),
                    charge=charge,
                    valence=valences[index],
                    limit=limit,
                )
            )
    return out

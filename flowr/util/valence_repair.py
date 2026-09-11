"""Valence-constrained decode: the model's own most probable CHEMICALLY VALID assignment.

`MolBuilder` turns the three discrete heads into a molecule by argmaxing them
INDEPENDENTLY (`_mol_extract_atomics` / `_mol_extract_charges` / `_mol_extract_bonds`).
Independent argmaxes of three heads that are not independent is exactly how a chemically
impossible atom arises: RDKit then raises `AtomValenceException` and `mol_from_atoms`
returns None, so the molecule is lost entirely. On the donor branch this failure mode was
censused and found to be the dominant remaining build-failure mass, with the large majority
of over-valent molecules carrying exactly ONE offending atom. Those shares are the DONOR's
checkpoint and harness and have not been re-measured here; nothing in this module depends
on them, they only motivate the bounds chosen below.

This module does constrained decoding for that case. When the independent argmax is
over-valent it searches for the assignment with the highest joint probability UNDER THE
MODEL that satisfies `flowr.util.valence_report`'s RDKit-probed limit table, and returns
the edits that get there.

WHAT IT IS NOT
--------------
This is a decode change, not a rendering detail: a formal charge, a bond order or an
element can come back different from the argmax. Three lines are drawn hard, because
crossing any of them would be scoring a molecule the model did not predict:

* the ONLY inputs are the model's own head distributions and the limit table. No ground
  truth, no reference molecule, no geometry-derived bond perception (OpenBabel /
  `rdDetermineBonds`).
* no atom is ever deleted and no fragment is ever selected. A disconnected molecule is a
  different failure with a different cause and is not touched here.
* the valence checker only FILTERS candidates; it never RANKS them. Ranking is by the
  model's likelihood alone, which is why the tests include cases where the cheapest edit is
  rejected and the search must walk on to a more expensive one.

The search is admissible in the sense that matters: cost is `-log P(assignment)` relative
to the argmax under the same per-head factorisation the decode already assumes, so
uniform-cost search returns the MOST PROBABLE valid assignment reachable inside the bounds.

BOUNDS (and why they are safe)
------------------------------
Only the FIRST violating atom is expanded. That is not a heuristic: an over-valent atom can
only be relieved by lowering its valence (one of its own incident bonds) or by raising its
limit (its own charge or its own element), so every solution must contain an edit from that
atom's local set. Inside that set the search is exhaustive up to `top_k` alternatives per
head and `max_edits` total edits, capped at `max_states` expansions. When a cap binds it is
reported on the outcome (`cap_hit`) and logged -- a silently truncated search would read as
"repaired everything".

DELETING A BOND IS A DEMOTION IN ARITHMETIC ONLY
------------------------------------------------
"No bond" is a class of the bond head like any other, so `order(candidate) < order(current)`
happily offers it -- and a deletion is frequently the CHEAPEST escape from an over-valence.
It is not the same act as lowering an order: it can split the molecule, converting a
valence failure into a DISCONNECTED molecule, which lifts plain validity but not
fully-connected validity (a single fragment). The donor measured a substantial minority of
its repairs coming back disconnected for exactly this reason, and the same behaviour was
reproduced by hand in this repo's build path before the port. `allow_bond_deletion=False`
forbids anything->none while leaving triple->double->single alone.

The FUNCTION here still defaults to True, which is the donor's shipped behaviour and what
the "unguarded search" baseline in the tests is compared against. `MolBuilder` deliberately
defaults it to **False**: this repo reports fully-connected validity, so trading a valence
failure for a disconnection is a bad default here, and `action="store_true"` is the native
CLI idiom in this codebase. It is a knob and not a fix because dropping a bond IS sometimes
right -- a spurious bond the model should never have drawn -- and the search cannot tell
the two apart from likelihood alone.
"""

from __future__ import annotations
import heapq
import logging
import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence
import numpy as np

from flowr.util.valence_report import (
    UNLIMITED_VALENCE,
    bond_order,
    max_valence,
    valence_violations,
)

logger = logging.getLogger(__name__)

# The three independently-argmaxed heads, spelled the way `valence_report` publishes them.
CHARGES = "charges"
BONDS = "bonds"
TYPES = "types"
_CHANNEL_RANK = {CHARGES: 0, BONDS: 1, TYPES: 2}

# Defaults chosen from the donor's census, not from taste: the large majority of its
# valence failures were a SINGLE over-valent atom, so a budget of 2 covers that case plus
# one of the two-atom cases, and the runner-up class is where a mis-argmaxed head's mass
# actually is. (Donor measurement, not re-measured here -- see the module docstring.)
DEFAULT_MAX_EDITS = 2
DEFAULT_TOP_K = 4
DEFAULT_MAX_STATES = 200

# Floor for the log of the ORIGINAL (argmax) class probability. A zero there would only
# happen on an all-zero row, which is padding; it must not produce a NaN cost.
_TINY = 1e-30


@dataclass(frozen=True)
class Edit:
    """One class flip: which head, which slot, and what it costs in model log-probability.

    `index` is `(atom,)` for the atom-type and charge heads and `(i, j)` with `i > j` for
    the bond head -- the LOWER triangle, which is the half `smolF.bonds_from_adj` reads and
    therefore the half the decode actually used.
    """

    channel: str
    index: tuple[int, ...]
    from_class: int
    to_class: int
    cost: float

    def _key(self) -> tuple[int, ...]:
        return (_CHANNEL_RANK[self.channel], *self.index, self.to_class)

    def deletes_a_bond(self) -> bool:
        """Did this edit REMOVE a bond, rather than lower its order?

        The one edit kind that can turn a valence failure into a disconnection, so the
        caller's telemetry has to be able to count it. Asked through `bond_order`, the same
        arithmetic the checker uses, so "class 0" never has to be spelled out twice.
        """
        return self.channel == BONDS and _bond_order(self.to_class) <= 0.0


@dataclass(frozen=True)
class RepairOutcome:
    """The (possibly unchanged) assignment, the edits that produced it, and the bounds hit.

    `attempted` says the independent argmax WAS over-valent; `repaired` says a valid
    assignment was found. `attempted and not repaired` is the honest failure and returns the
    argmax bit-for-bit -- an unrepairable molecule is left broken rather than mangled.

    `deletion_blocked` says `allow_bond_deletion=False` actually SUPPRESSED a candidate on
    this molecule. Paired with `repaired` it is what separates "the guard cost this molecule
    its repair" from "nothing was repairable anyway"; it is an upper bound on the former,
    since a suppressed deletion need not have fixed the valence either.
    """

    atom_classes: np.ndarray
    charge_classes: np.ndarray
    bond_classes: np.ndarray
    edits: tuple[Edit, ...]
    attempted: bool
    repaired: bool
    cap_hit: Optional[str]
    states_expanded: int
    deletion_blocked: bool = False

    def bond_list(self) -> np.ndarray:
        """`[n_bonds, 3]` of `(i, j, class)`, exactly `smolF.bonds_from_adj(..., lower_tri=True)`."""
        return _bond_list(self.bond_classes)


def _bond_list(bond_classes: np.ndarray) -> np.ndarray:
    """Lower-triangular non-zero entries as `(start, end, class)` rows.

    Deliberately identical to `smolF.bonds_from_adj` with its default `lower_tri=True`
    (`torch.tril(..., diagonal=-1).nonzero()`, row-major); `test_the_bond_list_matches_the
    _shipped_adjacency_flattener` pins the two together over random adjacencies so this can
    never quietly describe a different molecule from the one that gets built.
    """
    matrix = np.asarray(bond_classes)
    if matrix.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    lower = np.tril(matrix, -1)
    indices = np.argwhere(lower != 0)
    if indices.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    orders = lower[indices[:, 0], indices[:, 1]]
    return np.concatenate([indices, orders[:, None]], axis=1).astype(np.int64)


def _normalise(scores) -> np.ndarray:
    """Per-row probabilities from whatever the heads hand over.

    The shipped path delivers `F.softmax` PROBABILITIES (`_get_predictions`), so the
    non-negative branch is the one that runs and it is the identity on them. Raw logits
    (any negative entry) go through a softmax instead, so a caller holding pre-softmax
    output gets the same answer -- pinned by a property test over both spellings.

    A `-inf` entry is the decode masks' own spelling of "this class is gone"
    (`MolBuilder._mask_class`), and it must come out as probability ZERO in both branches,
    never as a cheap alternative.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return arr.reshape(arr.shape)
    finite = np.isfinite(arr)

    positive = np.where(finite, np.maximum(arr, 0.0), 0.0)
    total = positive.sum(axis=-1, keepdims=True)
    normalised = np.divide(
        positive, total, out=np.zeros_like(positive), where=total > 0.0
    )
    is_distribution = (
        np.all(np.where(finite, arr, 0.0) >= 0.0, axis=-1, keepdims=True)
        & finite.any(axis=-1, keepdims=True)
        & (total > 0.0)
    )

    masked = np.where(finite, arr, -np.inf)
    peak = masked.max(axis=-1, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    exponentiated = np.where(finite, np.exp(masked - peak), 0.0)
    exp_total = exponentiated.sum(axis=-1, keepdims=True)
    softmaxed = np.divide(
        exponentiated,
        exp_total,
        out=np.full_like(exponentiated, 1.0 / max(arr.shape[-1], 1)),
        where=exp_total > 0.0,
    )
    return np.where(is_distribution, normalised, softmaxed)


def _cost(probabilities: np.ndarray, old_class: int, new_class: int) -> Optional[float]:
    """`log p(old) - log p(new)` >= 0, or None when the model gave the target no mass."""
    new = float(probabilities[new_class])
    if not (new > 0.0):
        return None
    old = max(float(probabilities[old_class]), _TINY)
    return max(math.log(old) - math.log(new), 0.0)


def _ranked_alternatives(
    probabilities: np.ndarray, current: int, top_k: int
) -> list[int]:
    """The `top_k` most probable classes other than `current`, most probable first."""
    order = np.argsort(-probabilities, kind="stable")
    return [int(c) for c in order if int(c) != int(current) and probabilities[c] > 0.0][
        :top_k
    ]


def _argmax(scores, n_atoms: int, n_leading_dims: int) -> np.ndarray:
    """`torch.argmax(scores, dim=-1)` as int64, with the empty-molecule shape handled."""
    if not n_atoms:
        return np.zeros((0,) * n_leading_dims, dtype=np.int64)
    return np.argmax(np.asarray(scores, dtype=np.float64), axis=-1).astype(np.int64)


def _finite_limit(token: Any, charge: Any) -> bool:
    """Is `(token, charge)` a state the shipped table actually constrains?

    `<PAD>`, `<NOATOM>` and a handful of exotic charge states (C(+3), Cl(-2), ...) probe as
    UNLIMITED -- RDKit enforces nothing there. Moving an atom INTO such a state would
    satisfy the checker by construction rather than by chemistry, which is the one way this
    search could turn into metric gaming, so those targets are refused. Staying in one is
    fine; it was never a violation to begin with.
    """
    return max_valence(token, charge) < UNLIMITED_VALENCE


@dataclass(frozen=True)
class _State:
    atom_classes: tuple[int, ...]
    charge_classes: tuple[int, ...]
    bonds: tuple[tuple[int, ...], ...]
    edits: tuple[Edit, ...]
    cost: float

    def key(self) -> tuple:
        return tuple(sorted(edit._key() for edit in self.edits))


def repair_valence(
    *,
    atom_probs,
    charge_probs,
    bond_probs,
    atom_tokens: Sequence[Any],
    charge_values: Sequence[Any],
    max_edits: int = DEFAULT_MAX_EDITS,
    top_k: int = DEFAULT_TOP_K,
    max_states: int = DEFAULT_MAX_STATES,
    allow_bond_deletion: bool = True,
) -> RepairOutcome:
    """Most probable valence-VALID assignment for one molecule, or the argmax unchanged.

    Parameters
    ----------
    atom_probs, charge_probs, bond_probs
        The head outputs for ONE molecule, `[n, n_atom_classes]`, `[n, n_charge_classes]`
        and `[n, n, n_bond_classes]`. Probabilities or logits; see `_normalise`. Read-only
        -- these are views into the caller's batch and are never written to.
    atom_tokens, charge_values
        Class index -> vocabulary entry, i.e. `[vocab.idx_token_map[i] for i in
        range(vocab.size)]`, so the search speaks the same language as the decode.
    max_edits, top_k, max_states
        The bounds. `max_edits=0` disables the search and returns the plain argmax, which is
        how the tests obtain an untouched baseline.
    allow_bond_deletion
        May a bond be demoted all the way to "no bond"? "No bond" is a class of the bond
        head like any other, so DELETING a bond is often the cheapest escape from an
        over-valence -- and it can split the molecule, converting a valence failure into a
        DISCONNECTED molecule. That lifts plain validity but not fully-connected validity.
        `False` allows triple->double->single and forbids anything->none; `True` (this
        function's default) is the donor's shipped behaviour. Note that `MolBuilder`
        defaults its own knob to False -- see the module docstring.
    """
    atom_probabilities = _normalise(atom_probs)
    charge_probabilities = _normalise(charge_probs)
    bond_probabilities = _normalise(bond_probs)

    n_atoms = int(atom_probabilities.shape[0])
    # The starting assignment is argmaxed from the RAW sheets, not from the normalised
    # ones. `_normalise` is monotone per row and so cannot reorder classes -- but dividing
    # two adjacent float32 values by the same sum CAN round them onto one float64, and a
    # tie broken differently from `torch.argmax` would silently start the search from a
    # different molecule than the one the decode delivered. Casting float32 -> float64 is
    # exact and both `np.argmax` and `torch.argmax` take the FIRST maximum, so taking it
    # here makes the agreement structural instead of probable.
    atom_classes = _argmax(atom_probs, n_atoms, 1)
    charge_classes = _argmax(charge_probs, n_atoms, 1)
    bond_classes = _argmax(bond_probs, n_atoms, 2)

    start = _State(
        atom_classes=tuple(int(c) for c in atom_classes.tolist()),
        charge_classes=tuple(int(c) for c in charge_classes.tolist()),
        bonds=tuple(tuple(int(c) for c in row) for row in bond_classes.tolist()),
        edits=(),
        cost=0.0,
    )

    def violations(state: _State):
        tokens = [atom_tokens[c] for c in state.atom_classes]
        charges = np.array([charge_values[c] for c in state.charge_classes])
        return valence_violations(
            tokens, charges, _bond_list(np.asarray(state.bonds, dtype=np.int64))
        )

    def outcome(
        state: _State, attempted, repaired, cap_hit, expanded, blocked=False
    ) -> RepairOutcome:
        return RepairOutcome(
            atom_classes=np.asarray(state.atom_classes, dtype=np.int64),
            charge_classes=np.asarray(state.charge_classes, dtype=np.int64),
            bond_classes=np.asarray(state.bonds, dtype=np.int64).reshape(
                n_atoms, n_atoms
            ),
            edits=state.edits,
            attempted=attempted,
            repaired=repaired,
            cap_hit=cap_hit,
            states_expanded=expanded,
            deletion_blocked=blocked,
        )

    if n_atoms == 0 or not violations(start):
        return outcome(start, attempted=False, repaired=False, cap_hit=None, expanded=0)
    if max_edits <= 0:
        # The plain argmax, which is how a caller (and the tests) obtain an untouched
        # baseline. No search ran, so no cap "bound" and nothing is logged.
        return outcome(start, attempted=True, repaired=False, cap_hit=None, expanded=0)

    heap: list[tuple] = [(0.0, (), 0)]
    states = [start]
    seen = {start.key()}
    expanded = 0
    cap_states = False
    cap_edits = False
    blocked = False

    while heap:
        _, _, state_index = heapq.heappop(heap)
        state = states[state_index]
        offenders = violations(state)
        if not offenders:
            return outcome(
                state,
                attempted=True,
                repaired=True,
                cap_hit=None,
                expanded=expanded,
                blocked=blocked,
            )
        if len(state.edits) >= max_edits:
            cap_edits = True
            continue
        if expanded >= max_states:
            cap_states = True
            break
        expanded += 1

        successors, suppressed = _successors(
            state,
            offenders[0].index,
            atom_probabilities,
            charge_probabilities,
            bond_probabilities,
            atom_tokens,
            charge_values,
            top_k,
            allow_bond_deletion=allow_bond_deletion,
        )
        blocked = blocked or suppressed
        for successor in successors:
            key = successor.key()
            if key in seen:
                continue
            seen.add(key)
            states.append(successor)
            heapq.heappush(heap, (successor.cost, key, len(states) - 1))

    cap_hit = "states" if cap_states else ("edits" if cap_edits else None)
    if cap_hit is not None:
        # DEBUG, not WARNING: a bound cap is per-molecule and a validation epoch has
        # hundreds of them, so the loud surface is the caller's aggregate counter
        # (`MolBuilder.repair_stats`), which warns once per cap kind.
        logger.debug(
            "valence repair gave up after %d expansions (%s cap); molecule left unrepaired",
            expanded,
            cap_hit,
        )
    return outcome(
        start,
        attempted=True,
        repaired=False,
        cap_hit=cap_hit,
        expanded=expanded,
        blocked=blocked,
    )


def _successors(
    state: _State,
    atom: int,
    atom_probabilities: np.ndarray,
    charge_probabilities: np.ndarray,
    bond_probabilities: np.ndarray,
    atom_tokens: Sequence[Any],
    charge_values: Sequence[Any],
    top_k: int,
    *,
    allow_bond_deletion: bool = True,
) -> tuple[list[_State], bool]:
    """Every single edit LOCAL to `atom`, each already priced in model log-probability.

    Local means: this atom's charge, this atom's element, and the orders of the bonds
    incident to it -- the complete set of edits that can change whether THIS atom is
    over-valent (see the module docstring). A slot already edited by an ancestor state is
    skipped, so an edit sequence never re-litigates a slot: two edits to one slot are always
    dominated by the single edit to the final class, which the search already enumerates.

    Returns `(successors, suppressed)`, where `suppressed` says at least one candidate was
    withheld by `allow_bond_deletion=False`. The guard lives HERE rather than on the answer:
    a post-filter would still let doomed deletion branches be expanded, and expansions are
    what `max_states` spends -- on a dense molecule the deletions would crowd the surviving
    repairs out of the budget and the guard would read as "nothing was repairable".
    """
    used = {edit.index for edit in state.edits}
    token = atom_tokens[state.atom_classes[atom]]
    charge = charge_values[state.charge_classes[atom]]
    out: list[_State] = []
    suppressed = False

    if (atom,) not in used:
        current = state.charge_classes[atom]
        for candidate in _ranked_alternatives(
            charge_probabilities[atom], current, top_k
        ):
            if not _finite_limit(token, charge_values[candidate]):
                continue
            cost = _cost(charge_probabilities[atom], current, candidate)
            if cost is None:
                continue
            charges = list(state.charge_classes)
            charges[atom] = candidate
            out.append(
                _State(
                    atom_classes=state.atom_classes,
                    charge_classes=tuple(charges),
                    bonds=state.bonds,
                    edits=state.edits
                    + (Edit(CHARGES, (atom,), int(current), candidate, cost),),
                    cost=state.cost + cost,
                )
            )

        current = state.atom_classes[atom]
        for candidate in _ranked_alternatives(atom_probabilities[atom], current, top_k):
            if not _finite_limit(atom_tokens[candidate], charge):
                continue
            cost = _cost(atom_probabilities[atom], current, candidate)
            if cost is None:
                continue
            atoms = list(state.atom_classes)
            atoms[atom] = candidate
            out.append(
                _State(
                    atom_classes=tuple(atoms),
                    charge_classes=state.charge_classes,
                    bonds=state.bonds,
                    edits=state.edits
                    + (Edit(TYPES, (atom,), int(current), candidate, cost),),
                    cost=state.cost + cost,
                )
            )

    n_atoms = len(state.atom_classes)
    for other in range(n_atoms):
        if other == atom:
            continue
        i, j = (atom, other) if atom > other else (other, atom)
        if (i, j) in used:
            continue
        current = state.bonds[i][j]
        order = _bond_order(current)
        if order <= 0.0:
            continue
        # DEMOTIONS ONLY: raising a bond order can never relieve an over-valence, so a
        # promotion would only ever be dead weight in the frontier.
        #
        # A pure `< order` test also admits the ZERO-order classes, i.e. deleting the bond,
        # which is a demotion in arithmetic but a different act in chemistry: it can split
        # the molecule. `allow_bond_deletion=False` keeps the ladder (triple->double->single)
        # and drops the bottom rung.
        demotions = []
        for candidate in _ranked_alternatives(bond_probabilities[i, j], current, top_k):
            candidate_order = _bond_order(candidate)
            if candidate_order >= order:
                continue
            if candidate_order <= 0.0 and not allow_bond_deletion:
                suppressed = True
                continue
            demotions.append(candidate)
        for candidate in demotions:
            cost = _cost(bond_probabilities[i, j], current, candidate)
            if cost is None:
                continue
            bonds = [list(row) for row in state.bonds]
            bonds[i][j] = candidate
            bonds[j][i] = candidate
            out.append(
                _State(
                    atom_classes=state.atom_classes,
                    charge_classes=state.charge_classes,
                    bonds=tuple(tuple(row) for row in bonds),
                    edits=state.edits
                    + (Edit(BONDS, (i, j), int(current), candidate, cost),),
                    cost=state.cost + cost,
                )
            )
    return out, suppressed


# `valence_report.bond_order` itself, not a copy of it: "which classes are a DEMOTION"
# has to be the same arithmetic as "how much valence does this bond contribute", or the
# search would be filtering candidates against a different rule from the one it must
# satisfy. Re-exported under the module's own name so `_successors` reads locally.
_bond_order = bond_order

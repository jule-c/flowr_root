"""Valence-constrained decode repair: the search itself, over hand-built probability sheets.

`MolBuilder` argmaxes the atom-type, charge and bond heads INDEPENDENTLY, so nothing stops
the three from agreeing on a chemically impossible atom -- and RDKit then refuses to build
the molecule at all. `repair_valence` searches the model's OWN predicted distributions for
the highest-joint-probability assignment that is valence-VALID under the RDKit-probed table
in `flowr.util.valence_report`.

The line these tests police is that the search is driven by the MODEL's likelihood and only
FILTERED by the valence checker -- never steered by it. So there are cases here where the
cheapest edit is rejected (it does not fix the valence) and the search must walk on to a
more expensive one, cases where two different repairs are both valid and the more probable
one must win in BOTH directions, and cases where nothing reachable is valid and the
molecule must come back untouched and still broken.

Nothing here consults a ground-truth molecule, infers bonds from geometry, or deletes an
atom: the only inputs are the head distributions and the RDKit-probed limit table.

Ported from the donor branch's `tests/flowr/util/test_valence_repair.py`. The donor file is
pytest-style; this repo has no pytest, so `@parametrize` became `subTest` loops and
`pytest.skip` became a guarded `continue` with an explicit non-vacuity count. No assertion
was dropped in the conversion.
"""

from __future__ import annotations

import functools
import inspect
import unittest
from pathlib import Path

import numpy as np
import torch

from flowr.constants import CORE_ATOMS
from flowr.util.valence_repair import (
    DEFAULT_MAX_EDITS,
    DEFAULT_MAX_STATES,
    DEFAULT_TOP_K,
    repair_valence,
)
from flowr.util.valence_report import valence_violations

# The vocabularies `scriptutil._build_vocab(virtual_nodes=False)` / `_build_vocab_charges`
# produce, spelled out so the sheets below are readable. `test_the_test_vocabularies_are
# _the_real_ones` pins them against the builders.
ATOM_TOKENS = ["<PAD>"] + list(CORE_ATOMS)
CHARGE_VALUES = ["<PAD>", 0, 1, 2, 3, -1, -2, -3]
N_BOND_CLASSES = 5  # 0 none, 1 single, 2 double, 3 triple, 4 aromatic

ATOM_IDX = {token: i for i, token in enumerate(ATOM_TOKENS)}
CHARGE_IDX = {value: i for i, value in enumerate(CHARGE_VALUES)}


# ----------------------------------------------------------------- sheet builders


def atom_sheet(rows: list[dict]) -> torch.Tensor:
    """`[n_atoms, n_atom_classes]` probabilities from per-atom `{token: p}` dicts."""
    out = torch.zeros(len(rows), len(ATOM_TOKENS), dtype=torch.float64)
    for i, row in enumerate(rows):
        for token, p in row.items():
            out[i, ATOM_IDX[token]] = p
    return out


def charge_sheet(rows: list[dict]) -> torch.Tensor:
    out = torch.zeros(len(rows), len(CHARGE_VALUES), dtype=torch.float64)
    for i, row in enumerate(rows):
        for value, p in row.items():
            out[i, CHARGE_IDX[value]] = p
    return out


def bond_sheet(n_atoms: int, pairs: dict) -> torch.Tensor:
    """`[n, n, 5]`; unlisted pairs are "no bond" with probability 1.

    Written symmetrically so a bug that reads the upper triangle instead of the lower one
    is not hidden by the fixture.
    """
    out = torch.zeros(n_atoms, n_atoms, N_BOND_CLASSES, dtype=torch.float64)
    out[..., 0] = 1.0
    for (i, j), row in pairs.items():
        out[i, j] = 0.0
        out[j, i] = 0.0
        for bond_class, p in row.items():
            out[i, j, bond_class] = p
            out[j, i, bond_class] = p
    return out


def run(atoms, charges, bonds, n_atoms=None, **kwargs):
    n_atoms = len(atoms) if n_atoms is None else n_atoms
    return repair_valence(
        atom_probs=atom_sheet(atoms),
        charge_probs=charge_sheet(charges),
        bond_probs=bond_sheet(n_atoms, bonds),
        atom_tokens=ATOM_TOKENS,
        charge_values=CHARGE_VALUES,
        **kwargs,
    )


def decoded(outcome):
    """`(tokens, charges, bond list)` of an outcome, for an independent valence check."""
    tokens = [ATOM_TOKENS[c] for c in outcome.atom_classes.tolist()]
    charges = np.array([CHARGE_VALUES[c] for c in outcome.charge_classes.tolist()])
    return tokens, charges, outcome.bond_list()


def violations_of(outcome):
    return valence_violations(*decoded(outcome))


def fragments(outcome, n_atoms: int) -> int:
    """Connected components of the returned graph -- what fully-connected validity counts."""
    parent = list(range(n_atoms))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _ in outcome.bond_list().tolist():
        parent[find(i)] = find(j)
    return len({find(x) for x in range(n_atoms)})


# --------------------------------------------------------------- the hand-built cases
#
# Every case is a star: atom 0 is the (potentially) over-valent centre.


def ammonium(charge_runner_up: float = 0.39):
    """N with FOUR single bonds. N(0) allows 3, N(+1) allows 4 -> only a charge flip fixes it.

    The bond and type alternatives are present but improbable, so if the search ever ranked
    by "what fixes it most easily" instead of by likelihood it would still find a repair --
    just the wrong one. The assertion is on WHICH edit came back.
    """
    atoms = [{"N": 0.98, "C": 0.005, "O": 0.005, "P": 0.005, "S": 0.005}] + [
        {"C": 0.99, "N": 0.01}
    ] * 4
    charges = [{0: 1.0 - charge_runner_up, 1: charge_runner_up}] + [{0: 1.0}] * 4
    bonds = {
        (0, k): {1: 0.97, 0: 0.01, 2: 0.01, 3: 0.005, 4: 0.005} for k in range(1, 5)
    }
    return atoms, charges, bonds


def hexavalent_carbon():
    """C with three single bonds and one TRIPLE bond -> valence 6, limit 4.

    No charge state of carbon reaches 6 (C(+3) is the table's "unlimited" escape hatch and
    is refused separately), and every atom alternative is zero-probability, so the ONLY
    repair is to demote the triple bond. The trap: demoting it to DOUBLE is strictly more
    probable than demoting it to SINGLE, and leaves the atom at 5 -- still invalid. A search
    that stopped at the cheapest edit would return a molecule that still fails.
    """
    atoms = [{"C": 1.0}] + [{"C": 1.0}] * 4
    charges = [{0: 0.9, 1: 0.04, -1: 0.04, 2: 0.02}] + [{0: 1.0}] * 4
    bonds = {(0, k): {1: 0.999, 0: 0.001} for k in range(1, 4)}
    bonds[(0, 4)] = {3: 0.5, 2: 0.3, 1: 0.15, 0: 0.05}
    return atoms, charges, bonds


def unfixable_carbon():
    """The same valence-6 carbon, but with every reachable state still invalid.

    The triple bond can only go aromatic (1.5 -> valence 4.5), the centre can only become
    N, and no (C|N, reachable charge) pair allows 4.5. Two edits do not help either, so the
    molecule must come back UNTOUCHED and still fail -- the honest outcome, and the one a
    repair that quietly "fixed" things by deleting an atom would never produce.
    """
    atoms = [{"C": 0.9, "N": 0.1}] + [{"C": 1.0}] * 4
    charges = [{0: 0.8, 1: 0.1, -1: 0.1}] + [{0: 1.0}] * 4
    bonds = {(0, k): {1: 1.0} for k in range(1, 4)}
    bonds[(0, 4)] = {3: 0.9, 4: 0.1}
    return atoms, charges, bonds


def two_ammoniums():
    """Two independent over-valent nitrogens: the multi-offender minority case.

    Atoms 0 and 5 are each an N with four single bonds; the two stars share no atom, so the
    fix is two independent charge flips and NOTHING inside a budget of one can work.
    """
    atoms = []
    charges = []
    bonds = {}
    for star in range(2):
        centre = star * 5
        atoms.append({"N": 0.98, "C": 0.02})
        charges.append({0: 0.6, 1: 0.4})
        for k in range(1, 5):
            atoms.append({"C": 1.0})
            charges.append({0: 1.0})
            bonds[(centre, centre + k)] = {1: 0.97, 0: 0.01, 2: 0.02}
    return atoms, charges, bonds


def bridged_ammonium():
    """N0 is over-valent and its CHEAPEST escape DELETES the bond that holds the molecule together.

    N0=C1 (double), N0-C2, N0-C3, C3-C4, C4-C5 -- so N0 reads 2+1+1 = 4 against a limit of 3,
    and bond (3,0) is a BRIDGE: dropping it leaves {0,1,2} and {3,4,5}, two real fragments
    rather than one orphaned terminal atom.

    The three edits the model puts on the table, priced in its own log-probability:

        delete (3,0)                    ln(0.55/0.45) = 0.20  <- cheapest, DISCONNECTS
        demote (1,0) double -> single   ln(0.90/0.10) = 2.20  <- valid AND still one fragment
        charge  N(0) -> N(+1)           ln(0.999/0.001) = 6.9 <- valid, expensive

    Both non-deleting repairs are reachable, so "forbid deletion" cannot be satisfied by
    accident: the search has to walk past a candidate it currently prefers by an order of
    magnitude.
    """
    atoms = [{"N": 1.0}] + [{"C": 1.0}] * 5
    charges = [{0: 0.999, 1: 0.001}] + [{0: 1.0}] * 5
    bonds = {
        (0, 1): {2: 0.9, 1: 0.1},  # demotable to single; class 0 has NO mass
        (0, 2): {1: 1.0},  # one-hot: nothing to offer
        (0, 3): {1: 0.55, 0: 0.45},  # the bridge, and the cheap deletion
        (3, 4): {1: 1.0},
        (4, 5): {1: 1.0},
    }
    return atoms, charges, bonds


def deletion_only_ammonium():
    """The same over-valent N, but DELETION is the only repair the model gave any mass to.

    Four single bonds on a neutral N; the charge and type heads are one-hot, and each bond
    can only go to class 0. Forbidding deletion must therefore produce the honest failure --
    the argmax back untouched and still broken -- and NOT a disconnected molecule.
    """
    atoms = [{"N": 1.0}] + [{"C": 1.0}] * 4
    charges = [{0: 1.0}] * 5
    bonds = {(0, k): {1: 0.9, 0: 0.1} for k in range(1, 5)}
    return atoms, charges, bonds


# --------------------------------------------------------------- the random battery


def _random_case(seed: int, n: int = 7, bond_bias: float = 1.0):
    """Random head sheets. `bond_bias` is the extra logit on "no bond" == graph sparsity.

    Sparsity is what decides whether the draw is over-valent at all, so the properties below
    need DIFFERENT biases to be non-vacuous: at 1.0 nearly every draw is broken (the repair
    paths), at 3.0 most are fine (the leave-it-alone path).
    `test_the_random_battery_is_not_vacuous` pins both.
    """
    generator = torch.Generator().manual_seed(seed)
    atom_probs = torch.softmax(
        torch.randn(n, len(ATOM_TOKENS), generator=generator).double(), dim=-1
    )
    charge_probs = torch.softmax(
        torch.randn(n, len(CHARGE_VALUES), generator=generator).double(), dim=-1
    )
    bond_logits = torch.randn(n, n, N_BOND_CLASSES, generator=generator).double()
    bond_logits[..., 0] += bond_bias
    bond_probs = torch.softmax(bond_logits, dim=-1)
    return atom_probs, charge_probs, bond_probs


def _kwargs_for(seed: int, bond_bias: float = 1.0):
    atom_probs, charge_probs, bond_probs = _random_case(seed, bond_bias=bond_bias)
    return dict(
        atom_probs=atom_probs,
        charge_probs=charge_probs,
        bond_probs=bond_probs,
        atom_tokens=ATOM_TOKENS,
        charge_values=CHARGE_VALUES,
    )


def _census_of(bond_bias: float, seeds=range(40)) -> dict:
    counts = {"valid": 0, "repaired": 0, "unrepaired": 0}
    for seed in seeds:
        kwargs = _kwargs_for(seed, bond_bias=bond_bias)
        before = repair_valence(**kwargs, max_edits=0)
        if not violations_of(before):
            counts["valid"] += 1
            continue
        counts["repaired" if repair_valence(**kwargs).repaired else "unrepaired"] += 1
    return counts


@functools.lru_cache(maxsize=1)
def _parent_valence_repair_module():
    """Import the PRE-GUARD `valence_repair.py` as a standalone module.

    Loaded from the VENDORED copy (`_valence_repair_baseline.py.txt`), which predates
    `allow_bond_deletion`, rather than from `git show HEAD:`. The donor learned both
    failure modes of the git spelling the hard way: `HEAD:` is the CURRENT commit, so the
    moment the change was committed the comparison became `module == itself` and every case
    passed vacuously; and CI runs with no git history, so the subprocess exited 128 and
    every case ERRORED. A vendored baseline is fixed, needs no git, and cannot drift onto
    the version under test. `test_the_vendored_baseline_is_not_the_current_source` keeps it
    honest.
    """
    import importlib.machinery
    import importlib.util
    import sys

    path = Path(__file__).resolve().parent / "_valence_repair_baseline.py.txt"
    # The vendored baseline deliberately carries a `.txt` suffix so no collector picks it up
    # and no linter rewrites it. importlib will not infer a loader for an unknown suffix and
    # returns spec=None, so name the loader explicitly.
    loader = importlib.machinery.SourceFileLoader("_parent_valence_repair", str(path))
    spec = importlib.util.spec_from_file_location(
        "_parent_valence_repair", path, loader=loader
    )
    module = importlib.util.module_from_spec(spec)
    # `from __future__ import annotations` makes every field annotation a STRING, and
    # `@dataclass` resolves those through `sys.modules[cls.__module__]`; without this the
    # parent module cannot even be executed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# =========================================================== 0. the fixtures are real


class FixturesAreRealTests(unittest.TestCase):
    def test_the_test_vocabularies_are_the_real_ones(self):
        """A sheet built over the wrong vocabulary would test a different decode entirely."""
        from flowr.scriptutil import _build_vocab, _build_vocab_charges

        vocab = _build_vocab(virtual_nodes=False)
        vocab_charges = _build_vocab_charges()
        self.assertEqual(
            [vocab.idx_token_map[i] for i in range(vocab.size)], ATOM_TOKENS
        )
        self.assertEqual(
            [vocab_charges.idx_token_map[i] for i in range(vocab_charges.size)],
            CHARGE_VALUES,
        )

    def test_the_limit_table_says_what_the_cases_assume(self):
        """The cases are only meaningful if these are the real RDKit-probed limits."""
        from flowr.util.valence_report import max_valence

        self.assertEqual(max_valence("N", 0), 3.0)
        self.assertEqual(max_valence("N", 1), 4.0)
        self.assertEqual(max_valence("C", 0), 4.0)
        self.assertEqual(max_valence("C", 1), 3.0)
        self.assertEqual(max_valence("C", -1), 3.0)
        # the escape hatch the search must refuse to use
        self.assertGreaterEqual(max_valence("C", 3), 1.0e4)

    def test_the_bond_list_matches_the_shipped_adjacency_flattener(self):
        """`bond_list()` must be `smolF.bonds_from_adj`, or the goal test scores other bonds."""
        import flowr.util.functional as smolF

        generator = torch.Generator().manual_seed(11)
        for trial in range(20):
            with self.subTest(trial=trial):
                n = 6
                classes = torch.randint(
                    0, N_BOND_CLASSES, (n, n), generator=generator
                )
                classes = torch.tril(classes, diagonal=-1)
                classes = classes + classes.T
                outcome = run(
                    [{"C": 1.0}] * n,
                    [{0: 1.0}] * n,
                    {
                        (i, j): {int(classes[i, j]): 1.0}
                        for i in range(n)
                        for j in range(i)
                        if int(classes[i, j]) != 0
                    },
                )
                expected = smolF.bonds_from_adj(classes).long().numpy()
                got = outcome.bond_list()
                self.assertEqual(
                    sorted(map(tuple, got.tolist())),
                    sorted(map(tuple, expected.tolist())),
                )


# =========================================================== 1. it leaves valid molecules alone


class LeavesValidMoleculesAloneTests(unittest.TestCase):
    def test_a_valid_molecule_is_never_touched(self):
        outcome = run(
            [{"C": 0.9, "N": 0.1}, {"C": 0.9, "N": 0.1}],
            [{0: 0.9, 1: 0.1}, {0: 0.9, 1: 0.1}],
            {(0, 1): {1: 0.8, 2: 0.2}},
        )
        self.assertIs(outcome.attempted, False)
        self.assertIs(outcome.repaired, False)
        self.assertEqual(outcome.edits, ())
        self.assertEqual(
            outcome.atom_classes.tolist(), [ATOM_IDX["C"], ATOM_IDX["C"]]
        )
        self.assertEqual(
            outcome.charge_classes.tolist(), [CHARGE_IDX[0], CHARGE_IDX[0]]
        )
        self.assertEqual(violations_of(outcome), [])

    def test_the_starting_assignment_is_the_shipped_argmax(self):
        """The search must start where the shipped decode landed, ties included."""
        outcome = run(
            [{"C": 0.5, "N": 0.5}, {"O": 1.0}],
            [{0: 0.4, 1: 0.3, -1: 0.3}, {0: 1.0}],
            {(0, 1): {1: 0.5, 2: 0.5}},
        )
        atom_probs = atom_sheet([{"C": 0.5, "N": 0.5}, {"O": 1.0}])
        self.assertEqual(
            outcome.atom_classes.tolist(), torch.argmax(atom_probs, dim=-1).tolist()
        )


# =========================================================== 2. the three hand-built repairs


class HandBuiltRepairTests(unittest.TestCase):
    def test_only_a_charge_flip_fixes_the_ammonium(self):
        outcome = run(*ammonium())
        self.assertIs(outcome.attempted, True)
        self.assertIs(outcome.repaired, True)
        self.assertEqual(violations_of(outcome), [])
        self.assertEqual(len(outcome.edits), 1)
        (edit,) = outcome.edits
        self.assertEqual(edit.channel, "charges")
        self.assertEqual(edit.index, (0,))
        self.assertEqual(CHARGE_VALUES[edit.to_class], 1)
        # nothing else moved
        self.assertEqual(outcome.atom_classes.tolist()[0], ATOM_IDX["N"])
        self.assertEqual(outcome.bond_list().shape[0], 4)

    def test_the_search_walks_past_a_cheaper_edit_that_does_not_fix_the_valence(self):
        outcome = run(*hexavalent_carbon())
        self.assertIs(outcome.repaired, True)
        self.assertEqual(violations_of(outcome), [])
        self.assertEqual(len(outcome.edits), 1)
        (edit,) = outcome.edits
        self.assertEqual(edit.channel, "bonds")
        self.assertEqual(edit.index, (4, 0))
        self.assertEqual(edit.from_class, 3)
        # DOUBLE is the more probable demotion and would leave the atom at valence 5
        self.assertEqual(edit.to_class, 1)

    def test_an_unrepairable_molecule_comes_back_untouched_and_still_broken(self):
        atoms, charges, bonds = unfixable_carbon()
        outcome = run(atoms, charges, bonds)
        self.assertIs(outcome.attempted, True)
        self.assertIs(outcome.repaired, False)
        self.assertEqual(outcome.edits, ())
        self.assertNotEqual(violations_of(outcome), [])
        baseline = run(atoms, charges, bonds, max_edits=0)
        self.assertEqual(
            outcome.atom_classes.tolist(), baseline.atom_classes.tolist()
        )
        self.assertEqual(
            outcome.charge_classes.tolist(), baseline.charge_classes.tolist()
        )
        self.assertEqual(
            outcome.bond_classes.tolist(), baseline.bond_classes.tolist()
        )

    def test_two_over_valent_atoms_take_two_edits(self):
        outcome = run(*two_ammoniums())
        self.assertIs(outcome.repaired, True)
        self.assertEqual(violations_of(outcome), [])
        self.assertEqual(len(outcome.edits), 2)
        self.assertEqual({edit.index for edit in outcome.edits}, {(0,), (5,)})
        self.assertTrue(all(edit.channel == "charges" for edit in outcome.edits))

    def test_the_edit_budget_binds_and_says_so(self):
        outcome = run(*two_ammoniums(), max_edits=1)
        self.assertIs(outcome.repaired, False)
        self.assertEqual(outcome.edits, ())
        self.assertEqual(outcome.cap_hit, "edits")
        self.assertEqual(len(violations_of(outcome)), 2)


# =========================================================== 3. ranked by the MODEL


class RankedByTheModelTests(unittest.TestCase):
    def test_the_winning_channel_follows_the_models_own_likelihood(self):
        """The SAME broken atom, two valid repairs, and the probabilities decide -- both ways.

        N with four single bonds is fixed either by N(0)->N(+1) or by dropping one bond.
        With a confident runner-up charge the charge edit is cheaper; starve that charge and
        the bond edit wins. If this only passed in one direction the search would be ranking
        by something other than the model.
        """
        for charge_p, expected_channel in [(0.45, "charges"), (0.02, "bonds")]:
            with self.subTest(charge_p=charge_p):
                atoms = [{"N": 0.999, "C": 0.001}] + [{"C": 1.0}] * 4
                charges = [{0: 1.0 - charge_p, 1: charge_p}] + [{0: 1.0}] * 4
                bonds = {(0, k): {1: 0.9, 0: 0.1} for k in range(1, 5)}
                outcome = repair_valence(
                    atom_probs=atom_sheet(atoms),
                    charge_probs=charge_sheet(charges),
                    bond_probs=bond_sheet(5, bonds),
                    atom_tokens=ATOM_TOKENS,
                    charge_values=CHARGE_VALUES,
                )
                self.assertIs(outcome.repaired, True)
                self.assertEqual(violations_of(outcome), [])
                self.assertEqual(
                    [edit.channel for edit in outcome.edits], [expected_channel]
                )

    def test_the_type_channel_can_win_and_is_not_privileged(self):
        """A type edit is on the table exactly like the other two, and wins only on likelihood.

        Atom 0 carries five single bonds: C(0) allows 4, P(0) allows 5. So C->P repairs it,
        and so does dropping a bond. Whichever the model likes more must come back.
        """
        for type_wins in (True, False):
            with self.subTest(type_wins=type_wins):
                p_type = 0.4 if type_wins else 0.001
                atoms = [{"C": 1.0 - p_type, "P": p_type}] + [{"C": 1.0}] * 5
                charges = [{0: 1.0}] * 6
                bonds = {(0, k): {1: 0.7, 0: 0.3} for k in range(1, 6)}
                outcome = repair_valence(
                    atom_probs=atom_sheet(atoms),
                    charge_probs=charge_sheet(charges),
                    bond_probs=bond_sheet(6, bonds),
                    atom_tokens=ATOM_TOKENS,
                    charge_values=CHARGE_VALUES,
                )
                self.assertIs(outcome.repaired, True)
                self.assertEqual(violations_of(outcome), [])
                self.assertEqual(
                    [edit.channel for edit in outcome.edits],
                    ["types" if type_wins else "bonds"],
                )

    def test_the_repair_refuses_to_escape_into_an_unlimited_valence_state(self):
        """C(+3) has NO enforceable limit in the probed table -- a free pass, not a repair.

        Moving an atom into a state the checker has no opinion about would satisfy the
        checker by construction rather than by chemistry, so those targets are refused even
        when the model ranks them first. Here C(+3) is the most probable alternative charge
        by a wide margin and the search must still take the bond demotion.
        """
        atoms = [{"C": 1.0}] + [{"C": 1.0}] * 5
        charges = [{0: 0.5, 3: 0.49, 1: 0.01}] + [{0: 1.0}] * 5
        bonds = {(0, k): {1: 0.9, 0: 0.1} for k in range(1, 6)}
        outcome = repair_valence(
            atom_probs=atom_sheet(atoms),
            charge_probs=charge_sheet(charges),
            bond_probs=bond_sheet(6, bonds),
            atom_tokens=ATOM_TOKENS,
            charge_values=CHARGE_VALUES,
        )
        self.assertIs(outcome.repaired, True)
        self.assertEqual([edit.channel for edit in outcome.edits], ["bonds"])
        self.assertEqual(CHARGE_VALUES[outcome.charge_classes[0]], 0)

    def test_a_pad_token_is_never_a_repair_target(self):
        """`<PAD>` is a guaranteed build failure; "fixing" a valence with it is no fix."""
        atoms = [{"C": 0.5, "<PAD>": 0.49, "P": 0.01}] + [{"C": 1.0}] * 5
        charges = [{0: 0.51, "<PAD>": 0.49}] + [{0: 1.0}] * 5
        bonds = {(0, k): {1: 0.9, 0: 0.1} for k in range(1, 6)}
        outcome = repair_valence(
            atom_probs=atom_sheet(atoms),
            charge_probs=charge_sheet(charges),
            bond_probs=bond_sheet(6, bonds),
            atom_tokens=ATOM_TOKENS,
            charge_values=CHARGE_VALUES,
        )
        self.assertIs(outcome.repaired, True)
        self.assertNotEqual(ATOM_TOKENS[outcome.atom_classes[0]], "<PAD>")
        self.assertNotEqual(CHARGE_VALUES[outcome.charge_classes[0]], "<PAD>")

    def test_bond_edits_are_demotions_only(self):
        """Promoting a bond can never relieve an over-valence, so it is not a candidate."""
        atoms = [{"N": 1.0}] + [{"C": 1.0}] * 4
        charges = [{0: 0.99, 1: 0.01}] + [{0: 1.0}] * 4
        # The model would happily promote these to DOUBLE (0.44 of the mass sits there),
        # which is the temptation: a promotion is a cheap, high-probability edit that makes
        # the over-valence strictly worse. Only the demotion may be offered.
        bonds = {(0, k): {1: 0.55, 2: 0.44, 0: 0.01} for k in range(1, 5)}
        outcome = repair_valence(
            atom_probs=atom_sheet(atoms),
            charge_probs=charge_sheet(charges),
            bond_probs=bond_sheet(5, bonds),
            atom_tokens=ATOM_TOKENS,
            charge_values=CHARGE_VALUES,
        )
        self.assertIs(outcome.repaired, True)
        for edit in outcome.edits:
            if edit.channel == "bonds":
                self.assertEqual(
                    edit.to_class, 0, "only a demotion can relieve an over-valence"
                )


# =========================================================== 4. bounds


class BoundsTests(unittest.TestCase):
    def test_a_one_hot_sheet_offers_nothing_to_search(self):
        """A one-hot categorical state carries no runner-up; say so, do not invent one."""
        atoms = [{"C": 1.0}] * 6
        charges = [{0: 1.0}] * 6
        bonds = {(0, k): {1: 1.0} for k in range(1, 6)}
        outcome = repair_valence(
            atom_probs=atom_sheet(atoms),
            charge_probs=charge_sheet(charges),
            bond_probs=bond_sheet(6, bonds),
            atom_tokens=ATOM_TOKENS,
            charge_values=CHARGE_VALUES,
        )
        self.assertIs(outcome.attempted, True)
        self.assertIs(outcome.repaired, False)
        self.assertEqual(outcome.edits, ())
        self.assertNotEqual(violations_of(outcome), [])

    def test_top_k_bounds_the_branching(self):
        """With top_k=1 the only candidate per channel is the runner-up, so the fix is missed."""
        # one-hot types and bonds, so the charge head is the only channel with anything to
        # offer
        atoms = [{"N": 1.0}] + [{"C": 1.0}] * 4
        # +1 is the THIRD most probable charge, so top_k=1 cannot see it
        charges = [{0: 0.5, 2: 0.3, 1: 0.2}] + [{0: 1.0}] * 4
        bonds = {(0, k): {1: 1.0} for k in range(1, 5)}
        kwargs = dict(
            atom_probs=atom_sheet(atoms),
            charge_probs=charge_sheet(charges),
            bond_probs=bond_sheet(5, bonds),
            atom_tokens=ATOM_TOKENS,
            charge_values=CHARGE_VALUES,
        )
        self.assertIs(repair_valence(**kwargs, top_k=1).repaired, False)
        wide = repair_valence(**kwargs, top_k=2)
        self.assertIs(wide.repaired, True)
        self.assertEqual(CHARGE_VALUES[wide.charge_classes[0]], 1)

    def test_the_state_cap_binds_and_says_so(self):
        """A pathological molecule must truncate LOUDLY, never look like a clean failure."""
        n = 12
        atoms = [{"C": 0.6, "N": 0.4}] * n
        charges = [{0: 0.6, 1: 0.2, -1: 0.2}] * n
        bonds = {(i, j): {1: 0.5, 2: 0.3, 0: 0.2} for i in range(n) for j in range(i)}
        outcome = repair_valence(
            atom_probs=atom_sheet(atoms),
            charge_probs=charge_sheet(charges),
            bond_probs=bond_sheet(n, bonds),
            atom_tokens=ATOM_TOKENS,
            charge_values=CHARGE_VALUES,
            max_edits=4,
            max_states=5,
        )
        self.assertIs(outcome.repaired, False)
        self.assertEqual(outcome.cap_hit, "states")
        self.assertLessEqual(outcome.states_expanded, 5)

    def test_the_defaults_are_bounded(self):
        self.assertTrue(0 < DEFAULT_MAX_EDITS <= 4)
        self.assertTrue(0 < DEFAULT_TOP_K <= 8)
        self.assertTrue(0 < DEFAULT_MAX_STATES <= 10_000)


# =========================================================== 5. determinism + properties


class DeterminismTests(unittest.TestCase):
    def test_repeated_calls_are_identical(self):
        atoms, charges, bonds = two_ammoniums()
        first = run(atoms, charges, bonds)
        for trial in range(5):
            with self.subTest(trial=trial):
                again = run(atoms, charges, bonds)
                self.assertEqual(
                    again.atom_classes.tolist(), first.atom_classes.tolist()
                )
                self.assertEqual(
                    again.charge_classes.tolist(), first.charge_classes.tolist()
                )
                self.assertEqual(
                    again.bond_classes.tolist(), first.bond_classes.tolist()
                )
                self.assertEqual(again.edits, first.edits)

    def test_exactly_tied_alternatives_resolve_deterministically(self):
        """Two indistinguishable repairs must not depend on dict/heap ordering luck."""
        atoms = [{"N": 1.0}] + [{"C": 1.0}] * 4
        charges = [{0: 0.9, 1: 0.05, 2: 0.05}] + [{0: 1.0}] * 4
        bonds = {(0, k): {1: 0.5, 0: 0.5} for k in range(1, 5)}
        outcomes = [
            repair_valence(
                atom_probs=atom_sheet(atoms),
                charge_probs=charge_sheet(charges),
                bond_probs=bond_sheet(5, bonds),
                atom_tokens=ATOM_TOKENS,
                charge_values=CHARGE_VALUES,
            )
            for _ in range(8)
        ]
        self.assertEqual(len({tuple(o.edits) for o in outcomes}), 1)

    def test_the_starting_assignment_is_torch_argmax_bit_for_bit(self):
        """Over float32 sheets with deliberate exact ties, on all three heads.

        If the search started from a different class than the decode did, the repair would
        be fixing a molecule nobody was ever going to be shown -- and it would fail
        silently, since both molecules argmax from the same numbers.
        """
        for seed in range(15):
            with self.subTest(seed=seed):
                generator = torch.Generator().manual_seed(seed)
                n = 6
                atom_probs = torch.softmax(
                    torch.randn(n, len(ATOM_TOKENS), generator=generator), dim=-1
                )
                charge_probs = torch.softmax(
                    torch.randn(n, len(CHARGE_VALUES), generator=generator), dim=-1
                )
                bond_probs = torch.softmax(
                    torch.randn(n, n, N_BOND_CLASSES, generator=generator), dim=-1
                )
                # plant exact ties, which is where a tie-break disagreement would show up
                atom_probs[0, 1] = atom_probs[0, 2] = 0.5
                charge_probs[1, 3] = charge_probs[1, 4] = 0.5
                bond_probs[3, 1, 1] = bond_probs[3, 1, 2] = 0.5

                outcome = repair_valence(
                    atom_probs=atom_probs,
                    charge_probs=charge_probs,
                    bond_probs=bond_probs,
                    atom_tokens=ATOM_TOKENS,
                    charge_values=CHARGE_VALUES,
                    max_edits=0,
                )
                self.assertEqual(
                    outcome.atom_classes.tolist(),
                    torch.argmax(atom_probs, dim=-1).tolist(),
                )
                self.assertEqual(
                    outcome.charge_classes.tolist(),
                    torch.argmax(charge_probs, dim=-1).tolist(),
                )
                self.assertEqual(
                    outcome.bond_classes.tolist(),
                    torch.argmax(bond_probs, dim=-1).tolist(),
                )


class RandomBatteryPropertyTests(unittest.TestCase):
    def test_the_random_battery_is_not_vacuous(self):
        """Each property below must actually exercise the branch it claims to.

        A property test over draws that are ALL broken never checks "leaves valid molecules
        alone", and one over draws that are all fine never checks the repair. Both branches
        are asserted to be populated here, so a future change to the generator cannot
        silently turn either property into a no-op.
        """
        dense = _census_of(1.0)
        sparse = _census_of(3.0)
        self.assertGreaterEqual(dense["repaired"], 5, dense)
        self.assertGreaterEqual(dense["unrepaired"], 5, dense)
        self.assertGreaterEqual(sparse["valid"], 10, sparse)
        self.assertGreaterEqual(sparse["repaired"], 5, sparse)

    def test_property_the_repair_never_leaves_more_violations_than_it_found(self):
        for bond_bias in (1.0, 3.0):
            for seed in range(40):
                with self.subTest(bond_bias=bond_bias, seed=seed):
                    kwargs = _kwargs_for(seed, bond_bias=bond_bias)
                    before = repair_valence(**kwargs, max_edits=0)
                    after = repair_valence(**kwargs)
                    n_before = len(violations_of(before))
                    n_after = len(violations_of(after))
                    if after.repaired:
                        self.assertEqual(n_after, 0)
                    else:
                        self.assertEqual(n_after, n_before)
                        self.assertEqual(after.edits, ())
                        self.assertEqual(
                            after.atom_classes.tolist(), before.atom_classes.tolist()
                        )
                        self.assertEqual(
                            after.charge_classes.tolist(),
                            before.charge_classes.tolist(),
                        )
                        self.assertEqual(
                            after.bond_classes.tolist(), before.bond_classes.tolist()
                        )

    def test_property_a_valid_molecule_is_returned_bit_identical(self):
        exercised = 0
        for seed in range(40):
            kwargs = _kwargs_for(seed, bond_bias=3.0)
            before = repair_valence(**kwargs, max_edits=0)
            if violations_of(before):
                # over-valent draw; covered by the property above
                continue
            exercised += 1
            with self.subTest(seed=seed):
                after = repair_valence(**kwargs)
                self.assertIs(after.attempted, False)
                self.assertEqual(after.edits, ())
                self.assertEqual(
                    after.atom_classes.tolist(), before.atom_classes.tolist()
                )
                self.assertEqual(
                    after.charge_classes.tolist(), before.charge_classes.tolist()
                )
                self.assertEqual(
                    after.bond_classes.tolist(), before.bond_classes.tolist()
                )
        # `continue` replaces the donor's `pytest.skip`, which would have shown up in the
        # report; an explicit count keeps a fully-skipped battery from passing silently.
        self.assertGreaterEqual(exercised, 10, "every draw was over-valent")

    def test_property_every_edit_is_a_real_change_the_model_gave_mass_to(self):
        for seed in range(20):
            with self.subTest(seed=seed):
                kwargs = _kwargs_for(seed)
                outcome = repair_valence(**kwargs)
                sheets = {
                    "types": kwargs["atom_probs"],
                    "charges": kwargs["charge_probs"],
                }
                for edit in outcome.edits:
                    self.assertNotEqual(edit.from_class, edit.to_class)
                    self.assertGreaterEqual(edit.cost, 0.0)
                    if edit.channel == "bonds":
                        i, j = edit.index
                        self.assertGreater(
                            kwargs["bond_probs"][i, j, edit.to_class], 0.0
                        )
                    else:
                        (i,) = edit.index
                        self.assertGreater(sheets[edit.channel][i, edit.to_class], 0.0)

    def test_property_logits_and_the_probabilities_they_produce_repair_alike(self):
        """The sheets arrive as softmax PROBABILITIES today; raw logits must not change the answer."""

        def decision(outcome):
            # the DECISION, not the last bit of the cost: `log(exp(log p))` round-trips to
            # within ~1e-16 and the two spellings differ there and only there
            return [
                (edit.channel, edit.index, edit.from_class, edit.to_class)
                for edit in outcome.edits
            ]

        for seed in range(20):
            with self.subTest(seed=seed):
                atom_probs, charge_probs, bond_probs = _random_case(seed)
                from_probs = repair_valence(
                    atom_probs=atom_probs,
                    charge_probs=charge_probs,
                    bond_probs=bond_probs,
                    atom_tokens=ATOM_TOKENS,
                    charge_values=CHARGE_VALUES,
                )
                from_logits = repair_valence(
                    atom_probs=torch.log(atom_probs),
                    charge_probs=torch.log(charge_probs),
                    bond_probs=torch.log(bond_probs),
                    atom_tokens=ATOM_TOKENS,
                    charge_values=CHARGE_VALUES,
                )
                self.assertEqual(decision(from_logits), decision(from_probs))
                for a, b in zip(from_logits.edits, from_probs.edits):
                    self.assertAlmostEqual(a.cost, b.cost, delta=1e-9)


class InputHandlingTests(unittest.TestCase):
    def test_a_masked_minus_infinity_column_is_never_selected(self):
        """A -inf entry is "this class is gone"; that column is dead, not cheap."""
        atoms = atom_sheet([{"C": 0.5, "P": 0.5}] + [{"C": 1.0}] * 5)
        atoms[0, ATOM_IDX["P"]] = float("-inf")
        outcome = repair_valence(
            atom_probs=atoms,
            charge_probs=charge_sheet([{0: 1.0}] * 6),
            bond_probs=bond_sheet(6, {(0, k): {1: 0.9, 0: 0.1} for k in range(1, 6)}),
            atom_tokens=ATOM_TOKENS,
            charge_values=CHARGE_VALUES,
        )
        self.assertEqual(ATOM_TOKENS[outcome.atom_classes[0]], "C")
        self.assertTrue(all(edit.channel != "types" for edit in outcome.edits))

    def test_it_does_not_write_into_the_callers_tensors(self):
        """`_extract_mols` hands out VIEWS of the caller's batch; an in-place edit corrupts them."""
        atoms, charges, bonds = ammonium()
        atom_probs = atom_sheet(atoms)
        charge_probs = charge_sheet(charges)
        bond_probs = bond_sheet(5, bonds)
        snapshots = [t.clone() for t in (atom_probs, charge_probs, bond_probs)]
        repair_valence(
            atom_probs=atom_probs,
            charge_probs=charge_probs,
            bond_probs=bond_probs,
            atom_tokens=ATOM_TOKENS,
            charge_values=CHARGE_VALUES,
        )
        for before, after in zip(snapshots, (atom_probs, charge_probs, bond_probs)):
            self.assertTrue(torch.equal(before, after))

    def test_numpy_sheets_are_accepted(self):
        atoms, charges, bonds = ammonium()
        outcome = repair_valence(
            atom_probs=atom_sheet(atoms).numpy(),
            charge_probs=charge_sheet(charges).numpy(),
            bond_probs=bond_sheet(5, bonds).numpy(),
            atom_tokens=ATOM_TOKENS,
            charge_values=CHARGE_VALUES,
        )
        self.assertIs(outcome.repaired, True)
        self.assertEqual([edit.channel for edit in outcome.edits], ["charges"])

    def test_an_empty_molecule_is_handled(self):
        outcome = repair_valence(
            atom_probs=torch.zeros(0, len(ATOM_TOKENS)),
            charge_probs=torch.zeros(0, len(CHARGE_VALUES)),
            bond_probs=torch.zeros(0, 0, N_BOND_CLASSES),
            atom_tokens=ATOM_TOKENS,
            charge_values=CHARGE_VALUES,
        )
        self.assertIs(outcome.attempted, False)
        self.assertIs(outcome.repaired, False)
        self.assertEqual(outcome.bond_list().shape, (0, 3))


class CandidateGeneratorTests(unittest.TestCase):
    @staticmethod
    def _start_state(atoms, charges, bonds, n_atoms):
        from flowr.util import valence_repair

        atom_probabilities = valence_repair._normalise(atom_sheet(atoms))
        charge_probabilities = valence_repair._normalise(charge_sheet(charges))
        bond_probabilities = valence_repair._normalise(bond_sheet(n_atoms, bonds))
        start = valence_repair._State(
            atom_classes=tuple(int(c) for c in atom_probabilities.argmax(-1)),
            charge_classes=tuple(int(c) for c in charge_probabilities.argmax(-1)),
            bonds=tuple(
                tuple(int(c) for c in row) for row in bond_probabilities.argmax(-1)
            ),
            edits=(),
            cost=0.0,
        )
        return start, atom_probabilities, charge_probabilities, bond_probabilities

    def test_the_frontier_never_contains_a_promotion(self):
        """A candidate that RAISES a valence is dead weight the bounds cannot afford.

        A promotion can never relieve an over-valence, so it can never appear in a solution
        -- which is why the outcome-level assertions above cannot see it. What it CAN do is
        multiply the branching factor, and the branching factor is what `max_states` spends:
        on a dense molecule the useful demotions would be pushed out of the budget by
        candidates that were never going to help. So the generator is asserted directly.
        """
        from flowr.util import valence_repair

        atoms = [{"N": 1.0}] + [{"C": 1.0}] * 4
        charges = [{0: 0.9, 1: 0.1}] + [{0: 1.0}] * 4
        # the model would rather have DOUBLE bonds here than the SINGLE ones it argmaxed
        bonds = {(0, k): {1: 0.5, 2: 0.45, 3: 0.04, 0: 0.01} for k in range(1, 5)}
        start, atom_p, charge_p, bond_p = self._start_state(atoms, charges, bonds, 5)
        successors, _ = valence_repair._successors(
            start,
            0,
            atom_p,
            charge_p,
            bond_p,
            ATOM_TOKENS,
            CHARGE_VALUES,
            top_k=4,
        )
        self.assertTrue(successors, "precondition: there is something to search")
        bond_successors = [
            successor
            for successor in successors
            if successor.edits[-1].channel == "bonds"
        ]
        self.assertTrue(
            bond_successors, "precondition: bond edits are among the candidates"
        )
        for successor in bond_successors:
            edit = successor.edits[-1]
            self.assertLess(
                valence_repair._bond_order(edit.to_class),
                valence_repair._bond_order(edit.from_class),
                f"{edit} raises a valence and cannot ever help",
            )

    def test_the_candidate_generator_itself_stops_offering_deletions(self):
        """The guard belongs on the GENERATOR, not on a post-filter of the answer.

        A post-filter would still let doomed deletion branches be expanded, and expansions
        are exactly what `max_states` spends: on a dense molecule the deletions would crowd
        the surviving repairs out of the budget. So the successor set is asserted directly.
        """
        from flowr.util import valence_repair

        atoms, charges, bonds = deletion_only_ammonium()
        start, atom_p, charge_p, bond_p = self._start_state(atoms, charges, bonds, 5)
        args = (start, 0, atom_p, charge_p, bond_p, ATOM_TOKENS, CHARGE_VALUES)
        allowed, blocked_any = valence_repair._successors(*args, top_k=4)
        self.assertTrue(
            allowed, "precondition: the unguarded generator offers the deletions"
        )
        self.assertIs(blocked_any, False)
        guarded, blocked = valence_repair._successors(
            *args, top_k=4, allow_bond_deletion=False
        )
        self.assertEqual(guarded, [])
        self.assertIs(blocked, True)


# =========================================================== 6. deleting a bond as an escape
#
# "No bond" is a class of the bond head like any other, so a DEMOTION filter
# (`order(candidate) < order(current)`) lets the search escape an over-valence by DELETING a
# bond -- which lifts plain validity but not fully-connected validity, because it can split
# the molecule. The donor measured a substantial minority of its repairs doing exactly that.
#
# `allow_bond_deletion=False` forbids anything->none while leaving triple->double->single
# alone. The FUNCTION defaults it to True (the donor's shipped behaviour, and what the
# vendored baseline below is compared against); `MolBuilder` in this repo defaults it to
# False.


class BondDeletionGuardTests(unittest.TestCase):
    def test_the_default_escapes_the_over_valence_by_deleting_the_bridge(self):
        """The unguarded behaviour, pinned: the cheapest edit is a deletion and it splits the molecule."""
        outcome = run(*bridged_ammonium())
        self.assertIs(outcome.repaired, True)
        self.assertEqual(violations_of(outcome), [])
        self.assertEqual(
            [(e.channel, e.index, e.to_class) for e in outcome.edits],
            [("bonds", (3, 0), 0)],
        )
        self.assertEqual(
            fragments(outcome, 6),
            2,
            "precondition: this repair DISCONNECTS the molecule",
        )
        self.assertIs(outcome.deletion_blocked, False)

    def test_forbidding_deletion_takes_the_non_deleting_repair_and_stays_connected(self):
        outcome = run(*bridged_ammonium(), allow_bond_deletion=False)
        self.assertIs(outcome.repaired, True)
        self.assertEqual(violations_of(outcome), [])
        # the second-cheapest edit, 11x more expensive than the deletion it was not allowed
        self.assertEqual(
            [(e.channel, e.index, e.from_class, e.to_class) for e in outcome.edits],
            [("bonds", (1, 0), 2, 1)],
        )
        self.assertEqual(fragments(outcome, 6), 1)
        self.assertIs(outcome.deletion_blocked, True)

    def test_forbidding_deletion_never_returns_a_molecule_it_disconnected(self):
        """When only a deletion was on offer the answer is `unrepaired`, not a split molecule."""
        atoms, charges, bonds = deletion_only_ammonium()
        guarded = run(atoms, charges, bonds, allow_bond_deletion=False)
        self.assertIs(guarded.repaired, False)
        self.assertEqual(guarded.edits, ())
        self.assertNotEqual(violations_of(guarded), [])
        self.assertEqual(
            fragments(guarded, 5),
            1,
            "the argmax came back untouched, so still one fragment",
        )
        self.assertIs(guarded.deletion_blocked, True)

        # ... and the unguarded search really would have deleted a bond here, so the case is
        # about the guard rather than about an unrepairable molecule.
        default = run(atoms, charges, bonds)
        self.assertIs(default.repaired, True)
        self.assertEqual(
            [(e.channel, e.to_class) for e in default.edits], [("bonds", 0)]
        )
        self.assertEqual(fragments(default, 5), 2)

    def test_the_guard_leaves_ordinary_demotions_alone(self):
        """triple -> double -> single is untouched; only anything -> none is refused."""
        outcome = run(*hexavalent_carbon(), allow_bond_deletion=False)
        self.assertIs(outcome.repaired, True)
        self.assertEqual(violations_of(outcome), [])
        (edit,) = outcome.edits
        self.assertEqual(
            (edit.channel, edit.from_class, edit.to_class), ("bonds", 3, 1)
        )

    def test_an_edit_knows_whether_it_deleted_a_bond(self):
        """The telemetry has to be able to say `edits_bond_deletions` without re-deriving orders."""
        from flowr.util.valence_repair import Edit

        self.assertIs(Edit("bonds", (1, 0), 1, 0, 0.0).deletes_a_bond(), True)
        self.assertIs(Edit("bonds", (1, 0), 3, 1, 0.0).deletes_a_bond(), False)
        # aromatic, 1.5
        self.assertIs(Edit("bonds", (1, 0), 2, 4, 0.0).deletes_a_bond(), False)
        self.assertIs(Edit("charges", (0,), 1, 2, 0.0).deletes_a_bond(), False)


# ----------------------------------------------------------- the function default is inert


class GuardIsInertByDefaultTests(unittest.TestCase):
    def test_the_function_flag_defaults_to_the_unguarded_search(self):
        """`repair_valence`'s own default stays True -- that is what the baseline pins.

        `MolBuilder` overrides it to False (this repo reports fully-connected validity), but
        the SEARCH's default must stay the donor's, or the bit-identity comparison below
        would be comparing two different configurations and prove nothing about the guard.
        """
        self.assertIs(
            inspect.signature(repair_valence).parameters["allow_bond_deletion"].default,
            True,
        )

    def test_the_vendored_baseline_is_not_the_current_source(self):
        """NON-VACUITY. If the baseline ever equals the file under test, every comparison
        below passes by construction and proves nothing -- which is precisely how a
        `git show HEAD:` version of this failed silently once it was committed."""
        import flowr.util.valence_repair as current

        baseline = (
            Path(__file__).resolve().parent / "_valence_repair_baseline.py.txt"
        ).read_text()
        self.assertNotEqual(
            baseline,
            Path(current.__file__).read_text(),
            "the vendored baseline is byte-identical to the module under test -- the "
            "bit-identity comparisons are vacuous",
        )
        self.assertNotIn(
            "allow_bond_deletion",
            baseline,
            "the vendored baseline already carries the guard it is supposed to predate",
        )

    def test_property_the_default_is_bit_identical_to_the_unguarded_search(self):
        """The knob is default-True on the function precisely so nothing changes until asked.

        Compared against the PRE-GUARD `valence_repair`, loaded from the vendored baseline
        and run side by side over the random battery -- "identical to itself" would prove
        nothing.
        """
        parent = _parent_valence_repair_module()
        for bond_bias in (1.0, 3.0):
            for seed in range(40):
                with self.subTest(bond_bias=bond_bias, seed=seed):
                    kwargs = _kwargs_for(seed, bond_bias=bond_bias)
                    mine = repair_valence(**kwargs)
                    theirs = parent.repair_valence(**kwargs)
                    self.assertEqual(
                        mine.atom_classes.tolist(), theirs.atom_classes.tolist()
                    )
                    self.assertEqual(
                        mine.charge_classes.tolist(), theirs.charge_classes.tolist()
                    )
                    self.assertEqual(
                        mine.bond_classes.tolist(), theirs.bond_classes.tolist()
                    )
                    self.assertEqual(mine.attempted, theirs.attempted)
                    self.assertEqual(mine.repaired, theirs.repaired)
                    self.assertEqual(mine.cap_hit, theirs.cap_hit)
                    self.assertEqual(mine.states_expanded, theirs.states_expanded)
                    self.assertEqual(
                        [
                            (e.channel, e.index, e.from_class, e.to_class)
                            for e in mine.edits
                        ],
                        [
                            (e.channel, e.index, e.from_class, e.to_class)
                            for e in theirs.edits
                        ],
                    )

    def test_property_the_guard_never_invents_a_worse_outcome(self):
        """Forbidding deletion may cost a repair, but it must never break an invariant.

        Whatever it returns is either valence-VALID or the untouched argmax, it never
        carries a deleted bond, and it never returns MORE fragments than the molecule
        started with -- the whole point of the knob.
        """
        for seed in range(40):
            with self.subTest(seed=seed):
                kwargs = _kwargs_for(seed, bond_bias=1.0)
                before = repair_valence(**kwargs, max_edits=0)
                guarded = repair_valence(**kwargs, allow_bond_deletion=False)
                if guarded.repaired:
                    self.assertEqual(violations_of(guarded), [])
                else:
                    self.assertEqual(guarded.edits, ())
                    self.assertEqual(
                        guarded.bond_classes.tolist(), before.bond_classes.tolist()
                    )
                self.assertTrue(
                    all(not edit.deletes_a_bond() for edit in guarded.edits)
                )
                n = int(before.atom_classes.shape[0])
                self.assertLessEqual(fragments(guarded, n), fragments(before, n))

    def test_the_guard_is_not_vacuous_on_the_random_battery(self):
        """At least some draws must actually differ, or the properties above prove nothing."""
        differed = 0
        deletions = 0
        for seed in range(40):
            kwargs = _kwargs_for(seed, bond_bias=1.0)
            default = repair_valence(**kwargs)
            guarded = repair_valence(**kwargs, allow_bond_deletion=False)
            deletions += any(edit.deletes_a_bond() for edit in default.edits)
            differed += default.edits != guarded.edits
        self.assertGreaterEqual(
            deletions, 5, f"the unguarded search deleted a bond only {deletions} times"
        )
        self.assertGreaterEqual(
            differed, 5, f"the guard changed the answer only {differed} times"
        )


if __name__ == "__main__":
    unittest.main()

"""`ligand_valence_repair` where it actually acts: `MolBuilder`'s decode.

The flag is default OFF, and the first section exists to prove that OFF means *the shipped
path*, not "a code path that happens to agree today": `repair_valence` is replaced by a stub
that RAISES and the whole battery is decoded, so with the flag off the stub is never reached
and the executed code is the pre-port one.

With the flag ON the contract is narrower but just as sharp: a molecule that already loads
must come back SMILES-identical (the repair may only ever act on a build that has ALREADY
failed), and the REFERENCE ligand must never be repaired at all.

The last section asks the CHEMISTRY question rather than the plumbing one, in both
directions: on a planted-corruption battery (a fine molecule, one head's argmax broken, the
truth left as the model's runner-up) the repair must return the molecule that was there;
and when a valid-but-wrong distractor is slipped above the truth it must return the
DISTRACTOR, because the search reads the model's ranking and nothing else.

Ported from the donor branch's `tests/flowr/models/test_valence_repair_no_deletion.py` and
`tests/flowr/models/test_valence_repair_decode.py`, with four deliberate changes:

* This repo has no build-failure census, so `ligs_from_complex(collect_reasons=True)` and
  the `CENSUS_*` labels do not exist. Label assertions became `mol is None` (a failed build)
  / `n_fragments(mol) == 1` (the donor's `ok`) / `== 2` (its `disconnected`).
* The donor's tests drive the builder through `ligs_from_complex`. In THIS repo that decodes
  the REFERENCE ligand and must never repair, so every case was re-pointed at
  `mols_from_tensors`; `ReferenceLigandTests` pins the difference.
* `ligand_valence_repair_allow_bond_deletion` defaults to **False** here, not to the donor's
  True. See `DefaultsTests`.
* The donor's `git show HEAD:` comparison against a parent commit is dropped -- there is no
  parent commit here that carries this code, so it could only ever compare a module with
  itself.
"""

from __future__ import annotations

import collections
import inspect
import random
import unittest
from unittest import mock

import numpy as np
import torch
from rdkit import Chem

import flowr.util.metrics as Metrics
import flowr.util.rdkit as smolRD
from flowr.models.mol_builder import MolBuilder
from flowr.scriptutil import _build_vocab, _build_vocab_charges

N_BOND_CLASSES = 5


def vocabs():
    return _build_vocab(virtual_nodes=False), _build_vocab_charges()


def make_builder(**kwargs) -> MolBuilder:
    vocab, vocab_charges = vocabs()
    return MolBuilder(vocab, vocab_charges, n_workers=2, **kwargs)


def _sheets(atoms, charges, bonds, n_atoms):
    """`(coords, mask, atom_dists, bond_dists, charge_dists)` for a ONE-molecule batch.

    `atoms` / `charges` are per-atom `{token: p}` dicts over the REAL vocabularies, so the
    argmax the builder takes is the argmax these tests reason about.
    """
    vocab, vocab_charges = vocabs()
    atom_dists = torch.zeros(1, n_atoms, vocab.size, dtype=torch.float32)
    charge_dists = torch.zeros(1, n_atoms, vocab_charges.size, dtype=torch.float32)
    bond_dists = torch.zeros(1, n_atoms, n_atoms, N_BOND_CLASSES, dtype=torch.float32)
    bond_dists[..., 0] = 1.0
    for i, row in enumerate(atoms):
        for token, p in row.items():
            atom_dists[0, i, vocab.token_idx_map[token]] = p
    for i, row in enumerate(charges):
        for value, p in row.items():
            charge_dists[0, i, vocab_charges.token_idx_map[value]] = p
    for (i, j), row in bonds.items():
        bond_dists[0, i, j] = 0.0
        bond_dists[0, j, i] = 0.0
        for bond_class, p in row.items():
            bond_dists[0, i, j, bond_class] = p
            bond_dists[0, j, i, bond_class] = p
    # a straight chain: geometry is carried through the repair untouched and never read by
    # it (no bond perception anywhere in this path), so any non-degenerate coords will do
    coords = torch.zeros(1, n_atoms, 3)
    coords[0, :, 0] = torch.arange(n_atoms, dtype=torch.float32) * 1.5
    mask = torch.ones(1, n_atoms, dtype=torch.bool)
    return coords, mask, atom_dists, bond_dists, charge_dists


def decode(builder: MolBuilder, tensors):
    """The GENERATED-molecule path -- the only one the repair is allowed to touch."""
    coords, mask, atom_dists, bond_dists, charge_dists = tensors
    return builder.mols_from_tensors(
        coords,
        atom_dists,
        mask,
        bond_dists=bond_dists,
        charge_dists=charge_dists,
        sanitise=True,
    )


def decode_reference(builder: MolBuilder, tensors):
    """The REFERENCE-ligand path, which must never repair."""
    coords, mask, atom_dists, bond_dists, charge_dists = tensors
    return builder.ligs_from_complex(
        coords,
        mask,
        atom_dists,
        bond_dists=bond_dists,
        charge_dists=charge_dists,
        sanitise=True,
    )


def smiles(mols):
    return [None if mol is None else Chem.MolToSmiles(mol) for mol in mols]


def n_fragments(mol) -> int:
    return len(Chem.GetMolFrags(mol))


def random_tensors(seed: int, batch: int = 6, n: int = 7):
    """Soft random sheets over the real vocabulary widths -- the general case.

    Almost every draw here FAILS to build (a uniform atom head argmaxes `<PAD>` about one
    atom in fifteen, and an unstructured adjacency is wildly over-valent), which is exactly
    what the repair paths want. `valid_tree_tensors` is the complement, for the properties
    that need molecules that already load.
    """
    vocab, vocab_charges = vocabs()
    generator = torch.Generator().manual_seed(seed)
    coords = torch.randn(batch, n, 3, generator=generator) * 1.5
    mask = torch.ones(batch, n, dtype=torch.bool)
    atom_dists = torch.softmax(
        torch.randn(batch, n, vocab.size, generator=generator), dim=-1
    )
    charge_dists = torch.softmax(
        torch.randn(batch, n, vocab_charges.size, generator=generator), dim=-1
    )
    bond_logits = torch.randn(batch, n, n, N_BOND_CLASSES, generator=generator)
    bond_logits[..., 0] += 1.5
    bond_dists = torch.softmax(bond_logits, dim=-1)
    return coords, mask, atom_dists, bond_dists, charge_dists


def soften(tensor: torch.Tensor, alpha: float = 0.2) -> torch.Tensor:
    """Mix a one-hot sheet toward uniform WITHOUT moving any argmax.

    A one-hot head has no runner-up for the repair to find, so a test that used it as-is
    would report "no repair happened" for the wrong reason.
    """
    width = tensor.shape[-1]
    return (1.0 - alpha) * tensor + alpha / width


def valid_tree_tensors(seed: int, size: int = 12):
    """`size` random valence-respecting molecules, softened so every head has runner-ups.

    Every one of these BUILDS, so they are the population for "the repair never touches a
    molecule that already loads" -- and because they are softened, a repair that fired
    would have plenty of alternatives to pick, which is what makes the test non-vacuous.
    """
    rng = random.Random(10_000 + seed)
    cases = [_random_tree(rng) for _ in range(size)]
    tensors = _tensors_from_trees([(tokens, edges) for tokens, edges, _ in cases])
    coords, mask, atom_dists, bond_dists, charge_dists = tensors
    return (
        coords,
        mask,
        soften(atom_dists),
        soften(bond_dists),
        soften(charge_dists),
    )


# ------------------------------------------------------------------ the hand-built cases


def ammonium_tensors(charge_runner_up: float = 0.4):
    """N with four single bonds -- a failed build under the argmax, fine at N(+1).

    Hand-built rather than softened, so exactly one repair is available and the assertion
    can name it.
    """
    atoms = [{"N": 1.0}] + [{"C": 1.0}] * 4
    charges = [{0: 1.0 - charge_runner_up, 1: charge_runner_up}] + [{0: 1.0}] * 4
    bonds = {(0, k): {1: 0.99, 0: 0.01} for k in range(1, 5)}
    return _sheets(atoms, charges, bonds, 5)


def one_hot_ammonium():
    """The same over-valent N, but with every head one-hot -- an integrator state.

    `ammonium_tensors(charge_runner_up=0.0)` is NOT this: it leaves 0.01 of bond mass on
    "no bond", which is a real (if unlikely) alternative and the repair correctly takes it.
    """
    atoms = [{"N": 1.0}] + [{"C": 1.0}] * 4
    charges = [{0: 1.0}] * 5
    bonds = {(0, k): {1: 1.0} for k in range(1, 5)}
    return _sheets(atoms, charges, bonds, 5)


def bridged_ammonium():
    """N0 over-valent, and its CHEAPEST repair DELETES the bond holding the molecule together.

    N0=C1, N0-C2, N0-C3, C3-C4, C4-C5. N0 reads 2+1+1 = 4 against a neutral-N limit of 3, so
    the argmax fails to build. Bond (3,0) is a BRIDGE: dropping it leaves C1=N0-C2 and
    C3-C4-C5, two real fragments rather than one orphaned atom.

        delete (3,0)                   ln(0.55/0.45) = 0.20  <- cheapest, DISCONNECTS
        demote (1,0) double -> single  ln(0.90/0.10) = 2.20  <- valid AND one fragment
        charge N(0) -> N(+1)           ln(0.999/0.001) = 6.9

    Both non-deleting repairs are reachable, so "forbid deletion" cannot be satisfied by
    accident -- the search has to walk past a candidate it prefers by an order of magnitude.
    """
    atoms = [{"N": 1.0}] + [{"C": 1.0}] * 5
    charges = [{0: 0.999, 1: 0.001}] + [{0: 1.0}] * 5
    bonds = {
        (0, 1): {2: 0.9, 1: 0.1},
        (0, 2): {1: 1.0},
        (0, 3): {1: 0.55, 0: 0.45},
        (3, 4): {1: 1.0},
        (4, 5): {1: 1.0},
    }
    return _sheets(atoms, charges, bonds, 6)


def deletion_only_ammonium():
    """The same over-valent N, but deletion is the ONLY repair with any mass behind it."""
    atoms = [{"N": 1.0}] + [{"C": 1.0}] * 4
    charges = [{0: 1.0}] * 5
    bonds = {(0, k): {1: 0.9, 0: 0.1} for k in range(1, 5)}
    return _sheets(atoms, charges, bonds, 5)


def ring_ammonium():
    """The counterexample: the cheapest deletion is a RING bond, so it does NOT disconnect.

    C0 carries five single bonds -- two of them the ring closure C0-C1-C2-C3-C0, three of
    them substituents C4, C5, C6 -- against a carbon limit of 4. Deleting the ring bond
    (3,0) relieves it and leaves ONE fragment, because C3 is still reachable through C2.
    Nothing else has any mass behind it.

    So forbidding deletion costs a genuine fully-connected-validity win here. That is the
    real trade and not a bug: a bond deletion is only harmful when the bond is a BRIDGE, and
    the search prices candidates by model likelihood alone -- it has no connectivity term
    and cannot tell a bridge from a ring bond. A cheaper knob would be "forbid deletion of a
    BRIDGE", which is strictly better on this case and strictly harder to justify (it steers
    the search by a graph property rather than by the model), so it is deliberately not what
    this flag does.
    """
    atoms = [{"C": 1.0}] * 7
    charges = [{0: 1.0}] * 7
    bonds = {
        (0, 1): {1: 1.0},
        (1, 2): {1: 1.0},
        (2, 3): {1: 1.0},
        (0, 3): {1: 0.6, 0: 0.4},  # the ring closure, and the only candidate
        (0, 4): {1: 1.0},
        (0, 5): {1: 1.0},
        (0, 6): {1: 1.0},
    }
    return _sheets(atoms, charges, bonds, 7)


def propane():
    """A molecule that builds cleanly: the repair must never be able to reach it."""
    atoms = [{"C": 1.0}] * 3
    charges = [{0: 1.0}] * 3
    bonds = {(0, 1): {1: 0.9, 0: 0.1}, (1, 2): {1: 0.9, 0: 0.1}}
    return _sheets(atoms, charges, bonds, 3)


ALL_CASES = {
    "ammonium": ammonium_tensors,
    "one_hot_ammonium": one_hot_ammonium,
    "bridged_ammonium": bridged_ammonium,
    "deletion_only_ammonium": deletion_only_ammonium,
    "ring_ammonium": ring_ammonium,
    "propane": propane,
}


# ================================================== 0. defaults


class DefaultsTests(unittest.TestCase):
    def test_the_repair_defaults_to_off_everywhere(self):
        builder = make_builder()
        self.assertIs(builder.ligand_valence_repair, False)
        self.assertIs(
            inspect.signature(MolBuilder.__init__)
            .parameters["ligand_valence_repair"]
            .default,
            False,
        )

    def test_bond_deletion_defaults_to_FORBIDDEN_here_unlike_the_donor(self):
        """The one deliberate divergence from the donor's shipped defaults.

        The donor defaults `allow_bond_deletion` to True. Here it is False, for three
        independent reasons: the donor's own measurement had a substantial minority of its
        repairs coming back DISCONNECTED (plain validity up, fully-connected validity not);
        the same was reproduced by hand in this repo's build path; and this repo has no
        default-True boolean CLI flag anywhere, so `action="store_true"` is the native
        idiom. `repair_valence`'s own default stays True, which is what the unguarded
        baseline in `flowr/util/tests/test_valence_repair_search.py` is compared against.
        """
        self.assertIs(
            inspect.signature(MolBuilder.__init__)
            .parameters["ligand_valence_repair_allow_bond_deletion"]
            .default,
            False,
        )
        self.assertIs(make_builder().ligand_valence_repair_allow_bond_deletion, False)

    def test_the_counters_start_at_zero_and_reset_to_zero(self):
        builder = make_builder(ligand_valence_repair=True)
        self.assertEqual(set(builder.repair_stats.values()), {0})
        decode(builder, ammonium_tensors())
        self.assertNotEqual(set(builder.repair_stats.values()), {0})
        builder.reset_repair_stats()
        self.assertEqual(set(builder.repair_stats.values()), {0})


# ================================================== 1. OFF is the shipped path


class FlagOffIsTheShippedPathTests(unittest.TestCase):
    def test_with_the_flag_off_the_search_is_never_even_reached(self):
        """Not "it agrees", but "it never ran": the stub raises if the gate leaks."""

        def exploding(**kwargs):
            raise AssertionError("repair_valence was reached with the flag off")

        with mock.patch(
            "flowr.models.mol_builder.repair_valence", side_effect=exploding
        ):
            for name, case in ALL_CASES.items():
                with self.subTest(case=name):
                    decode(make_builder(), case())
            for seed in range(4):
                with self.subTest(random_seed=seed):
                    decode(make_builder(), random_tensors(seed))

    def test_constructing_with_the_new_kwargs_at_their_defaults_changes_nothing(self):
        """Every case, decoded by a bare builder and by one that spells out the defaults."""
        bare = make_builder()
        spelled = make_builder(
            ligand_valence_repair=False,
            ligand_valence_repair_allow_bond_deletion=False,
        )
        for name, case in ALL_CASES.items():
            with self.subTest(case=name):
                tensors = case()
                self.assertEqual(
                    smiles(decode(bare, tensors)), smiles(decode(spelled, tensors))
                )
        for seed in range(4):
            with self.subTest(random_seed=seed):
                tensors = random_tensors(seed)
                self.assertEqual(
                    smiles(decode(bare, tensors)), smiles(decode(spelled, tensors))
                )


# ================================================== 2. ON leaves loadable molecules alone


class LoadableMoleculesAreUntouchedTests(unittest.TestCase):
    def test_property_the_repair_never_destroys_a_loadable_molecule(self):
        """Every molecule the OFF decode delivered must come back SMILES-identical.

        Two populations, because they exercise different halves: `valid_tree_tensors` is
        molecules that all build (so the "untouched" claim has something to be about), and
        `random_tensors` is the general case, where almost nothing builds and the few that
        do are the interesting ones.
        """
        checked = 0
        for seed in range(8):
            for name, tensors in (
                ("trees", valid_tree_tensors(seed)),
                ("random", random_tensors(seed)),
            ):
                off = smiles(decode(make_builder(), tensors))
                on = smiles(decode(make_builder(ligand_valence_repair=True), tensors))
                for index, before in enumerate(off):
                    if before is None:
                        continue
                    checked += 1
                    with self.subTest(population=name, seed=seed, molecule=index):
                        self.assertEqual(
                            on[index],
                            before,
                            "the repair acted on a molecule that already built",
                        )
        self.assertGreaterEqual(checked, 50, "no loadable molecule in the battery")

    def test_the_valid_tree_battery_really_does_all_build(self):
        """Non-vacuity guard for the population above: if these ever stopped building, the
        "never destroys a loadable molecule" property would quietly become a no-op."""
        for seed in range(8):
            with self.subTest(seed=seed):
                mols = decode(make_builder(), valid_tree_tensors(seed))
                self.assertTrue(all(mol is not None for mol in mols))

    def test_the_counters_stay_at_zero_over_the_valid_tree_battery(self):
        """The strongest statement of C5 available: a whole battery of loadable molecules
        decoded with the repair ON, and the search was never even entered."""
        builder = make_builder(ligand_valence_repair=True)
        for seed in range(8):
            decode(builder, valid_tree_tensors(seed))
        self.assertEqual(set(builder.repair_stats.values()), {0})

    def test_a_molecule_that_builds_never_reaches_the_search(self):
        """The gate is the FAILED build, so a loadable molecule cannot be touched at all."""
        for allow in (True, False):
            with self.subTest(allow_bond_deletion=allow):
                builder = make_builder(
                    ligand_valence_repair=True,
                    ligand_valence_repair_allow_bond_deletion=allow,
                )
                mols = decode(builder, propane())
                self.assertEqual(Chem.MolToSmiles(mols[0]), "CCC")
                self.assertEqual(set(builder.repair_stats.values()), {0})

    def test_a_non_valence_build_failure_is_left_alone(self):
        """A build that failed for another reason is not a repair opportunity.

        `repair_valence` classifies the failure itself: with no over-valent atom it comes
        back `attempted=False`, so the counters must stay at zero rather than record an
        `unrepaired` that was never a candidate. Here the atom head argmaxes to `<PAD>`,
        which is not an element -- the build dies on the token, not on a valence.
        """
        atoms = [{"<PAD>": 0.9, "C": 0.1}] + [{"C": 1.0}] * 2
        charges = [{0: 1.0}] * 3
        bonds = {(0, 1): {1: 0.9, 0: 0.1}, (1, 2): {1: 0.9, 0: 0.1}}
        builder = make_builder(ligand_valence_repair=True)
        mols = decode(builder, _sheets(atoms, charges, bonds, 3))
        self.assertIsNone(mols[0])
        self.assertEqual(
            set(builder.repair_stats.values()),
            {0},
            "a non-valence failure inflated the repair counters",
        )


# ================================================== 2b. the REFERENCE ligand is off limits


class ReferenceLigandTests(unittest.TestCase):
    """C1: `ligs_from_complex` decodes the REFERENCE ligand, which is ground truth."""

    def test_the_reference_path_never_repairs_even_with_the_flag_on(self):
        for name in ("ammonium", "bridged_ammonium", "deletion_only_ammonium"):
            with self.subTest(case=name):
                builder = make_builder(ligand_valence_repair=True)
                tensors = ALL_CASES[name]()
                self.assertIsNone(
                    decode_reference(builder, tensors)[0],
                    "the reference ligand was repaired",
                )
                self.assertEqual(set(builder.repair_stats.values()), {0})

    def test_the_same_builder_repairs_the_generated_path_and_not_the_reference(self):
        """ONE long-lived builder serves both paths, so the split must be per call site."""
        builder = make_builder(ligand_valence_repair=True)
        tensors = ammonium_tensors()
        self.assertIsNone(decode_reference(builder, tensors)[0])
        self.assertIsNotNone(decode(builder, tensors)[0])
        self.assertEqual(builder.repair_stats["attempted"], 1)
        self.assertIsNone(decode_reference(builder, tensors)[0])
        self.assertEqual(builder.repair_stats["attempted"], 1)


# ================================================== 3. ON repairs what it should


class RepairsWhatItShouldTests(unittest.TestCase):
    def test_the_argmax_ammonium_fails_to_build_without_the_repair(self):
        self.assertEqual(decode(make_builder(), ammonium_tensors()), [None])

    def test_the_repair_recovers_the_ammonium_as_the_charged_species(self):
        builder = make_builder(ligand_valence_repair=True)
        mols = decode(builder, ammonium_tensors())
        self.assertEqual(Chem.MolToSmiles(mols[0]), "C[N+](C)(C)C")
        self.assertEqual(n_fragments(mols[0]), 1)
        self.assertEqual(builder.repair_stats["attempted"], 1)
        self.assertEqual(builder.repair_stats["repaired"], 1)
        self.assertEqual(builder.repair_stats["repaired_ok"], 1)
        self.assertEqual(builder.repair_stats["edits_charges"], 1)
        self.assertEqual(builder.repair_stats["edits_bonds"], 0)
        self.assertEqual(builder.repair_stats["edits_types"], 0)

    def test_a_one_hot_decode_is_left_alone_because_there_is_no_runner_up(self):
        """A one-hot categorical state carries no alternatives; do not invent any."""
        builder = make_builder(ligand_valence_repair=True)
        self.assertEqual(decode(builder, one_hot_ammonium()), [None])
        self.assertEqual(builder.repair_stats["attempted"], 1)
        self.assertEqual(builder.repair_stats["repaired"], 0)
        self.assertEqual(builder.repair_stats["unrepaired"], 1)

    def test_deleting_a_bond_is_reported_as_disconnected_when_it_is_allowed(self):
        """An honest consequence, asserted rather than hidden.

        "No bond" is a class of the bond head like any other, so when the model's own
        runner-up for a bond is to not have it, dropping that bond is a legitimate -- and
        sometimes the cheapest -- way out of an over-valence. The result can be TWO
        fragments, which lifts plain validity but NOT fully-connected validity. The
        delivered molecule is still the whole graph; no fragment is ever selected.
        """
        builder = make_builder(
            ligand_valence_repair=True,
            ligand_valence_repair_allow_bond_deletion=True,
        )
        mols = decode(builder, ammonium_tensors(charge_runner_up=0.0))
        self.assertIsNotNone(mols[0])
        self.assertEqual(n_fragments(mols[0]), 2)
        self.assertEqual(builder.repair_stats["repaired"], 1)
        self.assertEqual(builder.repair_stats["edits_bonds"], 1)
        self.assertEqual(builder.repair_stats["edits_bond_deletions"], 1)
        self.assertEqual(builder.repair_stats["repaired_disconnected"], 1)
        self.assertEqual(builder.repair_stats["repaired_ok"], 0)

    def test_the_repair_is_refused_when_the_rebuild_still_will_not_load(self):
        """Zero valence violations is necessary, not sufficient -- RDKit has the last word.

        The guard is control flow, so it is tested as control flow: the SECOND build (the
        one on the repaired graph) is made to fail, and the delivered result must be the
        ORIGINAL failure, not a molecule we edited and then lost. This is also the shape of
        the stale-aromaticity case (C10): an element flip on an aromatic-flagged atom can
        fail kekulization on rebuild, and the honest answer is the original failure plus a
        `rejected` count.
        """
        real = smolRD.mol_from_atoms
        calls = []

        def failing_second_build(*args, **kwargs):
            calls.append(args)
            if len(calls) == 1:
                return real(*args, **kwargs)
            return None

        with mock.patch.object(smolRD, "mol_from_atoms", failing_second_build):
            builder = make_builder(ligand_valence_repair=True)
            mols = decode(builder, ammonium_tensors())
        self.assertEqual(len(calls), 2, "the repair must have been built and offered")
        self.assertEqual(mols, [None], "the ORIGINAL failure, not the repair's")
        self.assertEqual(builder.repair_stats["attempted"], 1)
        self.assertEqual(builder.repair_stats["rejected"], 1)
        self.assertEqual(builder.repair_stats["repaired"], 0)

    def test_the_builder_forwards_the_knob_into_the_search(self):
        """A knob stored on `self` and never passed through would be inert."""
        import flowr.models.mol_builder as module

        real_repair = module.repair_valence
        seen = []

        def spy(**kwargs):
            seen.append(kwargs["allow_bond_deletion"])
            return real_repair(**kwargs)

        with mock.patch.object(module, "repair_valence", spy):
            for allow in (True, False):
                builder = make_builder(
                    ligand_valence_repair=True,
                    ligand_valence_repair_allow_bond_deletion=allow,
                )
                decode(builder, bridged_ammonium())
        self.assertEqual(seen, [True, False])


# ================================================== 4. bookkeeping


class BookkeepingTests(unittest.TestCase):
    def test_the_stats_are_exact_under_the_worker_pool(self):
        """The decode fans out over a ThreadPoolExecutor; the counters must add up."""
        tensors = ammonium_tensors()
        batch = 40
        repeated = (
            tensors[0].repeat(batch, 1, 1),
            tensors[1].repeat(batch, 1),
            tensors[2].repeat(batch, 1, 1),
            tensors[3].repeat(batch, 1, 1, 1),
            tensors[4].repeat(batch, 1, 1),
        )
        builder = MolBuilder(*vocabs(), n_workers=8, ligand_valence_repair=True)
        mols = decode(builder, repeated)
        self.assertEqual(len(mols), batch)
        self.assertTrue(all(mol is not None for mol in mols))
        self.assertEqual(builder.repair_stats["attempted"], batch)
        self.assertEqual(builder.repair_stats["repaired"], batch)
        self.assertEqual(builder.repair_stats["repaired_ok"], batch)
        self.assertEqual(builder.repair_stats["edits_charges"], batch)

    def test_a_bound_cap_is_counted_and_not_silently_swallowed(self):
        """A truncated search must be visible; "no repair" and "gave up" are different facts."""
        # TWO over-valent nitrogens (atoms 0 and 5) sharing the bridging carbon 4, so the
        # molecule is a single connected fragment and only the valence is wrong. Two charge
        # flips fix it and NOTHING inside a budget of one can.
        atoms = [{} for _ in range(10)]
        charges = [{} for _ in range(10)]
        bonds = {}
        for centre, arms in ((0, (1, 2, 3, 4)), (5, (4, 6, 7, 8))):
            atoms[centre] = {"N": 1.0}
            charges[centre] = {0: 0.6, 1: 0.4}
            for other in arms:
                atoms[other] = {"C": 1.0}
                charges[other] = {0: 1.0}
                bonds[(min(centre, other), max(centre, other))] = {1: 0.98, 0: 0.02}
        atoms[9] = {"C": 1.0}
        charges[9] = {0: 1.0}
        bonds[(8, 9)] = {1: 0.98, 0: 0.02}
        tensors = _sheets(atoms, charges, bonds, 10)

        builder = make_builder(
            ligand_valence_repair=True, ligand_valence_repair_max_edits=1
        )
        self.assertEqual(decode(builder, tensors), [None])
        self.assertEqual(builder.repair_stats["cap_edits"], 1)
        self.assertEqual(builder.repair_stats["repaired"], 0)

        wider = make_builder(
            ligand_valence_repair=True, ligand_valence_repair_max_edits=2
        )
        mols = decode(wider, tensors)
        self.assertIsNotNone(mols[0])
        self.assertEqual(wider.repair_stats["cap_edits"], 0)
        self.assertEqual(wider.repair_stats["edits_charges"], 2)

    def test_the_virtual_node_class_is_never_a_repair_target(self):
        """`<NOATOM>` is not an element; "repairing" a valence into it is a guaranteed failure."""
        from flowr import constants

        vocab = _build_vocab(virtual_nodes=True)
        self.assertTrue(
            vocab.contains(constants.NOATOM_TOKEN), "precondition: the class exists"
        )
        vocab_charges = _build_vocab_charges()
        builder = MolBuilder(
            vocab, vocab_charges, n_workers=2, ligand_valence_repair=True
        )
        n = 6
        coords = torch.zeros(1, n, 3)
        coords[0, :, 0] = torch.arange(n, dtype=torch.float32) * 1.4
        mask = torch.ones(1, n, dtype=torch.bool)
        atom_dists = torch.zeros(1, n, vocab.size)
        # atom 0 is a pentavalent carbon whose most probable alternative type is <NOATOM>
        atom_dists[0, 0, vocab.token_idx_map["C"]] = 0.55
        atom_dists[0, 0, vocab.token_idx_map[constants.NOATOM_TOKEN]] = 0.43
        atom_dists[0, 0, vocab.token_idx_map["P"]] = 0.02
        for k in range(1, n):
            atom_dists[0, k, vocab.token_idx_map["C"]] = 1.0
        charge_dists = torch.zeros(1, n, vocab_charges.size)
        charge_dists[0, :, vocab_charges.token_idx_map[0]] = 1.0
        bond_dists = torch.zeros(1, n, n, N_BOND_CLASSES)
        bond_dists[..., 0] = 1.0
        for k in range(1, n):
            bond_dists[0, 0, k, 0] = bond_dists[0, k, 0, 0] = 0.0001
            bond_dists[0, 0, k, 1] = bond_dists[0, k, 0, 1] = 0.9999
        mols = decode(builder, (coords, mask, atom_dists, bond_dists, charge_dists))
        self.assertIsNotNone(mols[0])
        # `<NOATOM>` is 20x more probable than P, and is still refused: it is not an
        # element, so "repairing" the valence into it would only guarantee a different
        # build failure.
        self.assertEqual(mols[0].GetAtomWithIdx(0).GetSymbol(), "P")

    def test_the_repair_is_deterministic_across_repeated_decodes(self):
        tensors = random_tensors(3)
        reference = None
        for trial in range(4):
            with self.subTest(trial=trial):
                current = smiles(
                    decode(make_builder(ligand_valence_repair=True), tensors)
                )
                reference = current if reference is None else reference
                self.assertEqual(current, reference)

    def test_the_repair_does_not_write_into_the_batch_tensors(self):
        """`_extract_mols` hands out views; a mutated batch would poison later consumers."""
        tensors = random_tensors(1)
        snapshots = [t.clone() for t in tensors]
        decode(make_builder(ligand_valence_repair=True), tensors)
        for before, after in zip(snapshots, tensors):
            self.assertTrue(torch.equal(before, after))

    def test_the_repaired_molecule_sanitises_for_real(self):
        """Not merely "RDKit returned an object": the delivered mol must pass SanitizeMol."""
        for seed in range(8):
            with self.subTest(seed=seed):
                mols = decode(
                    make_builder(ligand_valence_repair=True), random_tensors(seed)
                )
                for mol in mols:
                    if mol is None:
                        continue
                    Chem.SanitizeMol(Chem.Mol(mol))
                    positions = np.asarray(mol.GetConformer().GetPositions())
                    self.assertTrue(np.isfinite(positions).all())


# ================================================== 5. the bond-deletion guard


class BondDeletionGuardTests(unittest.TestCase):
    def test_the_unrepaired_argmax_is_a_connected_valence_failure(self):
        """Precondition for everything below: the build fails, and it is ONE fragment."""
        self.assertEqual(decode(make_builder(), bridged_ammonium()), [None])
        unsanitised = smolRD.mol_from_atoms(
            np.asarray(bridged_ammonium()[0][0]),
            ["N"] + ["C"] * 5,
            bonds=np.array([[1, 0, 2], [2, 0, 1], [3, 0, 1], [4, 3, 1], [5, 4, 1]]),
            charges=np.zeros(6, dtype=np.int64),
            sanitise=False,
        )
        self.assertEqual(n_fragments(unsanitised), 1)

    def test_allowing_deletion_repairs_by_deleting_the_bridge_and_splits_the_molecule(
        self,
    ):
        """The donor's default, reachable here on request. This is the leak, not a bug."""
        builder = make_builder(
            ligand_valence_repair=True,
            ligand_valence_repair_allow_bond_deletion=True,
        )
        mols = decode(builder, bridged_ammonium())
        self.assertIsNotNone(mols[0])
        self.assertEqual(n_fragments(mols[0]), 2)
        self.assertEqual(builder.repair_stats["repaired"], 1)
        self.assertEqual(builder.repair_stats["repaired_disconnected"], 1)
        self.assertEqual(builder.repair_stats["repaired_ok"], 0)
        self.assertEqual(builder.repair_stats["edits_bonds"], 1)
        self.assertEqual(builder.repair_stats["edits_bond_deletions"], 1)
        self.assertEqual(builder.repair_stats["unrepaired_deletion_blocked"], 0)

    def test_forbidding_deletion_delivers_one_connected_molecule_instead(self):
        builder = make_builder(ligand_valence_repair=True)  # default: forbidden
        mols = decode(builder, bridged_ammonium())
        self.assertIsNotNone(mols[0])
        self.assertEqual(n_fragments(mols[0]), 1)
        # the double bond came down to a single; every atom is still there
        self.assertEqual(mols[0].GetNumAtoms(), 6)
        self.assertEqual(mols[0].GetNumBonds(), 5)
        self.assertTrue(
            all(
                bond.GetBondType() == Chem.BondType.SINGLE for bond in mols[0].GetBonds()
            )
        )
        self.assertEqual(builder.repair_stats["repaired"], 1)
        self.assertEqual(builder.repair_stats["repaired_ok"], 1)
        self.assertEqual(builder.repair_stats["repaired_disconnected"], 0)
        self.assertEqual(builder.repair_stats["edits_bonds"], 1)
        self.assertEqual(builder.repair_stats["edits_bond_deletions"], 0)

    def test_forbidding_deletion_reports_the_repairs_it_gave_up(self):
        """When only a deletion was on offer: `unrepaired`, and the reason is legible."""
        builder = make_builder(ligand_valence_repair=True)
        self.assertEqual(
            decode(builder, deletion_only_ammonium()),
            [None],
            "the molecule is left broken, never delivered split",
        )
        self.assertEqual(builder.repair_stats["attempted"], 1)
        self.assertEqual(builder.repair_stats["repaired"], 0)
        self.assertEqual(builder.repair_stats["unrepaired"], 1)
        self.assertEqual(builder.repair_stats["unrepaired_deletion_blocked"], 1)
        self.assertEqual(builder.repair_stats["edits_bond_deletions"], 0)

        # and an unguarded builder really would have split it, so this is the guard's COST
        unguarded = make_builder(
            ligand_valence_repair=True,
            ligand_valence_repair_allow_bond_deletion=True,
        )
        delivered = decode(unguarded, deletion_only_ammonium())
        self.assertIsNotNone(delivered[0])
        self.assertEqual(n_fragments(delivered[0]), 2)
        self.assertEqual(unguarded.repair_stats["edits_bond_deletions"], 1)
        self.assertEqual(unguarded.repair_stats["unrepaired_deletion_blocked"], 0)

    def test_forbidding_deletion_costs_a_real_win_when_the_bond_is_a_RING_bond(self):
        """Reported loudly rather than hidden: the guard is a trade, not a free improvement."""
        unguarded = make_builder(
            ligand_valence_repair=True,
            ligand_valence_repair_allow_bond_deletion=True,
        )
        mols = decode(unguarded, ring_ammonium())
        self.assertIsNotNone(mols[0])
        self.assertEqual(
            n_fragments(mols[0]), 1, "deleting a RING bond keeps the molecule in one piece"
        )
        # `edits_bond_deletions` is therefore NOT a synonym for `repaired_disconnected`: it
        # counts the mechanism, and the mechanism is sometimes harmless
        self.assertEqual(unguarded.repair_stats["edits_bond_deletions"], 1)
        self.assertEqual(unguarded.repair_stats["repaired_ok"], 1)
        self.assertEqual(unguarded.repair_stats["repaired_disconnected"], 0)

        guarded = make_builder(ligand_valence_repair=True)
        self.assertEqual(decode(guarded, ring_ammonium()), [None])
        self.assertEqual(guarded.repair_stats["unrepaired_deletion_blocked"], 1)

    def test_the_connected_rate_is_the_metric_that_moves(self):
        """The published telemetry, not the raw dict: this is the panel the arm is read off."""
        cases = [bridged_ammonium(), deletion_only_ammonium()]

        def epoch(allow: bool):
            builder = make_builder(
                ligand_valence_repair=True,
                ligand_valence_repair_allow_bond_deletion=allow,
            )
            for case in cases:
                decode(builder, case)
            metric = Metrics.ValenceRepairStats()
            metric.update(builder.repair_stats)
            return {key: float(value) for key, value in metric.compute().items()}

        unguarded = epoch(True)
        guarded = epoch(False)

        self.assertEqual(unguarded["repair-connected-rate"], 0.0)
        self.assertEqual(unguarded["repair-edits-bond-deletions"], 2.0)
        self.assertEqual(unguarded["repair-unrepaired-deletion-blocked"], 0.0)

        self.assertEqual(guarded["repair-connected-rate"], 1.0)
        self.assertEqual(guarded["repair-edits-bond-deletions"], 0.0)
        self.assertEqual(guarded["repair-unrepaired-deletion-blocked"], 1.0)
        # the honest trade: one fewer molecule repaired, and the panel says which and why
        self.assertEqual(
            guarded["repair-repaired"], unguarded["repair-repaired"] - 1.0
        )
        self.assertEqual(
            guarded["repair-unrepaired"], unguarded["repair-unrepaired"] + 1.0
        )


# ================================================== 6. is the repair CHEMICALLY sensible?
#
# "It sanitises" is a low bar -- a bad repair sanitises too. The planted-corruption battery
# below asks the real question: take a molecule that is chemically fine, corrupt ONE head's
# argmax so exactly one atom goes over-valent (the shape of the great majority of real
# failures on the donor's census), leave the TRUE class as the model's runner-up, and see
# whether the repair comes back with the molecule that was there all along.
#
# It also states the limit of that result, in
# `test_when_the_truth_is_not_the_models_runner_up_it_delivers_a_different_molecule`: the
# repair is only as chemically right as the model's own ranking. It restores the most
# probable VALID reading of the model's output, which is the intended molecule only when the
# model actually ranked it next.

ELEMENTS = ["C", "N", "O", "F", "S", "Cl"]
MAX_DEGREE = {"C": 4, "N": 3, "O": 2, "F": 1, "S": 2, "Cl": 1}


def _random_tree(rng, n=12):
    """A connected, single-bonded, valence-respecting molecule: the "before" picture."""
    while True:
        tokens = [rng.choice(ELEMENTS) for _ in range(n)]
        degree = [0] * n
        edges = []
        order = list(range(n))
        rng.shuffle(order)
        for position in range(1, n):
            child = order[position]
            parents = [
                order[k]
                for k in range(position)
                if degree[order[k]] < MAX_DEGREE[tokens[order[k]]]
            ]
            if not parents or degree[child] >= MAX_DEGREE[tokens[child]]:
                break
            parent = rng.choice(parents)
            edges.append((max(parent, child), min(parent, child)))
            degree[parent] += 1
            degree[child] += 1
        else:
            return tokens, edges, degree


def _tensors_from_trees(trees):
    """One-hot `(coords, mask, atom_dists, bond_dists, charge_dists)` for `(tokens, edges)`."""
    vocab, vocab_charges = vocabs()
    n_max = max(len(tokens) for tokens, _ in trees)
    coords = torch.zeros(len(trees), n_max, 3)
    mask = torch.zeros(len(trees), n_max, dtype=torch.bool)
    atom_dists = torch.zeros(len(trees), n_max, vocab.size)
    charge_dists = torch.zeros(len(trees), n_max, vocab_charges.size)
    bond_dists = torch.zeros(len(trees), n_max, n_max, N_BOND_CLASSES)
    bond_dists[..., 0] = 1.0
    for row, (tokens, edges) in enumerate(trees):
        n = len(tokens)
        mask[row, :n] = True
        coords[row, :n] = torch.arange(n, dtype=torch.float32)[:, None] * 1.5
        for index, token in enumerate(tokens):
            atom_dists[row, index, vocab.token_idx_map[token]] = 1.0
            charge_dists[row, index, vocab_charges.token_idx_map[0]] = 1.0
        for i, j in edges:
            bond_dists[row, i, j] = bond_dists[row, j, i] = 0.0
            bond_dists[row, i, j, 1] = bond_dists[row, j, i, 1] = 1.0
    return coords, mask, atom_dists, bond_dists, charge_dists


def _corruption(tokens, edges, degree, rng):
    """One head-argmax corruption that makes exactly one atom over-valent."""
    choices = []
    for i, j in edges:
        if degree[i] >= MAX_DEGREE[tokens[i]] or degree[j] >= MAX_DEGREE[tokens[j]]:
            choices.append(("bond", (i, j), 2))  # promote SINGLE -> DOUBLE
    for index, token in enumerate(tokens):
        if token == "C" and degree[index] == 4:
            choices.append(("charge", index, 1))  # C(+1) allows only 3
        if token == "N" and degree[index] == 3:
            choices.append(("charge", index, -1))  # N(-1) allows only 2
        if token == "C" and degree[index] >= 3:
            choices.append(("type", index, "N"))
        if token == "N" and degree[index] >= 2:
            choices.append(("type", index, "O"))
    return rng.choice(choices) if choices else None


def _planted_battery(seed, size, truth_rank=2):
    """`(tensors, true SMILES, cases)`.

    `truth_rank=2` puts the true class second (the model's runner-up); `truth_rank=3` slips
    a valid-but-wrong distractor above it.
    """
    rng = random.Random(seed)
    cases = []
    while len(cases) < size:
        tokens, edges, degree = _random_tree(rng)
        corruption = _corruption(tokens, edges, degree, rng)
        if corruption is not None:
            cases.append((tokens, edges, corruption))

    vocab, vocab_charges = vocabs()
    n_max = max(len(tokens) for tokens, _, _ in cases)
    coords = torch.zeros(len(cases), n_max, 3)
    mask = torch.zeros(len(cases), n_max, dtype=torch.bool)
    atom_dists = torch.zeros(len(cases), n_max, vocab.size)
    charge_dists = torch.zeros(len(cases), n_max, vocab_charges.size)
    bond_dists = torch.zeros(len(cases), n_max, n_max, N_BOND_CLASSES)
    bond_dists[..., 0] = 1.0
    truths = []

    for row, (tokens, edges, (kind, slot, wrong)) in enumerate(cases):
        n = len(tokens)
        mask[row, :n] = True
        coords[row, :n] = torch.arange(n, dtype=torch.float32)[:, None] * 1.5
        for index, token in enumerate(tokens):
            atom_dists[row, index, vocab.token_idx_map[token]] = 1.0
            charge_dists[row, index, vocab_charges.token_idx_map[0]] = 1.0
        for i, j in edges:
            bond_dists[row, i, j] = bond_dists[row, j, i] = 0.0
            bond_dists[row, i, j, 1] = bond_dists[row, j, i, 1] = 1.0

        truth = 0.10 if truth_rank == 3 else 0.25
        distractor = 0.15 if truth_rank == 3 else 0.0
        if kind == "bond":
            i, j = slot
            row_probs = torch.zeros(N_BOND_CLASSES)
            row_probs[wrong] = 1.0 - truth - distractor
            row_probs[1] = truth  # the truth: SINGLE
            row_probs[0] = distractor  # valid, but a different molecule
            bond_dists[row, i, j] = bond_dists[row, j, i] = row_probs
        elif kind == "charge":
            row_probs = torch.zeros(vocab_charges.size)
            row_probs[vocab_charges.token_idx_map[wrong]] = 1.0 - truth - distractor
            row_probs[vocab_charges.token_idx_map[0]] = truth
            row_probs[vocab_charges.token_idx_map[1 if wrong == -1 else -1]] = distractor
            charge_dists[row, slot] = row_probs
        else:
            row_probs = torch.zeros(vocab.size)
            row_probs[vocab.token_idx_map[wrong]] = 1.0 - truth - distractor
            row_probs[vocab.token_idx_map[tokens[slot]]] = truth
            row_probs[vocab.token_idx_map["S" if tokens[slot] != "S" else "P"]] = (
                distractor
            )
            atom_dists[row, slot] = row_probs

        editable = Chem.RWMol()
        for token in tokens:
            editable.AddAtom(Chem.Atom(token))
        for i, j in edges:
            editable.AddBond(i, j, Chem.BondType.SINGLE)
        reference = editable.GetMol()
        Chem.SanitizeMol(reference)
        truths.append(Chem.MolToSmiles(reference))

    return (coords, mask, atom_dists, bond_dists, charge_dists), truths, cases


class PlantedCorruptionTests(unittest.TestCase):
    def test_the_repair_recovers_the_molecule_the_model_nearly_predicted(self):
        """The chemistry answer: not "it sanitises", but "it is the molecule that was there".

        Every case here is a valid molecule with ONE head's argmax corrupted and the true
        class left as the model's runner-up. Every failed build must come back as the
        ORIGINAL molecule, and the repair must have edited the channel that was actually
        corrupted -- if it edited a different channel it would be papering over the error
        somewhere else.
        """
        for seed in (0, 1):
            with self.subTest(seed=seed):
                tensors, truths, cases = _planted_battery(seed, size=60)
                off_mols = decode(make_builder(), tensors)
                builder = make_builder(
                    ligand_valence_repair=True,
                    ligand_valence_repair_allow_bond_deletion=True,
                )
                on_mols = decode(builder, tensors)

                planted = [
                    index for index, mol in enumerate(off_mols) if mol is None
                ]
                self.assertGreaterEqual(
                    len(planted),
                    30,
                    "precondition: the corruptions must actually break things",
                )
                for index in planted:
                    self.assertIsNotNone(
                        on_mols[index], f"case {index} was not recovered"
                    )
                    self.assertEqual(n_fragments(on_mols[index]), 1)
                    self.assertEqual(Chem.MolToSmiles(on_mols[index]), truths[index])
                kinds = collections.Counter(cases[index][2][0] for index in planted)
                self.assertEqual(set(kinds), {"bond", "charge", "type"}, kinds)
                # the repair edited exactly the channels that were corrupted, in the same
                # proportions
                self.assertEqual(builder.repair_stats["edits_bonds"], kinds["bond"])
                self.assertEqual(
                    builder.repair_stats["edits_charges"], kinds["charge"]
                )
                self.assertEqual(builder.repair_stats["edits_types"], kinds["type"])
                self.assertEqual(builder.repair_stats["repaired_disconnected"], 0)

    def test_when_the_truth_is_not_the_models_runner_up_it_delivers_a_different_molecule(
        self,
    ):
        """The honest limit, asserted rather than left for someone to discover.

        Slip a valid-but-wrong distractor above the true class and the repair takes the
        distractor: it restores the most probable VALID reading of the model's output, which
        is the intended molecule only when the model ranked it next. It is therefore only as
        chemically right as the model's own ordering -- and when the distractor is "no bond",
        the delivered molecule comes back in two pieces, which the counters say out loud.
        """
        tensors, truths, _ = _planted_battery(0, size=60, truth_rank=3)
        off_mols = decode(make_builder(), tensors)
        builder = make_builder(
            ligand_valence_repair=True,
            ligand_valence_repair_allow_bond_deletion=True,
        )
        on_mols = decode(builder, tensors)

        planted = [index for index, mol in enumerate(off_mols) if mol is None]
        recovered = sum(
            1
            for index in planted
            if on_mols[index] is not None
            and Chem.MolToSmiles(on_mols[index]) == truths[index]
        )
        self.assertLess(
            recovered,
            len(planted) // 2,
            "with the truth demoted, most repairs must NOT be the original molecule -- "
            "if they were, the search would be reading something other than the model",
        )
        self.assertEqual(builder.repair_stats["repaired"], len(planted))
        self.assertGreater(
            builder.repair_stats["repaired_disconnected"],
            0,
            "dropping a bond splits the molecule; that must be counted, not hidden",
        )
        self.assertEqual(
            builder.repair_stats["repaired_ok"]
            + builder.repair_stats["repaired_disconnected"],
            builder.repair_stats["repaired"],
        )

    def test_the_guard_removes_every_bond_deletion_from_the_planted_battery(self):
        """The tripwire, on a realistic population rather than one hand-built case."""
        tensors, _, _ = _planted_battery(0, size=60, truth_rank=3)
        unguarded = make_builder(
            ligand_valence_repair=True,
            ligand_valence_repair_allow_bond_deletion=True,
        )
        decode(unguarded, tensors)
        guarded = make_builder(ligand_valence_repair=True)
        decode(guarded, tensors)
        self.assertGreater(
            unguarded.repair_stats["edits_bond_deletions"],
            0,
            "precondition: the unguarded search does delete bonds here",
        )
        self.assertEqual(guarded.repair_stats["edits_bond_deletions"], 0)
        self.assertEqual(guarded.repair_stats["repaired_disconnected"], 0)


if __name__ == "__main__":
    unittest.main()

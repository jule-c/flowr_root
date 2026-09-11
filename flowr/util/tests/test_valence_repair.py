"""Contract for the valence-constrained decode repair.

These pin the behaviour the port must have in THIS repo, independently of how the donor
branch spelled it. Each test names the constraint it enforces; all of them were derived
from behaviour verified by hand against this repo's own build path before the port existed.
"""

import unittest

import numpy as np
import torch

import flowr.util.rdkit as smolRD
from flowr.models.mol_builder import MolBuilder
from flowr.scriptutil import _build_vocab, _build_vocab_charges
from flowr.util.valence_repair import repair_valence
from flowr.util.valence_report import UNLIMITED_VALENCE, max_valence, valence_limits

VOCAB = _build_vocab()
VOCAB_CHARGES = _build_vocab_charges()
ATOM_TOKENS = [VOCAB.idx_token_map[i] for i in range(VOCAB.size)]
CHARGE_VALUES = [VOCAB_CHARGES.idx_token_map[i] for i in range(VOCAB_CHARGES.size)]
C = ATOM_TOKENS.index("C")
NEUTRAL = CHARGE_VALUES.index(0)
NONE_, SINGLE, DOUBLE, TRIPLE, AROMATIC = 0, 1, 2, 3, 4
N_BOND = 5


def _sheets(n, bond_spec, atom_class=C, charge_class=NEUTRAL):
    """One molecule's head outputs whose independent argmax is exactly `bond_spec`."""
    atom_probs = np.full((n, len(ATOM_TOKENS)), 1e-4, dtype=np.float64)
    atom_probs[:, atom_class] = 0.99
    charge_probs = np.full((n, len(CHARGE_VALUES)), 1e-4, dtype=np.float64)
    charge_probs[:, charge_class] = 0.99
    bond_probs = np.zeros((n, n, N_BOND), dtype=np.float64)
    bond_probs[:, :, NONE_] = 0.99
    bond_probs[:, :, 1:] = 0.0025
    for (i, j), dist in bond_spec.items():
        for k in range(N_BOND):
            bond_probs[i, j, k] = bond_probs[j, i, k] = dist.get(k, 0.0)
    return dict(
        atom_probs=atom_probs,
        charge_probs=charge_probs,
        bond_probs=bond_probs,
        atom_tokens=ATOM_TOKENS,
        charge_values=CHARGE_VALUES,
    )


def _overvalent(cheapest_escape):
    """C0 single-bonded to C1/C2/C3 and double-bonded to C4 -> valence 5 on neutral carbon.

    `cheapest_escape` decides whether demotion or DELETION is the higher-probability fix,
    which is what makes the bond-deletion guard observable.
    """
    c0_c4 = (
        {DOUBLE: 0.60, SINGLE: 0.35, NONE_: 0.05}
        if cheapest_escape == "demote"
        else {DOUBLE: 0.60, NONE_: 0.35, SINGLE: 0.05}
    )
    return _sheets(
        5,
        {
            (0, 1): {SINGLE: 0.99, NONE_: 0.01},
            (0, 2): {SINGLE: 0.99, NONE_: 0.01},
            (0, 3): {SINGLE: 0.99, NONE_: 0.01},
            (0, 4): c0_c4,
        },
    )


def _valid_five():
    return _sheets(
        5,
        {
            (0, 1): {SINGLE: 0.99, NONE_: 0.01},
            (0, 2): {SINGLE: 0.99, NONE_: 0.01},
            (0, 3): {SINGLE: 0.99, NONE_: 0.01},
            (0, 4): {SINGLE: 0.99, NONE_: 0.01},
        },
    )


def _batch(sheets):
    """Turn one molecule's sheets into the batched tensors MolBuilder decodes."""
    n = sheets["atom_probs"].shape[0]
    coords = torch.tensor(
        [[1.5 * i, 0.0, 0.0] for i in range(n)], dtype=torch.float32
    ).unsqueeze(0)
    return dict(
        coords=coords,
        atom_dists=torch.from_numpy(sheets["atom_probs"]).float().unsqueeze(0),
        bond_dists=torch.from_numpy(sheets["bond_probs"]).float().unsqueeze(0),
        charge_dists=torch.from_numpy(sheets["charge_probs"]).float().unsqueeze(0),
        mask=torch.ones(1, n, dtype=torch.int64),
    )


def _fragments(mol):
    return len(smolRD.Chem.GetMolFrags(mol)) if mol is not None else -1


class RepairSearchTests(unittest.TestCase):
    """The search itself: `flowr.util.valence_repair.repair_valence`."""

    def test_valid_molecule_is_never_touched(self):
        """C5: an argmax that already satisfies the limits comes back unedited."""
        out = repair_valence(**_valid_five(), allow_bond_deletion=False)
        self.assertFalse(out.attempted)
        self.assertEqual(out.edits, ())

    def test_max_edits_zero_returns_the_plain_argmax(self):
        """The untouched baseline the A/B measurement is read against."""
        out = repair_valence(**_overvalent("demote"), max_edits=0, allow_bond_deletion=False)
        self.assertEqual(out.edits, ())
        self.assertFalse(out.repaired)

    def test_repairs_an_overvalent_atom_by_demoting_a_bond(self):
        out = repair_valence(**_overvalent("demote"), allow_bond_deletion=False)
        self.assertTrue(out.attempted)
        self.assertTrue(out.repaired)
        self.assertEqual(out.bond_classes[0, 4], SINGLE)
        self.assertEqual(out.bond_classes[4, 0], SINGLE)

    def test_forbidden_deletion_is_not_used_even_when_cheapest(self):
        """C4: the guard sits on the candidate generator, and records that it bit."""
        out = repair_valence(**_overvalent("delete"), allow_bond_deletion=False)
        self.assertTrue(out.repaired)
        self.assertNotEqual(out.bond_classes[0, 4], NONE_, "repair escaped by DELETING a bond")
        self.assertTrue(out.deletion_blocked)
        self.assertFalse(any(e.deletes_a_bond() for e in out.edits))

    def test_deletion_is_used_when_explicitly_allowed(self):
        """The donor's shipped behaviour stays reachable, so the guard is a real choice."""
        out = repair_valence(**_overvalent("delete"), allow_bond_deletion=True)
        self.assertTrue(out.repaired)
        self.assertEqual(out.bond_classes[0, 4], NONE_)

    def test_does_not_write_into_the_caller_tensors(self):
        """The sheets are views into the caller's batch and must be read-only."""
        sheets = _overvalent("delete")
        before = {k: v.copy() for k, v in sheets.items() if isinstance(v, np.ndarray)}
        repair_valence(**sheets, allow_bond_deletion=False)
        for key, original in before.items():
            np.testing.assert_array_equal(sheets[key], original)


class BuilderGateTests(unittest.TestCase):
    """The gate: only a FAILED build is repaired, and never a reference ligand."""

    @staticmethod
    def _builder(**kwargs):
        return MolBuilder(VOCAB, VOCAB_CHARGES, n_workers=2, **kwargs)

    def test_repair_defaults_to_off(self):
        """C5: the flag is opt-in; with it off the over-valent build still fails."""
        builder = self._builder()
        self.assertFalse(builder.ligand_valence_repair)
        mols = builder.mols_from_tensors(**_batch(_overvalent("demote")), sanitise=True)
        self.assertIsNone(mols[0])

    def test_repair_rescues_a_generated_molecule(self):
        builder = self._builder(
            ligand_valence_repair=True,
            ligand_valence_repair_allow_bond_deletion=False,
        )
        mols = builder.mols_from_tensors(**_batch(_overvalent("demote")), sanitise=True)
        self.assertIsNotNone(mols[0])
        self.assertEqual(_fragments(mols[0]), 1)
        self.assertEqual(builder.repair_stats["attempted"], 1)
        self.assertEqual(builder.repair_stats["repaired"], 1)
        self.assertEqual(builder.repair_stats["repaired_ok"], 1)
        self.assertEqual(builder.repair_stats["edits_bond_deletions"], 0)

    def test_a_molecule_that_builds_is_never_repaired(self):
        """C5: the gate is the FAILED build, so a loadable molecule cannot be touched."""
        builder = self._builder(ligand_valence_repair=True)
        mols = builder.mols_from_tensors(**_batch(_valid_five()), sanitise=True)
        self.assertIsNotNone(mols[0])
        self.assertEqual(builder.repair_stats["attempted"], 0)

    def test_reference_ligands_are_never_repaired(self):
        """C1: `ligs_from_complex` decodes the REFERENCE ligand (every caller names it
        `ref_ligs` and it is written out as `out_dict["ref_lig"]`). Repairing it would
        silently alter the ground truth that RMSD and substructure matching compare to."""
        builder = self._builder(
            ligand_valence_repair=True,
            ligand_valence_repair_allow_bond_deletion=False,
        )
        batch = _batch(_overvalent("demote"))
        mols = builder.ligs_from_complex(
            batch["coords"],
            batch["mask"],
            batch["atom_dists"],
            bond_dists=batch["bond_dists"],
            charge_dists=batch["charge_dists"],
            sanitise=True,
        )
        self.assertIsNone(mols[0], "the reference ligand was repaired")
        self.assertEqual(builder.repair_stats["attempted"], 0)

    def test_counters_are_exact_under_the_worker_pool(self):
        """The build fans out over a ThreadPoolExecutor, so the counters need a lock."""
        builder = MolBuilder(
            VOCAB,
            VOCAB_CHARGES,
            n_workers=8,
            ligand_valence_repair=True,
            ligand_valence_repair_allow_bond_deletion=False,
        )
        one = _batch(_overvalent("demote"))
        n = 40
        mols = builder.mols_from_tensors(
            coords=one["coords"].repeat(n, 1, 1),
            atom_dists=one["atom_dists"].repeat(n, 1, 1),
            bond_dists=one["bond_dists"].repeat(n, 1, 1, 1),
            charge_dists=one["charge_dists"].repeat(n, 1, 1),
            mask=one["mask"].repeat(n, 1),
            sanitise=True,
        )
        self.assertEqual(len(mols), n)
        self.assertEqual(builder.repair_stats["attempted"], n)
        self.assertEqual(builder.repair_stats["repaired"], n)


class LimitTableTests(unittest.TestCase):
    def test_table_is_probed_from_rdkit_not_the_naive_rule(self):
        """C6: the naive rule disagrees with RDKit's sanitiser on most pairs here.

        Measured by hand in this venv: 54 of the 98 (element, charge) pairs differ under
        `max(GetValenceList) + charge`. A table built from that rule would describe
        different chemistry from the sanitiser that actually raises AtomValenceException.
        """
        from rdkit import Chem

        pt = Chem.GetPeriodicTable()
        limits = valence_limits()
        self.assertEqual(len(limits), 98, "vocabulary is 14 elements x 7 charges")
        disagreements = sum(
            1
            for (symbol, charge), limit in limits.items()
            if limit != float(max(v for v in pt.GetValenceList(symbol) if v >= 0) + charge)
        )
        self.assertGreater(disagreements, 40, "table looks like the naive rule, not a probe")

    def test_neutral_carbon_limit_is_four(self):
        self.assertEqual(max_valence("C", 0), 4.0)

    def test_no_table_entry_is_negative(self):
        """A negative limit makes `valence > limit` true at ZERO bonds -- a false positive."""
        self.assertTrue(all(v >= 0 for v in valence_limits().values()))

    def test_sentinel_tokens_are_unlimited_and_so_never_repair_targets(self):
        """`<PAD>` / `<NOATOM>` columns survive `_remove_virtual_atoms`, so this guard
        is load-bearing: an unlimited limit means the search refuses them as edit targets."""
        self.assertEqual(max_valence("<PAD>", 0), UNLIMITED_VALENCE)
        self.assertEqual(max_valence("<NOATOM>", 0), UNLIMITED_VALENCE)


if __name__ == "__main__":
    unittest.main()

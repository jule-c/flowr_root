"""`MolBuilder.repair_stats` is accumulated per epoch; this is the metric that publishes it.

Without it, turning `ligand_valence_repair` on gives a moved validity number with no
explanation attached. In particular "repaired 19, all connected" and "repaired 19, eight of
them by DELETING a bond" are the same number on plain validity and opposite answers on
fully-connected validity -- bond demotion to class 0 is a legal repair move, so the cheapest
escape from an over-valence is sometimes deleting a bond. `repaired_ok` vs
`repaired_disconnected` is what separates them, and `edits_{charges,bonds,types}` is the
head attribution.

Every expected value here is hand-computed from the counters fed in.

Ported from the donor branch's `tests/flowr/util/test_valence_repair_stats.py`; pytest
spellings became their `unittest` equivalents and nothing else changed.
"""

from __future__ import annotations

import unittest

import torch

import flowr.util.metrics as Metrics
from flowr.models.mol_builder import MolBuilder


def _stats() -> Metrics.ValenceRepairStats:
    return Metrics.ValenceRepairStats()


def _builder_counter_names() -> set[str]:
    """The counter set of a real `MolBuilder`, without paying for a vocabulary."""
    builder = MolBuilder.__new__(MolBuilder)
    builder.reset_repair_stats()
    return set(builder.repair_stats)


class CounterSetTests(unittest.TestCase):
    """The counter set must be exactly the builder's, or a counter stops being logged."""

    def test_the_metric_covers_every_counter_the_builder_accumulates(self):
        """A counter the builder tracks but the metric drops is invisible telemetry; a
        counter the metric invents would be logged as a permanent zero."""
        self.assertEqual(
            _builder_counter_names(), set(Metrics.ValenceRepairStats.COUNTERS)
        )

    def test_the_task_mandated_counters_are_all_present(self):
        for required in (
            "repaired_ok",
            "repaired_disconnected",
            "rejected",
            "unrepaired",
            "edits_charges",
            "edits_bonds",
            "edits_types",
        ):
            with self.subTest(counter=required):
                self.assertIn(required, Metrics.ValenceRepairStats.COUNTERS)


class AccumulationTests(unittest.TestCase):
    def test_a_fresh_metric_reports_all_zeros_and_never_nan(self):
        out = _stats().compute()
        self.assertTrue(all(torch.isfinite(value) for value in out.values()))
        self.assertTrue(all(float(value) == 0.0 for value in out.values()))

    def test_the_counters_come_back_exactly_as_fed(self):
        metric = _stats()
        metric.update({"attempted": 22, "repaired": 19, "repaired_ok": 11})
        out = metric.compute()
        self.assertEqual(float(out["repair-attempted"]), 22.0)
        self.assertEqual(float(out["repair-repaired"]), 19.0)
        self.assertEqual(float(out["repair-repaired-ok"]), 11.0)

    def test_two_updates_add_rather_than_replace(self):
        """One update per rank per epoch is the shipped call pattern, but a second
        batch-wise caller must not silently overwrite the first."""
        metric = _stats()
        metric.update({"attempted": 3, "edits_bonds": 2})
        metric.update({"attempted": 4, "edits_bonds": 5})
        out = metric.compute()
        self.assertEqual(float(out["repair-attempted"]), 7.0)
        self.assertEqual(float(out["repair-edits-bonds"]), 7.0)

    def test_an_absent_counter_is_treated_as_zero_not_as_an_error(self):
        metric = _stats()
        metric.update({"attempted": 1})
        self.assertEqual(float(metric.compute()["repair-rejected"]), 0.0)

    def test_an_unknown_counter_is_refused_loudly(self):
        """A typo'd key must not vanish into a metric whose whole job is to be complete."""
        with self.assertRaisesRegex(ValueError, "unknown"):
            _stats().update({"repaired_okay": 1})

    def test_reset_returns_the_metric_to_zero(self):
        metric = _stats()
        metric.update({"attempted": 5})
        metric.reset()
        self.assertEqual(float(metric.compute()["repair-attempted"]), 0.0)


class DerivedRateTests(unittest.TestCase):
    """The two numbers that actually answer the question."""

    def test_the_accept_rate_is_repaired_over_attempted(self):
        metric = _stats()
        metric.update({"attempted": 20, "repaired": 15})
        self.assertAlmostEqual(
            float(metric.compute()["repair-accept-rate"]), 0.75, places=6
        )

    def test_the_connected_rate_is_repaired_ok_over_repaired(self):
        """THE number this metric is about: of the molecules the repair delivered, how many
        came back in one piece. 19 repaired / 11 connected => 0.5789, i.e. 8 were bought by
        deleting a bond and lift plain validity while leaving fully-connected validity
        where it was."""
        metric = _stats()
        metric.update({"attempted": 22, "repaired": 19, "repaired_ok": 11})
        self.assertAlmostEqual(
            float(metric.compute()["repair-connected-rate"]), 11 / 19, places=6
        )

    def test_the_all_connected_and_the_bond_deleting_cases_are_distinguishable(self):
        """Same `repaired`, opposite fully-connected consequence, different metric."""
        clean = _stats()
        clean.update({"attempted": 19, "repaired": 19, "repaired_ok": 19})
        dirty = _stats()
        dirty.update(
            {
                "attempted": 19,
                "repaired": 19,
                "repaired_ok": 11,
                "repaired_disconnected": 8,
            }
        )
        self.assertEqual(
            float(clean.compute()["repair-repaired"]),
            float(dirty.compute()["repair-repaired"]),
        )
        self.assertEqual(float(clean.compute()["repair-connected-rate"]), 1.0)
        self.assertAlmostEqual(
            float(dirty.compute()["repair-connected-rate"]), 11 / 19, places=6
        )

    def test_a_zero_denominator_gives_zero_not_nan(self):
        """A NaN would poison the logged series and read as a regression on a plot."""
        for rate in ("repair-accept-rate", "repair-connected-rate"):
            with self.subTest(rate=rate):
                value = _stats().compute()[rate]
                self.assertTrue(torch.isfinite(value))
                self.assertEqual(float(value), 0.0)


class DistributedReductionTests(unittest.TestCase):
    def test_every_state_reduces_with_sum(self):
        """A "mean" reduction would divide the per-rank counts by world_size, so an 8-GPU
        eval would under-report every count 8x with nothing to flag it."""
        from torchmetrics.utilities.data import dim_zero_sum

        metric = _stats()
        self.assertEqual(
            set(metric._reductions),
            set(Metrics.ValenceRepairStats.COUNTERS),
            "a state exists that this test does not check",
        )
        wrong = {
            name: metric._reductions[name]
            for name in metric._reductions
            if metric._reductions[name] is not dim_zero_sum
        }
        self.assertFalse(wrong, f"non-sum reduction on {wrong}")

    def test_a_simulated_two_rank_merge_sums_the_counters(self):
        """Replays what `Metric._sync_dist` does: gather into a leading rank axis, then
        apply the REGISTERED reduction (not a hardcoded sum), so this follows the product
        code."""
        rank_a, rank_b = _stats(), _stats()
        rank_a.update({"attempted": 4, "repaired": 3, "repaired_ok": 2})
        rank_b.update({"attempted": 6, "repaired": 5, "repaired_ok": 1})
        merged = _stats()
        for name in Metrics.ValenceRepairStats.COUNTERS:
            gathered = torch.stack(
                [getattr(rank, name).double() for rank in (rank_a, rank_b)]
            )
            setattr(merged, name, merged._reductions[name](gathered))
        out = merged.compute()
        self.assertEqual(float(out["repair-attempted"]), 10.0)
        self.assertEqual(float(out["repair-repaired"]), 8.0)
        self.assertAlmostEqual(
            float(out["repair-connected-rate"]), 3 / 8, places=6
        )


class LoggedShapeTests(unittest.TestCase):
    """Shape contract with a `self.log` loop."""

    def test_compute_returns_tensors_under_a_repair_prefix(self):
        out = _stats().compute()
        self.assertIsInstance(out, dict)
        self.assertTrue(all(key.startswith("repair-") for key in out))
        self.assertTrue(all(isinstance(value, torch.Tensor) for value in out.values()))
        self.assertTrue(all(value.dtype.is_floating_point for value in out.values()))

    def test_the_logged_key_set_is_fixed_regardless_of_the_counts(self):
        """A key set that depends on the data breaks a panel the first quiet epoch."""
        empty = set(_stats().compute())
        busy = _stats()
        busy.update({name: 1 for name in Metrics.ValenceRepairStats.COUNTERS})
        self.assertEqual(empty, set(busy.compute()))

    def test_no_key_carries_an_underscore(self):
        """An underscore here would be the only one in the logged namespace."""
        self.assertTrue(all("_" not in key for key in _stats().compute()))


class ZeroCostWhenOffTests(unittest.TestCase):
    def test_with_the_repair_off_the_builder_reports_an_all_zero_tally(self):
        builder = MolBuilder.__new__(MolBuilder)
        builder.reset_repair_stats()
        metric = _stats()
        metric.update(builder.repair_stats)
        self.assertTrue(
            all(float(value) == 0.0 for value in metric.compute().values())
        )

    def test_the_metric_registers_no_persistent_state(self):
        """torchmetrics states are non-persistent, so a checkpoint's state_dict is unchanged."""
        self.assertFalse(_stats().state_dict())


if __name__ == "__main__":
    unittest.main()

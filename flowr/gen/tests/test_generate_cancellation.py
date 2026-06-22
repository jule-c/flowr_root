"""Plumbing tests: generate_* must forward ``should_cancel`` into the model's
``_generate`` so cancellation reaches the integration loop.

These use a stub model at the real dependency boundary (the trained model),
exercising the real ``flowr.gen.generate`` plumbing without loading weights.
"""

import unittest

import torch

from flowr.gen.cancellation import GenerationCancelled, raise_if_cancelled
from flowr.gen.generate import generate_molecules


class _StubArgs:
    integration_steps = 25
    ode_sampling_strategy = "linear"
    solver = "euler"
    corrector_iters = 0


class _StubModel:
    """Stands in for a trained model; its ``_generate`` mimics the real loop's
    per-step cancellation check so we can assert the predicate is threaded in."""

    def __init__(self):
        self.received_should_cancel = "MISSING"
        self.steps_run = 0

    def _generate(self, prior, times, steps, strategy, solver, corr_iters,
                  save_traj, iter, should_cancel=None):
        self.received_should_cancel = should_cancel
        for i in range(steps):
            raise_if_cancelled(should_cancel, i)  # same call the real loops make
            self.steps_run += 1
        return {}

    def _generate_mols(self, output):
        return []


class GenerateMoleculesCancellationTests(unittest.TestCase):
    def _prior(self):
        return {"coords": torch.zeros(2, 3)}

    def test_forwards_predicate_to_generate(self):
        model = _StubModel()
        generate_molecules(_StubArgs(), model, self._prior(), device="cpu",
                           should_cancel=lambda: False)
        self.assertTrue(callable(model.received_should_cancel))

    def test_cancellation_propagates_and_stops_early(self):
        model = _StubModel()
        with self.assertRaises(GenerationCancelled):
            generate_molecules(_StubArgs(), model, self._prior(), device="cpu",
                               should_cancel=lambda: True)
        # Must abort on the very first step, not run all 25.
        self.assertEqual(model.steps_run, 0)

    def test_default_no_predicate_runs_to_completion(self):
        model = _StubModel()
        generate_molecules(_StubArgs(), model, self._prior(), device="cpu")
        self.assertIsNone(model.received_should_cancel)
        self.assertEqual(model.steps_run, _StubArgs.integration_steps)


if __name__ == "__main__":
    unittest.main()

"""Contract for the terminal categorical-noise guard in `Integrator._uniform_sample_step`.

The uniform-noise sampler adds `step_size * cat_noise_level * p_current` to EVERY category,
including a perfectly converged prediction. Near t=1 the off-diagonal mass can reach 1.0, at
which point `diags = (1 - sum).clamp(min=0.0)` floors the CURRENT category at zero and the
sampler is forced to move: the argmax is starved, not merely perturbed. The existing line

    noise[times + step_size < 1.0] = self.cat_noise_level

silences the noise on exactly ONE step, the last. The guard widens that window to the step's
actual validity condition, `1 - t > step_size * (1 + noise * n_categories)`, which is where
the Euler jump probability for a confidently predicted category first exceeds 1.

It is default-OFF and, with the flag off, the expression is LITERALLY the one shipped today.
"""

import unittest

import torch

from flowr.models.integrator import Integrator

K = 15  # atomics: "<PAD>" + 14 CORE_ATOMS
N = 100  # every generation entrypoint in this repo defaults to 100 integration steps
STEP = 1.0 / N
N_ATOMS = 20000


def _integrator(guard, noise=1):
    return Integrator(
        steps=N,
        type_strategy="uniform-sample",
        bond_strategy="uniform-sample",
        cat_noise_level=noise,
        cat_noise_euler_guard=guard,
    )


def _converged(n_atoms=N_ATOMS, category=3):
    """A perfectly converged prediction: current == predicted == one-hot at `category`."""
    one_hot = torch.zeros(1, n_atoms, K)
    one_hot[:, :, category] = 1.0
    return one_hot, one_hot.clone(), category


def _flips(integrator, t, seed=0):
    """How many atoms leave the argmax on a single step at time `t`."""
    torch.manual_seed(seed)
    curr, pred, category = _converged()
    out = integrator._uniform_sample_step(
        curr, pred, torch.tensor([t]), STEP
    )
    return int((out.argmax(dim=-1) != category).sum())


class TerminalNoiseGuardTests(unittest.TestCase):
    def test_guard_defaults_to_off(self):
        self.assertFalse(Integrator(steps=N).cat_noise_euler_guard)

    def test_without_the_guard_a_converged_prediction_is_still_kicked_near_t1(self):
        """The behaviour being fixed. P(leave) = (K-1)*noise/N = 0.14 for K=15, N=100."""
        flips = _flips(_integrator(guard=False), t=0.95)
        self.assertGreater(flips, 0)
        self.assertAlmostEqual(flips / N_ATOMS, (K - 1) / N, delta=0.02)

    def test_with_the_guard_a_converged_prediction_is_left_alone_near_t1(self):
        """noise=0 makes step_probs == pred_dist exactly, so the state cannot move."""
        self.assertEqual(_flips(_integrator(guard=True), t=0.95), 0)

    def test_the_guard_does_not_touch_mid_trajectory_stochasticity(self):
        """Only the terminal window is silenced; the noise the model trains under stays."""
        for t in (0.1, 0.5, 0.75):
            with self.subTest(t=t):
                self.assertGreater(_flips(_integrator(guard=True), t=t), 0)
                self.assertGreater(_flips(_integrator(guard=False), t=t), 0)

    def test_the_last_step_was_already_protected_either_way(self):
        """`times + step_size < 1.0` is False at t = 1 - step_size, with or without."""
        last = 1.0 - STEP
        self.assertEqual(_flips(_integrator(guard=False), t=last), 0)
        self.assertEqual(_flips(_integrator(guard=True), t=last), 0)

    def test_guard_is_inert_when_there_is_no_noise_to_guard(self):
        for t in (0.5, 0.95):
            with self.subTest(t=t):
                self.assertEqual(_flips(_integrator(guard=False, noise=0), t=t), 0)
                self.assertEqual(_flips(_integrator(guard=True, noise=0), t=t), 0)

    def test_flag_off_is_bit_identical_to_the_shipped_expression(self):
        """The strongest statement available: same seed, same sample, every step."""
        shipped = _integrator(guard=False)
        for t in (0.05, 0.33, 0.5, 0.84, 0.9, 0.95, 0.99):
            with self.subTest(t=t):
                torch.manual_seed(7)
                curr, pred, _ = _converged(n_atoms=500)
                got = shipped._uniform_sample_step(curr, pred, torch.tensor([t]), STEP)
                # Reference: the expression exactly as it stands on main.
                torch.manual_seed(7)
                curr2, pred2, _ = _converged(n_atoms=500)
                want = _reference_uniform_sample_step(
                    shipped, curr2, pred2, torch.tensor([t]), STEP
                )
                self.assertTrue(torch.equal(got, want))

    def test_guard_widens_the_silent_window_to_the_validity_condition(self):
        """`1 - t > step_size * (1 + noise * K)` -- auto-scaling per feature, no constant."""
        guarded = _integrator(guard=True)
        # K=15, noise=1, step=0.01 -> guard = 0.16, so noise stops once t >= 0.84.
        self.assertGreater(_flips(guarded, t=0.83), 0)
        self.assertEqual(_flips(guarded, t=0.85), 0)


def _reference_uniform_sample_step(integrator, curr_dist, pred_dist, t, step_size):
    """`_uniform_sample_step` verbatim as it stands on main, for the equivalence test."""
    import flowr.util.functional as smolF

    n_categories = pred_dist.size(-1)
    curr = torch.argmax(curr_dist, dim=-1).unsqueeze(-1)
    pred_probs_curr = torch.gather(pred_dist, -1, curr)
    ones = [1] * (len(pred_dist.shape) - 1)
    times = t.view(-1, *ones).clamp(min=integrator.eps, max=1.0 - integrator.eps)
    noise = torch.zeros_like(times)
    noise[times + step_size < 1.0] = integrator.cat_noise_level
    mult = (1 + noise + (noise * (n_categories - 1) * times)) / (1 - times)
    first_term = step_size * mult * pred_dist
    second_term = step_size * noise * pred_probs_curr
    step_probs = (first_term + second_term).clamp(max=1.0)
    step_probs.scatter_(-1, curr, 0.0)
    diags = (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
    step_probs.scatter_(-1, curr, diags)
    samples = torch.distributions.Categorical(step_probs).sample()
    return smolF.one_hot_encode_tensor(samples, n_categories)


if __name__ == "__main__":
    unittest.main()

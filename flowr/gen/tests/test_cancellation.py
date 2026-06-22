"""Tests for cooperative generation cancellation (flowr.gen.cancellation)."""

import unittest

from flowr.gen.cancellation import GenerationCancelled, raise_if_cancelled


class RaiseIfCancelledTests(unittest.TestCase):
    def test_raises_when_predicate_true(self):
        with self.assertRaises(GenerationCancelled):
            raise_if_cancelled(lambda: True)

    def test_noop_when_predicate_false(self):
        # Should simply return without raising.
        self.assertIsNone(raise_if_cancelled(lambda: False))

    def test_noop_when_predicate_none(self):
        # Default (no callback) must never raise, so existing callers are safe.
        self.assertIsNone(raise_if_cancelled(None))

    def test_message_includes_step_when_given(self):
        with self.assertRaises(GenerationCancelled) as ctx:
            raise_if_cancelled(lambda: True, step=7)
        self.assertIn("7", str(ctx.exception))

    def test_only_calls_predicate_once(self):
        calls = {"n": 0}

        def pred():
            calls["n"] += 1
            return False

        raise_if_cancelled(pred)
        self.assertEqual(calls["n"], 1)


if __name__ == "__main__":
    unittest.main()

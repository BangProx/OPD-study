from __future__ import annotations

import unittest

import torch

from opd_study.diagnostics.support import support_diagnostics
from opd_study.diagnostics.test_time_scaling import (
    gained_and_lost_solvability,
    scaling_metrics,
)
from opd_study.envs import CalculatorEnvironment


class DiagnosticsTest(unittest.TestCase):
    def test_identical_support_is_perfectly_aligned(self) -> None:
        logits = torch.tensor([[[3.0, 2.0, 1.0, 0.0]]])
        result = support_diagnostics(
            logits, logits.clone(), torch.tensor([[True]]), top_k=2
        )
        self.assertEqual(result.overlap_ratio, 1.0)
        self.assertAlmostEqual(result.overlap_token_advantage, 0.0, places=7)
        self.assertAlmostEqual(result.absolute_entropy_gap, 0.0, places=7)

    def test_avg_and_pass_at_k_are_not_interchangeable(self) -> None:
        outcomes = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=torch.bool)
        result = scaling_metrics(outcomes, k=4)
        self.assertEqual(result.avg_at_k, 0.25)
        self.assertEqual(result.pass_at_k, 1.0)
        changed = gained_and_lost_solvability(
            outcomes,
            torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.bool),
            k=4,
        )
        self.assertEqual(changed, {"gained": 0, "lost": 1, "retained": 1, "never_solved": 0})


class CalculatorEnvironmentTest(unittest.TestCase):
    def test_an_early_error_changes_later_state(self) -> None:
        environment = CalculatorEnvironment(2, (("+", 3), ("*", 4)))
        initial = environment.reset()
        self.assertEqual(initial.value, 2)
        after_error = environment.step(6)
        self.assertIn("expected 5", after_error.observation)
        final = environment.step(24)
        self.assertTrue(final.terminal)
        self.assertFalse(final.success)
        self.assertEqual(final.target, 20)


if __name__ == "__main__":
    unittest.main()

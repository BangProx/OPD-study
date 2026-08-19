from __future__ import annotations

import math
import unittest

import torch

from opd_study.math import (
    forward_kl_from_logits,
    generalized_jsd_from_logits,
    log_probs,
    masked_mean,
    reverse_kl_from_logits,
)


class DistillationMathTest(unittest.TestCase):
    def setUp(self) -> None:
        self.teacher_probabilities = torch.tensor([[0.8, 0.2]])
        self.student_probabilities = torch.tensor([[0.5, 0.5]])
        self.teacher_logits = self.teacher_probabilities.log()
        self.student_logits = self.student_probabilities.log()

    def test_forward_kl_matches_hand_calculation(self) -> None:
        expected = 0.8 * math.log(0.8 / 0.5) + 0.2 * math.log(0.2 / 0.5)
        actual = forward_kl_from_logits(self.teacher_logits, self.student_logits)
        self.assertAlmostEqual(actual.item(), expected, places=6)

    def test_reverse_kl_matches_hand_calculation(self) -> None:
        expected = 0.5 * math.log(0.5 / 0.8) + 0.5 * math.log(0.5 / 0.2)
        actual = reverse_kl_from_logits(self.teacher_logits, self.student_logits)
        self.assertAlmostEqual(actual.item(), expected, places=6)

    def test_gjsd_boundaries_have_named_kl_direction(self) -> None:
        forward = forward_kl_from_logits(self.teacher_logits, self.student_logits)
        reverse = reverse_kl_from_logits(self.teacher_logits, self.student_logits)
        self.assertTrue(
            torch.allclose(
                generalized_jsd_from_logits(
                    self.teacher_logits, self.student_logits, beta=0.0
                ),
                forward,
            )
        )
        self.assertTrue(
            torch.allclose(
                generalized_jsd_from_logits(
                    self.teacher_logits, self.student_logits, beta=1.0
                ),
                reverse,
            )
        )

    def test_gjsd_is_zero_for_identical_distributions(self) -> None:
        actual = generalized_jsd_from_logits(
            self.teacher_logits, self.teacher_logits, beta=0.37
        )
        self.assertLess(abs(actual.item()), 1e-7)

    def test_temperature_and_empty_masks_fail_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "temperature"):
            log_probs(self.teacher_logits, temperature=0.0)
        with self.assertRaisesRegex(ValueError, "empty"):
            masked_mean(torch.ones(2), torch.zeros(2, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()

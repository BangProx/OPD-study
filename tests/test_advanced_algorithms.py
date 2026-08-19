from __future__ import annotations

import unittest

import torch

from opd_study.algorithms import opd2_loss, sage_opd_loss, sod_loss, tcod_loss, vopd_loss
from opd_study.algorithms.sage_opd import sage_token_weights
from opd_study.algorithms.sod import step_divergence_weights
from opd_study.algorithms.tcod import curriculum_depth, temporal_curriculum_mask
from opd_study.data import CharacterTokenizer, collate_multiturn_text, generate_tiny_arithmetic
from opd_study.types import TeacherSignals


class AdvancedAlgorithmTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(5)
        tokenizer = CharacterTokenizer()
        rows = generate_tiny_arithmetic(
            seed=5, train_rows=3, validation_rows=1, test_rows=1
        ).train[:2]
        self.batch = collate_multiturn_text(
            [(row.prompt, tuple(row.response.splitlines())) for row in rows], tokenizer
        )
        vocabulary_size = tokenizer.vocab_size
        shape = (*self.batch.token_ids.shape, vocabulary_size)
        self.student_logits = torch.randn(shape, requires_grad=True)
        self.teacher_logits = torch.randn(shape)

    def test_tcod_curriculum_and_directions(self) -> None:
        self.assertEqual(
            [
                curriculum_depth(step, start_depth=1, pacing_steps=2, maximum_depth=3)
                for step in range(6)
            ],
            [1, 1, 2, 2, 3, 3],
        )
        early = temporal_curriculum_mask(self.batch, depth=1, direction="f2b")
        late = temporal_curriculum_mask(self.batch, depth=1, direction="b2f")
        self.assertFalse((early & late).any().item())
        output = tcod_loss(
            self.student_logits,
            self.batch,
            TeacherSignals(logits=self.teacher_logits),
            depth=1,
            direction="f2b",
        )
        output.loss.backward(retain_graph=True)
        self.assertTrue(torch.isfinite(output.loss).item())

    def test_sod_downweights_a_divergence_jump(self) -> None:
        divergences, weights = step_divergence_weights(
            self.student_logits.detach(),
            self.teacher_logits,
            self.batch.token_ids,
            self.batch.response_mask,
            self.batch.step_ids,
        )
        self.assertFalse(divergences.requires_grad)
        self.assertFalse(weights.requires_grad)
        output = sod_loss(
            self.student_logits,
            self.batch,
            TeacherSignals(logits=self.teacher_logits),
        )
        self.assertTrue(torch.isfinite(output.loss).item())
        self.assertEqual(output.metrics["sod/distillation_only"], 1.0)

    def test_sage_weights_normalize_and_skip(self) -> None:
        assert self.batch.turn_ids is not None
        turns = int(self.batch.turn_ids.max().item()) + 1
        intervention = torch.ones((self.batch.token_ids.shape[0], turns))
        intervention[:, 0] = 0.0
        weights, confidence = sage_token_weights(
            self.teacher_logits, self.batch, intervention
        )
        prediction_tokens = self.batch.response_mask[:, 1:].sum()
        self.assertAlmostEqual(weights.sum().item(), prediction_tokens.item(), places=4)
        self.assertTrue((confidence >= 0).all().item())
        output = sage_opd_loss(
            self.student_logits,
            self.batch,
            TeacherSignals(logits=self.teacher_logits, intervene=intervention),
        )
        self.assertTrue(torch.isfinite(output.loss).item())

    def test_vopd_is_zero_when_teacher_equals_student(self) -> None:
        teacher = self.student_logits.detach().clone()
        output = vopd_loss(
            self.student_logits, self.batch, TeacherSignals(logits=teacher)
        )
        self.assertAlmostEqual(output.loss.item(), 0.0, places=6)
        output.loss.backward(retain_graph=True)

    def test_opd2_gate_closes_when_teacher_equals_base(self) -> None:
        same = self.teacher_logits.clone()
        output = opd2_loss(
            self.student_logits,
            self.batch,
            TeacherSignals(logits=same),
            TeacherSignals(logits=same.clone()),
        )
        self.assertAlmostEqual(output.loss.item(), 0.0, places=6)
        self.assertEqual(output.metrics["opd2/gate_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()

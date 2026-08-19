from __future__ import annotations

import unittest

import torch

from opd_study.algorithms import (
    collect_multiturn_trajectories,
    collect_student_trajectories,
    generalized_kd_loss,
    score_teacher,
    supervised_fine_tuning_loss,
)
from opd_study.data import CharacterTokenizer, collate_examples, generate_tiny_arithmetic
from opd_study.models import TinyCausalLM, TinyTransformerConfig
from opd_study.utils import model_state_hash, seed_everything


def tiny_config(vocab_size: int) -> TinyTransformerConfig:
    return TinyTransformerConfig(
        vocab_size=vocab_size,
        max_sequence_length=128,
        number_of_layers=1,
        hidden_size=32,
        number_of_heads=4,
        feed_forward_size=64,
    )


class TinyModelTest(unittest.TestCase):
    def setUp(self) -> None:
        seed_everything(123)
        self.tokenizer = CharacterTokenizer()
        self.splits = generate_tiny_arithmetic(
            seed=3, train_rows=8, validation_rows=2, test_rows=2
        )
        self.batch = collate_examples(self.splits.train[:2], self.tokenizer)

    def test_future_token_does_not_change_past_logits(self) -> None:
        model = TinyCausalLM(tiny_config(self.tokenizer.vocab_size)).eval()
        first = self.batch.token_ids[:1, :20].clone()
        second = first.clone()
        second[0, -1] = (second[0, -1] + 1) % self.tokenizer.vocab_size
        with torch.no_grad():
            first_logits = model(first)
            second_logits = model(second)
        self.assertTrue(torch.allclose(first_logits[:, :-1], second_logits[:, :-1]))

    def test_teacher_is_frozen_and_student_updates(self) -> None:
        student = TinyCausalLM(tiny_config(self.tokenizer.vocab_size))
        teacher = TinyCausalLM(tiny_config(self.tokenizer.vocab_size))
        teacher_before = model_state_hash(teacher)
        student_before = model_state_hash(student)
        signals = score_teacher(teacher, self.batch)
        output = generalized_kd_loss(
            student(self.batch.token_ids, self.batch.attention_mask),
            self.batch,
            signals,
        )
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        optimizer.step()
        self.assertEqual(model_state_hash(teacher), teacher_before)
        self.assertNotEqual(model_state_hash(student), student_before)
        self.assertFalse(any(parameter.grad is not None for parameter in teacher.parameters()))

    def test_sft_counts_only_response_targets(self) -> None:
        student = TinyCausalLM(tiny_config(self.tokenizer.vocab_size))
        output = supervised_fine_tuning_loss(
            student(self.batch.token_ids, self.batch.attention_mask), self.batch
        )
        self.assertEqual(
            int(output.effective_mask.sum()), int(self.batch.response_mask.sum())
        )
        self.assertFalse(output.effective_mask[:, 0].any().item())

    def test_rollout_snapshots_are_detached_and_mode_is_restored(self) -> None:
        student = TinyCausalLM(tiny_config(self.tokenizer.vocab_size))
        student.train()
        trajectories = collect_student_trajectories(
            student,
            [row.prompt for row in self.splits.train[:2]],
            self.tokenizer,
            max_new_tokens=3,
            temperature=0.0,
        )
        self.assertTrue(student.training)
        self.assertIsNotNone(trajectories.student_logprobs)
        assert trajectories.student_logprobs is not None
        self.assertFalse(trajectories.student_logprobs.requires_grad)
        self.assertEqual(int(trajectories.response_mask.sum()), 6)

    def test_b2f_teacher_prefix_is_context_not_student_loss(self) -> None:
        student = TinyCausalLM(tiny_config(self.tokenizer.vocab_size))
        teacher = TinyCausalLM(tiny_config(self.tokenizer.vocab_size))
        trajectories = collect_multiturn_trajectories(
            student,
            [self.splits.train[0].prompt],
            self.tokenizer,
            number_of_turns=3,
            tokens_per_turn=1,
            temperature=0.0,
            teacher_prefix=teacher,
            teacher_prefix_turns=2,
        )
        assert trajectories.turn_ids is not None
        self.assertEqual(int(trajectories.response_mask.sum()), 1)
        self.assertEqual(
            trajectories.turn_ids[trajectories.response_mask].tolist(), [2]
        )
        self.assertTrue(trajectories.terminal.item())


if __name__ == "__main__":
    unittest.main()

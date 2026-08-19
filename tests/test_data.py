from __future__ import annotations

import unittest

from opd_study.data import CharacterTokenizer, collate_examples, generate_tiny_arithmetic


class TinyArithmeticTest(unittest.TestCase):
    def test_generation_is_deterministic_unique_and_split_safe(self) -> None:
        first = generate_tiny_arithmetic(
            seed=42, train_rows=60, validation_rows=15, test_rows=15
        )
        second = generate_tiny_arithmetic(
            seed=42, train_rows=60, validation_rows=15, test_rows=15
        )
        self.assertEqual(first, second)
        expressions = [row.expression for row in first.train + first.validation + first.test]
        self.assertEqual(len(expressions), len(set(expressions)))
        self.assertNotEqual(first.split_hash("train"), first.split_hash("test"))
        self.assertEqual({row.number_of_steps for row in first.train}, {1, 2, 3})

    def test_tokenizer_round_trip_and_response_boundary(self) -> None:
        splits = generate_tiny_arithmetic(
            seed=9, train_rows=3, validation_rows=1, test_rows=1
        )
        tokenizer = CharacterTokenizer()
        example = splits.train[0]
        self.assertEqual(
            tokenizer.decode(tokenizer.encode(example.prompt)), example.prompt
        )
        batch = collate_examples(splits.train, tokenizer)
        for row, prompt_length in enumerate(batch.prompt_lengths.tolist()):
            self.assertFalse(batch.response_mask[row, :prompt_length].any().item())
            self.assertTrue(batch.response_mask[row, prompt_length].item())
        self.assertTrue(
            (batch.response_mask.sum(dim=1) > 0).all().item(),
            "every demonstration needs at least one response target",
        )

    def test_default_dataset_fits_the_frozen_context(self) -> None:
        splits = generate_tiny_arithmetic()
        tokenizer = CharacterTokenizer()
        all_rows = splits.train + splits.validation + splits.test
        lengths = [
            len(tokenizer.encode(row.prompt, bos=True))
            + len(tokenizer.encode(row.response, eos=True))
            for row in all_rows
        ]
        self.assertEqual(len(set(row.expression for row in all_rows)), 5120)
        self.assertLessEqual(max(lengths), 128)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path

import torch

from opd_study.algorithms import available_algorithms
from opd_study.cli import main as cli_main
from opd_study.data import CharacterTokenizer, generate_tiny_arithmetic
from opd_study.demo import compare_expression, run_demo
from opd_study.evaluation import evaluate_model
from opd_study.models import TinyCausalLM, TinyTransformerConfig
from opd_study.reporting import load_checkpoint, save_checkpoint
from opd_study.training import (
    train_advanced_distillation,
    train_distillation,
    train_sft,
)
from opd_study.utils import model_state_hash, seed_everything


def _config(vocab_size: int) -> TinyTransformerConfig:
    return TinyTransformerConfig(
        vocab_size=vocab_size,
        max_sequence_length=128,
        number_of_layers=1,
        hidden_size=16,
        number_of_heads=2,
        feed_forward_size=32,
    )


class TrainingAndArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = CharacterTokenizer()
        self.splits = generate_tiny_arithmetic(
            seed=17, train_rows=8, validation_rows=2, test_rows=2
        )
        seed_everything(99)
        self.initial = TinyCausalLM(_config(self.tokenizer.vocab_size)).state_dict()
        self.teacher = TinyCausalLM(_config(self.tokenizer.vocab_size))
        self.teacher.requires_grad_(False)

    def _student(self) -> TinyCausalLM:
        model = TinyCausalLM(_config(self.tokenizer.vocab_size))
        model.load_state_dict(copy.deepcopy(self.initial))
        return model

    def test_sft_resume_matches_uninterrupted_training(self) -> None:
        full = self._student()
        train_sft(
            full,
            self.splits.train,
            self.tokenizer,
            steps=2,
            batch_size=2,
            tokens_per_step=2,
        )

        resumed = self._student()
        first = train_sft(
            resumed,
            self.splits.train,
            self.tokenizer,
            steps=1,
            batch_size=2,
            tokens_per_step=2,
        )
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "resume.pt"
            save_checkpoint(
                checkpoint,
                {
                    "model_state": resumed.state_dict(),
                    "optimizer_state": first.optimizer_state,
                    "torch_rng_state": torch.get_rng_state(),
                    "step": 1,
                },
            )
            payload = load_checkpoint(checkpoint)
        resumed.load_state_dict(payload["model_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        second = train_sft(
            resumed,
            self.splits.train,
            self.tokenizer,
            steps=1,
            batch_size=2,
            tokens_per_step=2,
            optimizer_state=payload["optimizer_state"],
            start_step=payload["step"],
        )
        self.assertEqual(second.history[0]["step"], 2.0)
        self.assertEqual(model_state_hash(resumed), model_state_hash(full))

    def test_every_required_algorithm_can_resume_and_evaluate(self) -> None:
        algorithms = ("sft", "opd", "tcod_f2b", "tcod_b2f", "sod", "sage_opd")
        for algorithm in algorithms:
            with self.subTest(algorithm=algorithm):
                student = self._student()
                generator = torch.Generator(device="cpu").manual_seed(123)
                if algorithm == "sft":
                    first = train_sft(
                        student,
                        self.splits.train,
                        self.tokenizer,
                        steps=1,
                        batch_size=1,
                        tokens_per_step=1,
                    )
                    second = train_sft(
                        student,
                        self.splits.train,
                        self.tokenizer,
                        steps=1,
                        batch_size=1,
                        tokens_per_step=1,
                        optimizer_state=first.optimizer_state,
                        start_step=1,
                    )
                elif algorithm == "opd":
                    first = train_distillation(
                        student,
                        self.teacher,
                        self.splits.train,
                        self.tokenizer,
                        algorithm=algorithm,
                        steps=1,
                        batch_size=1,
                        tokens_per_step=1,
                        generator=generator,
                    )
                    second = train_distillation(
                        student,
                        self.teacher,
                        self.splits.train,
                        self.tokenizer,
                        algorithm=algorithm,
                        steps=1,
                        batch_size=1,
                        tokens_per_step=1,
                        generator=generator,
                        optimizer_state=first.optimizer_state,
                        start_step=1,
                    )
                else:
                    first = train_advanced_distillation(
                        student,
                        self.teacher,
                        self.splits.train,
                        self.tokenizer,
                        algorithm=algorithm,
                        steps=1,
                        batch_size=1,
                        number_of_turns=1,
                        tokens_per_turn=1,
                        generator=generator,
                        curriculum_total_steps=2,
                    )
                    second = train_advanced_distillation(
                        student,
                        self.teacher,
                        self.splits.train,
                        self.tokenizer,
                        algorithm=algorithm,
                        steps=1,
                        batch_size=1,
                        number_of_turns=1,
                        tokens_per_turn=1,
                        generator=generator,
                        optimizer_state=first.optimizer_state,
                        start_step=1,
                        curriculum_total_steps=2,
                    )
                self.assertTrue(torch.isfinite(torch.tensor(second.history[0]["loss"])))
                evaluation = evaluate_model(
                    student,
                    self.teacher,
                    self.splits.test[:1],
                    self.tokenizer,
                    max_new_tokens=1,
                )
                self.assertEqual(evaluation.evaluated_rows, 1)

    def test_every_registered_algorithm_runs_through_train_and_eval_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for algorithm in available_algorithms():
                with self.subTest(algorithm=algorithm):
                    run_dir = root / algorithm
                    captured = io.StringIO()
                    with contextlib.redirect_stdout(captured):
                        train_status = cli_main(
                            [
                                "train",
                                "--algorithm",
                                algorithm,
                                "--smoke",
                                "--device",
                                "cpu",
                                "--output",
                                str(run_dir),
                            ]
                        )
                        eval_status = cli_main(
                            ["eval", "--run", str(run_dir), "--rows", "1"]
                        )
                    self.assertEqual(train_status, 0, captured.getvalue())
                    self.assertEqual(eval_status, 0, captured.getvalue())
                    self.assertTrue((run_dir / "evaluation.json").is_file())
                    self.assertTrue((run_dir / "index.html").is_file())

    def test_demo_writes_all_learner_facing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = run_demo(
                root,
                smoke=True,
                methods=("sft", "opd"),
                requested_device="cpu",
            )
            expected = (
                "summary.json",
                "metrics.jsonl",
                "index.html",
                "loss_curves.png",
                "distribution_diagnostics.png",
                "checkpoints/sft.pt",
                "checkpoints/opd.pt",
            )
            for relative in expected:
                self.assertTrue((root / relative).is_file(), relative)
            self.assertEqual(
                summary["runs"]["sft"]["response_tokens"],
                summary["runs"]["opd"]["response_tokens"],
            )
            self.assertEqual(summary["fairness"]["same_initial_student"], True)
            for method in ("no_train", "sft", "opd"):
                evaluation = summary["runs"][method]["evaluation"]
                self.assertGreaterEqual(evaluation["student_entropy"], 0.0)
                self.assertGreaterEqual(evaluation["forward_kl_teacher_student"], 0.0)
                self.assertGreaterEqual(evaluation["reverse_kl_student_teacher"], 0.0)
            report = (root / "index.html").read_text(encoding="utf-8")
            self.assertIn("Side-by-side generated responses", report)
            comparison = compare_expression(root, "2 + 3", max_new_tokens=2)
            self.assertEqual(comparison["expression"], "2 + 3")
            self.assertEqual(
                set(comparison["responses"]), {"teacher", "no_train", "sft", "opd"}
            )

    def test_playground_rejects_unsafe_or_missing_input(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            self.assertRaisesRegex(ValueError, "expression must contain"),
        ):
            compare_expression(temporary, "__import__('os')")


if __name__ == "__main__":
    unittest.main()

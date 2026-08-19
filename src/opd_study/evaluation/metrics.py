"""Evaluation helpers for TinyArithmetic-OPD."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from opd_study.algorithms import score_teacher, supervised_fine_tuning_loss
from opd_study.data import ArithmeticExample, CharacterTokenizer, collate_examples
from opd_study.math import (
    entropy_from_logits,
    forward_kl_from_logits,
    masked_mean,
    reverse_kl_from_logits,
)
from opd_study.models import TinyCausalLM

_ANSWER_PATTERN = re.compile(r"Answer:\s*(-?\d+)")


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    exact_answer_accuracy: float
    teacher_argmax_agreement: float
    student_entropy: float
    teacher_entropy: float
    forward_kl_teacher_student: float
    reverse_kl_student_teacher: float
    evaluated_rows: int
    samples: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "loss": self.loss,
            "exact_answer_accuracy": self.exact_answer_accuracy,
            "teacher_argmax_agreement": self.teacher_argmax_agreement,
            "student_entropy": self.student_entropy,
            "teacher_entropy": self.teacher_entropy,
            "forward_kl_teacher_student": self.forward_kl_teacher_student,
            "reverse_kl_student_teacher": self.reverse_kl_student_teacher,
            "evaluated_rows": self.evaluated_rows,
            "samples": list(self.samples),
        }


def extract_answer(text: str) -> int | None:
    matches = _ANSWER_PATTERN.findall(text)
    return int(matches[-1]) if matches else None


@torch.no_grad()
def evaluate_model(
    model: TinyCausalLM,
    teacher: TinyCausalLM,
    examples: Sequence[ArithmeticExample],
    tokenizer: CharacterTokenizer,
    *,
    max_new_tokens: int = 64,
) -> EvaluationResult:
    """Measure gold NLL, teacher top-1 agreement and greedy exact-answer accuracy."""

    if not examples:
        raise ValueError("evaluation examples must not be empty")
    device = next(model.parameters()).device
    if next(teacher.parameters()).device != device:
        raise ValueError("teacher and evaluated model must share a device")
    batch = collate_examples(examples, tokenizer, device=device)
    was_training = model.training
    model.eval()
    logits = model(batch.token_ids, batch.attention_mask)
    loss = supervised_fine_tuning_loss(logits, batch).loss
    teacher_logits = score_teacher(teacher, batch).logits
    assert teacher_logits is not None
    target_mask = batch.response_mask[:, 1:]
    agreement = (
        logits[:, :-1].argmax(dim=-1) == teacher_logits[:, :-1].argmax(dim=-1)
    )
    teacher_agreement = float(agreement[target_mask].float().mean().cpu())
    shifted_student = logits[:, :-1]
    shifted_teacher = teacher_logits[:, :-1]
    student_entropy = float(
        masked_mean(entropy_from_logits(shifted_student), target_mask).cpu()
    )
    teacher_entropy = float(
        masked_mean(entropy_from_logits(shifted_teacher), target_mask).cpu()
    )
    forward_kl = float(
        masked_mean(
            forward_kl_from_logits(shifted_teacher, shifted_student), target_mask
        ).cpu()
    )
    reverse_kl = float(
        masked_mean(
            reverse_kl_from_logits(shifted_teacher, shifted_student), target_mask
        ).cpu()
    )

    correct = 0
    samples: list[dict[str, object]] = []
    for example in examples:
        prompt_ids = torch.tensor(
            tokenizer.encode(example.prompt, bos=True), dtype=torch.long, device=device
        ).unsqueeze(0)
        generated = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            temperature=0.0,
        )
        response_ids = generated[0, prompt_ids.shape[1] :].tolist()
        response = tokenizer.decode(response_ids)
        prediction = extract_answer(response)
        is_correct = prediction == example.answer
        correct += int(is_correct)
        samples.append(
            {
                "example_id": example.example_id,
                "prompt": example.prompt,
                "response": response,
                "predicted_answer": prediction,
                "expected_answer": example.answer,
                "correct": is_correct,
            }
        )
    model.train(was_training)
    return EvaluationResult(
        loss=float(loss.cpu()),
        exact_answer_accuracy=correct / len(examples),
        teacher_argmax_agreement=teacher_agreement,
        student_entropy=student_entropy,
        teacher_entropy=teacher_entropy,
        forward_kl_teacher_student=forward_kl,
        reverse_kl_student_teacher=reverse_kl,
        evaluated_rows=len(examples),
        samples=tuple(samples),
    )

"""Opt-in, single-device Qwen/GSM8K OPD smoke runner for the research backend."""

from __future__ import annotations

import re
import time
import tracemalloc
from pathlib import Path
from typing import Any

import torch

from opd_study.algorithms import on_policy_distillation_loss
from opd_study.config import ExperimentConfig
from opd_study.data import fetch_gsm8k, load_gsm8k_rows
from opd_study.reporting import (
    environment_record,
    file_sha256,
    git_commit,
    save_checkpoint,
    write_json,
    write_jsonl,
)
from opd_study.research.hf_backend import load_model_pair
from opd_study.research.preflight import research_preflight
from opd_study.tensorboard import SummaryWriter
from opd_study.training.optim import clip_gradient_norm
from opd_study.types import TeacherSignals, TrajectoryBatch
from opd_study.utils import seed_everything

_FINAL_ANSWER = re.compile(r"####\s*([-+]?\d[\d,]*)")
_BOXED_ANSWER = re.compile(r"\\boxed\{\s*([-+]?\d[\d,]*)\s*\}")
_NUMBER = re.compile(r"[-+]?\d[\d,]*")


def extract_gsm8k_answer(text: str) -> int | None:
    """Read a GSM8K marker, boxed answer, or final integer—in that order."""

    for pattern in (_FINAL_ANSWER, _BOXED_ANSWER):
        match = pattern.search(text)
        if match is not None:
            return int(match.group(1).replace(",", ""))
    matches = _NUMBER.findall(text)
    return int(matches[-1].replace(",", "")) if matches else None


def _prompt(tokenizer: Any, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    return str(
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )


def _trajectory(
    student: Any,
    tokenizer: Any,
    prompt: str,
    maximum_tokens: int,
    temperature: float,
) -> TrajectoryBatch:
    device = next(student.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    prompt_ids = encoded["input_ids"].to(device)
    prompt_length = prompt_ids.shape[1]
    was_training = student.training
    student.eval()
    try:
        with torch.no_grad():
            generated = student.generate(
                prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                max_new_tokens=maximum_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            snapshot_logits = student(
                generated,
                attention_mask=torch.ones_like(generated),
            ).logits
            target_ids = generated[:, 1:]
            selected = torch.log_softmax(snapshot_logits[:, :-1].float(), dim=-1)
            selected = selected.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    finally:
        student.train(was_training)
    attention = torch.ones_like(generated, dtype=torch.bool)
    response = torch.zeros_like(generated, dtype=torch.bool)
    response[:, prompt_length:] = True
    old_logprobs = torch.zeros_like(generated, dtype=torch.float32)
    old_logprobs[:, 1:] = selected.detach()
    return TrajectoryBatch(
        token_ids=generated,
        attention_mask=attention,
        response_mask=response,
        prompt_lengths=torch.tensor([prompt_length], device=device),
        student_logprobs=old_logprobs,
    )


@torch.no_grad()
def _evaluate(
    student: Any,
    tokenizer: Any,
    rows: list[dict[str, str]],
    *,
    maximum_tokens: int,
) -> dict[str, Any]:
    device = next(student.parameters()).device
    samples: list[dict[str, Any]] = []
    correct = 0
    for row in rows:
        prompt = _prompt(tokenizer, row["question"])
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        prompt_ids = encoded["input_ids"].to(device)
        output_ids = student.generate(
            prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            max_new_tokens=maximum_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(
            output_ids[0, prompt_ids.shape[1] :],
            skip_special_tokens=True,
        )
        expected = extract_gsm8k_answer(row["answer"])
        predicted = extract_gsm8k_answer(response)
        is_correct = predicted == expected
        correct += int(is_correct)
        samples.append(
            {
                "question": row["question"],
                "response": response,
                "predicted_answer": predicted,
                "expected_answer": expected,
                "correct": is_correct,
            }
        )
    return {
        "exact_answer_accuracy": correct / len(rows),
        "evaluated_rows": len(rows),
        "samples": samples,
    }


def run_research_smoke(
    config: ExperimentConfig,
    cache_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run opt-in sampled-OPD and save a PEFT adapter plus an honest experiment card."""

    if config.backend != "research":
        raise ValueError("research training requires backend=research")
    if config.algorithm.name != "opd":
        raise NotImplementedError(
            "the real-model single-device runner currently supports algorithm=opd; "
            "other research recipes remain explicitly unverified"
        )
    if config.data.id != "openai/gsm8k":
        raise NotImplementedError("the executable research runner currently supports GSM8K")
    if config.training.batch_size != 1:
        raise ValueError("research sampled-OPD currently requires batch_size=1")
    if config.algorithm.temperature != 1.0:
        raise ValueError(
            "research sampled reverse-KL currently requires temperature=1.0 so the "
            "rollout and optimized policy distributions match"
        )
    report = research_preflight(config)
    if not report.ready:
        raise RuntimeError("research preflight failed: " + "; ".join(report.blockers))

    output = Path(output_dir or config.output.root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = fetch_gsm8k(
        cache_dir,
        accept_dataset_license=config.data.accept_dataset_license,
    )
    dataset = load_gsm8k_rows(paths, seed=config.data.seed)
    train_rows = dataset["train"][: config.data.train_rows]
    validation_rows = dataset["validation"][: config.data.validation_rows]
    steps = 1 if smoke else config.training.steps
    rollout_tokens = (
        min(config.training.tokens_per_step, 16)
        if smoke
        else config.training.tokens_per_step
    )
    evaluation_rows = validation_rows[: 1 if smoke else config.evaluation.rows]
    evaluation_tokens = (
        min(config.evaluation.max_new_tokens, 32)
        if smoke
        else config.evaluation.max_new_tokens
    )

    seed_everything(config.training.seed)
    tracemalloc.start()
    started = time.perf_counter()
    tokenizer, student, teacher = load_model_pair(config, cache_dir)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in student.parameters() if parameter.requires_grad),
        lr=config.training.learning_rate,
    )
    history: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        row = train_rows[(step - 1) % len(train_rows)]
        trajectory = _trajectory(
            student,
            tokenizer,
            _prompt(tokenizer, row["question"]),
            rollout_tokens,
            config.algorithm.temperature,
        )
        teacher_device = next(teacher.parameters()).device
        if trajectory.token_ids.device != teacher_device:
            raise RuntimeError("student and teacher must be on one device in laptop mode")
        teacher.eval()
        with torch.no_grad():
            teacher_logits = teacher(
                trajectory.token_ids,
                attention_mask=trajectory.attention_mask,
            ).logits.detach()
        student.train()
        student_logits = student(
            trajectory.token_ids,
            attention_mask=trajectory.attention_mask,
        ).logits
        loss_output = on_policy_distillation_loss(
            student_logits,
            trajectory,
            TeacherSignals(logits=teacher_logits),
            estimator="sampled_reverse_kl",
        )
        optimizer.zero_grad(set_to_none=True)
        loss_output.loss.backward()
        gradient_norm = clip_gradient_norm(student.parameters(), 1.0)
        optimizer.step()
        history.append(
            {
                "step": step,
                "loss": float(loss_output.loss.detach().cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "response_tokens": int(loss_output.effective_mask.sum().item()),
                "wall_seconds": time.perf_counter() - started,
            }
        )

    student.eval()
    evaluation = _evaluate(
        student,
        tokenizer,
        evaluation_rows,
        maximum_tokens=evaluation_tokens,
    )
    tensorboard_path: Path | None = None
    if config.output.tensorboard:
        tensorboard_path = output / "tensorboard"
        writer = SummaryWriter(str(tensorboard_path))
        try:
            for row in history:
                writer.add_scalar("loss/sampled_reverse_kl_opd", row["loss"], row["step"])
                writer.add_scalar(
                    "optimization/gradient_norm", row["gradient_norm"], row["step"]
                )
            writer.add_scalar(
                "evaluation/exact_answer_accuracy",
                evaluation["exact_answer_accuracy"],
                steps,
            )
        finally:
            writer.close()
    adapter = output / "adapter"
    student.save_pretrained(adapter, safe_serialization=True)
    tokenizer.save_pretrained(adapter)
    checkpoint = output / "checkpoints" / "optimizer.pt"
    save_checkpoint(
        checkpoint,
        {
            "schema_version": 1,
            "optimizer_state": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "completed_steps": steps,
            "adapter": str(adapter),
            "config_hash": config.config_hash,
        },
    )
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    repository = Path(__file__).resolve().parents[3]
    peak_cuda = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    )
    card: dict[str, Any] = {
        "schema_version": 1,
        "status": "EXECUTED",
        "profile": config.profile,
        "algorithm": "sampled_reverse_kl_opd",
        "smoke": smoke,
        "git_commit": git_commit(repository),
        "config_hash": config.config_hash,
        "environment": environment_record(),
        "models": {
            "student": config.model.student,
            "student_revision": config.model.student_revision,
            "teacher": config.model.teacher,
            "teacher_revision": config.model.teacher_revision,
            "finetuning": config.model.finetuning,
        },
        "dataset": {
            "id": config.data.id,
            "revision": config.data.revision,
            "license": config.data.license,
            "split_rows": {name: len(rows) for name, rows in dataset.items()},
            "train_rows_available_to_run": len(train_rows),
            "validation_rows_available_to_run": len(validation_rows),
            "test_used": False,
            "downloaded_shards": {
                name: {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                for name, path in paths.items()
            },
        },
        "training": {
            "seed": config.training.seed,
            "steps": steps,
            "history": history,
            "wall_seconds": time.perf_counter() - started,
            "python_tracemalloc_peak_bytes": peak_bytes,
            "cuda_peak_allocated_bytes": peak_cuda,
        },
        "evaluation": evaluation,
        "artifacts": {
            "adapter": str(adapter),
            "optimizer_checkpoint": str(checkpoint),
            "tensorboard": None if tensorboard_path is None else str(tensorboard_path),
        },
        "limitations": [
            "This is a single-device plumbing/sanity run, not a paper-scale reproduction.",
            "The sampled reverse-KL estimator is high variance and a smoke run does "
            "not establish convergence.",
            "Official GSM8K test is untouched; validation is derived only from official train.",
        ],
    }
    write_jsonl(output / "metrics.jsonl", history)
    write_json(output / "experiment-card.json", card)
    return card

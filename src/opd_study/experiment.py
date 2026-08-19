"""Single-algorithm toy train/eval commands used by CLI and notebook checks."""

from __future__ import annotations

import copy
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import torch

from opd_study.algorithms import available_algorithms
from opd_study.data import CharacterTokenizer, generate_tiny_arithmetic
from opd_study.device import resolve_device
from opd_study.evaluation import evaluate_model
from opd_study.models import TinyCausalLM, TinyTransformerConfig
from opd_study.reporting import (
    create_static_report,
    environment_record,
    git_commit,
    load_checkpoint,
    save_checkpoint,
    write_json,
    write_jsonl,
)
from opd_study.tensorboard import SummaryWriter
from opd_study.training import (
    train_advanced_distillation,
    train_distillation,
    train_sft,
)
from opd_study.utils import model_state_hash, seed_everything, stable_json_hash


def run_toy_algorithm(
    algorithm: str,
    output_dir: str | Path,
    *,
    smoke: bool = False,
    requested_device: str = "auto",
    allow_device_fallback: bool = False,
) -> dict[str, Any]:
    """Train any registered method and write a checkpoint plus honest run card."""

    if algorithm not in available_algorithms():
        raise ValueError(f"unknown algorithm: {algorithm}")
    tracemalloc.start()
    overall_started = time.perf_counter()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device_report = resolve_device(
        requested_device, allow_fallback=allow_device_fallback
    )
    device = torch.device(device_report.selected)
    seed = 42
    generator = seed_everything(seed)
    splits = generate_tiny_arithmetic()
    tokenizer = CharacterTokenizer()

    teacher_base = TinyCausalLM(
        TinyTransformerConfig.teacher(vocab_size=tokenizer.vocab_size)
    ).to(device)
    teacher = TinyCausalLM(teacher_base.config).to(device)
    teacher.load_state_dict(copy.deepcopy(teacher_base.state_dict()))
    teacher_result = train_sft(
        teacher,
        splits.train,
        tokenizer,
        steps=1 if smoke else 20,
        batch_size=2 if smoke else 8,
        tokens_per_step=4 if smoke else 64,
    )
    teacher.requires_grad_(False)
    teacher_base.requires_grad_(False)

    seed_everything(seed + 1)
    student = TinyCausalLM(
        TinyTransformerConfig.student(vocab_size=tokenizer.vocab_size)
    ).to(device)
    initial_hash = model_state_hash(student)
    steps = 1 if smoke else 10
    batch_size = 2 if smoke else 8
    tokens_per_step = 4 if smoke else 64
    if algorithm == "sft":
        result = train_sft(
            student,
            splits.train,
            tokenizer,
            steps=steps,
            batch_size=batch_size,
            tokens_per_step=tokens_per_step,
        )
    elif algorithm in {"off_policy_kd", "gkd", "opd"}:
        result = train_distillation(
            student,
            teacher,
            splits.train,
            tokenizer,
            algorithm=algorithm,
            steps=steps,
            batch_size=batch_size,
            tokens_per_step=tokens_per_step,
            generator=generator,
        )
    else:
        result = train_advanced_distillation(
            student,
            teacher,
            splits.train,
            tokenizer,
            algorithm=algorithm,
            steps=steps,
            batch_size=batch_size,
            number_of_turns=2 if smoke else 3,
            tokens_per_turn=2 if smoke else 4,
            teacher_base=teacher_base,
            generator=generator,
        )

    evaluation = evaluate_model(
        student,
        teacher,
        splits.test[: 2 if smoke else 8],
        tokenizer,
        max_new_tokens=4 if smoke else 48,
    )
    checkpoints = output / "checkpoints"
    student_checkpoint = checkpoints / f"{algorithm}.pt"
    teacher_checkpoint = checkpoints / "teacher.pt"
    base_checkpoint = checkpoints / "teacher_base.pt"
    common = {
        "schema_version": 1,
        "seed": seed,
        "split_hashes": {
            "train": splits.split_hash("train"),
            "test": splits.split_hash("test"),
        },
    }
    run_configuration = {
        "algorithm": algorithm,
        "smoke": smoke,
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "tokens_per_step": tokens_per_step,
        "device": device_report.to_dict(),
    }
    config_hash = stable_json_hash(run_configuration)
    common["config_hash"] = config_hash
    save_checkpoint(
        student_checkpoint,
        {
            **common,
            "role": "student",
            "algorithm": algorithm,
            "model_config": student.config.to_dict(),
            "model_state": student.state_dict(),
            "initial_student_hash": initial_hash,
            "optimizer_steps": result.optimizer_steps,
            "optimizer_state": result.optimizer_state,
            "response_tokens": result.response_tokens,
            "torch_rng_state": torch.get_rng_state(),
            "sampling_generator_state": generator.get_state(),
        },
    )
    for path, role, model in (
        (teacher_checkpoint, "teacher", teacher),
        (base_checkpoint, "teacher_base", teacher_base),
    ):
        save_checkpoint(
            path,
            {
                **common,
                "role": role,
                "model_config": model.config.to_dict(),
                "model_state": model.state_dict(),
                "optimizer_state": (
                    teacher_result.optimizer_state if role == "teacher" else None
                ),
                "torch_rng_state": torch.get_rng_state(),
            },
        )

    history = [dict(row) for row in result.history]
    writer = SummaryWriter(str(output / "tensorboard"))
    try:
        for row in history:
            writer.add_scalar(f"loss/{algorithm}", row["loss"], int(row["step"]))
        writer.add_scalar(
            f"evaluation/exact_answer_accuracy/{algorithm}",
            evaluation.exact_answer_accuracy,
            steps,
        )
    finally:
        writer.close()
    _, python_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    repository = Path(__file__).resolve().parents[2]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "educational_run": True,
        "smoke": smoke,
        "algorithm": algorithm,
        "seed": seed,
        "git_commit": git_commit(repository),
        "command": [sys.executable, *sys.argv],
        "config": run_configuration,
        "config_hash": config_hash,
        "environment": environment_record(),
        "device": device_report.to_dict(),
        "initial_student_hash": initial_hash,
        "teacher_bootstrap": {
            "optimizer_steps": teacher_result.optimizer_steps,
            "response_tokens": teacher_result.response_tokens,
            "wall_seconds": teacher_result.wall_seconds,
        },
        "dataset": {
            "id": "TinyArithmetic-OPD",
            "revision": splits.revision,
            "license": "generated/Apache-2.0",
            "train_rows": len(splits.train),
            "validation_rows": len(splits.validation),
            "test_rows": len(splits.test),
            "split_hashes": {
                split_name: splits.split_hash(split_name)
                for split_name in ("train", "validation", "test")
            },
        },
        "runs": {
            algorithm: {
                "history": history,
                "response_tokens": result.response_tokens,
                "optimizer_steps": result.optimizer_steps,
                "wall_seconds": result.wall_seconds,
                "checkpoint": str(student_checkpoint),
                "evaluation": evaluation.to_dict(),
            }
        },
        "wall_seconds": time.perf_counter() - overall_started,
        "python_tracemalloc_peak_bytes": python_peak_bytes,
        "limitations": [
            "SAGE uses a token-agreement proxy judge in the mini backend; research "
            "mode requires environment feedback and a semantic teacher query."
            if algorithm == "sage_opd"
            else "This educational toy run does not reproduce paper-scale benchmark results."
        ],
    }
    write_jsonl(output / "metrics.jsonl", history)
    html_path, plot_path, diagnostic_path = create_static_report(output, summary)
    summary["artifacts"] = {
        "student_checkpoint": str(student_checkpoint),
        "teacher_checkpoint": str(teacher_checkpoint),
        "teacher_base_checkpoint": str(base_checkpoint),
        "html": str(html_path),
        "plot": str(plot_path),
        "distribution_diagnostics": str(diagnostic_path),
        "tensorboard": str(output / "tensorboard"),
    }
    write_json(output / "summary.json", summary)
    write_json(output / "experiment-card.json", summary)
    return summary


def evaluate_saved_run(run_dir: str | Path, *, rows: int = 8) -> dict[str, object]:
    """Reconstruct student/teacher from local checkpoints and rerun held-out eval."""

    directory = Path(run_dir).resolve()
    summary = load_checkpoint(directory / "checkpoints" / "teacher.pt")
    teacher = TinyCausalLM(TinyTransformerConfig.from_dict(summary["model_config"]))
    teacher.load_state_dict(summary["model_state"])
    student_paths = [
        path for path in (directory / "checkpoints").glob("*.pt")
        if path.name not in {"teacher.pt", "teacher_base.pt"}
    ]
    if len(student_paths) != 1:
        raise ValueError("run directory must contain exactly one student checkpoint")
    student_payload = load_checkpoint(student_paths[0])
    student = TinyCausalLM(
        TinyTransformerConfig.from_dict(student_payload["model_config"])
    )
    student.load_state_dict(student_payload["model_state"])
    splits = generate_tiny_arithmetic()
    result = evaluate_model(
        student,
        teacher,
        splits.test[:rows],
        CharacterTokenizer(),
        max_new_tokens=48,
    ).to_dict()
    write_json(directory / "evaluation.json", result)
    return result

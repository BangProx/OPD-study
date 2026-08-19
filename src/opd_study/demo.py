"""Offline mini playground comparing identical-initialization SFT, KD and OPD."""

from __future__ import annotations

import argparse
import copy
import re
import sys
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

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
from opd_study.training import TrainResult, train_distillation, train_sft
from opd_study.utils import model_state_hash, seed_everything

_SAFE_EXPRESSION = re.compile(r"[0-9+\-*/() ]{1,48}")


def _train_method(
    name: str,
    student: TinyCausalLM,
    teacher: TinyCausalLM,
    examples: tuple[Any, ...],
    tokenizer: CharacterTokenizer,
    *,
    steps: int,
    batch_size: int,
    tokens_per_step: int,
    learning_rate: float,
    generator: torch.Generator,
) -> TrainResult | None:
    if name == "no_train":
        return None
    if name == "sft":
        return train_sft(
            student,
            examples,
            tokenizer,
            steps=steps,
            batch_size=batch_size,
            tokens_per_step=tokens_per_step,
            learning_rate=learning_rate,
        )
    return train_distillation(
        student,
        teacher,
        examples,
        tokenizer,
        algorithm=name,
        steps=steps,
        batch_size=batch_size,
        tokens_per_step=tokens_per_step,
        learning_rate=learning_rate,
        beta=0.5,
        lambda_on_policy=0.5,
        rollout_temperature=1.0,
        generator=generator,
    )


def run_demo(
    output_dir: str | Path = "artifacts/demo",
    *,
    smoke: bool = False,
    methods: tuple[str, ...] = ("no_train", "sft", "off_policy_kd", "opd"),
    requested_device: str = "auto",
    allow_device_fallback: bool = False,
) -> dict[str, Any]:
    """Run a fair toy comparison and return the exact JSON summary written to disk."""

    allowed = {"no_train", "sft", "off_policy_kd", "gkd", "opd"}
    unknown = set(methods) - allowed
    if unknown:
        raise ValueError(f"unknown demo methods: {', '.join(sorted(unknown))}")
    if "no_train" not in methods:
        methods = ("no_train", *methods)

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tracemalloc.start()
    started = time.perf_counter()
    seed = 42
    seed_everything(seed)
    device_report = resolve_device(
        requested_device, allow_fallback=allow_device_fallback
    )
    device = torch.device(device_report.selected)
    splits = generate_tiny_arithmetic()
    tokenizer = CharacterTokenizer()

    teacher = TinyCausalLM(
        TinyTransformerConfig.teacher(vocab_size=tokenizer.vocab_size)
    ).to(device)
    teacher_steps = 1 if smoke else 30
    method_steps = 1 if smoke else 12
    batch_size = 2 if smoke else 8
    tokens_per_step = 4 if smoke else 64
    evaluation_rows = 2 if smoke else 8
    evaluation_tokens = 4 if smoke else 48
    teacher_result = train_sft(
        teacher,
        splits.train,
        tokenizer,
        steps=teacher_steps,
        batch_size=batch_size,
        tokens_per_step=tokens_per_step,
        learning_rate=3e-4,
    )
    teacher.requires_grad_(False)
    teacher_hash = model_state_hash(teacher)
    save_checkpoint(
        output / "checkpoints" / "teacher.pt",
        {
            "schema_version": 1,
            "role": "teacher",
            "model_config": teacher.config.to_dict(),
            "model_state": teacher.state_dict(),
            "model_hash": teacher_hash,
            "train_result": {
                "history": list(teacher_result.history),
                "optimizer_steps": teacher_result.optimizer_steps,
                "response_tokens": teacher_result.response_tokens,
                "wall_seconds": teacher_result.wall_seconds,
            },
            "optimizer_state": teacher_result.optimizer_state,
            "seed": seed,
            "torch_rng_state": torch.get_rng_state(),
        },
    )

    seed_everything(seed + 1)
    initial_student = TinyCausalLM(
        TinyTransformerConfig.student(vocab_size=tokenizer.vocab_size)
    ).to(device)
    initial_state = copy.deepcopy(initial_student.state_dict())
    initial_hash = model_state_hash(initial_student)
    runs: dict[str, Any] = {}
    all_metric_rows: list[dict[str, Any]] = []
    tensorboard = SummaryWriter(log_dir=str(output / "tensorboard"))
    try:
        for name in methods:
            # Keep stochastic comparisons independent of method order.
            method_generator = torch.Generator(device="cpu").manual_seed(seed + 1_000)
            student = TinyCausalLM(initial_student.config).to(device)
            student.load_state_dict(initial_state)
            if model_state_hash(student) != initial_hash:
                raise RuntimeError("fairness invariant failed: student initialization changed")
            result = _train_method(
                name,
                student,
                teacher,
                splits.train,
                tokenizer,
                steps=method_steps,
                batch_size=batch_size,
                tokens_per_step=tokens_per_step,
                learning_rate=3e-4,
                generator=method_generator,
            )
            history = [] if result is None else [dict(row) for row in result.history]
            response_tokens = 0 if result is None else result.response_tokens
            evaluation = evaluate_model(
                student,
                teacher,
                splits.test[:evaluation_rows],
                tokenizer,
                max_new_tokens=evaluation_tokens,
            )
            for row in history:
                metric_row = {**row, "method": name}
                all_metric_rows.append(metric_row)
                tensorboard.add_scalar(f"loss/{name}", row["loss"], int(row["step"]))
            tensorboard.add_scalar(
                f"evaluation/exact_answer_accuracy/{name}",
                evaluation.exact_answer_accuracy,
                method_steps,
            )
            tensorboard.add_scalar(
                f"evaluation/teacher_agreement/{name}",
                evaluation.teacher_argmax_agreement,
                method_steps,
            )
            for metric_name in (
                "student_entropy",
                "teacher_entropy",
                "forward_kl_teacher_student",
                "reverse_kl_student_teacher",
            ):
                tensorboard.add_scalar(
                    f"evaluation/{metric_name}/{name}",
                    float(getattr(evaluation, metric_name)),
                    method_steps,
                )
            checkpoint_path = output / "checkpoints" / f"{name}.pt"
            save_checkpoint(
                checkpoint_path,
                {
                    "schema_version": 1,
                    "algorithm": name,
                    "model_config": student.config.to_dict(),
                    "model_state": student.state_dict(),
                    "initial_student_hash": initial_hash,
                    "response_tokens": response_tokens,
                    "optimizer_steps": 0 if result is None else result.optimizer_steps,
                    "optimizer_state": None if result is None else result.optimizer_state,
                    "seed": seed,
                    "split_hashes": {
                        "train": splits.split_hash("train"),
                        "test": splits.split_hash("test"),
                    },
                    "torch_rng_state": torch.get_rng_state(),
                    "sampling_generator_state": method_generator.get_state(),
                },
            )
            runs[name] = {
                "history": history,
                "response_tokens": response_tokens,
                "optimizer_steps": 0 if result is None else result.optimizer_steps,
                "wall_seconds": 0.0 if result is None else result.wall_seconds,
                "final_model_hash": model_state_hash(student),
                "checkpoint": str(checkpoint_path),
                "evaluation": evaluation.to_dict(),
            }
    finally:
        tensorboard.close()

    _, python_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    repository = Path(__file__).resolve().parents[2]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "educational_run": True,
        "claim_scope": "offline TinyArithmetic smoke" if smoke else "offline TinyArithmetic demo",
        "seed": seed,
        "git_commit": git_commit(repository),
        "command": [sys.executable, *sys.argv],
        "environment": environment_record(),
        "device": device_report.to_dict(),
        "dataset": {
            "id": "TinyArithmetic-OPD",
            "revision": splits.revision,
            "license": "generated/Apache-2.0",
            "train_rows": len(splits.train),
            "validation_rows": len(splits.validation),
            "test_rows": len(splits.test),
            "split_hashes": {
                split: splits.split_hash(split)
                for split in ("train", "validation", "test")
            },
        },
        "model": {
            "teacher_parameters": teacher.number_of_parameters(),
            "student_parameters": initial_student.number_of_parameters(),
            "teacher_hash": teacher_hash,
        },
        "fairness": {
            "same_initial_student": True,
            "same_train_split": True,
            "same_eval_split": True,
            "same_optimizer": "AdamW",
            "same_learning_rate": 3e-4,
            "trained_method_token_budget": method_steps * tokens_per_step,
        },
        "teacher_bootstrap": {
            "optimizer_steps": teacher_result.optimizer_steps,
            "response_tokens": teacher_result.response_tokens,
            "wall_seconds": teacher_result.wall_seconds,
            "history": list(teacher_result.history),
        },
        "initial_student_hash": initial_hash,
        "runs": runs,
        "wall_seconds": time.perf_counter() - started,
        "python_tracemalloc_peak_bytes": python_peak_bytes,
        "limitations": [
            "Toy arithmetic does not estimate large-language-model benchmark quality.",
            "A smoke run validates plumbing only; it is not a convergence result."
            if smoke
            else "The quick teacher bootstrap is intentionally smaller than a research run.",
        ],
    }
    write_json(output / "summary.json", summary)
    write_jsonl(output / "metrics.jsonl", all_metric_rows)
    html_path, plot_path, diagnostic_path = create_static_report(output, summary)
    summary["artifacts"] = {
        "html": str(html_path),
        "plot": str(plot_path),
        "distribution_diagnostics": str(diagnostic_path),
        "tensorboard": str(output / "tensorboard"),
    }
    write_json(output / "summary.json", summary)
    write_json(output / "experiment-card.json", summary)
    return summary


@torch.no_grad()
def compare_expression(
    run_dir: str | Path,
    expression: str,
    *,
    max_new_tokens: int = 48,
) -> dict[str, object]:
    """Generate teacher and student responses for one safe arithmetic expression."""

    normalized = expression.strip()
    if _SAFE_EXPRESSION.fullmatch(normalized) is None:
        raise ValueError(
            "expression must contain 1-48 characters from digits, spaces, +, -, *, /, (, )"
        )
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    directory = Path(run_dir).resolve()
    checkpoint_dir = directory / "checkpoints"
    checkpoint_paths = [
        checkpoint_dir / "teacher.pt",
        *(sorted(path for path in checkpoint_dir.glob("*.pt") if path.name != "teacher.pt")),
    ]
    if len(checkpoint_paths) < 2 or any(not path.is_file() for path in checkpoint_paths):
        raise FileNotFoundError("run directory needs teacher and student checkpoints")
    tokenizer = CharacterTokenizer()
    prompt = f"Compute: {normalized}\nReasoning:\n"
    prompt_ids = torch.tensor(tokenizer.encode(prompt, bos=True), dtype=torch.long).unsqueeze(0)
    responses: dict[str, str] = {}
    for checkpoint_path in checkpoint_paths:
        payload = load_checkpoint(checkpoint_path)
        model = TinyCausalLM(TinyTransformerConfig.from_dict(payload["model_config"]))
        model.load_state_dict(payload["model_state"])
        model.eval()
        generated = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            temperature=0.0,
        )
        response_ids = generated[0, prompt_ids.shape[1] :].tolist()
        responses[checkpoint_path.stem] = tokenizer.decode(response_ids)
    return {"expression": normalized, "prompt": prompt, "responses": responses}


def print_comparison(comparison: dict[str, object]) -> None:
    """Print one comparison without depending on a rich-terminal package."""

    print(f"Expression: {comparison['expression']}")
    responses = comparison["responses"]
    if not isinstance(responses, dict):
        raise TypeError("comparison responses must be a dictionary")
    for name, response in responses.items():
        print(f"\n[{name}]\n{response or '[empty response]'}")


def run_interactive_playground(
    run_dir: str | Path,
    *,
    input_fn: Callable[[str], str] = input,
) -> None:
    """Read expressions until blank/quit; enabled only by an explicit CLI flag."""

    print("Enter an arithmetic expression, or press Enter/type quit to stop.")
    while True:
        try:
            expression = input_fn("expression> ").strip()
        except EOFError:
            break
        if not expression or expression.lower() in {"q", "quit", "exit"}:
            break
        try:
            print_comparison(compare_expression(run_dir, expression))
        except ValueError as error:
            print(f"Invalid expression: {error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/demo", help="artifact directory")
    parser.add_argument("--smoke", action="store_true", help="one-step plumbing check")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["no_train", "sft", "off_policy_kd", "opd"],
        choices=["no_train", "sft", "off_policy_kd", "gkd", "opd"],
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--allow-device-fallback", action="store_true")
    parser.add_argument("--prompt", help="compare checkpoints on one arithmetic expression")
    parser.add_argument("--interactive", action="store_true", help="open the terminal playground")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    summary = run_demo(
        arguments.output,
        smoke=arguments.smoke,
        methods=tuple(arguments.methods),
        requested_device=arguments.device,
        allow_device_fallback=arguments.allow_device_fallback,
    )
    print(f"Report: {summary['artifacts']['html']}")
    print(f"TensorBoard: python -m tensorboard.main --logdir {summary['artifacts']['tensorboard']}")
    if arguments.prompt:
        print_comparison(compare_expression(arguments.output, arguments.prompt))
    if arguments.interactive:
        run_interactive_playground(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

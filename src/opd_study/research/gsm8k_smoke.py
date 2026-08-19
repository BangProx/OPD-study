"""Actual pinned-GSM8K plumbing smoke using the offline mini model."""

from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import torch

from opd_study.algorithms import supervised_fine_tuning_loss
from opd_study.data import (
    CharacterTokenizer,
    collate_examples,
    fetch_gsm8k,
    gsm8k_to_mini_examples,
    load_gsm8k_rows,
)
from opd_study.models import TinyCausalLM, TinyTransformerConfig
from opd_study.reporting import (
    environment_record,
    file_sha256,
    git_commit,
    save_checkpoint,
    write_json,
)
from opd_study.training import train_sft
from opd_study.utils import model_state_hash, seed_everything, stable_json_hash


def run_gsm8k_mini_smoke(
    cache_dir: str | Path,
    output_dir: str | Path,
    *,
    accept_dataset_license: bool,
) -> dict[str, Any]:
    tracemalloc.start()
    started = time.perf_counter()
    run_configuration = {
        "profile": "gsm8k-mini-plumbing-smoke",
        "dataset_revision": "740312add88f781978c0658806c59bc2815b9866",
        "seed": 42,
        "steps": 1,
        "batch_size": 4,
        "tokens_per_step": 4,
    }
    config_hash = stable_json_hash(run_configuration)
    paths = fetch_gsm8k(
        cache_dir, accept_dataset_license=accept_dataset_license
    )
    splits = load_gsm8k_rows(paths)
    train_examples = gsm8k_to_mini_examples(splits["train"][:16])
    validation_examples = gsm8k_to_mini_examples(splits["validation"][:8])
    tokenizer = CharacterTokenizer()
    seed_everything(42)
    model = TinyCausalLM(
        TinyTransformerConfig.student(vocab_size=tokenizer.vocab_size)
    )
    initial_hash = model_state_hash(model)
    result = train_sft(
        model,
        train_examples,
        tokenizer,
        steps=1,
        batch_size=4,
        tokens_per_step=4,
    )
    model.eval()
    batch = collate_examples(validation_examples, tokenizer)
    with torch.no_grad():
        validation_loss = float(
            supervised_fine_tuning_loss(
                model(batch.token_ids, batch.attention_mask), batch
            ).loss
        )
    output = Path(output_dir).resolve()
    checkpoint = output / "checkpoints" / "gsm8k-mini-sft.pt"
    save_checkpoint(
        checkpoint,
        {
            "schema_version": 1,
            "model_config": model.config.to_dict(),
            "model_state": model.state_dict(),
            "dataset_revision": "740312add88f781978c0658806c59bc2815b9866",
            "config_hash": config_hash,
            "seed": 42,
            "optimizer_steps": result.optimizer_steps,
            "optimizer_state": result.optimizer_state,
            "response_tokens": result.response_tokens,
            "torch_rng_state": torch.get_rng_state(),
        },
    )
    _, python_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    card: dict[str, Any] = {
        "schema_version": 1,
        "status": "EXECUTED",
        "profile": "gsm8k-mini-plumbing-smoke",
        "git_commit": git_commit(Path(__file__).resolve().parents[3]),
        "command": [sys.executable, *sys.argv],
        "config": run_configuration,
        "config_hash": config_hash,
        "environment": environment_record(),
        "dataset": {
            "id": "openai/gsm8k",
            "config": "main",
            "revision": "740312add88f781978c0658806c59bc2815b9866",
            "license": "MIT",
            "official_rows": {"train": 7473, "test": 1319},
            "derived_rows": {"train": 6961, "validation": 512, "test": 1319},
            "smoke_rows": {"train": 16, "validation": 8, "test": 0},
            "download_bytes": 2_725_633,
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
            "algorithm": "sft",
            "seed": 42,
            "optimizer_steps": result.optimizer_steps,
            "response_tokens": result.response_tokens,
            "initial_student_hash": initial_hash,
            "final_student_hash": model_state_hash(model),
            "history": list(result.history),
        },
        "evaluation": {"validation_gold_nll": validation_loss, "test_used": False},
        "checkpoint": str(checkpoint),
        "wall_seconds": time.perf_counter() - started,
        "python_tracemalloc_peak_bytes": python_peak_bytes,
        "limitations": [
            "This run validates real-data loading and training plumbing with a tiny "
            "character model.",
            "Questions are bounded to 56 characters and only final answers are targets.",
            "It is not the Qwen3 LoRA OPD result and makes no GSM8K accuracy claim.",
        ],
    }
    write_json(output / "experiment-card.json", card)
    return card

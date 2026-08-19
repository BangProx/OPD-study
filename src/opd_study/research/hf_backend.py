"""Lazy Transformers/PEFT construction for the pinned laptop/server presets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from opd_study.config import ExperimentConfig
from opd_study.device import resolve_device
from opd_study.research.preflight import research_preflight


def load_model_pair(
    config: ExperimentConfig, cache_dir: str | Path | None = None
) -> tuple[Any, Any, Any]:
    """Load tokenizer, LoRA/QLoRA student, and frozen teacher after all gates pass."""

    report = research_preflight(config)
    if not report.ready:
        raise RuntimeError("research preflight failed: " + "; ".join(report.blockers))
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    device_report = resolve_device(
        config.training.device,
        allow_fallback=config.training.allow_device_fallback,
    )
    cache = None if cache_dir is None else str(Path(cache_dir).resolve())
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.student,
        revision=config.model.student_revision,
        trust_remote_code=False,
        cache_dir=cache,
    )
    quantization = None
    if config.model.finetuning == "qlora":
        compute_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(config.training.precision)
        if compute_dtype is None:
            raise ValueError("precision must be float16, bfloat16 or float32")
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    student_load_options: dict[str, Any] = {
        "revision": config.model.student_revision,
        "trust_remote_code": False,
        "quantization_config": quantization,
        "torch_dtype": "auto",
        "low_cpu_mem_usage": True,
        "cache_dir": cache,
    }
    if quantization is not None:
        student_load_options["device_map"] = {"": 0}
    student: Any = AutoModelForCausalLM.from_pretrained(
        config.model.student,
        **student_load_options,
    )
    if quantization is not None:
        student = prepare_model_for_kbit_training(
            student,
            use_gradient_checkpointing=False,
        )
    student = get_peft_model(
        student,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    teacher: Any = AutoModelForCausalLM.from_pretrained(
        config.model.teacher,
        revision=config.model.teacher_revision,
        trust_remote_code=False,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        cache_dir=cache,
    ).eval()
    if quantization is None:
        student.to(torch.device(device_report.selected))
    teacher.to(torch.device(device_report.selected))
    teacher.requires_grad_(False)
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        config.model.teacher,
        revision=config.model.teacher_revision,
        trust_remote_code=False,
        cache_dir=cache,
    )
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("student and teacher tokenizers differ; cross-tokenizer KD is unsupported")
    if tokenizer.chat_template != teacher_tokenizer.chat_template:
        raise ValueError("student and teacher chat templates differ")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, student, teacher

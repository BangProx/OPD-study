"""Strict experiment configuration shared by CLI, notebooks and run cards."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from opd_study.utils import stable_json_hash


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown {section} configuration keys: {names}")


@dataclass(frozen=True)
class AlgorithmConfig:
    name: str = "opd"
    divergence: str = "reverse_kl"
    lambda_on_policy: float = 1.0
    beta_jsd: float = 0.5
    temperature: float = 1.0


@dataclass(frozen=True)
class DataConfig:
    id: str = "tiny_arithmetic"
    revision: str = "generated-v1"
    config: str | None = None
    license: str = "generated/Apache-2.0"
    expected_download_bytes: int = 0
    accept_dataset_license: bool = False
    seed: int = 42
    train_rows: int = 4096
    validation_rows: int = 512
    test_rows: int = 512


@dataclass(frozen=True)
class ModelConfig:
    student: str = "tiny-student"
    teacher: str = "tiny-teacher"
    student_revision: str | None = None
    teacher_revision: str | None = None
    teacher_base: str | None = None
    finetuning: str = "full"
    trust_remote_code: bool = False
    expected_download_bytes: int = 0
    accept_model_license: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    steps: int = 20
    batch_size: int = 8
    tokens_per_step: int = 64
    learning_rate: float = 0.0003
    device: str = "auto"
    allow_device_fallback: bool = False
    precision: str = "float32"


@dataclass(frozen=True)
class EvaluationConfig:
    rows: int = 16
    max_new_tokens: int = 64
    k_values: tuple[int, ...] = (1, 2, 4, 8)


@dataclass(frozen=True)
class OutputConfig:
    root: str = "artifacts/demo"
    tensorboard: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int = 1
    profile: str = "toy"
    backend: str = "mini"
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only schema_version=1 is supported")
        if self.profile not in {"toy", "laptop", "server"}:
            raise ValueError("profile must be toy, laptop or server")
        if self.backend not in {"mini", "research"}:
            raise ValueError("backend must be mini or research")
        allowed_algorithms = {
            "sft",
            "off_policy_kd",
            "gkd",
            "opd",
            "vopd",
            "opd2",
            "tcod_f2b",
            "tcod_b2f",
            "sod",
            "sage_opd",
        }
        if self.algorithm.name not in allowed_algorithms:
            raise ValueError(f"unsupported algorithm: {self.algorithm.name}")
        if not 0.0 <= self.algorithm.lambda_on_policy <= 1.0:
            raise ValueError("lambda_on_policy must be in [0, 1]")
        if not 0.0 <= self.algorithm.beta_jsd <= 1.0:
            raise ValueError("beta_jsd must be in [0, 1]")
        if self.algorithm.temperature <= 0:
            raise ValueError("temperature must be positive")
        if min(
            self.data.train_rows,
            self.data.validation_rows,
            self.data.test_rows,
            self.training.steps,
            self.training.batch_size,
            self.training.tokens_per_step,
            self.evaluation.rows,
        ) < 1:
            raise ValueError("row, step, batch and token counts must be positive")
        if self.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.model.trust_remote_code:
            raise ValueError("trust_remote_code=true is not allowed by the security contract")
        if self.profile != "toy" and (
            self.model.student_revision is None or self.model.teacher_revision is None
        ):
            raise ValueError("non-toy model revisions must be pinned")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        return stable_json_hash(self.to_dict())


def _section(cls: type[Any], value: Any, name: str) -> Any:
    if value is None:
        return cls()
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    allowed = set(cls.__dataclass_fields__)
    _reject_unknown(value, allowed, name)
    normalized = dict(value)
    if cls is EvaluationConfig and "k_values" in normalized:
        normalized["k_values"] = tuple(normalized["k_values"])
    return cls(**normalized)


def config_from_mapping(value: Mapping[str, Any]) -> ExperimentConfig:
    allowed = set(ExperimentConfig.__dataclass_fields__)
    _reject_unknown(value, allowed, "root")
    scalar = {
        key: value[key]
        for key in ("schema_version", "profile", "backend")
        if key in value
    }
    return ExperimentConfig(
        **scalar,
        algorithm=_section(AlgorithmConfig, value.get("algorithm"), "algorithm"),
        data=_section(DataConfig, value.get("data"), "data"),
        model=_section(ModelConfig, value.get("model"), "model"),
        training=_section(TrainingConfig, value.get("training"), "training"),
        evaluation=_section(EvaluationConfig, value.get("evaluation"), "evaluation"),
        output=_section(OutputConfig, value.get("output"), "output"),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise TypeError("configuration root must be a mapping")
    return config_from_mapping(value)

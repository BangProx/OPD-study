"""Stable public names and fidelity labels for implemented algorithms."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlgorithmMetadata:
    name: str
    state_source: str
    objective: str
    fidelity: str
    source_id: str
    multi_turn: bool = False


_ALGORITHMS = {
    item.name: item
    for item in (
        AlgorithmMetadata("sft", "fixed demonstration", "hard CE", "reference", "sft"),
        AlgorithmMetadata(
            "off_policy_kd", "fixed demonstration", "forward KL", "reference", "gkd"
        ),
        AlgorithmMetadata("gkd", "lambda mixture", "generalized JSD", "paper", "gkd"),
        AlgorithmMetadata("opd", "student rollout", "sampled/full reverse KL", "paper", "vopd"),
        AlgorithmMetadata("vopd", "student rollout", "control-variate OPD", "paper", "vopd"),
        AlgorithmMetadata("opd2", "student rollout", "centered delta advantage", "paper", "opd2"),
        AlgorithmMetadata("tcod_f2b", "early student turns", "forward KL", "paper", "tcod", True),
        AlgorithmMetadata(
            "tcod_b2f",
            "teacher prefix + student suffix",
            "forward KL",
            "paper",
            "tcod",
            True,
        ),
        AlgorithmMetadata(
            "sod",
            "student tool trajectory",
            "step-weighted KL",
            "distillation-term",
            "sod",
            True,
        ),
        AlgorithmMetadata(
            "sage_opd",
            "student agent trajectory",
            "selective normalized RKL",
            "paper",
            "sage_opd",
            True,
        ),
    )
}


def available_algorithms() -> tuple[str, ...]:
    return tuple(_ALGORITHMS)


def algorithm_metadata(name: str) -> AlgorithmMetadata:
    try:
        return _ALGORITHMS[name]
    except KeyError as error:
        raise KeyError(f"unknown algorithm '{name}'") from error

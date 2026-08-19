"""Deterministic, offline arithmetic trajectories used by every core lesson.

The dataset deliberately exposes its generation procedure.  It is a teaching fixture,
not a benchmark: test rows must still remain read-only within an experiment.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ArithmeticExample:
    """One left-associated integer-arithmetic problem and worked solution."""

    example_id: str
    prompt: str
    response: str
    expression: str
    answer: int
    number_of_steps: int


@dataclass(frozen=True)
class TinyArithmeticSplits:
    """Immutable train/validation/test partitions and their provenance."""

    train: tuple[ArithmeticExample, ...]
    validation: tuple[ArithmeticExample, ...]
    test: tuple[ArithmeticExample, ...]
    seed: int
    revision: str = "generated-v1"

    def split_hash(self, split: str) -> str:
        """Return a stable SHA-256 over a named split."""

        rows = getattr(self, split, None)
        if not isinstance(rows, tuple):
            raise ValueError("split must be one of: train, validation, test")
        payload = json.dumps(
            [asdict(row) for row in rows],
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _apply(left: int, operator: str, right: int) -> int | None:
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if operator == "/" and right != 0 and left % right == 0:
        return left // right
    return None


def _candidate(rng: random.Random, number_of_steps: int) -> tuple[str, str, int] | None:
    """Generate a bounded problem; return ``None`` for an invalid division/result."""

    current = rng.randint(-30, 50)
    expression = str(current)
    worked_steps: list[str] = []
    for step_index in range(1, number_of_steps + 1):
        operator = rng.choice(("+", "-", "*", "/"))
        right = rng.randint(1, 20)
        result = _apply(current, operator, right)
        if result is None or abs(result) > 999:
            return None
        worked_steps.append(f"{step_index}) {current} {operator} {right} = {result}")
        expression = f"({expression} {operator} {right})"
        current = result
    return expression, "\n".join(worked_steps), current


def generate_tiny_arithmetic(
    *,
    seed: int = 42,
    train_rows: int = 4096,
    validation_rows: int = 512,
    test_rows: int = 512,
) -> TinyArithmeticSplits:
    """Generate unique, deterministic one-to-three-step arithmetic examples.

    The split is made only after all unique rows have been generated, so changing an
    evaluation size cannot silently move an existing row into training.
    """

    sizes = (train_rows, validation_rows, test_rows)
    if any(size < 1 for size in sizes):
        raise ValueError("all split sizes must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    total = sum(sizes)
    rng = random.Random(seed)
    rows: list[ArithmeticExample] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = total * 100
    while len(rows) < total and attempts < max_attempts:
        attempts += 1
        number_of_steps = 1 + len(rows) % 3
        candidate = _candidate(rng, number_of_steps)
        if candidate is None:
            continue
        expression, worked, answer = candidate
        if expression in seen:
            continue
        seen.add(expression)
        row_hash = hashlib.sha256(f"generated-v1:{seed}:{expression}".encode()).hexdigest()
        rows.append(
            ArithmeticExample(
                example_id=f"ta-{row_hash[:12]}",
                prompt=f"Compute: {expression}\nReasoning:\n",
                response=f"{worked}\nAnswer: {answer}",
                expression=expression,
                answer=answer,
                number_of_steps=number_of_steps,
            )
        )
    if len(rows) != total:
        raise RuntimeError(
            f"could only generate {len(rows)} of {total} unique rows after {attempts} attempts"
        )

    train_end = train_rows
    validation_end = train_end + validation_rows
    return TinyArithmeticSplits(
        train=tuple(rows[:train_end]),
        validation=tuple(rows[train_end:validation_end]),
        test=tuple(rows[validation_end:]),
        seed=seed,
    )

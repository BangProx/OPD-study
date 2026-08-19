"""A deterministic calculator environment exposing error compounding turn by turn."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CalculatorState:
    turn: int
    value: int
    target: int
    observation: str
    terminal: bool
    success: bool


class CalculatorEnvironment:
    """Apply a fixed operation chain while the agent proposes each intermediate value."""

    def __init__(self, start: int, operations: tuple[tuple[str, int], ...]) -> None:
        if not operations:
            raise ValueError("operations must not be empty")
        self._start = start
        self._operations = operations
        expected = start
        for operator, operand in operations:
            expected = self._apply(expected, operator, operand)
        self._target = expected
        self._turn = 0
        self._value = start
        self._terminal = False

    @staticmethod
    def _apply(value: int, operator: str, operand: int) -> int:
        if operator == "+":
            return value + operand
        if operator == "-":
            return value - operand
        if operator == "*":
            return value * operand
        if operator == "/" and operand != 0 and value % operand == 0:
            return value // operand
        raise ValueError(f"invalid integer operation: {value} {operator} {operand}")

    def reset(self) -> CalculatorState:
        self._turn = 0
        self._value = self._start
        self._terminal = False
        operator, operand = self._operations[0]
        return CalculatorState(
            turn=0,
            value=self._value,
            target=self._target,
            observation=f"Current {self._value}; apply {operator} {operand}",
            terminal=False,
            success=False,
        )

    def step(self, proposed_value: int) -> CalculatorState:
        if self._terminal:
            raise RuntimeError("cannot step a terminal environment; call reset")
        operator, operand = self._operations[self._turn]
        correct_value = self._apply(self._value, operator, operand)
        correct = proposed_value == correct_value
        # The environment uses the student's actual value. One wrong turn therefore
        # changes every later state and makes error compounding visible.
        self._value = proposed_value
        self._turn += 1
        self._terminal = self._turn == len(self._operations)
        success = self._terminal and self._value == self._target
        if self._terminal:
            observation = (
                f"Done; value {self._value}; target {self._target}; "
                f"{'success' if success else 'failure'}"
            )
        else:
            next_operator, next_operand = self._operations[self._turn]
            status = "correct" if correct else f"expected {correct_value}"
            observation = (
                f"Tool says {status}; current {self._value}; "
                f"apply {next_operator} {next_operand}"
            )
        return CalculatorState(
            turn=self._turn,
            value=self._value,
            target=self._target,
            observation=observation,
            terminal=self._terminal,
            success=success,
        )

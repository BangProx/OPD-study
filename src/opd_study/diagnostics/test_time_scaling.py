"""Problem-level avg@K/pass@K diagnostics from arXiv:2608.11829v1."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class ScalingMetrics:
    k: int
    avg_at_k: float
    pass_at_k: float
    solved_problem_count: int
    problem_count: int


def scaling_metrics(outcomes: Tensor, *, k: int) -> ScalingMetrics:
    """Aggregate boolean outcomes shaped ``[problems, independent samples]``."""

    if outcomes.ndim != 2 or outcomes.shape[0] < 1:
        raise ValueError("outcomes must have shape [problems, samples]")
    if not 1 <= k <= outcomes.shape[1]:
        raise ValueError(f"k must be within [1, {outcomes.shape[1]}]")
    selected = outcomes[:, :k].bool()
    solved = selected.any(dim=1)
    return ScalingMetrics(
        k=k,
        avg_at_k=float(selected.float().mean().item()),
        pass_at_k=float(solved.float().mean().item()),
        solved_problem_count=int(solved.sum().item()),
        problem_count=outcomes.shape[0],
    )


def gained_and_lost_solvability(
    before: Tensor, after: Tensor, *, k: int
) -> dict[str, int]:
    if before.shape != after.shape:
        raise ValueError("before and after outcome matrices must match")
    if not 1 <= k <= before.shape[1]:
        raise ValueError(f"k must be within [1, {before.shape[1]}]")
    solved_before = before[:, :k].bool().any(dim=1)
    solved_after = after[:, :k].bool().any(dim=1)
    return {
        "gained": int((~solved_before & solved_after).sum().item()),
        "lost": int((solved_before & ~solved_after).sum().item()),
        "retained": int((solved_before & solved_after).sum().item()),
        "never_solved": int((~solved_before & ~solved_after).sum().item()),
    }

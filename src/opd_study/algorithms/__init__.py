"""Reference objectives and rollout utilities for distillation lessons."""

from opd_study.algorithms.losses import (
    generalized_kd_loss,
    off_policy_kd_loss,
    on_policy_distillation_loss,
    sampled_reverse_kl_loss,
    supervised_fine_tuning_loss,
)
from opd_study.algorithms.opd2 import opd2_loss
from opd_study.algorithms.registry import algorithm_metadata, available_algorithms
from opd_study.algorithms.rollout import (
    collect_multiturn_trajectories,
    collect_student_trajectories,
    score_teacher,
)
from opd_study.algorithms.sage_opd import sage_opd_loss
from opd_study.algorithms.sod import sod_loss
from opd_study.algorithms.tcod import tcod_loss
from opd_study.algorithms.vopd import vopd_loss

__all__ = [
    "algorithm_metadata",
    "available_algorithms",
    "collect_multiturn_trajectories",
    "collect_student_trajectories",
    "generalized_kd_loss",
    "off_policy_kd_loss",
    "on_policy_distillation_loss",
    "opd2_loss",
    "sage_opd_loss",
    "sampled_reverse_kl_loss",
    "score_teacher",
    "sod_loss",
    "supervised_fine_tuning_loss",
    "tcod_loss",
    "vopd_loss",
]

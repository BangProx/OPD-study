"""Small typed wrappers around optimizer helpers with version-varying PyTorch stubs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import torch
from torch import Tensor, nn


def clip_gradient_norm(
    parameters: Iterable[nn.Parameter], maximum_norm: float
) -> Tensor:
    implementation: Any = vars(torch.nn.utils)["clip_grad_norm_"]
    return cast(Tensor, implementation(parameters, maximum_norm))

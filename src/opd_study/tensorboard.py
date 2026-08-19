"""Version-neutral lazy access to PyTorch's TensorBoard writer."""

from __future__ import annotations

import importlib
from typing import Any

SummaryWriter: Any = vars(importlib.import_module("torch.utils.tensorboard"))[
    "SummaryWriter"
]

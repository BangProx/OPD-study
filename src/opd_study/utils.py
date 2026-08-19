"""Small deterministic and hashing utilities shared by examples and tests."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


def seed_everything(seed: int, *, deterministic: bool = True) -> torch.Generator:
    """Seed Python and PyTorch and return a CPU generator for explicit sampling."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def model_state_hash(model: nn.Module) -> str:
    """Hash tensor names, dtypes, shapes and bytes for fairness auditing."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        detached = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("ascii"))
        digest.update(str(tuple(detached.shape)).encode("ascii"))
        # NumPy cannot represent every PyTorch dtype (notably older bfloat16).
        # Hashing the raw byte view keeps the audit stable across those dtypes.
        digest.update(detached.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def stable_json_hash(value: Mapping[str, Any]) -> str:
    """Hash JSON-compatible metadata with deterministic ordering."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

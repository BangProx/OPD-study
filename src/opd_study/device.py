"""Cross-platform device capability checks with opt-in fallback only."""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class DeviceReport:
    requested: str
    selected: str
    fallback_used: bool
    operating_system: str
    cuda_available: bool
    mps_available: bool
    qlora_supported: bool
    note: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(requested: str = "auto", *, allow_fallback: bool = False) -> DeviceReport:
    requested = requested.lower()
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda or mps")
    available = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": _mps_available(),
    }
    if requested == "auto":
        selected = "cuda" if available["cuda"] else "mps" if available["mps"] else "cpu"
        fallback_used = False
        note = "auto selected the best available backend"
    elif available[requested]:
        selected = requested
        fallback_used = False
        note = "requested backend is available"
    elif allow_fallback:
        selected = "cpu"
        fallback_used = True
        note = f"{requested} unavailable; explicit fallback selected CPU"
    else:
        raise RuntimeError(
            f"requested device '{requested}' is unavailable; use device=cpu or explicitly "
            "set allow_device_fallback=true"
        )
    system = platform.system()
    qlora_supported = selected == "cuda" and system != "Darwin"
    return DeviceReport(
        requested=requested,
        selected=selected,
        fallback_used=fallback_used,
        operating_system=system,
        cuda_available=available["cuda"],
        mps_available=available["mps"],
        qlora_supported=qlora_supported,
        note=note,
    )


def require_qlora(report: DeviceReport) -> None:
    if not report.qlora_supported:
        raise RuntimeError(
            "QLoRA requires a validated NVIDIA CUDA + bitsandbytes environment in this "
            "project; macOS/MPS and CPU never fall back to full fine-tuning"
        )

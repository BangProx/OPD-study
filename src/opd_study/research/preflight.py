"""Fail-fast checks before any research-backend model or dataset download."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version

from opd_study.config import ExperimentConfig
from opd_study.device import require_qlora, resolve_device


@dataclass(frozen=True)
class ResearchPreflight:
    ready: bool
    missing_packages: tuple[str, ...]
    package_versions: dict[str, str]
    import_errors: tuple[str, ...]
    dataset_license_accepted: bool
    model_license_accepted: bool
    expected_download_bytes: int
    selected_device: str
    qlora_supported: bool
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def research_preflight(config: ExperimentConfig) -> ResearchPreflight:
    if config.backend != "research":
        raise ValueError("research preflight requires backend=research")
    required = ["transformers", "datasets", "accelerate", "peft"]
    if config.model.finetuning == "qlora":
        required.append("bitsandbytes")
    missing = tuple(name for name in required if importlib.util.find_spec(name) is None)
    blockers: list[str] = []
    if missing:
        blockers.append("missing optional packages: " + ", ".join(missing))
    package_versions: dict[str, str] = {}
    for distribution in ("torch", *required):
        try:
            package_versions[distribution] = version(distribution)
        except PackageNotFoundError:
            continue
    if Version(package_versions["torch"]) < Version("2.2"):
        blockers.append(
            f"research backend requires torch>=2.2; found {package_versions['torch']}"
        )
    import_errors: list[str] = []
    for module_name in required:
        if module_name in missing:
            continue
        try:
            importlib.import_module(module_name)
        except Exception as error:  # package binary/API incompatibility, not user code
            message = f"{module_name}: {type(error).__name__}: {error}"
            import_errors.append(message)
    if import_errors:
        blockers.append("optional package import failed: " + " | ".join(import_errors))
    try:
        device = resolve_device(
            config.training.device,
            allow_fallback=config.training.allow_device_fallback,
        )
        selected_device = device.selected
        qlora_supported = device.qlora_supported
    except RuntimeError as error:
        blockers.append(str(error))
        device = None
        selected_device = "unavailable"
        qlora_supported = False
    if config.data.expected_download_bytes and not config.data.accept_dataset_license:
        blockers.append("dataset license/download not accepted")
    if config.model.expected_download_bytes and not config.model.accept_model_license:
        blockers.append("model license/download not accepted")
    if config.model.finetuning == "qlora" and device is not None:
        try:
            require_qlora(device)
        except RuntimeError as error:
            blockers.append(str(error))
    return ResearchPreflight(
        ready=not blockers,
        missing_packages=missing,
        package_versions=package_versions,
        import_errors=tuple(import_errors),
        dataset_license_accepted=config.data.accept_dataset_license,
        model_license_accepted=config.model.accept_model_license,
        expected_download_bytes=(
            config.data.expected_download_bytes + config.model.expected_download_bytes
        ),
        selected_device=selected_device,
        qlora_supported=qlora_supported,
        blockers=tuple(blockers),
    )

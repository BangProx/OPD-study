from __future__ import annotations

import unittest
from pathlib import Path

from opd_study.config import config_from_mapping, load_config
from opd_study.device import require_qlora, resolve_device


class ConfigAndDeviceTest(unittest.TestCase):
    def test_all_checked_in_presets_parse(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        for path in (repository / "configs").glob("*/*.yaml"):
            with self.subTest(path=path):
                self.assertEqual(load_config(path).schema_version, 1)

    def test_unknown_keys_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown training"):
            config_from_mapping({"training": {"steps": 1, "silent_typo": True}})

    def test_unavailable_accelerator_never_silently_falls_back(self) -> None:
        report = resolve_device("cpu")
        self.assertEqual(report.selected, "cpu")
        self.assertFalse(report.fallback_used)
        with self.assertRaisesRegex(RuntimeError, "QLoRA"):
            require_qlora(report)

    def test_qlora_preset_is_explicit_cuda_and_opt_in(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        config = load_config(repository / "configs/laptop/gsm8k_qlora.yaml")
        self.assertEqual(config.model.finetuning, "qlora")
        self.assertEqual(config.training.device, "cuda")
        self.assertEqual(config.training.precision, "float16")
        self.assertFalse(config.training.allow_device_fallback)
        self.assertFalse(config.data.accept_dataset_license)
        self.assertFalse(config.model.accept_model_license)


if __name__ == "__main__":
    unittest.main()

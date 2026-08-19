from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from opd_study.cli import main as cli_main
from opd_study.config import load_config
from opd_study.data import fetch_gsm8k, gsm8k_to_mini_examples
from opd_study.research import extract_gsm8k_answer, research_preflight


class ResearchGateTest(unittest.TestCase):
    def test_gsm8k_download_requires_explicit_acceptance_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(PermissionError, "MIT licensed"):
                fetch_gsm8k(root, accept_dataset_license=False)
            self.assertEqual(list(root.iterdir()), [])

    def test_gsm8k_conversion_extracts_final_answer_without_chain_of_thought(self) -> None:
        rows = [
            {
                "question": "If Mina has 2 apples and buys 3, how many?",
                "answer": "Add them carefully. 2 + 3 = 5. #### 5",
            }
        ]
        example = gsm8k_to_mini_examples(rows)[0]
        self.assertEqual(example.answer, 5)
        self.assertEqual(example.response, "Answer: 5")
        self.assertNotIn("Add them carefully", example.response)

    def test_research_preflight_keeps_both_acceptance_gates_closed(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        config = load_config(repository / "configs/laptop/gsm8k_lora.yaml")
        report = research_preflight(config)
        self.assertFalse(report.ready)
        self.assertIn("dataset license/download not accepted", report.blockers)
        self.assertIn("model license/download not accepted", report.blockers)
        accepted = replace(
            config,
            data=replace(config.data, accept_dataset_license=True),
            model=replace(config.model, accept_model_license=True),
        )
        accepted_report = research_preflight(accepted)
        self.assertTrue(accepted_report.dataset_license_accepted)
        self.assertTrue(accepted_report.model_license_accepted)
        # Optional packages/hardware may still block this host; acceptance alone is not
        # misreported as readiness.
        self.assertEqual(accepted_report.ready, not accepted_report.blockers)

    def test_research_answer_parser_has_explicit_precedence(self) -> None:
        self.assertEqual(extract_gsm8k_answer("work 1 #### 2 and then 99"), 2)
        self.assertEqual(extract_gsm8k_answer(r"result is \boxed{-1,234}"), -1234)
        self.assertEqual(extract_gsm8k_answer("guess 3 then 7"), 7)
        self.assertIsNone(extract_gsm8k_answer("no numeric answer"))

    def test_research_cli_fails_before_creating_cache_without_consent(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            output = root / "output"
            status = cli_main(
                [
                    "research-train",
                    "--config",
                    str(repository / "configs/laptop/gsm8k_lora.yaml"),
                    "--cache",
                    str(cache),
                    "--output",
                    str(output),
                    "--smoke",
                ]
            )
            self.assertEqual(status, 2)
            self.assertFalse(cache.exists())
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

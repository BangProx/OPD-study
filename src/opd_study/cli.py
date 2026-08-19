"""Command-line entry point for OPD-study."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from opd_study import __version__
from opd_study.algorithms import available_algorithms
from opd_study.config import load_config
from opd_study.data import fetch_gsm8k, load_gsm8k_rows
from opd_study.demo import (
    compare_expression,
    print_comparison,
    run_demo,
    run_interactive_playground,
)
from opd_study.device import resolve_device
from opd_study.experiment import evaluate_saved_run, run_toy_algorithm
from opd_study.research import research_preflight, run_research_smoke
from opd_study.research.gsm8k_smoke import run_gsm8k_mini_smoke


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opd-study")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")
    demo = subparsers.add_parser("demo", help="run the offline fair-comparison playground")
    demo.add_argument("--output", default="artifacts/demo")
    demo.add_argument("--smoke", action="store_true")
    demo.add_argument(
        "--methods",
        nargs="+",
        default=["no_train", "sft", "off_policy_kd", "opd"],
        choices=["no_train", "sft", "off_policy_kd", "gkd", "opd"],
    )
    demo.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    demo.add_argument("--allow-device-fallback", action="store_true")
    demo.add_argument("--prompt", help="compare checkpoints on one arithmetic expression")
    demo.add_argument("--interactive", action="store_true", help="open terminal playground")
    train = subparsers.add_parser("train", help="train one toy algorithm and save artifacts")
    train.add_argument("--algorithm", required=True, choices=available_algorithms())
    train.add_argument("--output", default="artifacts/train")
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    train.add_argument("--allow-device-fallback", action="store_true")
    evaluate = subparsers.add_parser("eval", help="evaluate a saved toy train run")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--rows", type=int, default=8)
    download = subparsers.add_parser("download-data", help="download a pinned dataset")
    download.add_argument("--dataset", choices=["gsm8k"], required=True)
    download.add_argument("--cache", default="artifacts/cache")
    download.add_argument("--accept-dataset-license", action="store_true")
    preflight = subparsers.add_parser("research-preflight", help="validate a research preset")
    preflight.add_argument("--config", required=True)
    research_train = subparsers.add_parser(
        "research-train",
        help="run an explicitly accepted single-device research preset",
    )
    research_train.add_argument("--config", required=True)
    research_train.add_argument("--cache", default="artifacts/cache")
    research_train.add_argument("--output")
    research_train.add_argument("--smoke", action="store_true")
    research_train.add_argument("--accept-dataset-license", action="store_true")
    research_train.add_argument("--accept-model-license", action="store_true")
    gsm_smoke = subparsers.add_parser("gsm8k-smoke", help="run real-data mini plumbing smoke")
    gsm_smoke.add_argument("--cache", default="artifacts/cache")
    gsm_smoke.add_argument("--output", default="artifacts/gsm8k-mini-smoke")
    gsm_smoke.add_argument("--accept-dataset-license", action="store_true")
    subparsers.add_parser("doctor", help="print detected hardware capability")
    return parser


def _run(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "demo":
        summary = run_demo(
            arguments.output,
            smoke=arguments.smoke,
            methods=tuple(arguments.methods),
            requested_device=arguments.device,
            allow_device_fallback=arguments.allow_device_fallback,
        )
        print(f"Report: {summary['artifacts']['html']}")
        if arguments.prompt:
            print_comparison(compare_expression(arguments.output, arguments.prompt))
        if arguments.interactive:
            run_interactive_playground(arguments.output)
        return 0
    if arguments.command == "doctor":
        print(resolve_device().to_dict())
        return 0
    if arguments.command == "train":
        summary = run_toy_algorithm(
            arguments.algorithm,
            arguments.output,
            smoke=arguments.smoke,
            requested_device=arguments.device,
            allow_device_fallback=arguments.allow_device_fallback,
        )
        print(f"Report: {summary['artifacts']['html']}")
        return 0
    if arguments.command == "eval":
        print(evaluate_saved_run(arguments.run, rows=arguments.rows))
        return 0
    if arguments.command == "download-data":
        paths = fetch_gsm8k(
            arguments.cache,
            accept_dataset_license=arguments.accept_dataset_license,
        )
        splits = load_gsm8k_rows(paths)
        print({name: len(rows) for name, rows in splits.items()})
        print({name: str(path) for name, path in paths.items()})
        return 0
    if arguments.command == "research-preflight":
        print(research_preflight(load_config(arguments.config)).to_dict())
        return 0
    if arguments.command == "research-train":
        config = load_config(arguments.config)
        config = replace(
            config,
            data=replace(
                config.data,
                accept_dataset_license=(
                    config.data.accept_dataset_license
                    or arguments.accept_dataset_license
                ),
            ),
            model=replace(
                config.model,
                accept_model_license=(
                    config.model.accept_model_license or arguments.accept_model_license
                ),
            ),
        )
        card = run_research_smoke(
            config,
            arguments.cache,
            arguments.output,
            smoke=arguments.smoke,
        )
        print(f"Experiment card: {card['artifacts']}")
        return 0
    if arguments.command == "gsm8k-smoke":
        card = run_gsm8k_mini_smoke(
            arguments.cache,
            arguments.output,
            accept_dataset_license=arguments.accept_dataset_license,
        )
        print(card)
        return 0
    build_parser().print_help()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and turn expected safety/configuration failures into concise errors."""

    try:
        return _run(argv)
    except (
        FileNotFoundError,
        ImportError,
        NotImplementedError,
        PermissionError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"opd-study: error: {error}", file=sys.stderr)
        return 2

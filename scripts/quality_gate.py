"""Run the same cross-platform core quality gate used by CI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]


def run(arguments: list[str]) -> None:
    print("+", " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=REPOSITORY, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-notebooks", action="store_true")
    parser.add_argument("--external-links", action="store_true")
    arguments = parser.parse_args()
    python = sys.executable
    run([python, "-m", "pytest", "-q"])
    run([python, "-m", "ruff", "check", "src", "tests", "scripts"])
    run([python, "-m", "mypy", "src/opd_study"])
    if arguments.execute_notebooks:
        run([python, "scripts/execute_notebooks.py", "--language", "all"])
    notebook_check = [python, "scripts/check_notebooks.py", "--require-executed"]
    run(notebook_check)
    run([python, "scripts/check_colab.py"])
    run([python, "scripts/check_sources.py"])
    link_check = [python, "scripts/check_links.py"]
    if not arguments.external_links:
        link_check.append("--local-only")
    run(link_check)
    print("OPD-study quality gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

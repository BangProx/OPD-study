"""Execute course notebooks top-to-bottom in a repository-root working directory."""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPOSITORY = Path(__file__).resolve().parents[1]


def execute(path: Path, timeout: int) -> float:
    notebook = nbformat.read(path, as_version=4)
    started = time.perf_counter()
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPOSITORY)}},
    )
    client.execute()
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".ipynb"
    ) as handle:
        temporary = Path(handle.name)
        nbformat.write(notebook, handle)
    os.replace(temporary, path)
    return time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["ko", "en", "all", "colab"], default="all")
    parser.add_argument("--timeout", type=int, default=180)
    arguments = parser.parse_args()
    if arguments.language == "colab":
        paths = [REPOSITORY / "notebooks/colab/quickstart.ipynb"]
    else:
        languages = ("ko", "en") if arguments.language == "all" else (arguments.language,)
        paths = [
            path
            for language in languages
            for path in sorted((REPOSITORY / "notebooks" / language).glob("*.ipynb"))
        ]
    for path in paths:
        duration = execute(path, arguments.timeout)
        print(f"{path.relative_to(REPOSITORY)}: {duration:.2f}s")
    print(f"executed {len(paths)} notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

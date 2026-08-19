"""Validate notebook structure, bilingual parity, sources, and stored execution state."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import nbformat
import yaml

REPOSITORY = Path(__file__).resolve().parents[1]
REQUIRED_KO = [
    "## Goal", "## Setup", "## Steps", "## Checks",
    "## 내가 자주 틀리는 것", "## 60초 요약", "## Next Steps",
]
REQUIRED_EN = [
    "## Goal", "## Setup", "## Steps", "## Checks",
    "## My recurring mistakes", "## 60-second summary", "## Next Steps",
]
ALLOWED_ROLES = {
    "objective", "map", "explain", "predict", "demo", "check", "exercise",
    "solution", "mistake-note", "summary", "source", "next",
}


def fail(message: str) -> None:
    raise AssertionError(message)


def normalized_code_hash(notebook: Any) -> str:
    source = "\n\n".join(
        "\n".join(line.rstrip() for line in cell.source.strip().splitlines())
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    return hashlib.sha256(source.encode()).hexdigest()


def validate_one(path: Path, *, require_executed: bool, known_sources: set[str]) -> Any:
    notebook = nbformat.read(path, as_version=4)
    metadata = notebook.metadata.get("opd_study", {})
    language = metadata.get("language")
    if language not in {"ko", "en"}:
        fail(f"{path}: invalid language metadata")
    if metadata.get("schema_version") != 1 or metadata.get("profile") != "toy":
        fail(f"{path}: invalid schema/profile metadata")
    if any(source not in known_sources for source in metadata.get("source_ids", [])):
        fail(f"{path}: metadata references an unknown source ID")
    expected_headings = REQUIRED_KO if language == "ko" else REQUIRED_EN
    headings = [
        cell.source.strip().splitlines()[0]
        for cell in notebook.cells
        if cell.cell_type == "markdown" and cell.source.strip().startswith("## ")
    ]
    if headings != expected_headings:
        fail(f"{path}: required section order mismatch: {headings}")
    roles: Counter[str] = Counter()
    cell_ids: list[str] = []
    code_counts: list[int] = []
    for cell in notebook.cells:
        cell_metadata = cell.metadata.get("opd_study", {})
        role = cell_metadata.get("role")
        cell_id = cell_metadata.get("cell_id")
        if role not in ALLOWED_ROLES or not isinstance(cell_id, str):
            fail(f"{path}: invalid cell metadata")
        roles[role] += 1
        cell_ids.append(cell_id)
        if cell.cell_type == "code":
            if len(cell.source.splitlines()) > 25:
                fail(f"{path}:{cell_id}: code cell exceeds 25 lines")
            if require_executed:
                if cell.execution_count is None:
                    fail(f"{path}:{cell_id}: code cell was not executed")
                code_counts.append(cell.execution_count)
                for output in cell.get("outputs", []):
                    if output.get("output_type") == "error":
                        fail(f"{path}:{cell_id}: stored traceback {output.get('ename')}")
                    if len(str(output)) > 1_000_000:
                        fail(f"{path}:{cell_id}: output exceeds 1MB")
    if len(cell_ids) != len(set(cell_ids)):
        fail(f"{path}: duplicate cell IDs")
    for role in ("map", "predict", "check", "exercise", "mistake-note", "source"):
        if roles[role] < 1:
            fail(f"{path}: missing role {role}")
    markdown_text = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    minimum_markdown_characters = 2_200 if language == "ko" else 3_000
    if len(markdown_text) < minimum_markdown_characters:
        fail(
            f"{path}: tutorial prose is too shallow ({len(markdown_text)} < "
            f"{minimum_markdown_characters} characters)"
        )
    design_headings = (
        ("### 실제 구현: 왜 이렇게 만들었나", "### 다른 선택지는 없나?")
        if language == "ko"
        else (
            "### Production implementation: why this design",
            "### Alternatives and trade-offs",
        )
    )
    for heading in design_headings:
        if heading not in markdown_text:
            fail(f"{path}: missing design section {heading}")
    if "Alt text:" not in markdown_text:
        fail(f"{path}: missing equivalent figure/map alt text")
    if markdown_text.count("### M") < 2:
        fail(f"{path}: needs two recurring-mistake entries")
    if require_executed and code_counts != sorted(code_counts):
        fail(f"{path}: execution counts are not monotonic")
    source_probe_cells = [
        cell for cell in notebook.cells
        if cell.cell_type == "code" and "inspect.getsource" in cell.source
    ]
    if len(source_probe_cells) != 1:
        fail(f"{path}: expected one bounded production-source probe")
    if require_executed:
        output_text = str(source_probe_cells[0].get("outputs", []))
        if "# opd_study." not in output_text:
            fail(f"{path}: production-source probe has no stored source output")
    source_markdown = "\n".join(
        cell.source
        for cell in notebook.cells
        if cell.metadata.opd_study.role == "source"
    )
    if "https://" not in source_markdown or "license `" not in source_markdown:
        fail(f"{path}: sources need direct URLs, exact versions, and licenses")
    return notebook


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-executed", action="store_true")
    arguments = parser.parse_args()
    manifest = yaml.safe_load((REPOSITORY / "docs/sources.yml").read_text())
    known_sources = {
        item["id"]
        for section in ("papers", "code_sources", "datasets", "models")
        for item in manifest[section]
    } | {"sft"}
    notebooks: dict[tuple[str, str], Any] = {}
    for language in ("ko", "en"):
        paths = sorted((REPOSITORY / "notebooks" / language).glob("*.ipynb"))
        if len(paths) != 12:
            fail(f"expected 12 {language} notebooks, found {len(paths)}")
        for path in paths:
            notebook = validate_one(
                path,
                require_executed=arguments.require_executed,
                known_sources=known_sources,
            )
            lesson_id = notebook.metadata.opd_study.lesson_id
            notebooks[(language, lesson_id)] = notebook
    for index in range(12):
        lesson_id = f"L{index:02d}"
        korean = notebooks[("ko", lesson_id)]
        english = notebooks[("en", lesson_id)]
        if normalized_code_hash(korean) != normalized_code_hash(english):
            fail(f"{lesson_id}: bilingual code hash mismatch")
        korean_code_ids = [
            cell.metadata.opd_study.cell_id for cell in korean.cells if cell.cell_type == "code"
        ]
        english_code_ids = [
            cell.metadata.opd_study.cell_id for cell in english.cells if cell.cell_type == "code"
        ]
        if korean_code_ids != english_code_ids:
            fail(f"{lesson_id}: bilingual code-cell order mismatch")
        role_counts = []
        for notebook in (korean, english):
            role_counts.append(
                Counter(cell.metadata.opd_study.role for cell in notebook.cells)
            )
        if role_counts[0] != role_counts[1]:
            fail(f"{lesson_id}: bilingual role-count mismatch")
    for language in ("ko", "en"):
        exercise_texts: set[str] = set()
        mistake_texts: set[str] = set()
        for index in range(12):
            notebook = notebooks[(language, f"L{index:02d}")]
            exercise_texts.update(
                cell.source
                for cell in notebook.cells
                if cell.metadata.opd_study.role == "exercise"
            )
            mistake_texts.update(
                cell.source
                for cell in notebook.cells
                if cell.metadata.opd_study.cell_id.endswith("M02")
            )
        if len(exercise_texts) != 12 or len(mistake_texts) != 12:
            fail(f"{language}: exercises and mistake notes must be lesson-specific")
    print(
        f"validated 24 notebooks; bilingual code parity passed; "
        f"executed={arguments.require_executed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

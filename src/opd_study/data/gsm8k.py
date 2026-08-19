"""License-gated, revision-pinned GSM8K parquet loader."""

from __future__ import annotations

import hashlib
import random
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from opd_study.data.tiny_arithmetic import ArithmeticExample

REVISION = "740312add88f781978c0658806c59bc2815b9866"
BASE_URL = f"https://huggingface.co/datasets/openai/gsm8k/resolve/{REVISION}/main"


@dataclass(frozen=True)
class GSM8KAsset:
    split: str
    filename: str
    bytes: int
    sha256: str

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.filename}"


ASSETS = {
    "train": GSM8KAsset(
        "train",
        "train-00000-of-00001.parquet",
        2_306_545,
        "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d",
    ),
    "test": GSM8KAsset(
        "test",
        "test-00000-of-00001.parquet",
        419_088,
        "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_gsm8k(
    cache_dir: str | Path,
    *,
    accept_dataset_license: bool,
) -> dict[str, Path]:
    """Download exact MIT-licensed shards only after an explicit caller opt-in."""

    if not accept_dataset_license:
        raise PermissionError(
            "GSM8K is MIT licensed; rerun with --accept-dataset-license after reviewing "
            "docs/sources.yml (2,725,633 bytes, pinned revision)."
        )
    directory = Path(cache_dir).resolve() / "openai-gsm8k" / REVISION
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, asset in ASSETS.items():
        path = directory / asset.filename
        if path.exists() and path.stat().st_size == asset.bytes and _sha256(path) == asset.sha256:
            paths[split] = path
            continue
        temporary = path.with_suffix(path.suffix + ".partial")
        request = urllib.request.Request(asset.url, headers={"User-Agent": "OPD-study/0.1"})
        try:
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                temporary.open("wb") as handle,
            ):
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            if temporary.stat().st_size != asset.bytes:
                raise OSError(
                    f"{split} size mismatch: {temporary.stat().st_size} != {asset.bytes}"
                )
            actual_hash = _sha256(temporary)
            if actual_hash != asset.sha256:
                raise OSError(f"{split} SHA-256 mismatch: {actual_hash}")
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        paths[split] = path
    return paths


def load_gsm8k_rows(
    paths: dict[str, Path], *, validation_rows: int = 512, seed: int = 42
) -> dict[str, list[dict[str, str]]]:
    """Read pinned shards and split validation only from official train."""

    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise ImportError(
            "GSM8K parquet reading needs pyarrow; install opd-study[research]"
        ) from error
    train_table = parquet.read_table(paths["train"], columns=["question", "answer"])
    test_table = parquet.read_table(paths["test"], columns=["question", "answer"])
    train_rows = train_table.to_pylist()
    test_rows = test_table.to_pylist()
    if len(train_rows) != 7473 or len(test_rows) != 1319:
        raise ValueError("GSM8K row counts differ from the audited manifest")
    if not 1 <= validation_rows < len(train_rows):
        raise ValueError("validation_rows must leave at least one training row")
    indices = list(range(len(train_rows)))
    random.Random(seed).shuffle(indices)
    validation_indices = set(indices[:validation_rows])
    return {
        "train": [row for index, row in enumerate(train_rows) if index not in validation_indices],
        "validation": [row for index, row in enumerate(train_rows) if index in validation_indices],
        "test": test_rows,
    }


def gsm8k_to_mini_examples(
    rows: list[dict[str, str]], *, maximum_question_characters: int = 56
) -> tuple[ArithmeticExample, ...]:
    """Create bounded final-answer examples for a mini-backend plumbing smoke.

    This deliberately does not claim to reproduce chain-of-thought Qwen training.  It
    proves that pinned real rows, split discipline, masks, optimization, and artifacts
    connect before a multi-gigabyte research run is authorized.
    """

    if maximum_question_characters < 16:
        raise ValueError("maximum_question_characters must be at least 16")
    examples: list[ArithmeticExample] = []
    for index, row in enumerate(rows):
        match = re.search(r"####\s*([-+]?\d[\d,]*)", row["answer"])
        if match is None:
            raise ValueError(f"row {index} lacks a GSM8K final-answer marker")
        answer = int(match.group(1).replace(",", ""))
        question = " ".join(row["question"].split())[:maximum_question_characters]
        examples.append(
            ArithmeticExample(
                example_id=f"gsm8k-main-{index:05d}",
                prompt=f"Question: {question}\nAnswer:\n",
                response=f"Answer: {answer}",
                expression=question,
                answer=answer,
                number_of_steps=1,
            )
        )
    return tuple(examples)

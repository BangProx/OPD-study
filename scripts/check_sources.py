"""Validate source/provenance metadata and locked asset revisions."""

from __future__ import annotations

from pathlib import Path

import yaml

REPOSITORY = Path(__file__).resolve().parents[1]


def main() -> int:
    manifest = yaml.safe_load((REPOSITORY / "docs/sources.yml").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["project_license"] == "Apache-2.0"
    all_ids: list[str] = []
    for section in ("papers", "code_sources", "datasets", "models"):
        items = manifest[section]
        all_ids.extend(item["id"] for item in items)
    assert len(all_ids) == len(set(all_ids)), "source IDs must be unique"

    approved = {"rethinking_opd", "vopd", "opd2", "opd2_multilingual", "opd_test_time_scaling"}
    actual_approved = {
        item["id"] for item in manifest["papers"]
        if item.get("approval_status") == "approved"
    }
    assert actual_approved == approved
    for item in manifest["code_sources"]:
        if item["repository"] is not None and item["revision"] is not None:
            assert len(item["revision"]) == 40
        if item["license"] is None:
            assert item["reuse"].startswith("clean-room")
    for item in manifest["datasets"]:
        assert len(item["revision"]) == 40
        assert item["parquet_bytes"] > 0
        if item["parquet_bytes"] > 100_000_000:
            assert item.get("requires_explicit_download_acceptance") is True
    for item in manifest["models"]:
        assert len(item["revision"]) == 40
        assert item["license"] == "Apache-2.0"
        assert item["approximate_bf16_weight_bytes"] == item["parameters"] * 2
    print(f"validated {len(all_ids)} source records and all approval/download gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

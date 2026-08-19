"""Fetch a dated, non-curating metadata snapshot for papers already in scope."""

from __future__ import annotations

import json
import os
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPOSITORY = Path(__file__).resolve().parents[1]


def main() -> int:
    sources = yaml.safe_load((REPOSITORY / "docs/sources.yml").read_text())
    papers = []
    for item in sources["papers"]:
        paper_id = item["arxiv"].split("v", 1)[0]
        url = f"https://huggingface.co/api/papers/{paper_id}"
        request = urllib.request.Request(url, headers={"User-Agent": "OPD-study-snapshot"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                metadata = json.load(response)
            lookup_status = "available"
            lookup_error = None
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            # HF Papers can lag arXiv. A missing enrichment record must remain visible,
            # but it must not remove a locked primary source or fail the weekly audit.
            metadata = {}
            lookup_status = "not-indexed"
            lookup_error = "HTTP 404 from Hugging Face Papers API"
        papers.append(
            {
                "source_id": item["id"],
                "locked_arxiv": item["arxiv"],
                "lookup_status": lookup_status,
                "lookup_error": lookup_error,
                "title": metadata.get("title"),
                "publishedAt": metadata.get("publishedAt"),
                "githubRepo": metadata.get("githubRepo"),
            }
        )
    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope_changes_automatically": False,
        "papers": papers,
    }
    output = REPOSITORY / "docs/research/literature-snapshot.json"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(snapshot, handle, ensure_ascii=False, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, output)
    print(f"wrote {output.relative_to(REPOSITORY)} with {len(papers)} locked papers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

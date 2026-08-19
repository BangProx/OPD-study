"""Check repository-relative Markdown links; optionally probe external HTTP links."""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-only", action="store_true")
    arguments = parser.parse_args()
    local_count = 0
    external: set[str] = set()
    failures: list[str] = []
    markdown_sources: list[tuple[Path, str]] = []
    for document in sorted(REPOSITORY.rglob("*.md")):
        relative_document = document.relative_to(REPOSITORY)
        if any(part.startswith(".") for part in relative_document.parts):
            continue
        if "artifacts" in relative_document.parts:
            continue
        markdown_sources.append((document, document.read_text(encoding="utf-8")))
    for notebook_path in sorted((REPOSITORY / "notebooks").rglob("*.ipynb")):
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        markdown_text = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        markdown_sources.append((notebook_path, markdown_text))
    for document, markdown_text in markdown_sources:
        for target in LINK.findall(markdown_text):
            target = target.strip().split()[0].strip("<>")
            if target.startswith(("http://", "https://")):
                external.add(target)
                continue
            if target.startswith("#"):
                continue
            relative = target.split("#", 1)[0]
            if not relative:
                continue
            local_count += 1
            if not (document.parent / relative).resolve().exists():
                failures.append(f"{document.relative_to(REPOSITORY)} -> {target}")
    if not arguments.local_only:
        for url in sorted(external):
            request = urllib.request.Request(
                url,
                method="HEAD",
                headers={"User-Agent": "OPD-study-link-check"},
            )
            try:
                with urllib.request.urlopen(request, timeout=15) as response:
                    if response.status >= 400:
                        failures.append(f"HTTP {response.status}: {url}")
            except urllib.error.HTTPError as error:
                if error.code not in {403, 405}:
                    failures.append(f"{url}: {error}")
                    continue
                fallback = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "OPD-study-link-check",
                        "Range": "bytes=0-0",
                    },
                )
                try:
                    with urllib.request.urlopen(fallback, timeout=15) as response:
                        response.read(1)
                except (urllib.error.URLError, TimeoutError) as fallback_error:
                    failures.append(f"{url}: {fallback_error}")
            except (urllib.error.URLError, TimeoutError) as error:
                failures.append(f"{url}: {error}")
    if failures:
        raise SystemExit("broken links:\n" + "\n".join(failures))
    external_count = 0 if arguments.local_only else len(external)
    print(f"checked {local_count} local links and {external_count} external links")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

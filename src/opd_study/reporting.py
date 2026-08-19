"""Atomic JSON, TensorBoard and static HTML/PNG reporting."""

from __future__ import annotations

import hashlib
import html
import json
import os
import platform
import subprocess
import sys
import tempfile
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNCOMMITTED"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".pt") as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path: Path, *, map_location: str = "cpu") -> dict[str, Any]:
    """Load only tensors/primitives, suppressing one harmless PyTorch 2.1 warning."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="TypedStorage is deprecated.*",
            category=UserWarning,
        )
        payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint root must be a dictionary")
    return payload


def environment_record() -> dict[str, object]:
    dependencies: dict[str, str] = {}
    for distribution in (
        "accelerate",
        "bitsandbytes",
        "datasets",
        "matplotlib",
        "peft",
        "PyYAML",
        "tensorboard",
        "torch",
        "transformers",
    ):
        try:
            dependencies[distribution] = version(distribution)
        except PackageNotFoundError:
            continue
    dependency_payload = json.dumps(
        dependencies, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ),
        "dependencies": dependencies,
        "dependency_fingerprint_sha256": hashlib.sha256(
            dependency_payload
        ).hexdigest(),
    }


def create_static_report(output_dir: Path, summary: dict[str, Any]) -> tuple[Path, Path, Path]:
    """Create a labeled, color-blind-safe loss plot and escaped standalone HTML."""

    cache_dir = output_dir / ".runtime-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    colors = {
        "sft": "#0072B2",
        "off_policy_kd": "#E69F00",
        "opd": "#009E73",
    }
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for name, run in summary["runs"].items():
        history = run.get("history", [])
        if not history:
            continue
        axis.plot(
            [row["step"] for row in history],
            [row["loss"] for row in history],
            marker="o",
            label=name,
            color=colors.get(name, "#CC79A7"),
        )
    axis.set(title="TinyArithmetic training loss", xlabel="Optimizer step", ylabel="Loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    plot_path = output_dir / "loss_curves.png"
    figure.savefig(plot_path, dpi=140)
    plt.close(figure)

    method_names = list(summary["runs"])
    diagnostic_names = (
        "student_entropy",
        "forward_kl_teacher_student",
        "reverse_kl_student_teacher",
    )
    diagnostic_labels = ("student entropy", "KL(teacher || student)", "KL(student || teacher)")
    diagnostic_figure, diagnostic_axes = plt.subplots(1, 3, figsize=(11.4, 3.8))
    for axis, metric_name, label in zip(
        diagnostic_axes, diagnostic_names, diagnostic_labels, strict=True
    ):
        values = [summary["runs"][name]["evaluation"][metric_name] for name in method_names]
        axis.bar(
            method_names,
            values,
            color=[colors.get(name, "#CC79A7") for name in method_names],
        )
        axis.set_title(label)
        axis.tick_params(axis="x", labelrotation=35)
        axis.grid(axis="y", alpha=0.25)
    diagnostic_figure.suptitle("Held-out distribution diagnostics")
    diagnostic_figure.tight_layout()
    diagnostic_path = output_dir / "distribution_diagnostics.png"
    diagnostic_figure.savefig(diagnostic_path, dpi=140)
    plt.close(diagnostic_figure)

    rows: list[str] = []
    for name, run in summary["runs"].items():
        evaluation = run["evaluation"]
        rows.append(
            "<tr>"
            f"<th>{html.escape(name)}</th>"
            f"<td>{evaluation['loss']:.4f}</td>"
            f"<td>{evaluation['exact_answer_accuracy']:.3f}</td>"
            f"<td>{evaluation['teacher_argmax_agreement']:.3f}</td>"
            f"<td>{evaluation['student_entropy']:.3f}</td>"
            f"<td>{evaluation['forward_kl_teacher_student']:.3f}</td>"
            f"<td>{evaluation['reverse_kl_student_teacher']:.3f}</td>"
            f"<td>{run['response_tokens']}</td>"
            "</tr>"
        )
    sample_rows: list[str] = []
    if method_names:
        first_samples = summary["runs"][method_names[0]]["evaluation"]["samples"]
        for sample_index, sample in enumerate(first_samples):
            responses = []
            for method_name in method_names:
                method_sample = summary["runs"][method_name]["evaluation"]["samples"][
                    sample_index
                ]
                response = str(method_sample["response"])
                prediction = method_sample["predicted_answer"]
                responses.append(
                    f"<dt>{html.escape(method_name)} → {html.escape(str(prediction))}</dt>"
                    f"<dd><pre>{html.escape(response or '[empty response]')}</pre></dd>"
                )
            sample_rows.append(
                "<section class='sample'>"
                f"<h3>{html.escape(str(sample['example_id']))}</h3>"
                f"<p><strong>Prompt:</strong> <code>{html.escape(str(sample['prompt']))}</code></p>"
                "<p><strong>Expected answer:</strong> "
                f"{html.escape(str(sample['expected_answer']))}</p>"
                f"<dl>{''.join(responses)}</dl></section>"
            )
    html_path = output_dir / "index.html"
    html_text = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>OPD-study mini playground</title>
<style>body{{font:16px system-ui;max-width:900px;margin:2rem auto;padding:0 1rem;line-height:1.5}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #bbb;padding:.5rem;text-align:right}}
th:first-child{{text-align:left}}img{{max-width:100%;height:auto}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f5f5;padding:.5rem}}
.sample{{border-top:1px solid #bbb;margin-top:1rem}}</style></head>
<body><h1>OPD-study mini playground</h1>
<p>This is an educational CPU run, not a claim about large-model benchmark quality.</p>
<img src="loss_curves.png" alt="Training loss curves labeled SFT, off-policy KD and OPD">
<table><caption>Evaluation on the same held-out prompts</caption><thead><tr>
<th>method</th><th>gold NLL</th><th>exact answer</th><th>teacher agreement</th>
<th>student entropy</th><th>forward KL</th><th>reverse KL</th><th>train tokens</th>
</tr></thead><tbody>{''.join(rows)}</tbody></table>
<img src="distribution_diagnostics.png"
alt="Three labeled charts: student entropy, forward KL, and reverse KL">
<h2>Side-by-side generated responses</h2>
<p>These are greedy generations on the same held-out prompts. Empty or malformed
responses are visible evidence, not silently repaired.</p>
{''.join(sample_rows)}
<p>Seed: {summary['seed']}; initial student SHA-256:
<code>{html.escape(summary['initial_student_hash'])}</code></p>
</body></html>"""
    html_path.write_text(html_text, encoding="utf-8")
    return html_path, plot_path, diagnostic_path

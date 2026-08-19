"""Build the installation-free Colab quickstart notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat

REPOSITORY = Path(__file__).resolve().parents[1]


def md(cell_id: str, role: str, source: str):
    return nbformat.v4.new_markdown_cell(
        source, metadata={"opd_study": {"cell_id": cell_id, "role": role}}
    )


def code(cell_id: str, role: str, source: str):
    return nbformat.v4.new_code_cell(
        source, metadata={"opd_study": {"cell_id": cell_id, "role": role}}
    )


def main() -> None:
    notebook = nbformat.v4.new_notebook(
        cells=[
            md(
                "COLAB-T00",
                "objective",
                "# OPD-study Colab quickstart\n\n"
                "Run a CPU-safe SFT/KD/OPD plumbing comparison, create checkpoints, "
                "a static report, and TensorBoard logs. A smoke run verifies the path; "
                "remove `smoke=True` for a longer toy experiment. No Qwen weights are downloaded.",
            ),
            code(
                "COLAB-S01",
                "demo",
                "import sys\nIN_COLAB = 'google.colab' in sys.modules\nprint({'in_colab': IN_COLAB, 'python': sys.version.split()[0]})",
            ),
            code(
                "COLAB-S02",
                "demo",
                "import subprocess\nfrom pathlib import Path\nif IN_COLAB:\n"
                "    repo_root = Path('/content/OPD-study')\n"
                "    if not (repo_root / '.git').exists():\n"
                "        subprocess.check_call(['git', 'clone', '--depth', '1',\n"
                "            'https://github.com/BangProx/OPD-study.git', str(repo_root)])\n"
                "    subprocess.check_call([sys.executable, '-m', 'pip', 'install',\n"
                "        '-e', str(repo_root)])\n"
                "else:\n"
                "    repo_root = Path.cwd()\n"
                "    if not (repo_root / 'src').exists():\n"
                "        repo_root = Path.cwd().parents[1]\n"
                "    print('Local validation: using repository src/; no network call.')\n"
                "# An editable install writes a .pth hook that this already-running Colab kernel\n"
                "# does not process until restart, so make the freshly cloned source importable now.\n"
                "sys.path.insert(0, str(repo_root / 'src'))",
            ),
            code(
                "COLAB-S03",
                "demo",
                "from opd_study.demo import run_demo\nprint('OPD-study imported')",
            ),
            code(
                "COLAB-P01",
                "demo",
                "output = Path('/content/opd-study-demo' if IN_COLAB else 'artifacts/colab-local')\n"
                "summary = run_demo(output, smoke=True, requested_device='cpu')\n"
                "print({name: {'loss': round(run['evaluation']['loss'], 4),\n"
                "              'tokens': run['response_tokens']}\n"
                "       for name, run in summary['runs'].items()})",
            ),
            code(
                "COLAB-C01",
                "check",
                "required = {'summary.json', 'experiment-card.json', 'index.html',\n"
                "            'loss_curves.png', 'distribution_diagnostics.png'}\n"
                "assert required.issubset({path.name for path in output.iterdir()})\n"
                "assert summary['fairness']['same_initial_student']\n"
                "print('Report:', output / 'index.html')\n"
                "print('TensorBoard:', output / 'tensorboard')",
            ),
            md(
                "COLAB-Q01",
                "optional",
                "## Optional CUDA LoRA and QLoRA smoke (manual opt-in)\n\n"
                "The default above downloads no model. The cell below stays disabled. "
                "Enabling it accepts the MIT GSM8K and Apache-2.0 Qwen terms and permits "
                "about 5.57GB of pinned weights plus 2.73MB of data. Use a CUDA runtime; "
                "each successful one-step update is a plumbing check, not a benchmark result. "
                "Run LoRA first, then QLoRA; both subprocesses reuse the same download cache.",
            ),
            code(
                "COLAB-L02",
                "optional",
                "RUN_OPTIONAL_QWEN_LORA = False\n"
                "lora_output = Path('/content/opd-study-qwen-lora')\n"
                "if RUN_OPTIONAL_QWEN_LORA:\n"
                "    import torch\n"
                "    if not torch.cuda.is_available():\n"
                "        raise RuntimeError('Select a CUDA runtime before LoRA.')\n"
                "    subprocess.check_call([sys.executable, '-m', 'pip', 'install',\n"
                "        '-e', f'{repo_root}[research]'])\n"
                "    subprocess.check_call([sys.executable, '-m', 'opd_study',\n"
                "        'research-train', '--config',\n"
                "        str(repo_root / 'configs/laptop/gsm8k_lora.yaml'),\n"
                "        '--cache', '/content/opd-study-cache', '--output', str(lora_output),\n"
                "        '--smoke', '--accept-dataset-license', '--accept-model-license'],\n"
                "        cwd=repo_root)\n"
                "else:\n"
                "    print('SKIPPED: set RUN_OPTIONAL_QWEN_LORA=True only after review.')",
            ),
            code(
                "COLAB-L03",
                "check",
                "if RUN_OPTIONAL_QWEN_LORA:\n"
                "    assert (lora_output / 'experiment-card.json').is_file()\n"
                "    assert (lora_output / 'metrics.jsonl').is_file()\n"
                "    assert (lora_output / 'adapter').is_dir()\n"
                "    assert (lora_output / 'checkpoints/optimizer.pt').is_file()\n"
                "    assert any((lora_output / 'tensorboard').iterdir())\n"
                "    print('LoRA update/save/eval/TensorBoard artifact contract passed.')\n"
                "else:\n"
                "    print('LoRA smoke remained disabled by default.')",
            ),
            code(
                "COLAB-Q02",
                "optional",
                "RUN_OPTIONAL_QWEN_QLORA = False\n"
                "qlora_output = Path('/content/opd-study-qwen-qlora')\n"
                "if RUN_OPTIONAL_QWEN_QLORA:\n"
                "    import torch\n"
                "    if not torch.cuda.is_available():\n"
                "        raise RuntimeError('Select a CUDA runtime before QLoRA.')\n"
                "    subprocess.check_call([sys.executable, '-m', 'pip', 'install',\n"
                "        '-e', f'{repo_root}[research,qlora]'])\n"
                "    subprocess.check_call([sys.executable, '-m', 'opd_study',\n"
                "        'research-train', '--config',\n"
                "        str(repo_root / 'configs/laptop/gsm8k_qlora.yaml'),\n"
                "        '--cache', '/content/opd-study-cache', '--output', str(qlora_output),\n"
                "        '--smoke', '--accept-dataset-license', '--accept-model-license'],\n"
                "        cwd=repo_root)\n"
                "else:\n"
                "    print('SKIPPED: set RUN_OPTIONAL_QWEN_QLORA=True only after review.')",
            ),
            code(
                "COLAB-Q03",
                "check",
                "if RUN_OPTIONAL_QWEN_QLORA:\n"
                "    assert (qlora_output / 'experiment-card.json').is_file()\n"
                "    assert (qlora_output / 'metrics.jsonl').is_file()\n"
                "    assert (qlora_output / 'adapter').is_dir()\n"
                "    assert (qlora_output / 'checkpoints/optimizer.pt').is_file()\n"
                "    assert any((qlora_output / 'tensorboard').iterdir())\n"
                "    print('QLoRA update/save/eval/TensorBoard artifact contract passed.')\n"
                "else:\n"
                "    print('QLoRA smoke remained disabled by default.')",
            ),
            code(
                "COLAB-Q04",
                "check",
                "if not RUN_OPTIONAL_QWEN_LORA and not RUN_OPTIONAL_QWEN_QLORA:\n"
                "    print('Default Colab path remained CPU-safe and model-download-free.')\n"
                "else:\n"
                "    print('Selected CUDA smoke paths completed.')",
            ),
            md(
                "COLAB-N01",
                "next",
                "## Next\n\nOpen the generated `index.html`, then continue with Korean or English "
                "Lesson 00. This notebook validates Colab only after the repository has "
                "been published; local stored output is not evidence of a hosted-runtime run.",
            ),
        ]
    )
    notebook.metadata.update(
        {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "opd_study": {
                "lesson_id": "COLAB-QUICKSTART",
                "language": "en",
                "track": ["quickstart"],
                "profile": "toy",
                "source_ids": ["gkd"],
                "schema_version": 1,
            },
            "colab": {"name": "OPD-study quickstart", "provenance": []},
        }
    )
    output = REPOSITORY / "notebooks/colab/quickstart.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    print(f"wrote {output.relative_to(REPOSITORY)}")


if __name__ == "__main__":
    main()

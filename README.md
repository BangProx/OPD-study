# OPD-study

[한국어](README.ko.md) · English

A notebook-first, bilingual course and reference implementation for **LLM
On-Policy Distillation (OPD)**. Start with distributions and SFT, build the original
GKD loop, then study modern sampled-token OPD, vOPD, OPD², TCOD, SOD, and SAGE-OPD.

The default path is a real PyTorch Transformer and generated arithmetic dataset that
runs offline on CPU. Pinned Qwen3/GSM8K research presets are isolated behind explicit
dependency, hardware, license, and download gates.

> [!NOTE]
> Paper-scale scores are not reproduced here. Every artifact says whether it was
> executed, which profile produced it, and what it cannot establish.

## 15-minute start

Requires Python 3.10–3.12.

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[notebooks]"
python -m opd_study.demo --smoke --device cpu --output artifacts/quickstart
```

Open `artifacts/quickstart/index.html`, or inspect the same run in TensorBoard:

```bash
python -m tensorboard.main --logdir artifacts/quickstart/tensorboard
```

The HTML report compares generated answers, held-out NLL/accuracy/agreement, entropy,
and both KL directions. Try a new expression directly or open the terminal playground:

```bash
python -m opd_study.demo --smoke --prompt "(2 + 3) * 4"
python -m opd_study.demo --smoke --interactive
```

Then open [Lesson 00](notebooks/en/00_opd_in_15_minutes.ipynb). For a meaningful
longer toy comparison, omit `--smoke`.

Windows PowerShell equivalent:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[notebooks]"
python -m opd_study.demo --smoke --device cpu --output artifacts/quickstart
```

No-install path: [open the Colab quickstart](https://colab.research.google.com/github/BangProx/OPD-study/blob/main/notebooks/colab/quickstart.ipynb).

## Course

| Lesson | Topic | Path |
|---:|---|---|
| 00 | OPD in 15 minutes and the full concept map | fast/full |
| 01 | Tokens, probabilities, autoregressive LMs | full |
| 02 | CE, entropy, KL, and KD | fast/full |
| 03 | SFT, off-policy, and on-policy state sources | fast/full |
| 04 | Original GKD: lambda mixing and generalized JSD | fast/full |
| 05 | Modern OPD from rollout to update | fast/full |
| 06 | Real train/eval, checkpoints, LoRA/QLoRA gates | full |
| 07 | Failure diagnostics, overlap, and vOPD | full |
| 08 | Multi-turn error compounding | full |
| 09 | TCOD F2B/B2F temporal curricula | full |
| 10 | SOD and SAGE-OPD | full |
| 11 | OPD², multilingual retention, avg@K/pass@K | fast/full |

All [12 English notebooks](notebooks/en) and [12 Korean notebooks](notebooks/ko) have
matching code, source IDs, checks, exercises, and stored clean executions. The
[curriculum specification](docs/design/curriculum.md) explains the 2–3 hour fast path
and 7–9 hour full path.

Every lesson exposes the bounded production function source it discusses, then explains
why that implementation was chosen, viable alternatives, and lesson-specific failure
notes. This keeps the notebooks self-contained without duplicating a second hidden
implementation.

## Train and evaluate from the CLI

```bash
python -m opd_study train --algorithm opd --smoke --device cpu --output artifacts/opd
python -m opd_study train --algorithm tcod_f2b --smoke --device cpu --output artifacts/tcod
python -m opd_study train --algorithm sod --smoke --device cpu --output artifacts/sod
python -m opd_study train --algorithm sage_opd --smoke --device cpu --output artifacts/sage
python -m opd_study eval --run artifacts/sage --rows 8
```

Available names: `sft`, `off_policy_kd`, `gkd`, `opd`, `vopd`, `opd2`, `tcod_f2b`,
`tcod_b2f`, `sod`, `sage_opd`.

## Real data and models

The actual pinned GSM8K shards total 2,725,633 bytes and require an explicit license
flag:

```bash
python -m opd_study download-data --dataset gsm8k \
  --cache artifacts/cache --accept-dataset-license
python -m opd_study gsm8k-smoke --cache artifacts/cache \
  --output artifacts/gsm8k-mini-smoke --accept-dataset-license
```

The mini smoke validates real rows and split discipline; it is not Qwen training. Check
the 5.57GB Qwen3-0.6B/1.7B LoRA preset before any model load:

```bash
python -m opd_study research-preflight --config configs/laptop/gsm8k_lora.yaml
```

The preset intentionally ships with model/download acceptance set to false. See the
[hardware guide](docs/hardware-guide.md) and [model compatibility table](docs/model-compatibility.md).

After reviewing those terms and confirming enough memory, the executable one-step
Qwen/GSM8K sampled-OPD path is:

```bash
python -m pip install -e ".[research]"
python -m opd_study research-train \
  --config configs/laptop/gsm8k_lora.yaml --smoke \
  --accept-dataset-license --accept-model-license
```

The `research` extra requires PyTorch 2.2+ and preflight verifies package imports before
any download. Core toy lessons remain compatible with PyTorch 2.1+.

The two acceptance flags authorize roughly 5.57GB of pinned model weights plus the
2.73MB dataset. The local macOS host is CPU-only, while one LoRA update was
**EXECUTED** on a hosted Colab T4. It is a wiring check, not a benchmark result.

On a validated NVIDIA runtime, select the checked QLoRA preset instead:

```bash
python -m opd_study research-train \
  --config configs/laptop/gsm8k_qlora.yaml --smoke \
  --accept-dataset-license --accept-model-license
```

It never falls back to full fine-tuning. The Colab notebook contains the same command
behind `RUN_OPTIONAL_QWEN_QLORA = False`. One QLoRA update was also **EXECUTED** on
the same hosted T4. See the exact revisions, environment, metrics and limitations in
the [CUDA smoke evidence](docs/research/colab-cuda-smoke-2026-08-19.json).

## Fidelity and educational simplifications

- Original GKD math follows arXiv:2306.13649v3; beta boundary behavior was
  cross-checked against pinned Apache-2.0 TRL behavior without copying source.
- vOPD, SAGE-OPD, and unlicensed-source diagnostics are clean-room implementations
  from paper equations.
- TCOD and OPD² preserve the paper's core temporal/delta semantics. The mini backend
  replaces distributed rollout infrastructure with a synchronous toy collector.
- SOD exposes its step-weighted distillation term; the paper's separate GRPO term is
  deliberately absent from core and labeled in metrics.
- Mini SAGE uses a documented token-agreement proxy judge. Research mode must supply
  environment failures and a semantic teacher judgment query.

Read [math conventions](docs/math.md), [implementation notes](docs/implementation-notes.md),
and the machine-readable [source manifest](docs/sources.yml) before extending an
algorithm.

## Quality checks

```bash
python -m pip install -e ".[dev,notebooks]"
python scripts/quality_gate.py
# Full local rerun, including all 24 notebook kernels:
python scripts/quality_gate.py --execute-notebooks
```

CI runs core tests on Linux, macOS, and Windows. Scheduled CI executes all notebooks
and checks external links/source drift. See [results and verification status](docs/results.md).

## Contributing and safety

Start with [CONTRIBUTING.md](CONTRIBUTING.md). Report vulnerabilities through
[SECURITY.md](SECURITY.md), not a public issue. Downloaded weights, datasets, keys, and
run artifacts are ignored by Git.

## License

Project code and original course material are Apache-2.0. Papers, datasets, models,
and referenced upstream projects retain their own terms. See [LICENSE](LICENSE),
[NOTICE](NOTICE), and [third-party notices](docs/third-party-notices.md).

# Executed results and verification status

> Date: 2026-08-19 Asia/Seoul. Artifacts are generated and git-ignored; commands below
> are the durable reproduction record. No benchmark score is inferred from a smoke run.

## CPU evidence on the current host

| Run | Actual scope | Result |
|---|---|---|
| test suite | math, masks, data/model, algorithms, CLI, resume, artifacts, research gates | 35 tests passed |
| bilingual notebooks | 12 Korean + 12 English | all 24 executed top-to-bottom; parity passed |
| `demo --smoke` | exact tiny teacher/student; no-train/SFT/off-policy KD/OPD | safe checkpoints, JSON/JSONL, 2 PNGs, response-comparison HTML, TensorBoard produced |
| `train sage_opd --smoke` + `eval` | multi-turn collector and mini proxy judge | checkpoint/report/evaluation produced; one-step accuracy 0 as expected |
| GSM8K mini smoke | pinned actual Parquet; tiny SFT plumbing | 6961/512/1319 split validated; one update; test unused |

The recorded GSM8K validation NLL was 4.3742 after one deliberately tiny update. It is
not an accuracy or convergence claim. Exact values and environment are in the generated
`artifacts/gsm8k-mini-smoke/experiment-card.json` after rerunning the documented command.

The demo report shows the same held-out prompts and raw responses side by side, plus
student/teacher entropy and both named KL directions. `--prompt` and explicit
`--interactive` modes reconstruct locally generated weights-only-safe checkpoints;
invalid expression characters are rejected.

## Not executed

- Qwen3-0.6B/1.7B LoRA: executable sampled-OPD runner exists; the research dependency
  set passed real imports and the complete quality gate in an isolated environment.
  The 5.57GB acceptance remains false, so no model load/update was attempted.
- QLoRA: current host has no CUDA; macOS/CPU is blocked.
- Qwen3-4B server/SOD distributed run: no multi-GPU hardware provided.
- Fresh hosted Colab run and 3-OS GitHub Actions: workflows/notebook are prepared, but
  evidence requires a published commit and remote CI execution.
- Paper-scale TCOD/SOD/SAGE/OPD² benchmark tables.

These items remain `UNVERIFIED`, not failures and not fabricated successes.

The checked `configs/laptop/gsm8k_qlora.yaml` and default-disabled Colab cell provide
the exact CUDA command. Local headless execution proves only that the default path makes
no Qwen download; it does not count as the optional GPU cell having run.

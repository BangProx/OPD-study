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
| hosted Colab default | published commit `372c19f`; T4 runtime, CPU-safe toy mode | clone/import, fair comparison, HTML/PNG/TensorBoard artifacts passed |
| Qwen LoRA smoke | Qwen3-0.6B student + Qwen3-1.7B teacher; pinned GSM8K; Tesla T4 | sampled reverse-KL update, adapter/checkpoint/eval/TensorBoard contract passed |
| Qwen QLoRA smoke | same pinned pair/cache; NF4 student; Tesla T4 | quantized load, forward/backward/update/save/eval/TensorBoard contract passed |

The recorded GSM8K validation NLL was 4.3742 after one deliberately tiny update. It is
not an accuracy or convergence claim. Exact values and environment are in the generated
`artifacts/gsm8k-mini-smoke/experiment-card.json` after rerunning the documented command.

The demo report shows the same held-out prompts and raw responses side by side, plus
student/teacher entropy and both named KL directions. `--prompt` and explicit
`--interactive` modes reconstruct locally generated weights-only-safe checkpoints;
invalid expression characters are rejected.

## Hosted CUDA evidence

The hosted notebook was loaded from exact commit `372c19fec5a4609663348c58217e79f097840143`
on a fresh Colab Tesla T4 runtime. The default path remained model-download-free. The
opt-in LoRA and QLoRA cells then ran one update each and printed their artifact-contract
success messages plus `Selected CUDA smoke paths completed.`

| Mode | Loss | Gradient norm | Response tokens | Peak CUDA allocation | Wall time |
|---|---:|---:|---:|---:|---:|
| LoRA | 0.0026496 | 0.1956 | 16 | 5,026,640,384B | 95.30s |
| QLoRA | -0.7205186 | 26.6793 | 16 | 4,736,268,800B | 41.86s |

Both one-row validation checks scored 0.0 exact-answer accuracy. That deliberately
visible failure reinforces that a one-step smoke validates wiring, not capability.
Exact package versions, revisions, shard checksums, config hashes and limitations are
recorded in
[`colab-cuda-smoke-2026-08-19.json`](research/colab-cuda-smoke-2026-08-19.json).
The 2026-08-19 Colab image also exposed an optional-package conflict: preinstalled
`torchao 0.10.0` was incompatible with Transformers 4.57.6. OPD-study does not use
torchao, so the opt-in cells now remove it only when its version is below 0.16.

## Still not executed

- Qwen3-4B server/SOD distributed run: no multi-GPU hardware provided.
- Paper-scale TCOD/SOD/SAGE/OPD² benchmark tables.

These items remain `UNVERIFIED`, not failures and not fabricated successes.

The checked laptop configs and default-disabled Colab cells remain opt-in despite the
successful smoke. Neither the one-step CUDA result nor the mini backend should be cited
as a paper-scale reproduction.

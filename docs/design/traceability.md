# Requirement traceability matrix

Status values: `designed`, `implemented`, `verified`, `external-manual`. This current audit
links each claim to checked-in, executable evidence; generated artifacts stay ignored.

| Req | User/PLAN requirement | Evidence | Primary files/tests | Current status |
|---:|---|---|---|---|
| 1 | Korean/English courses match | 12+12 code/source/check parity; bounded production source, design alternatives, unique exercises/mistakes | `scripts/check_notebooks.py` | verified |
| 2 | notebooks run top-to-bottom | 24 stored clean executions with no traceback/warning/local path, rerun on 2026-08-19 | `scripts/execute_notebooks.py`, notebook CI | verified |
| 3 | CPU toy fair SFT/OPD comparison | same init, token budget, eval, checkpoints/reports | `tests/test_training_and_artifacts.py`, `opd_study.demo` | verified |
| 4 | CLI train/eval for SFT, OPD, TCOD, SOD, SAGE | all required methods train→resume→eval | `tests/test_training_and_artifacts.py`, `experiment.py` | verified |
| 5 | original GKD vs modern OPD distinction | separate state/estimator APIs and analytic tests | L04/L05, `tests/test_math.py`, `test_model_and_losses.py` | verified |
| 6 | laptop and server presets | pinned configs, guarded real runner, honest status | `configs/`, `research/`, hardware guide | implemented |
| 7 | source traceability | 24 unique records and notebook source IDs | `docs/sources.yml`, `scripts/check_sources.py` | verified |
| 8 | compatible licenses and attribution | Apache clean-room policy and source license gates | `LICENSE`, `NOTICE`, third-party notices | verified |
| 9 | test/lint/type/notebook/link/parity quality | hermetic venv plus passing Linux/macOS/Windows × Python 3.10/3.12 CI; nightly reruns 24 notebooks/source/link/snapshot | `scripts/quality_gate.py`, `.github/workflows/ci.yml`, `PROGRESS.md` | verified |
| 10 | discoverable install/quickstart/paths/hardware/help | bilingual entrypoints and local link audit | `README.md`, `README.ko.md`, docs | verified |
| 11 | path/repo/origin match | origin verified; final workspace-safe rename pending | `PROGRESS.md` C0/C9 | implemented |
| 12 | honest experiment record | env/dependency/config hashes, memory/time, limits | `reporting.py`, generated cards, artifact test | verified |
| 13 | toy and GSM8K actual train; dataset metadata | deterministic toy and pinned real-shard mini train | `test_data.py`, `test_research_gates.py`, results | verified |
| 14 | `python -m opd_study.demo` report/playground/TB | raw response matrix, entropy/FKL/RKL, prompt/interactive mode, safe checkpoints, HTML/2×PNG/JSON/JSONL/TB assertions | `tests/test_training_and_artifacts.py` | verified |
| 15 | Linux/macOS/Windows CPU + Colab; device guards | 3-OS remote CI; hosted T4 default, LoRA and QLoRA runs; incompatible optional torchao guard | workflows, Colab notebook, CUDA evidence, hardware guide | verified |

## G1–G8 mapping

| ID | Requirement | Lesson | Runtime/document evidence |
|---|---|---|---|
| G1 | dataset license/size/split | L06 | `docs/sources.yml`, guarded loader tests |
| G2 | SFT baseline | L03/L05 | fairness contract and comparison integration test |
| G3 | visible learner result | L00/L05/L11 | demo HTML/PNG/JSON + TensorBoard |
| G4 | fixed toy/model/data | L00–L06 | configs and deterministic generation/model tests |
| G5 | Windows/Colab | L00/L06 | 3-OS CI, pathlib tests, Colab execution manifest |
| G6 | concept map | L00 and every notebook | notebook schema/parity checker |
| G7 | recurring mistakes | every lesson | mistake-note count/format checker |
| G8 | LoRA/QLoRA accessibility | L06 | capability probe, no unsafe fallback tests |

## Algorithm detail coverage

| Detail | Explanation | Executable evidence |
|---|---|---|
| sampled prefix treated as state | L04/L05 | rollout tensors detached; update logits recomputed |
| sampling non-differentiability | L05 | rollout has no grad; student update does |
| temperature/normalization | L02/L04 | analytic distributions and temperature tests |
| FKL/RKL/JSD directions | L02/L04 | known-value unit tests and argument-swap mistake note |
| full vs sampled estimator | L05/L07 | expectation/variance toy experiment |
| prompt/pad/EOS/tool masks | L03/L05/L08 | mask truth tables and tests |
| reduction/length normalization | L02/L05/L10 | masked reduction and SAGE normalization tests |
| teacher eval/no_grad | L05 | invariant teacher parameter/grad test |
| current/old policy and stale rollouts | L05/L07 | policy version and resume checks |
| tokenizer/template compatibility | L06 | mismatch fails before model forward |
| top-k approximation | L06/L07 | mass/error metrics and edge-case tests |
| mixed-precision stability | L06 | float32 log-softmax and finite probes |
| collapse/KL/support diagnostics | L07 | synthetic failure scenarios |
| seed/checkpoint/resume | L03/L05/L06 | deterministic split and resume integration test |

## Approval record

- 2026-08-19: A1–A4, A7–A8 and Apache-2.0 approved; Kaggle and MkDocs deferred.
- 2026-08-19: Rethinking OPD, vOPD, OPD²+multilingual and test-time scaling
  analysis all approved. The first/last are required diagnostics/evaluation; vOPD and
  OPD² are advanced implementations within the existing 12 lessons.
- 2026-08-19: preserving remote `6f1e395` as parent and fast-forward publishing approved;
  free Colab CUDA plus pinned Qwen3-0.6B/1.7B LoRA and QLoRA smoke approved.

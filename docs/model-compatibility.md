# Model and tokenizer compatibility

| Profile | Student | Teacher | Finetuning | Local status |
|---|---|---|---|---|
| toy | 2×64 tiny Transformer | 4×128 tiny Transformer | full | **EXECUTED CPU** |
| laptop | Qwen3-0.6B @ `c1899de…` | Qwen3-1.7B @ `70d244c…` | LoRA | **EXECUTED** one-step on Colab T4 |
| laptop CUDA | same | same | QLoRA | **EXECUTED** one-step on Colab T4 |
| server SOD-like | Qwen3-0.6B | Qwen3-4B @ `1cfa9a7…` | LoRA/distributed adapter | schema only; **NOT EXECUTED** |

Qwen variants were chosen because they share a model family, but the loader still
compares the complete vocabulary and chat template. A mismatch fails before training.
Cross-tokenizer distillation, API-only teachers without full log-probabilities, and
silent model substitutions are unsupported.

Exact 40-character revisions, approximate BF16 bytes, parameter counts, and licenses
are in [`sources.yml`](sources.yml). Model weights are never committed.

## Download consent

The laptop pair is approximately 5.57GB in BF16 before optimizer, activation, KV-cache,
and framework overhead. Presets ship with `accept_model_license: false`; changing that
field is an explicit decision after running `research-preflight`. More than 100MB is
never downloaded as a hidden fallback.

The executable research runner currently accepts only GSM8K, `algorithm=opd`, one
device, and batch size 1. TCOD/SOD/SAGE remain executable in the mini backend; their
paper-scale distributed Qwen recipes are not mislabeled as laptop support. A successful
run saves the PEFT adapter, safe optimizer/RNG checkpoint, validation samples, experiment
card, JSONL metrics, and TensorBoard events when the config enables them.

The core package supports PyTorch 2.1+, while the `research` extra requires PyTorch
2.2+ because current Transformers/PEFT releases use newer PyTorch APIs. The dependency
set was import- and API-checked in an isolated Python 3.12 environment with PyTorch
2.13.0. A separate hosted Colab T4 run executed both adapters with Python 3.12.13,
PyTorch 2.11.0+cu128, Transformers 4.57.6, PEFT 0.20.0, Datasets 4.0.0,
Accelerate 1.14.0, and bitsandbytes 0.50.1 for QLoRA. The Colab image's unrelated
`torchao 0.10.0` conflicts with current Transformers; the opt-in notebook removes only
that incompatible optional version because this project does not use torchao. See the
checked [CUDA smoke evidence](research/colab-cuda-smoke-2026-08-19.json). These one-step
runs verify wiring, not model quality or convergence.

# Hardware guide

## Safe default

`--device auto` selects CUDA, then MPS, then CPU. An explicitly requested unavailable
device raises an error. CPU fallback happens only with `--allow-device-fallback`.

| Platform | Toy | LoRA research | QLoRA |
|---|---|---|---|
| Linux CPU | supported/CI | possible but impractical for pinned pair | blocked |
| Linux + NVIDIA CUDA | supported | intended | intended after bitsandbytes update probe |
| Windows CPU | supported/CI | possible but impractical | blocked |
| Windows + NVIDIA CUDA | supported target | intended | requires CI/manual CUDA evidence |
| macOS CPU/MPS | CPU supported; MPS capability-probed | LoRA only if memory permits | blocked; no full-FT fallback |
| Colab | CPU toy quickstart | GPU LoRA opt-in | only after runtime probe |

Current local evidence (2026-08-19): macOS arm64, Python 3.10.12, PyTorch 2.1.0,
CPU selected; CUDA and MPS unavailable. Qwen training and QLoRA were not executed.
An isolated Python 3.12/PyTorch 2.13.0 environment passed real imports for the pinned
`research` dependency ranges and the full quality gate. The research extra deliberately
requires PyTorch 2.2+; preflight reports version and import incompatibilities before any
dataset or model download.

## Commands

```bash
python -m opd_study doctor
python -m opd_study research-preflight --config configs/laptop/gsm8k_lora.yaml
python -m opd_study research-train \
  --config configs/laptop/gsm8k_lora.yaml --smoke \
  --accept-dataset-license --accept-model-license
python -m opd_study research-train \
  --config configs/laptop/gsm8k_qlora.yaml --smoke \
  --accept-dataset-license --accept-model-license
```

`research-train` is currently a single-device, batch-size-1 sampled reverse-KL OPD
runner. It saves a PEFT adapter, optimizer/RNG checkpoint, metrics, validation samples,
environment, memory counters, TensorBoard events when enabled, and an experiment card.
The checked server SOD config is
still a schema/documentation target, not a working distributed launcher.

Do not treat `bitsandbytes` import success as enough. A QLoRA-supported status requires
an actual quantized load, forward, backward, optimizer update, save, and reload on the
target environment. A failed probe never switches to full fine-tuning.

Paths use `pathlib`; code assumes neither `/tmp` nor symlinks, fork, POSIX signals, or
bash quoting. README lists separate virtual-environment activation commands for Windows.

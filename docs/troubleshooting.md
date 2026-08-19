# Troubleshooting

## Requested CUDA/MPS is unavailable

Run `python -m opd_study doctor`. Choose `--device cpu`, or use
`--allow-device-fallback` only when slower CPU execution is acceptable. The error is a
safety feature, not a detection bug by default.

## QLoRA is blocked on macOS or CPU

This project validates QLoRA only on NVIDIA CUDA with bitsandbytes. Use LoRA, a CUDA
host, or the toy path. It will not silently allocate full model/optimizer weights.

## Colab reports an incompatible torchao version

The 2026-08-19 Colab image included optional `torchao 0.10.0`, while its current
Transformers build required at least 0.16. OPD-study does not use torchao. The hosted
notebook's opt-in LoRA/QLoRA cells therefore remove only an installed version below
0.16 after installing the research dependencies. The default path does not alter it.
If the error appears outside Colab, inspect the environment instead of copying that
Colab-specific action blindly.

## Dataset download refuses to start

Read `docs/sources.yml`, confirm the revision/size/license, then pass
`--accept-dataset-license`. A `.partial` file is not accepted as a cache hit; final size
and SHA-256 must match.

## Teacher and student tokenizer mismatch

Full-distribution KD needs matching vocabulary IDs and chat templates. Select the pinned
same-family preset. Cross-tokenizer projection is intentionally outside this course.

## Loss is NaN or an empty-mask error appears

Inspect `response_mask.sum()`, EOS placement, and padding. The reducer rejects zero
effective tokens. Teacher signals should be detached and log-softmax should run in
float32.

## Notebook cannot import `opd_study`

Launch Jupyter from the repository root after `pip install -e ".[notebooks]"`. The
checked-in notebook setup also finds `src/` when executed from the repository root.

## SFT beats OPD

Keep the result. Audit same initialization, prompt IDs, update/token budget, teacher
quality, support overlap, and rollout depth. OPD winning is not a release criterion.

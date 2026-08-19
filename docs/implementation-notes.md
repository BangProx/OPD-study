# Implementation fidelity and alternatives

## Decision table

| Method | Evidence | This repository | Deliberate alternative or omission |
|---|---|---|---|
| GKD | arXiv:2306.13649v3; TRL commit `1e3ba4e…` behavior | clean equation-shaped PyTorch; lambda state mix; generalized JSD | no distributed Trainer dependency |
| modern OPD | sampled reverse-KL policy-gradient formulation | sampled and exact full-vocabulary objectives | full-V is the toy default for stability, sampled remains explicit |
| vOPD | arXiv:2605.07865v1 Eqs. 9–15 | detached full/top-k RKL control variate | no unavailable/unlicensed upstream code copied |
| OPD² | arXiv:2607.15161v1; NAVER commit `2dac53…` | centered delta and joint-sign gate | synchronous mini rollout; no 8×H100 claim |
| TCOD | arXiv:2604.24005v3; commit `465eef…` | pacing, F2B states, teacher-prefix B2F states | no async actor/learner or replay infrastructure |
| SOD | arXiv:2605.07725v3; commit `110c4…` predates v3 | sampled gap, detached step weights, weighted KL | GRPO term omitted from core and labeled `distillation_only` |
| SAGE-OPD | arXiv:2606.19659v1; no official repo found | intervention × confidence × normalization | mini judge is a labeled agreement proxy; research needs semantic judge |

No upstream source file has been copied. Apache repositories were used for behavioral
and semantic comparison; paper equations were independently expressed for a small
backend. The unlicensed THUNLP repository is never a code source.

The optional Hugging Face runner implements only single-device GSM8K sampled reverse-KL
OPD. It uses the pinned tokenizer family, saves a PEFT adapter and optimizer/RNG state,
and evaluates only the validation split. It is an executable laptop sanity path, not an
implementation of the distributed TCOD/SOD/SAGE training stacks.

## Why a character tokenizer?

Teacher and student must share vocabulary and state boundaries for full-distribution
KD. A fixed repository-owned character vocabulary makes every ID visible, removes
download/template drift, and supports Windows/macOS/Linux offline tests. It is not a
recommendation for production LLMs. Research presets instead pin one Qwen family and
fail if vocabulary or chat templates differ.

## Why use full reverse KL in the default toy OPD?

The tiny vocabulary makes its exact conditional expectation cheap and low variance.
Lesson 05 and vOPD expose sampled estimators separately, so learners can compare the
gradient/cost tradeoff without conflating state source with estimator choice.

## Teacher and rollout invariants

- Teacher calls use `eval()` and `no_grad()`, then restore the previous mode.
- Teacher signals and rollout-time student log-probabilities reject gradients.
- Update-time student logits are recomputed after collection.
- Prompt, padding, separator, and environment tokens are excluded from response loss.
- B2F collection uses a teacher prefix; stale policy versions belong in future async
  adapters, not silently in the synchronous mini path.

## SOD v3/code audit limitation

The pinned official code predates the 2026-08-18 paper v3. We therefore treat the v3
paper equations as normative and use the repository only as an Apache-licensed semantic
cross-check. A future upstream release must be diffed before changing defaults.

# Technical architecture contract

> This document freezes C2 behavior. Later implementation may change internals, but a
> contract change needs a rationale, migration note and tests.

## Backends and dependency boundary

`opd_study` has two explicit backends:

- `mini`: pure PyTorch, offline toy data, one process, CPU/MPS/CUDA where available.
  It owns the educational reference implementation and all default notebooks/CI.
- `research`: optional Transformers/PEFT/Accelerate adapters and server recipes. It may
  depend on CUDA-only upstream stacks, but importing `opd_study` never imports them.

Algorithms depend on typed batches and model protocols, never on a server framework.
Profiles select adapters; algorithm names do not silently select hardware.

```text
CLI / notebooks
  → validated ExperimentConfig
  → data + model + device preflight
  → DistillationAlgorithm
      collect → score → loss → metrics
  → Trainer (budget audit, checkpoint, logging)
  → Evaluator (task + distillation + K-scaling metrics)
  → JSON / PNG / HTML / TensorBoard artifacts
```

## Public data contracts

```python
@dataclass(frozen=True)
class TrajectoryBatch:
    token_ids: Tensor          # int64 [B, T]
    attention_mask: Tensor     # bool  [B, T]
    response_mask: Tensor      # bool  [B, T]
    prompt_lengths: Tensor     # int64 [B]
    student_logprobs: Tensor | None  # float [B, T], rollout-time detached
    turn_ids: Tensor | None    # int64 [B, T], -1 outside turns
    step_ids: Tensor | None    # int64 [B, T], -1 outside reasoning steps
    terminal: Tensor | None    # bool [B]

@dataclass(frozen=True)
class TeacherSignals:
    logits: Tensor | None      # float [B, T, V], always detached
    logprobs: Tensor | None    # float [B, T, V] or selected [B, T]
    confidence: Tensor | None  # float [B, T] or [B, turns]
    intervene: Tensor | None   # bool [B, turns]

@dataclass(frozen=True)
class LossOutput:
    loss: Tensor               # scalar, student-grad enabled
    token_loss: Tensor         # float [B, T]
    effective_mask: Tensor     # bool/float [B, T]
    metrics: Mapping[str, float]
```

Validation rejects mismatched rank, device, vocabulary, non-prefix attention, response
tokens outside attention, negative prompt lengths, or turn/step IDs in padding. Teacher
parameters and all `TeacherSignals` are gradient-free. Rollout-time log-probabilities are
detached snapshots; update-time student logits are recomputed unless an estimator
explicitly requires an old policy.

## Algorithm interface

```python
class DistillationAlgorithm(Protocol):
    name: str
    def collect(self, student, prompts, *, generator) -> TrajectoryBatch: ...
    def score(self, teacher, trajectories) -> TeacherSignals: ...
    def loss(self, student, trajectories, signals) -> LossOutput: ...
    def metrics(self, output: LossOutput) -> dict[str, float]: ...
```

Built-in registry keys:

| Key | State source | Loss/weight | Required mode |
|---|---|---|---|
| `sft` | fixed demonstration | hard token CE | single-turn |
| `off_policy_kd` | fixed demonstration | teacher distribution divergence | single-turn |
| `gkd` | lambda mixture | generalized JSD/full distribution | single-turn |
| `opd` | current student | sampled-token/reverse-KL choices | single-turn |
| `vopd` | current student | detached control-variate advantage | single-turn |
| `opd2` | current student | teacher-minus-teacher-base delta | single-turn |
| `tcod_f2b`, `tcod_b2f` | student multi-turn | curriculum slice + OPD | multi-turn |
| `sod` | student multi-turn | detached step-divergence weights | multi-turn |
| `sage_opd` | student multi-turn | intervention × confidence × normalized loss | multi-turn |

## Loss conventions

Course convention:

- forward KL: `KL(p_teacher || p_student)`;
- reverse KL: `KL(p_student || p_teacher)`;
- `JSD_beta(p_teacher, p_student)` uses mixture
  `m = beta * p_teacher + (1-beta) * p_student` and
  `beta * KL(p_teacher || m) + (1-beta) * KL(p_student || m)`.

Every implementation names both distributions. `masked_mean` divides by the effective
mask sum, not padded length. Empty response masks fail with an explanatory error. Log
softmax and reductions use float32 even when model matmuls use a lower precision.

## Configuration schema

Configuration is parsed into frozen dataclasses; unknown keys are errors. YAML is only a
serialization format. Environment variables never silently override experiment values.

```yaml
schema_version: 1
profile: toy                 # toy | laptop | server
backend: mini                # mini | research
algorithm:
  name: opd
  divergence: reverse_kl
  lambda_on_policy: 1.0
  beta_jsd: 0.5
  temperature: 1.0
data:
  id: tiny_arithmetic
  revision: generated-v1
  config: null
  license: generated/Apache-2.0
  expected_download_bytes: 0
  accept_dataset_license: false
  seed: 42
  train_rows: 4096
  validation_rows: 512
  test_rows: 512
model:
  student: tiny-student
  teacher: tiny-teacher
  student_revision: null
  teacher_revision: null
  teacher_base: null
  finetuning: full
  trust_remote_code: false
  expected_download_bytes: 0
  accept_model_license: false
training:
  seed: 42
  steps: 20
  batch_size: 8
  tokens_per_step: 64
  learning_rate: 0.0003
  device: auto
  allow_device_fallback: false
  precision: float32
evaluation:
  rows: 16
  max_new_tokens: 64
  k_values: [1, 2, 4, 8]
output:
  root: artifacts/demo
  tensorboard: true
```

Profile-specific config may only tighten/replace values declared in the schema. Laptop
and server configs additionally require full model/dataset revisions and explicit
dataset/model acceptance flags before network access.

## Fair-comparison contract

SFT, off-policy KD and OPD comparisons share:

1. a serialized initial student state hash;
2. prompt IDs and train/eval split hashes;
3. optimizer class, hyperparameters and scheduler;
4. maximum optimizer steps and response-token budget;
5. evaluation prompts, decoding settings and metric definitions;
6. seed set and checkpoint cadence.

The trainer records actual optimizer steps, sampled/fixed response tokens, examples,
wall-clock and peak memory. A budget auditor marks a comparison invalid instead of
normalizing after the fact. SFT winning is a valid result and is retained.

## Profiles and compatibility

| Profile | Default data/models | Network | Expected validation |
|---|---|---|---|
| toy | generated TinyArithmetic + tiny Transformers | forbidden | all algorithms CPU smoke, notebooks, CI |
| laptop | pinned GSM8K + Qwen3 0.6B/1.7B | explicit opt-in | preflight; LoRA; QLoRA only after real update probe |
| server | MATH/OpenR1 + Qwen3 0.6B/4B or config override | explicit opt-in | recipe/schema tests; hardware results only when run |

Tokenizer vocabulary and chat-template identity are required for full-distribution KD.
Cross-tokenizer distillation is outside the current algorithm track and fails fast.
Requested device fallback happens only with `allow_device_fallback=true`. Failed QLoRA
never falls back to full fine-tuning.

## Checkpoint and artifact contract

- Checkpoints contain config, model/optimizer state, RNG states, global step,
  processed token counts and provenance hashes.
- Scheduler state is required only when a scheduler is configured; current mini and
  research smoke loops use constant learning rates and therefore have none to save.
- Resume validates schema and immutable fairness fields before loading.
- Each run writes `summary.json`, `metrics.jsonl`, an experiment card and optional
  TensorBoard events. The demo additionally writes `index.html` and PNG charts.
- Paths use `pathlib`; temporary files use `tempfile`; writes are atomic within one
  filesystem. No symlink, POSIX signal or fork is required.

## Security and download contract

- Core runs execute no downloaded code (`trust_remote_code=false`).
- Asset metadata is shown before download. More than 100MB needs a CLI flag or an
  already-cached, checksum-verified asset.
- Dataset test splits are read-only evaluation inputs.
- Generated reports escape prompts/completions before HTML rendering.
- No API keys, credentials, model weights or datasets are committed.

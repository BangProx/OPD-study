# Notebook authoring and parity contract

## Mode and audience

All course notebooks are tutorials for first-time distillation learners. They are
reader-facing artifacts, not development scratchpads. Korean is reviewed first; English
mirrors concepts and code without mechanical word-for-word constraints.

## Required top-to-bottom skeleton

Each notebook contains these level-two sections in this exact order:

1. `## Goal`
2. `## Setup`
3. `## Steps`
4. `## Checks`
5. `## 내가 자주 틀리는 것` / `## My recurring mistakes`
6. `## 60초 요약` / `## 60-second summary`
7. `## Next Steps`

`Goal` includes at most three objectives, estimated time, path label and current map
position. `Setup` prints versions, device, seed, profile and network requirement before
work begins. `Checks` must execute assertions, not only state expected answers. Every
lesson includes bounded inspection of the production function being taught, an
implementation-choice explanation, alternatives/trade-offs, a lesson-specific
exercise, and two lesson-specific recurring mistakes.

## Stable metadata

Notebook metadata:

```json
{
  "opd_study": {
    "lesson_id": "L05",
    "language": "ko",
    "track": ["fast", "full"],
    "profile": "toy",
    "source_ids": ["gkd"],
    "schema_version": 1
  }
}
```

Cell metadata contains a stable ID and role:

```json
{"opd_study": {"cell_id": "L05-S03-C02", "role": "demo"}}
```

Allowed roles are `objective`, `map`, `explain`, `predict`, `demo`, `check`,
`exercise`, `solution`, `mistake-note`, `summary`, `source`, and `next`. Code cell IDs
and order must match across languages. Markdown IDs match by role and section, while
translated text may differ.

## Code and output rules

- One code cell performs one conceptual action and normally stays under 25 lines.
- Imports and seed/device selection occur once in Setup through package helpers.
- Variables are descriptive English identifiers. User-facing strings may be localized.
- Outputs are bounded: tables ≤20 rows, no full model repr, no progress-bar spam.
- Charts use the Okabe–Ito color-blind-safe palette plus line style and direct labels.
- Every figure has alt text in nearby Markdown and a plain-text result summary.
- Network/server cells are tagged `optional-network` or `server-only`, disabled by
  default, and have a deterministic toy fallback.
- A clean execution stores no traceback and updates `execution_count` monotonically.

## Required mistake-note format

```markdown
### M2 — prompt까지 loss에 넣기

- 틀린 형태: `loss = nll(logits, all_tokens)`
- 왜 틀렸나: prompt token 수가 긴 sample이 distillation budget을 지배한다.
- 고친 형태: `masked_mean(token_loss, response_mask)`
- 관련 검사: `test_response_mask_excludes_prompt`
```

## Validation

The notebook checker proves:

- 12 notebooks per language and matching lesson order;
- required section order and metadata;
- normalized code-cell hash equality;
- equal numbers of predict/check/exercise/mistake-note/source cells;
- map position and figure alt text;
- top-to-bottom toy execution with bounded output.

An unexecuted optional network/server cell is acceptable only when its tag, resource
requirements and exact reproduction command are visible. It must not support a stated
local result.

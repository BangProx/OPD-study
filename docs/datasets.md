# Dataset cards and download policy

| ID/config | Revision | License | Rows | Parquet bytes | Role |
|---|---|---|---:|---:|---|
| TinyArithmetic-OPD | `generated-v1`, seed 42 | project Apache-2.0 | 4096/512/512 | 0 | offline core |
| openai/gsm8k/main | `740312add8…` | MIT | 7473 train, 1319 test | 2,725,633 | real default |
| EleutherAI/hendrycks_math | `21a5633873…` | MIT | 7500/5000 | 4,883,857 | advanced/server |
| open-r1/OpenR1-Math-220k/default | `e4e141ec9d…` | Apache-2.0 | 93,733 train | 2,149,897,914 | explicit server opt-in |
| roneneldan/TinyStories | `f54c09fd23…` | CDLA-Sharing-1.0 | 2,119,719/21,990 | 1,000,775,442 | optional LM foundation |

GSM8K validation is a deterministic seed-42 subset of official train. Official test is
read-only evaluation data. The real-data mini smoke reads all row counts but trains only
16 derived train rows, validates on 8, and does not touch test.

OpenR1 and TinyStories exceed 100MB and are never automatically downloaded. A dataset
repository license does not erase restrictions or provenance of original problem
sources; users must inspect the linked dataset card for their application.

Exact metadata and official card URLs: [`sources.yml`](sources.yml).

# OPD-study

한국어 · [English](README.md)

**LLM 온폴리시 증류(On-Policy Distillation, OPD)**를 코드로 배우는 한·영
노트북 강좌이자 참고 구현입니다. 확률과 SFT에서 시작해 원 GKD를 직접 만들고,
현대 sampled-token OPD, vOPD, OPD², TCOD, SOD, SAGE-OPD까지 이어집니다.

기본 경로는 실제 PyTorch Transformer와 생성 산술 데이터로 CPU에서 완전
오프라인 실행됩니다. Qwen3/GSM8K 연구 경로는 dependency·하드웨어·라이선스·
다운로드 동의를 모두 확인한 뒤에만 열립니다.

> [!NOTE]
> 이 저장소는 논문의 대규모 GPU 점수를 재현했다고 주장하지 않습니다. 모든
> 결과에는 실제 실행 여부, profile, seed와 한계가 함께 기록됩니다.

## 15분 시작

Python 3.10–3.12가 필요합니다.

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[notebooks]"
python -m opd_study.demo --smoke --device cpu --output artifacts/quickstart
```

`artifacts/quickstart/index.html`을 열거나 같은 run을 TensorBoard에서 봅니다.

```bash
python -m tensorboard.main --logdir artifacts/quickstart/tensorboard
```

HTML report에서 생성 답변, held-out NLL/정확도/teacher agreement, entropy와 양방향
KL을 비교할 수 있습니다. 새 문제 하나를 바로 비교하거나 터미널 playground를
열 수도 있습니다.

```bash
python -m opd_study.demo --smoke --prompt "(2 + 3) * 4"
python -m opd_study.demo --smoke --interactive
```

그다음 [00번 강의](notebooks/ko/00_opd_in_15_minutes.ipynb)를 여세요. 수렴을
관찰할 toy 비교는 `--smoke`를 빼고 실행합니다.

Windows PowerShell에서는 다음처럼 실행합니다.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[notebooks]"
python -m opd_study.demo --smoke --device cpu --output artifacts/quickstart
```

설치 없는 경로: [Colab quickstart 열기](https://colab.research.google.com/github/BangProx/OPD-study/blob/main/notebooks/colab/quickstart.ipynb).

## 학습 경로

| 강의 | 주제 | 경로 |
|---:|---|---|
| 00 | 15분 OPD 체험과 전체 개념 지도 | 빠른/전체 |
| 01 | 토큰·확률·자기회귀 LM | 전체 |
| 02 | CE·엔트로피·KL·KD | 빠른/전체 |
| 03 | SFT·오프폴리시·온폴리시 state | 빠른/전체 |
| 04 | 원 GKD: lambda 혼합과 일반화 JSD | 빠른/전체 |
| 05 | rollout부터 update까지 현대 OPD | 빠른/전체 |
| 06 | 실제 train/eval·checkpoint·LoRA/QLoRA | 전체 |
| 07 | 실패 진단·support overlap·vOPD | 전체 |
| 08 | 멀티턴 오류 누적 | 전체 |
| 09 | TCOD F2B/B2F temporal curriculum | 전체 |
| 10 | SOD와 SAGE-OPD | 전체 |
| 11 | OPD²·다국어 유지·avg@K/pass@K | 빠른/전체 |

[한국어 12개](notebooks/ko)와 [영어 12개](notebooks/en)는 코드·source ID·검사·
연습문제가 같고 모두 깨끗하게 실행된 출력이 저장되어 있습니다. 빠른 경로는
약 2–3시간, 전체 경로는 약 7–9시간입니다.

각 강의는 다루는 production 함수 원문을 길이 제한과 함께 직접 출력하고, 그
구현을 선택한 이유·가능한 대안·강의별 실패 사례를 설명합니다. 별도의 숨은
구현을 복제하지 않으면서 notebook만으로 코드와 설명을 함께 볼 수 있습니다.

## CLI 학습과 평가

```bash
python -m opd_study train --algorithm opd --smoke --device cpu --output artifacts/opd
python -m opd_study train --algorithm tcod_f2b --smoke --device cpu --output artifacts/tcod
python -m opd_study train --algorithm sod --smoke --device cpu --output artifacts/sod
python -m opd_study train --algorithm sage_opd --smoke --device cpu --output artifacts/sage
python -m opd_study eval --run artifacts/sage --rows 8
```

선택 가능: `sft`, `off_policy_kd`, `gkd`, `opd`, `vopd`, `opd2`, `tcod_f2b`,
`tcod_b2f`, `sod`, `sage_opd`.

## 실제 데이터와 모델

GSM8K의 고정된 실제 shard는 총 2,725,633B이며 라이선스 동의 flag가 필요합니다.

```bash
python -m opd_study download-data --dataset gsm8k \
  --cache artifacts/cache --accept-dataset-license
python -m opd_study gsm8k-smoke --cache artifacts/cache \
  --output artifacts/gsm8k-mini-smoke --accept-dataset-license
```

mini smoke는 실제 row와 split 배관을 검사할 뿐 Qwen 결과가 아닙니다. 5.57GB
Qwen3-0.6B/1.7B LoRA 조합은 먼저 다음을 실행합니다.

```bash
python -m opd_study research-preflight --config configs/laptop/gsm8k_lora.yaml
```

기본 preset은 모델 다운로드 동의를 `false`로 둡니다. [하드웨어 가이드](docs/hardware-guide.md)와
[모델 호환표](docs/model-compatibility.md)를 확인하세요.

라이선스와 메모리를 확인한 뒤 실행하는 Qwen/GSM8K sampled-OPD one-step 경로는
다음과 같습니다.

```bash
python -m pip install -e ".[research]"
python -m opd_study research-train \
  --config configs/laptop/gsm8k_lora.yaml --smoke \
  --accept-dataset-license --accept-model-license
```

`research` extra는 PyTorch 2.2 이상이 필요하며 preflight가 다운로드 전에 package
import까지 확인합니다. CPU toy 강의는 계속 PyTorch 2.1 이상에서 동작합니다.

두 동의 flag는 약 5.57GB의 고정 모델 weight와 2.73MB 데이터 다운로드를
허용합니다. 명령은 구현했지만 현재 CPU-only 호스트에서는 **UNVERIFIED**이며,
Qwen 결과를 실행한 것처럼 제공하지 않습니다.

검증된 NVIDIA 환경에서는 별도 QLoRA preset을 선택합니다.

```bash
python -m opd_study research-train \
  --config configs/laptop/gsm8k_qlora.yaml --smoke \
  --accept-dataset-license --accept-model-license
```

실패 시 full fine-tuning으로 전환하지 않습니다. Colab에도 같은 명령이
`RUN_OPTIONAL_QWEN_QLORA = False` 뒤에 기본 비활성 상태로 들어 있습니다.

## 논문 충실도와 교육용 단순화

- 원 GKD 수학은 arXiv:2306.13649v3을 따르고 beta 경계는 고정 TRL 동작과
  교차 검증했지만 코드를 복사하지 않았습니다.
- vOPD·SAGE-OPD·라이선스 없는 upstream의 진단은 논문 식 기반 clean-room입니다.
- TCOD·OPD²의 핵심 시간/델타 의미는 보존하되 분산 rollout을 동기식 toy로
  바꿨습니다.
- SOD core는 step-weighted distillation 항만 구현하며 별도 GRPO 항이 없음을
  metric에 표시합니다.
- mini SAGE의 judge는 token-agreement proxy입니다. research mode에서는 실제
  환경 실패와 의미 판단 teacher query가 필요합니다.

[수학 convention](docs/math.md), [구현 노트](docs/implementation-notes.md),
[출처 manifest](docs/sources.yml)를 먼저 읽으세요.

## 전체 검사

```bash
python -m pip install -e ".[dev,notebooks]"
python scripts/quality_gate.py
# 24개 노트북 커널 재실행까지 포함한 전체 검사:
python scripts/quality_gate.py --execute-notebooks
```

Linux/macOS/Windows core CI와 예약 notebook/link/source 검사가 포함됩니다. 실제
실행 범위는 [결과와 검증 상태](docs/results.md)에 있습니다.

기여는 [CONTRIBUTING.md](CONTRIBUTING.md), 보안 제보는 [SECURITY.md](SECURITY.md)를
따르세요. 프로젝트 코드는 Apache-2.0이며 외부 자산은 각 라이선스를 유지합니다.

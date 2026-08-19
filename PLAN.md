# OPD-study 구현 Goal 계약서

> 상태: **Goal 실행 중 / C8–C9 공개 전 최종 감사**<br>
> 작성일: 2026-08-19 (Asia/Seoul)<br>
> 대상 저장소: `https://github.com/BangProx/OPD-study.git`<br>
> 현재 로컬 경로: `/Users/bangbyeonghun/Documents/nlp/OPD-study`

이 문서는 Codex의 장기 실행 Goal에 전달할 구현 계약서다. 단순 할 일 목록이
아니라 범위, 근거, 승인 지점, 검증 방법, 종료 조건을 정의한다.

## 0. Goal 시작 명령

아래의 **승인 게이트 A**가 해결된 뒤 사용한다.

```text
/goal PLAN.md를 유일한 구현 계약으로 삼아 OPD-study를 완성하라. 승인된 범위만 구현하고, 각 체크포인트의 검증을 통과시키며, 완료 조건을 모두 증명할 때까지 계속 진행하라. 승인되지 않은 추가 범위, 라이선스 충돌, 원격 이력 충돌, 또는 검증 불가능한 연구 주장을 만나면 임의로 결정하지 말고 멈춰 보고하라.
```

공식 Goal 가이드에 맞춰 하나의 목표, 검증 가능한 종료 조건, 선행 문서,
체크포인트별 증거를 명시한다.

- Goal 참고: <https://learn.chatgpt.com/use-cases/follow-goals>

---

## 1. 단일 목표와 종료 조건

### 단일 목표

초심자가 이 저장소만으로 LLM **On-Policy Distillation(OPD)**의 기초부터
원 논문, 실제 구현, 학습·평가, 최근 single-turn 및 multi-turn agent 변형까지
배우고 실행할 수 있는 한국어/영어 이중언어 오픈소스 강좌와 재사용 가능한
Python 패키지를 구축한다.

### 완료의 정의

다음 항목이 모두 참일 때만 Goal을 완료로 간주한다.

1. 한국어와 영어의 전체 강좌가 동일한 학습 목표·코드·실험을 제공한다.
2. 모든 기본 노트북이 깨끗한 환경에서 위에서 아래로 실행된다.
3. 최소 CPU 오프라인 toy 학습에서 같은 초기 student와 예산으로 SFT와 OPD를
   비교하고, loss·정답률·teacher agreement 및 checkpoint를 남긴다.
4. 노트북 밖 CLI에서도 SFT, OPD, TCOD, SOD, SAGE-OPD의 최소 교육용 구현을
   선택하여 train/eval할 수 있다.
5. 원 GKD/OPD 구현과 현대 reasoning OPD 구현의 차이가 코드와 문서에
   명시되고 테스트된다.
6. 실제 소형 Hugging Face 모델용 laptop preset과 다중 GPU server preset이
   있으며, 지원 조건과 실행 검증 상태가 정직하게 표시된다.
7. 주장·수식·구현 선택마다 논문, 공식 코드, 공식 문서 또는 신뢰할 수 있는
   기술 자료의 출처를 추적할 수 있다.
8. 가져오거나 변형한 모든 코드의 라이선스가 호환되고 출처가 보존된다.
9. 테스트, lint, 타입 검사, 노트북 구조 검사, 링크 검사, 한영 동기화 검사가
   모두 통과한다.
10. 설치, 15분 quickstart, 학습 경로, 모델 선택, 하드웨어별 실행, 문제 해결,
    기여 방법이 README에서 발견 가능하다.
11. 로컬 디렉터리 이름과 저장소 이름이 `OPD-study`로 일치하고 `origin`이
    지정된 GitHub 저장소를 가리킨다.
12. 실행한 실험의 명령, 환경, seed, 소요 시간, 메모리, 결과 및 한계를
    재현 기록으로 남긴다. 실행하지 않은 결과를 실행한 것처럼 쓰지 않는다.
13. 확정된 toy data와 GSM8K 기본 preset이 실제 학습되며, 모든 외부 dataset의
    라이선스·revision·행 수·다운로드 크기·사용 split이 문서화된다.
14. `python -m opd_study.demo`가 SFT와 OPD를 비교하는 mini playground와
    정적 결과 report를 만들고, TensorBoard에서 같은 run을 확인할 수 있다.
15. Linux, macOS, Windows CPU smoke test가 통과하고, Colab quickstart가 새
    runtime에서 실행된다. CUDA/MPS/QLoRA의 미지원 조합은 안전하게 차단된다.

별 개수는 직접 통제할 수 있는 완료 조건이 아니다. 대신 정확성, 학습 경험,
재현성, 유지보수성, 접근성, 공개 저장소 품질을 5만-star 수준의 품질 기준으로
삼는다.

---

## 2. 고정 요구사항

아래 항목은 추가 승인 없이 구현 대상이다.

### 콘텐츠

- 확률분포, autoregressive LM, entropy/cross-entropy, KL divergence,
  teacher/student, knowledge distillation, sampling의 기초
- off-policy와 on-policy, exposure bias와 distribution shift
- ICLR 2024 원 논문의 Generalized Knowledge Distillation(GKD)와 그 안의
  on-policy distillation
- 최근 reasoning LLM 문헌에서 흔히 OPD라고 부르는 student rollout 기반
  token-level distillation과 policy-gradient 관점
- 원 구현의 수식, tensor shape, masking, gradient 흐름, 구현 이유, 대안 및
  trade-off
- TCOD, SOD, SAGE-OPD의 논문 기반 개념과 최소 동작 구현
- 동일 초기화·데이터·학습 예산의 SFT baseline과 OPD 비교
- 실제 학습, 평가, 진단, 실패 사례, 결과 해석
- 한국어 우선 집필 후 자연스러운 영어판 제공

### 실행 환경

- laptop-first: Windows/Linux CPU·CUDA와 macOS CPU·MPS를 기본 지원
  대상으로 삼는다.
- NVIDIA CUDA는 선택 가능한 server profile로 제공한다.
- core 설치는 FlashAttention, Ray, vLLM 같은 무거운 의존성을 요구하지 않는다.
- toy/offline, laptop/Hugging Face, server/research 세 실행 프로필을 분리한다.
- student/teacher 모델은 config로 선택할 수 있게 한다.
- 실제 대형 논문 수치의 완전 재현과 laptop 교육용 재현을 명확히 구분한다.
- Colab을 설치 없는 필수 quickstart 경로로 제공한다. Kaggle은 선택 사항이다.
- 모든 경로는 `pathlib`을 사용하고 bash 전용 실행에 의존하지 않는다.

### 확정 toy 모델과 데이터

교육용 toy 실험은 외부 다운로드 없이 다음 사양으로 고정한다. 구현 중 수치가
바뀌어야 한다면 기존 test/runtime 증거와 변경 이유를 먼저 제시한다.

| 항목 | 확정값 | 이유 |
|---|---|---|
| dataset | seed 42로 생성하는 `TinyArithmetic-OPD` | 정답·중간 step·오류 주입을 완전히 통제하고 재배포 라이선스 문제를 피함 |
| split | train 4,096 / validation 512 / test 512 | CPU에서 빠르면서 seed별 비교가 가능한 크기 |
| task | 1~3단계 정수 사칙연산과 명시적 scratchpad; 별도 multi-turn calculator environment | single-turn과 error compounding을 같은 개념으로 연결 |
| tokenizer | 고정 vocabulary의 저장소 내 symbol tokenizer, context 128 | teacher/student vocabulary를 완전히 일치시키고 tensor를 눈으로 추적 가능 |
| teacher | decoder-only Transformer, 4 layers, `d_model=128`, 4 heads, FFN 512 | CPU에서 SFT bootstrap 가능한 실제 neural teacher |
| student | decoder-only Transformer, 2 layers, `d_model=64`, 4 heads, FFN 256 | 명확한 capacity gap과 수분 이내 학습 |
| baseline | 동일 student 초기화의 no-train, hard-label SFT, soft off-policy KD, OPD | OPD가 필요한 이유를 공정하게 비교 |

teacher는 같은 생성 데이터로 deterministic SFT bootstrap하고 cache한다. cache가
없어도 각 notebook이 top-to-bottom으로 teacher를 재생성할 수 있어야 하며,
실제 parameter 수와 bootstrap 시간은 코드가 출력한다. toy 결과는 실제 LLM
benchmark 성능을 대신하지 않는다.

### 확정 실제 모델

같은 Qwen3 tokenizer/chat template를 사용하는 다음 조합을 기준 preset으로
고정한다. 명칭의 B 수치가 실제 parameter 수와 다를 수 있으므로 둘 다 쓴다.

| profile | student | teacher | BF16 weight download | license | 용도 |
|---|---|---|---:|---|---|
| laptop/Colab | `Qwen/Qwen3-0.6B` (751.6M) | `Qwen/Qwen3-1.7B` (2.032B) | 약 1.50GB + 4.06GB | Apache-2.0 | GSM8K 소규모 LoRA/QLoRA |
| server/SOD-like | `Qwen/Qwen3-0.6B` | `Qwen/Qwen3-4B` (4.022B) | 약 1.50GB + 8.05GB | Apache-2.0 | 논문형 확장 실험 |

작성 시점 기준 revision은 각각 `c1899de`, `70d244c`, `1cfa9a7`이다. 구현 시
전체 commit SHA를 lock하고 model card/license 변경을 다시 감사한다. laptop
기본은 post-trained Qwen3를 써서 초기 student가 reasoning format과 teacher
support에 어느 정도 들어오게 한다. Base 모델 사용은 별도 failure 실습이다.

- <https://huggingface.co/Qwen/Qwen3-0.6B>
- <https://huggingface.co/Qwen/Qwen3-1.7B>
- <https://huggingface.co/Qwen/Qwen3-4B>

### 확정 dataset 역할과 라이선스

크기는 2026-08-19 Hugging Face Dataset Viewer의 Parquet download 기준이며,
실행 시 revision을 pin하고 다시 확인한다.

| dataset / config | license | 행 수 | download | 확정 역할 |
|---|---|---:|---:|---|
| `openai/gsm8k` / `main` | MIT | 7,473 train + 1,319 test | 약 2.73MB | **기본 실제 train/eval** |
| `EleutherAI/hendrycks_math` / 7 configs | MIT | 7,500 train + 5,000 test | 약 4.88MB | 심화 평가와 server train |
| `open-r1/OpenR1-Math-220k` / `default` | Apache-2.0 | 93,733 train | 약 2.15GB | server 선택 실습; 자동 다운로드 금지 |
| `roneneldan/TinyStories` / `default` | CDLA-Sharing-1.0 | 2,119,719 train + 21,990 validation | 약 1.00GB | lesson 01 선택형 LM 예제; reasoning 비교에는 사용하지 않음 |

Dataset cards:

- <https://huggingface.co/datasets/openai/gsm8k>
- <https://huggingface.co/datasets/EleutherAI/hendrycks_math>
- <https://huggingface.co/datasets/open-r1/OpenR1-Math-220k>
- <https://huggingface.co/datasets/roneneldan/TinyStories>

OpenR1-Math의 `all` config는 약 4.22GB이고 Dataset Viewer의 세 config 합계는
약 8.44GB다. TinyStories Hub 저장소 전체는 원본 txt/tar 등으로 Parquet보다
훨씬 크므로 필요한 Parquet split만 받는다. 모든 loader는 다운로드 전 예상
크기, cache 위치, 라이선스를 출력하고 `--accept-dataset-license` 또는 문서화된
명시적 동의 없이 100MB 초과 dataset을 자동으로 받지 않는다.

GSM8K는 `main` train에서 seed 42로 validation을 분리하고 official test에는
학습하지 않는다. `smoke`, `laptop`, `full` preset의 정확한 row/token 예산을
config에 기록하고 SFT/OPD 비교에는 동일 prompt subset과 평가 split을 쓴다.

### 연구 근거 우선순위

1. 논문 최종본 또는 arXiv 원문
2. 저자 공식 저장소와 해당 commit/tag
3. 사용 프레임워크 공식 문서와 코드
4. 저자·연구기관의 기술 블로그
5. 보조 설명 자료

블로그나 awesome-list는 발견과 설명 보조에는 쓸 수 있으나 핵심 수식이나
성능 수치의 단독 근거로 사용하지 않는다.

### 필수 기준 문헌

- Agarwal et al., *On-Policy Distillation of Language Models: Learning from
  Self-Generated Mistakes*, ICLR 2024 / arXiv:2306.13649
  - <https://arxiv.org/abs/2306.13649>
- TCOD, arXiv:2604.24005
  - <https://arxiv.org/abs/2604.24005>
  - <https://github.com/kokolerk/TCOD>
- SOD, arXiv:2605.07725
  - <https://arxiv.org/abs/2605.07725>
  - <https://github.com/YoungZ365/SOD>
- SAGE-OPD, arXiv:2606.19659
  - <https://arxiv.org/abs/2606.19659>
- 구현 시점의 최신 OPD survey 및 원 논문/공식 구현

구현 시작 시 각 문헌의 최신 버전, 정정, 공식 코드 공개 여부와 라이선스를
다시 확인하고 `docs/sources.yml`에 확인 날짜와 commit SHA를 기록한다.

---

## 3. 승인 게이트 A — 구현 전에 제안하고 결정받을 추가 사항

아래는 품질과 유지보수성을 크게 높일 것으로 예상되지만 원 요청을 넘어설 수
있으므로 **사용자의 명시적 승인 전에는 구현하지 않는다**. Goal은 첫
체크포인트에서 예상 작업량과 이유를 간단히 보고하고 선택을 기다린다.

| ID | 제안 | 이유 | 기본 제안 |
|---|---|---|---|
| A1 | `mini`와 `research` 이중 백엔드 | 노트북은 laptop에서 즉시 돌고, 서버에서는 실제 프레임워크로 확장 가능 | 승인 권장 |
| A2 | 한영 구조 동기화 자동 검사 | 번역본의 코드·수식·결과가 시간이 지나며 어긋나는 문제 방지 | 승인 권장 |
| A3 | 논문·코드 provenance 및 라이선스 manifest | “실제 레포에서 가져오기”를 합법적이고 검증 가능하게 수행 | 승인 권장 |
| A4 | 확장 GitHub Actions: scheduled 전체 notebook·외부 링크·license drift 검사 | G5의 3-OS toy smoke는 필수; 긴 nightly 검사는 추가 비용이 있음 | 승인 권장 |
| A5 | Kaggle 실행 경로와 배지 | Colab은 G5에 따라 필수로 승격됨; 추가 hosted 경로 제공 | 선택 |
| A6 | MkDocs 기반 정적 강의 사이트 | notebook만 볼 때보다 탐색·검색·공유가 쉬움 | 선택 |
| A7 | 재현 experiment card와 결과 대시보드 | 논문 수치와 laptop 결과를 혼동하지 않고 비교 | 승인 권장 |
| A8 | 최신 문헌 snapshot 생성 스크립트 | 매우 빠르게 변하는 OPD 분야의 범위와 기준일을 투명하게 유지 | 선택 |

### 최신 변형의 범위 결정

TCOD, SOD, SAGE-OPD는 필수다. 그 밖의 “etc.”는 구현 시작일 기준 문헌 조사를
먼저 수행하고 다음 기준으로 후보표를 제시한다.

- 기존 필수 알고리즘과 다른 학습 아이디어를 가르치는가?
- 원 논문과 공식 구현 또는 구현에 충분한 알고리즘 설명이 있는가?
- 라이선스가 호환되는가?
- laptop 축약 실험으로 핵심 현상을 보여줄 수 있는가?
- 단순 유행이 아니라 재현·비판적 분석 가치가 있는가?

조사 후보 예시는 mechanism/diagnostics, variance reduction, 효율화,
cross-tokenizer, black-box teacher, representation distillation, OPD+RL,
multilingual OPD, long-horizon validation 및 OPD의 한계를 다루는 최신 연구다.
후보 이름을 곧바로 구현 약속으로 간주하지 않는다. 3~5개를 우선순위와 예상
비용까지 제시하고 사용자 승인 후 필수/심화/문헌 소개 중 하나로 분류한다.

### 승인을 기록하는 형식

사용자 결정 후 이 표를 갱신하고 commit에 포함한다.

| 항목 | 결정 | 비고 |
|---|---|---|
| A1~A4 | 승인 | A4는 3-OS core CI와 확장 nightly 검사 포함 |
| A5 | 보류 | 필수 Colab은 G5로 구현, Kaggle만 2단계로 보류 |
| A6 | 보류 | MkDocs 사이트는 2단계 |
| A7~A8 | 승인 | experiment card/dashboard와 문헌 snapshot 구현 |
| 추가 OPD 변형 | 승인 | 2026-08-19: Rethinking OPD 진단, vOPD, OPD²+다국어, test-time scaling 비판 실험 |
| 저장소 라이선스 | Apache-2.0 | 2026-08-19 사용자 승인; third-party 호환성은 별도 감사 |

### 사용자 검토에서 고정된 추가 요구사항

2026-08-19 검토에서 G1~G8은 승인 게이트가 아닌 고정 범위로 편입되었다.

- G1: dataset/license/크기/split 문서화와 다운로드 안전장치
- G2: 같은 toy task와 공정한 예산의 SFT baseline
- G3: `python -m opd_study.demo`, 정적 report, TensorBoard 결과 확인
- G4: `TinyArithmetic-OPD`, tiny Transformer, Qwen3 모델 조합 확정
- G5: Windows 안전장치와 필수 Colab quickstart
- G6: lesson 00의 OPD 개념 지도와 모든 lesson의 현재 위치 표시
- G7: 모든 lesson 말미의 “내가 자주 틀리는 것” 오답노트
- G8: LoRA와 QLoRA, platform별 bitsandbytes 지원 상태의 정직한 구분

---

## 4. 범위 밖과 금지 사항

- “OPD를 전부 안다”는 표현을 무기한 모든 미래 연구를 포함한다는 뜻으로
  과장하지 않는다. 문헌 기준일과 다루지 않은 영역을 명시한다.
- 논문 표의 대형 GPU 결과를 laptop에서 동일하게 재현한다고 약속하지 않는다.
- 실행하지 않은 셀 출력, benchmark 점수, 속도, 메모리 수치를 만들지 않는다.
- 출처 불명 또는 라이선스 비호환 코드를 복사하지 않는다.
- upstream 저장소 전체를 무분별하게 vendor하지 않는다.
- 원격 Git 이력을 force-push, 덮어쓰기, 삭제하지 않는다.
- 사용자 승인 없이 API 유료 호출, cloud GPU job, 모델·데이터 업로드를 하지
  않는다.
- API teacher가 충분한 token distribution/logprob을 제공하지 않는데도
  white-box OPD와 동일하다고 부르지 않는다.
- 같은 tokenizer가 필요한 구현에서 다른 tokenizer 모델 조합을 조용히
  허용하지 않는다. 지원하지 않으면 빠르고 설명적인 오류를 낸다.
- 교육용 단순화와 논문 충실 구현을 같은 결과로 포장하지 않는다.
- dataset의 저장소 license와 각 원천 문제의 추가 조건을 같은 것으로 단정하지
  않는다. dataset card가 밝힌 source와 알려진 제한을 함께 기록한다.
- 100MB가 넘는 dataset/model을 크기 고지와 명시적 선택 없이 자동으로 받지
  않는다.
- QLoRA를 모든 OS에서 같은 속도와 안정성으로 지원한다고 쓰지 않는다.
- Windows에서 POSIX path, shell quoting, symlink, `/tmp`를 가정하지 않는다.

---

## 5. 학습 설계

### 대상 독자

- Python 기초는 알지만 knowledge distillation이나 RL은 처음인 학습자
- 수식만으로는 이해하기 어렵고 코드를 실행하며 배우는 학습자
- 주의 전환이 잦아 짧고 명확한 학습 단위와 즉각적인 피드백이 필요한 학습자

### 강의 길이

- 총 12개 lesson, lesson당 약 20~45분
- 빠른 경로 약 2~3시간, 전체 경로 약 7~9시간
- 한국어 12개 + 영어 12개의 mirrored notebook
- 긴 부록과 전체 실험은 notebook 본문 대신 `docs/`와 CLI로 분리

### 모든 notebook의 고정 리듬

1. **이번 30분의 목표**: 최대 3개
2. **개념 지도에서 현재 위치**: lesson 00의 OPD 지도를 작게 다시 표시
3. **5~8분 micro-section**: 설명 → 작은 코드 → 눈에 보이는 결과
4. **예측하기**: 실행 전에 결과를 고르는 짧은 질문
5. **실행하기**: 한 셀은 한 개념만 담당
6. **체크포인트**: 1~3분 회상 문제와 즉시 확인 가능한 답
7. **왜 이렇게 구현했나**: 대안과 trade-off
8. **흔한 함정**: shape/mask/gradient/분포 방향 오류
9. **내가 자주 틀리는 것**: 이번 lesson의 오답 패턴 2~4개와 자가 점검
10. **60초 요약**: 핵심 3줄
11. **다음 lesson 연결** 및 선택형 심화 과제

추가 접근성 원칙:

- 긴 문단을 피하고 코드 셀은 한 화면 안에 들어오도록 나눈다.
- 예상 소요 시간과 남은 구간을 각 section에 표시한다.
- 색만으로 의미를 구분하지 않고 color-blind-safe palette와 직접 label을 쓴다.
- 그림에는 alt text 또는 바로 아래 텍스트 설명을 둔다.
- 장식 animation, 불필요한 경고 출력, 거대한 table/output을 피한다.
- “필수”와 “심화”를 시각적으로 일관되게 구분한다.
- 해답은 접을 수 있는 `<details>` 또는 별도 solution section으로 제공한다.
- lesson 00에는 `Data → Student rollout → Teacher signal → Loss → Update →
  Evaluation`과 `SFT/off-policy/OPD/agent variants`의 관계를 보여주는 Mermaid
  개념 지도를 둔다. Mermaid가 렌더링되지 않는 환경을 위한 동등한 text/ASCII
  fallback과 alt text도 제공한다.
- 각 lesson의 오답노트는 단순 경고가 아니라 작은 틀린 코드/수식, 왜 틀렸는지,
  고친 형태, 관련 test 이름을 함께 보여준다.

### 강좌 지도

| # | 제목 | 핵심 산출물 | 시간 | 경로 |
|---:|---|---|---:|---|
| 00 | 15분 OPD 체험과 전체 지도 | 전체 개념 지도와 첫 SFT/OPD loss curve | 20분 | 빠른/전체 |
| 01 | 토큰·확률·autoregressive LM | next-token distribution 직접 계산 | 30분 | 전체 |
| 02 | CE, entropy, KL, KD | FKL/RKL/JSD 방향을 그림과 tensor로 비교 | 40분 | 빠른/전체 |
| 03 | SFT/off-policy vs on-policy | 같은 toy task에서 SFT와 exposure bias를 관찰 | 40분 | 빠른/전체 |
| 04 | 원 논문 GKD 해부 | λ 혼합, divergence 선택, Algorithm 구현 | 45분 | 빠른/전체 |
| 05 | 현대 OPD from scratch | 동일 초기 student의 SFT/OPD 공정 비교와 전체 루프 | 50분 | 빠른/전체 |
| 06 | 실제 train/eval | GSM8K+Qwen3 preset, checkpoint, report, TensorBoard | 50분 | 전체 |
| 07 | 왜 성공/실패하는가 | support overlap, KL, entropy, gradient 진단 | 40분 | 전체 |
| 08 | multi-turn agent OPD | error compounding toy environment | 35분 | 전체 |
| 09 | TCOD | F2B/B2F temporal curriculum 비교 | 40분 | 전체 |
| 10 | SOD와 SAGE-OPD | step/turn 신뢰도와 selective intervention | 45분 | 전체 |
| 11 | 통합 비교와 다음 연구 | 공정 비교표, 의사결정 트리, capstone | 45분 | 빠른/전체 |

각 lesson은 `Goal → Setup → Steps → Checks → Next Steps`의 top-to-bottom 구조를
따르고, 모든 결론은 같은 notebook의 실행 결과나 명시된 출처에 연결한다.

---

## 6. 기술 설계

### 제안 디렉터리 구조

승인된 옵션에 따라 조정할 수 있으나 책임 분리는 유지한다.

```text
OPD-study/
├── README.md
├── README.ko.md
├── PLAN.md
├── pyproject.toml
├── LICENSE
├── CITATION.cff
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── notebooks/
│   ├── ko/00_...ipynb ... 11_...ipynb
│   ├── en/00_...ipynb ... 11_...ipynb
│   └── colab/quickstart.ipynb
├── src/opd_study/
│   ├── __main__.py
│   ├── demo.py
│   ├── algorithms/
│   │   ├── gkd.py
│   │   ├── opd.py
│   │   ├── tcod.py
│   │   ├── sod.py
│   │   └── sage_opd.py
│   ├── models/
│   ├── data/
│   ├── rollout/
│   ├── envs/
│   ├── training/
│   ├── evaluation/
│   └── diagnostics/
├── configs/
│   ├── toy/
│   ├── laptop/
│   └── server/
├── scripts/
├── tests/
├── docs/
│   ├── math.md
│   ├── implementation-notes.md
│   ├── model-compatibility.md
│   ├── hardware-guide.md
│   ├── troubleshooting.md
│   ├── literature-map.md
│   ├── sources.yml
│   └── third-party-notices.md
└── artifacts/
    ├── experiment-cards/
    └── reports/
```

### 세 실행 프로필

#### 1. `toy` — 기본, 완전 오프라인

- 확정된 `TinyArithmetic-OPD`, symbol tokenizer, tiny teacher/student를
  저장소 안의 코드로 생성한다.
- CPU에서 수분 내 rollout, teacher scoring, update, eval을 완료한다.
- 동일 초기 student로 no-train/SFT/off-policy KD/OPD를 비교한다.
- 모든 알고리즘의 핵심 아이디어를 동일한 toy task에서 비교한다.
- CI와 notebook 기본 실행은 이 프로필만으로 완결된다.

#### 2. `laptop` — 실제 Qwen3 + GSM8K

- Transformers 기반 local teacher/student와 PEFT/LoRA를 지원한다.
- Windows/Linux/macOS에서 CPU/MPS/CUDA device를 preflight하고 사용자가
  override할 수 있다. 요청한 device가 없으면 경고 후 조용히 바꾸지 말고
  `--allow-device-fallback`일 때만 CPU로 fallback한다.
- low-memory 옵션: 짧은 context, 작은 batch, gradient accumulation,
  checkpointing, teacher `no_grad`, 선택적 top-k 근사.
- `lora`와 `qlora`를 별도 preset으로 제공한다. QLoRA는 student를 4-bit로
  load하고 adapter만 학습하며 teacher quantization 여부를 별도로 설정한다.
- 모델 preset마다 tokenizer 호환성, 예상 RAM/VRAM, dtype, 검증 장치를
  표로 제공한다.
- 다운로드 크기와 예상 실행 시간을 명령 실행 전에 보여준다.
- GSM8K `smoke` subset을 기본으로 하고 `laptop`/`full`은 명시적으로 고른다.

#### 3. `server` — 논문형 실험

- multi-GPU backend는 별도 optional dependency로 격리한다.
- upstream이 verl/Trinity/AReaL 등을 사용한다면 버전과 commit을 pin하고,
  thin adapter 또는 recipe로 연결한다.
- laptop API와 가능한 한 같은 config 의미를 유지하되, 동일 동작을 보장할 수
  없는 차이는 문서화한다.

### LoRA/QLoRA와 platform 계약

| platform | 기본 fine-tuning | QLoRA 상태 | 안전한 동작 |
|---|---|---|---|
| Linux + NVIDIA CUDA | LoRA BF16/FP16 | bitsandbytes NF4 지원 경로 | capability와 CUDA/bnb version 검사 |
| Windows + NVIDIA CUDA | LoRA BF16/FP16 | bitsandbytes 지원 경로 | native Windows CI smoke; shell/path 독립 |
| Windows/Linux CPU | toy 또는 작은 LoRA smoke | 매우 느린 CPU 경로, 기본 비활성 | 예상 시간 경고와 명시적 opt-in |
| macOS Apple Silicon MPS | LoRA FP16/BF16 지원 여부를 runtime probe | macOS 14+ bitsandbytes preview/slow fallback이므로 **experimental** | 안정 PyPI 경로로 가정하지 않고 실패 시 plain LoRA 제안 |
| macOS Intel/CPU | toy 우선 | 기본 미지원 | 실제 모델 실행 전 메모리·시간 경고 |

지원표는 구현 시점의 공식 PyTorch, Transformers, PEFT, bitsandbytes 문서와
release로 다시 검증한다. import 성공만으로 지원 판정하지 않고 4-bit load,
한 forward, 한 backward/update를 수행하는 preflight test를 둔다. QLoRA 실패를
plain full fine-tuning으로 자동 전환해 OOM을 유발하지 않는다.

- bitsandbytes 설치/preview 지원표: <https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/docs/source/installation.mdx>
- PyTorch MPS backend: <https://docs.pytorch.org/docs/stable/notes/mps>

### Windows와 Colab 계약

- file/cache/temp 경로는 `pathlib.Path`, `platformdirs`, `tempfile`을 사용한다.
- `spawn` multiprocessing과 `if __name__ == "__main__"`을 지킨다.
- 문서 명령은 `python -m ...`를 우선하고 Bash와 PowerShell 예시를 함께 둔다.
- symlink, executable bit, fork, POSIX signal에 핵심 경로가 의존하지 않는다.
- GitHub Actions에서 Ubuntu/macOS/Windows CPU toy smoke를 실행한다.
- `notebooks/colab/quickstart.ipynb`는 새 Colab runtime에서 clone → install →
  toy demo → 선택형 GSM8K QLoRA smoke 순서로 실행되며, 유료 runtime을
  요구하지 않는다. accelerator가 없으면 toy 경로는 계속 작동한다.

### 학습 결과를 확인하는 방법

`python -m opd_study.demo --profile toy`는 5분 이내 목표로 다음을 한 번에
수행한다.

1. 동일 초기 student의 SFT와 OPD 짧은 학습
2. 고정 test prompt의 teacher/SFT/OPD 응답 나란히 비교
3. loss, exact match, teacher agreement, entropy, KL curve 생성
4. `artifacts/reports/latest/index.html`, PNG, JSON summary 생성
5. `runs/`에 TensorBoard event 기록
6. 터미널에서 새 toy 문제를 입력해 두 student를 비교하는 mini playground

TensorBoard는 `python -m tensorboard.main --logdir runs`로 열 수 있게 한다.
Weights & Biases는 계정 없는 `offline` mode만 선택 지원하며 core 경로가
의존하지 않는다. notebook은 report의 핵심 chart와 실패 예시를 inline으로
보여주고 artifact 경로를 마지막 cell에 출력한다.

### 공통 알고리즘 인터페이스

각 알고리즘은 적어도 다음 계약을 공유한다.

```python
class DistillationAlgorithm(Protocol):
    def collect(self, student, batch, generator) -> TrajectoryBatch: ...
    def score(self, teacher, trajectories) -> TeacherSignals: ...
    def loss(self, student, trajectories, signals) -> LossOutput: ...
    def metrics(self, ...) -> dict[str, float]: ...
```

`TrajectoryBatch`는 token IDs, attention mask, prompt/response mask,
student logprob, turn/step boundary, environment observation과 terminal 정보를
명시적으로 가진다. 각 tensor의 shape와 gradient 소유권을 type/docstring과
테스트에서 고정한다.

### 원 구현에서 반드시 설명·검증할 세부사항

- student가 생성한 prefix를 어떻게 고정된 training state로 다루는지
- sampling의 비미분성과 어떤 estimator에 gradient가 흐르는지
- teacher/student logits의 temperature와 normalization
- forward KL, reverse KL, JSD의 정의 방향과 실제 코드 인자 순서
- full-vocabulary loss와 sampled-token/policy-gradient estimator의 차이
- prompt token, padding, EOS, tool/environment token의 loss mask
- sequence 길이 normalization과 batch reduction
- teacher `eval()`/`no_grad()` 및 dropout 처리
- current policy와 old rollout policy, stale trajectory 문제
- tokenizer/chat template 일치와 vocabulary mapping 제약
- top-k teacher logits 근사가 무엇을 버리는지
- mixed precision에서 log-softmax/KL 수치 안정성
- entropy collapse, KL 폭증, support mismatch, reward hacking이 아닌
  teacher imitation failure의 진단
- seed, deterministic 옵션, checkpoint/resume

모든 항목에 최소 하나의 unit test 또는 executable assertion을 둔다.

### 알고리즘별 최소 구현 계약

#### SFT / off-policy baseline

- student initialization, optimizer, prompt subset, response-token budget,
  evaluation split과 seed를 OPD와 일치시킨다.
- hard-label SFT는 고정 teacher demonstration, soft off-policy KD는 같은 고정
  prefix의 teacher distribution을 사용한다.
- 단순히 더 많은 token/update를 본 방법이 이기지 않도록 optimizer step과
  processed response token을 모두 보고한다.
- base/SFT/KD/OPD의 train distribution과 inference distribution을 그림과
  실행 trace로 비교한다.
- OPD 승리를 완료 조건으로 만들거나 유리한 seed만 고르지 않는다. SFT가 이긴
  run도 보존하고 support overlap, teacher 품질, 분산과 함께 해석한다.

#### GKD / vanilla OPD

- 원 논문의 on/off-policy mixture λ와 flexible divergence를 구현한다.
- λ=0, λ=1 경계 동작을 테스트한다.
- 원 논문 방식과 최근 reasoning OPD 구현의 estimator 차이를 별도 API/이름으로
  분리한다.

#### TCOD

- vanilla multi-turn OPD와 동일 환경·seed에서 비교한다.
- F2B와 B2F schedule을 모두 구현하고 현재 curriculum depth를 log한다.
- 잘린 trajectory의 observation/action 정합성과 loss mask를 테스트한다.

#### SOD

- step boundary를 명시적으로 보존한다.
- 논문에 정의된 step divergence와 adaptive weight를 충실히 구현한다.
- weight 계산에 gradient가 흐르는지/끊기는지를 논문과 코드 근거로 설명하고
  테스트한다.

#### SAGE-OPD

- 공개 공식 코드가 있으면 라이선스 확인 후 그 의미를 보존한 최소 port를 한다.
- 공식 코드가 없으면 Algorithm/appendix를 기준으로 clean-room 구현한다.
- teacher-judged intervention, teacher confidence weighting, loss-scale
  normalization을 각각 분리하고 ablation 가능하게 한다.
- toy teacher-judge는 deterministic하게 검증하고, 실제 LLM judge 경로는
  backend 제약과 비용을 명확히 밝힌다.

---

## 7. 연구·코드 수집과 라이선스 절차

각 필수 알고리즘에 대해 다음 표를 `docs/sources.yml`과 사람이 읽는 문서에
남긴다.

- paper title, authors, venue/status, arXiv/version, 확인 날짜
- official repository URL, commit SHA, release/tag
- license SPDX identifier와 확인 경로
- 실제로 참고한 file/function/line 또는 algorithm/equation
- `copied`, `adapted`, `reimplemented` 중 provenance 분류
- 우리 구현과 upstream의 의도적 차이
- 재현한 실험과 재현하지 못한 실험

각 dataset/model에는 추가로 다음을 남긴다.

- Hub ID, exact revision, card URL과 license file URL
- config/split, row 수, compressed download와 예상 memory/cache 크기
- 원천 dataset과 생성 trace의 lineage 및 서로 다른 license/terms
- 학습 허용 범위, 재배포 여부, attribution/citation 요구사항
- loader가 실제로 받은 shard와 checksum
- test contamination 방지 split 정책과 dedup 여부

라이선스가 없거나 불명확하면 코드를 복사하지 않는다. 논문의 알고리즘 설명을
바탕으로 clean-room 구현하고 그 사실을 표시한다. 호환 가능한 코드를
가져오더라도 저작권 고지, LICENSE/NOTICE 요구사항, 원 링크를 보존한다.

성능 수치는 원 논문 결과, upstream 결과, 이 저장소 실행 결과를 서로 다른
열과 시각 스타일로 구분한다.

dataset/model license 표는 법률 자문으로 표현하지 않는다. 불명확하거나 card와
원 저장소가 충돌하면 다운로드·재배포를 멈추고 사용자에게 근거와 선택지를
제시한다.

---

## 8. 이중언어 정책

- 한국어 markdown을 먼저 기술 검토한 뒤 영어를 번역한다.
- 코드 셀, seed, config, 수식, figure data와 assertion은 언어판 간 동일하다.
- 코드의 identifier/docstring은 국제 기여를 위해 영어로 작성한다.
- 한국어 전문 용어 첫 등장에는 영어 원어를 병기한다.
- 기계적 직역 대신 학습 목표와 설명의 난이도를 보존한다.
- notebook metadata에 stable lesson/cell ID를 넣고 CI에서 다음을 검사한다.
  - lesson 수와 순서
  - code cell hash
  - exercise/checkpoint 수
  - concept-map position과 오답노트 item 수
  - source/citation ID
  - figure와 alt text 존재

README 기본 언어는 영어로 하되 문서 상단에서 한국어 README와 첫 강의로
바로 이동할 수 있게 한다. 한국어 사용자에게는 `README.ko.md`가 완전한
진입점이어야 한다.

---

## 9. 검증 전략

### 단위 테스트

- analytic categorical distribution으로 FKL/RKL/JSD 값 검산
- padding/prompt/response/turn mask
- reduction과 length normalization
- gradient 존재/부재 및 teacher parameter 불변성
- λ 경계, temperature, top-k 근사
- TCOD schedule과 trajectory slice
- SOD step weight
- SAGE intervention/confidence/normalization
- tokenizer/model compatibility validator
- dataset config/split/schema/revision과 download-size guard
- SFT/OPD 비교 config의 초기 weight·prompt·token/update budget 일치
- device fallback, Windows path, QLoRA capability preflight

### 통합 테스트

- toy OPD 한 step에서 finite loss와 student parameter update
- 짧은 학습에서 고정 평가셋 loss 또는 합의율이 합리적으로 개선
- 모든 필수 알고리즘 train→checkpoint→resume→eval
- 같은 초기 student의 SFT와 OPD가 같은 평가 harness와 예산을 사용했는지 감사
- `python -m opd_study.demo --profile toy`가 HTML/PNG/JSON/TensorBoard artifact 생성
- 같은 seed/config의 재실행 구조 검증
- CLI의 잘못된 모델 조합이 설명적인 오류로 종료
- 명시적 동의 없이 100MB 초과 download를 시도하지 않음

학습은 stochastic하므로 무리한 단일 수치 threshold 대신 작은 analytic test,
여러 seed의 넓고 정당한 sanity bound, NaN/Inf 및 회귀 검사를 결합한다.

### Notebook 검증

- `nbformat`으로 구조 검증
- 기본 toy profile로 모두 top-to-bottom 실행
- 출력 크기 제한, traceback/경고 잔존 여부 확인
- markdown 결론과 실제 셀 출력 대조
- concept-map position, “내가 자주 틀리는 것”, 결과 artifact 링크 존재
- 실행 시간 기록
- network나 server가 필요한 선택 셀은 기본 경로를 막지 않으며 정확한 실행
  명령과 필요한 자원을 제공

권장 실행 명령의 형태:

```bash
python -m pytest
python -m ruff check .
python -m mypy src
python scripts/check_bilingual_parity.py
python scripts/execute_notebooks.py --profile toy --language all
python scripts/check_links.py --local
python -m opd_study.demo --profile toy --non-interactive
```

실제 도구 선택 후 README와 CI에서 단일 명령으로 묶는다.

### Platform 검증

- Ubuntu latest: CPU toy 전체 + 가능하면 CUDA 별도 job
- macOS Apple Silicon: MPS probe, toy 전체, Qwen3 LoRA one-step smoke
- Windows latest: CPU toy 전체, path/cache/checkpoint/resume, PowerShell quickstart
- Colab free runtime: toy 전체와 accelerator가 있을 때 QLoRA one-step smoke

MPS/CUDA hardware가 CI에 없으면 해당 job은 “통과”로 꾸미지 않고 exact manual
test와 마지막 검증 날짜를 compatibility 표에 둔다. Colab notebook에는 CI용
headless 경로를 두고, 최소 release 전에는 새 runtime 수동 실행 증거를 남긴다.

### Experiment card

각 실행 기록에는 다음을 포함한다.

- git commit, config hash, dependency lock hash
- OS, CPU/GPU/MPS, RAM/VRAM, dtype
- model IDs와 exact revision, dataset와 revision
- dataset license/config/split/rows/downloaded shards와 checksum
- seed, train/eval split, metric 정의
- wall-clock, peak memory, checkpoint 경로
- 결과와 오차/분산
- 알려진 편차와 실패

---

## 10. 구현 체크포인트

Goal은 아래 순서로 진행하고 `PROGRESS.md`에 짧게 기록한다. 각 체크포인트
보고에는 현재 단계, 생성한 산출물, 실행한 검증, 남은 일, blocker를 쓴다.

### C0. 안전한 저장소 부트스트랩

- 현재 디렉터리·파일·Git 상태와 대상 원격 상태를 다시 확인한다.
- 부모 경로 쓰기 권한과 대상 경로 충돌을 확인한 뒤 `opd`를 `OPD-study`로
  변경한다. 작업 환경 제약으로 불가능하면 우회하지 말고 보고한다.
- 원격이 비어 있는지/기존 이력이 있는지 확인한다.
- Git 초기화 또는 기존 이력의 안전한 clone/merge 전략을 선택한다.
- `origin`을 `https://github.com/BangProx/OPD-study.git`으로 설정한다.
- force push와 원격 덮어쓰기는 금지한다.

검증: 실제 경로, `git status`, `git remote -v`, 기본 브랜치와 원격 이력 보고.

### C1. 문헌 조사와 승인 게이트

- 필수 문헌과 공식 코드를 수집하고 sources/provenance 초안을 만든다.
- 확정 dataset/model의 revision, 크기, license와 원천 lineage를 재검증한다.
- 최신 변형 후보 3~5개와 A1~A8의 비용/효과를 제안한다.
- 사용자 승인 전에는 scaffold 이상의 구현으로 진행하지 않는다.

검증: 출처 URL/version/license가 채워진 후보표와 승인 기록.

### C2. 교육·기술 설계 고정

- 12개 lesson별 objective, prerequisite, demo, exercise, source를 설계한다.
- 알고리즘 API, config schema, SFT fairness contract, profile, model/dataset
  compatibility를 확정한다.
- notebook style guide와 한영 용어집을 만든다.
- lesson 00 전체 개념 지도와 lesson별 위치/오답노트 template을 확정한다.

검증: 각 사용자 요구가 lesson/test/file에 매핑된 traceability matrix.

### C3. 핵심 수학과 toy runtime

- distribution/KL/mask utilities, tiny LM, toy tasks, logging을 구현한다.
- toy teacher bootstrap, SFT/off-policy KD baseline, 원 GKD와 vanilla/modern
  OPD를 구현한다.
- mini playground, 정적 report와 TensorBoard logging을 연결한다.
- unit/integration test를 우선 통과시킨다.

검증: offline CPU 공정 비교, demo artifact, analytic loss/gradient test.

### C4. 필수 최신 알고리즘

- multi-turn toy environment, TCOD, SOD, SAGE-OPD를 순서대로 구현한다.
- 각 구현 직후 vanilla 대비 동일 조건 sanity experiment를 실행한다.
- 논문 충실 부분과 교육용 단순화를 표시한다.

검증: 알고리즘별 unit/integration test, ablation, experiment card.

### C5. 한국어 강좌

- 00~11을 순서대로 작성하되 공통 패키지를 import한다.
- notebook 안에 거대한 중복 구현을 넣지 않고 핵심 코드만 작은 cell로 다시
  구성하거나 링크한다.
- 각 notebook을 작성 즉시 실행하고 결과 기반 설명을 쓴다.
- lesson 00 개념 지도와 각 lesson의 현재 위치/오답노트를 검토한다.

검증: 한국어 전체 notebook top-to-bottom 실행.

### C6. 영어 강좌와 parity

- 기술 검토가 끝난 한국어판을 자연스러운 영어로 옮긴다.
- 코드·수식·결과·exercise가 동일함을 자동 검사한다.

검증: 영어 전체 실행 및 bilingual parity 통과.

### C7. 실제 모델과 서버 경로

- 현재 hardware preflight 결과에 맞춘 laptop preset을 검증한다.
- GSM8K와 Qwen3-0.6B/1.7B LoRA의 최소 실제 smoke train/eval을 실행한다.
- 지원되는 CUDA/Windows 또는 Colab에서 QLoRA one-step 이상을 검증하고,
  macOS bitsandbytes는 preview 상태와 실제 검증 결과를 구분한다.
- server optional backend와 config를 문서화하고 가능한 범위에서 정적/단위
  검증한다. 실제 multi-GPU를 실행하지 못했으면 명확히 `UNVERIFIED`로 표시한다.

검증: 실행 가능한 명령, checkpoint/eval 결과, hardware 기록 또는 정확한
미검증 사유.

### C8. 공개 저장소 품질

- README, citation, contribution, security, troubleshooting, changelog를 완성한다.
- Windows/Linux/macOS CI와 필수 Colab quickstart를 추가한다. 승인된 경우
  Kaggle, site, literature snapshot을 추가한다.
- 링크, 설치, clean checkout quickstart를 검증한다.

검증: 새 임시 환경에서 문서의 quickstart를 그대로 실행.

### C9. 최종 감사와 인계

- 완료 조건 15개를 각각 증거 링크와 함께 점검한다.
- 코드·노트북·문서의 모순, stale output, 큰 artifact, secret을 감사한다.
- 원격 연결을 확인하되 commit/push/PR은 사용자가 별도로 승인한 범위에서만
  수행한다.

검증: 최종 test 명령 전체 통과, 남은 미검증 항목과 한계의 명시적 목록.

---

## 11. 중단·질문 조건

다음 상황에서 Goal은 추측으로 진행하지 않고 안전한 조사 결과와 선택지를
사용자에게 제시한다.

- 승인 게이트 A 또는 추가 알고리즘 범위가 미결정
- 저장소 LICENSE 선택이 미결정인데 제3자 코드를 가져와야 함
- target GitHub 저장소에 보존해야 할 기존 이력이 있음
- 로컬 디렉터리 이름 변경에 추가 권한 또는 workspace 재연결이 필요함
- upstream code license가 없거나 서로 충돌함
- 논문 알고리즘과 공식 구현이 의미 있게 다름
- 실제 모델 다운로드, 유료 API, cloud/GPU 사용이 필요함
- laptop 메모리 때문에 검증 범위를 바꿔야 함
- 성능 주장을 재현할 수 없거나 여러 출처가 충돌함
- 사용자 원본 파일 또는 변경사항과 겹침

질문 없이 진행 가능한 경우에는 합리적인 보수적 기본값을 사용하고 결정과
이유를 `docs/implementation-notes.md`에 기록한다.

---

## 12. 최종 인계 형식

완료 보고는 다음만 간결하게 포함한다.

1. 무엇이 완성되었는지와 추천 첫 링크
2. laptop에서 시작하는 정확한 명령
3. 전체 검증 명령과 결과
4. 실제 실행한 모델/알고리즘과 실행하지 못한 server 항목
5. 논문 구현과 교육용 구현의 알려진 차이
6. Git 원격·브랜치·commit/push 상태
7. 다음 유지보수 우선순위

Goal을 “완료”로 표시하기 전에 이 문서의 완료 조건과 체크포인트 증거를 다시
대조한다.

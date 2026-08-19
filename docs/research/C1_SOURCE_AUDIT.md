# C1 문헌·코드·라이선스 감사와 추가 범위 제안

> 감사 기준일: 2026-08-19 (Asia/Seoul)<br>
> 상태: 2026-08-19 추천안 4개 모두 사용자 승인<br>
> 원칙: 논문 내용과 코드 재사용 권한은 별개로 판단한다.

## 결론 먼저

필수 범위인 GKD, TCOD, SOD, SAGE-OPD는 그대로 구현한다. 추가 범위는 아래
4개를 추천한다. 새 노트북을 무작정 늘리지 않고 기존 12개 lesson 안에 배치해
학습 시간을 약 60~90분만 늘린다.

| 순위 | 추천 범위 | 분류 | 추가 학습 가치 | laptop 비용 |
|---:|---|---|---|---|
| 1 | Rethinking OPD의 성공 조건·overlap 진단·회복 recipe | 필수 진단 | OPD가 왜 실패하는지 관측하고 고치는 기준 | 낮음 |
| 2 | vOPD control-variate baseline | 심화 구현 | policy-gradient 관점과 분산 감소를 코드로 연결 | 낮음~중간 |
| 3 | OPD²와 영어/한국어 언어 유지 실험 | 심화 구현 | teacher-base delta와 한국어 학습의 실제 함정 | 중간 |
| 4 | test-time scaling 비판 실험 | 필수 평가 | avg@K 향상과 능력 경계 확장을 구분 | 낮음 |

추천 승인 문구는 다음과 같다.

```text
C1 추천안 4개를 모두 승인한다. Rethinking OPD와 test-time scaling은 필수
진단/평가로, vOPD와 OPD²+다국어는 심화 구현으로 12개 lesson에 통합한다.
라이선스 없는 upstream 코드는 복사하지 않고 clean-room으로 구현한다.
```

## 필수 문헌과 공식 코드

| 항목 | 고정 버전 | 공식 코드 | 코드 라이선스 | 구현 결정 |
|---|---|---|---|---|
| GKD / 원 OPD | arXiv `2306.13649v3`, ICLR 2024 | 원 저자 코드 링크 없음; 비교 기준으로 Hugging Face TRL commit `1e3ba4e80dfd8c64f11022a7ae47de6a58255ca5` | TRL Apache-2.0 | 논문 수식을 독립 구현하고 TRL의 API/행동과 교차 검증. 코드를 그대로 옮기지 않음 |
| TCOD | arXiv `2604.24005v3` | `kokolerk/TCOD@465eef4406ad0cff675b36bd46f37f28b1736ff9` | Apache-2.0 | 핵심 temporal curriculum을 의미 보존한 작은 port로 구현하고 NOTICE/출처 보존 |
| SOD | arXiv `2605.07725v3` | `YoungZ365/SOD@110c4b8e843aee274d3cd648199569369ee2403e` | Apache-2.0 | v3 논문이 2026-08-18 갱신됐지만 코드 최신 commit은 2026-05-22. 식·기본값 차이를 먼저 대조하고 차이가 있으면 논문 v3 우선 |
| SAGE-OPD | arXiv `2606.19659v1` | 기준일 현재 저자가 연결한 공식 저장소를 찾지 못함 | 해당 없음 | CC BY 4.0 논문을 근거로 selective intervention, confidence weighting, loss normalization을 clean-room 구현 |

논문 라이선스는 GKD·TCOD·SAGE-OPD가 CC BY 4.0이다. SOD 논문은 arXiv의
비독점 배포 라이선스이므로 본문·그림을 재배포하지 않고 수식과 아이디어를
독립적으로 설명한다. SOD 코드의 Apache-2.0 권한과 논문 라이선스를 혼동하지
않는다.

## 추가 후보 상세

### 1. Rethinking OPD — 승인 추천

- 출처: arXiv `2604.13016v2`, 공식 `thunlp/OPD` commit
  `4532fd35ccfdde82adc918b265e4c964534e83d1`.
- 배울 것: thinking-pattern compatibility, teacher novelty, student/teacher top-k
  support overlap, off-policy cold start, teacher-aligned prompt selection.
- 배치: lesson 07 실패 진단과 lesson 08 recipe.
- 구현: overlap ratio와 token advantage를 toy/실제-model 공통 metric으로 제공.
- 라이선스 판단: 논문은 CC BY 4.0이지만 공식 저장소 최상위에 `LICENSE`가
  없다. 저장소 코드는 복사하지 않고 논문 기반 clean-room 진단만 작성한다.
- 예상 비용: 코드 약 0.5일, 노트북 설명 약 0.5일, CPU toy 수 초.

### 2. vOPD — 승인 추천

- 출처: arXiv `2605.07865v1` (CC BY 4.0).
- 배울 것: sampled-token OPD를 policy gradient로 보는 법, action-independent
  control variate, detached reverse-KL baseline, top-k 근사와 편향/분산.
- 배치: lesson 06 현대 OPD와 lesson 07 안정성 실험.
- 구현: 정확한 full-vocabulary baseline과 top-k baseline을 모두 제공하고,
  평균 gradient와 gradient variance를 vanilla OPD와 seed 반복 비교.
- 코드 상태: 저자 project page는 `Riasok/vOPD`를 가리키지만 기준일 현재
  GitHub API에서 404이며 project-page 저장소에도 라이선스가 없다. 논문 식으로
  clean-room 구현한다.
- 예상 비용: 코드/테스트 1일, 실험·설명 0.5일, tiny 모델 5분 이내 목표.

### 3. OPD² + 다국어 — 승인 추천

- 출처: arXiv `2607.15161v1`, 다국어 확장 `2608.05802v1`, 공식
  `naver-ai/opd2@2dac53fbc80673677fb3bd1d8690392761ee3e52`.
- 배울 것: post-trained teacher와 teacher base의 확률 차이를 목표 신호로 쓰는
  이유, expectation correction/gating, 영어 학습이 한국어 정답을 영어로
  바꿀 수 있는 language-retention 문제.
- 배치: lesson 08 single-turn 변형과 lesson 12 capstone 평가.
- 구현: Apache-2.0 코드의 핵심 의미를 작은 backend에 이식하되 NOTICE와 변경
  내역을 남긴다. toy의 두 teacher 상태로 delta를 검증하고 실제 모델 실험은
  server profile로 둔다. 한국어/영어 소형 평가 fixture는 저장소가 직접 만든다.
- 주의: 공식 대규모 recipe는 최소 8×H100을 전제로 하므로 laptop에서 논문
  결과 재현을 약속하지 않는다.
- 예상 비용: 코드/테스트 1.5일, 다국어 lab 1일, laptop은 축약 smoke만.

### 4. Test-time scaling 관점의 한계 — 승인 추천

- 출처: arXiv `2608.11829v1`; 기준일 현재 공식 코드 미발견, 논문은 arXiv
  비독점 배포 라이선스.
- 배울 것: `avg@K`와 `pass@K`, 작은 K의 sampling efficiency와 큰 K의
  capability boundary, 평균 점수만으로 “새 능력을 배웠다”고 결론 내리는 오류.
- 배치: lesson 07 오답노트와 lesson 12 최종 report.
- 구현: 우리 checkpoint들에 공통 `K={1,2,4,8,...}` 평가와 문제별
  gained/lost-solvability 표를 추가. 논문 수치는 복제하지 않고 우리 실험만 표시.
- 예상 비용: metric/테스트 0.5일, toy 평가 수 분. 큰 K 실제-model 평가는
  server opt-in.

## 데이터셋 재검증

크기는 Hugging Face Dataset Viewer의 현재 Parquet 합계이며, 실행 시 고정 SHA를
사용한다. 100MB 초과 데이터는 자동 다운로드하지 않는다.

| 역할 | dataset/config | revision | license | 행 수 | 다운로드 크기 |
|---|---|---|---|---:|---:|
| 기본 실제 train/eval | `openai/gsm8k/main` | `740312add88f781978c0658806c59bc2815b9866` | MIT | train 7,473 + test 1,319 | 2,725,633 B |
| 심화 eval/server train | `EleutherAI/hendrycks_math` 전체 7 config | `21a5633873b6a120296cce3e2df9d5550074f4a3` | MIT | train 7,500 + test 5,000 | 4,883,857 B |
| server-only 선택 train | `open-r1/OpenR1-Math-220k/default` | `e4e141ec9dea9f8326f4d347be56105859b2bd68` | Apache-2.0 | train 93,733 | 2,149,897,914 B |
| 선택 LM 기초 예제 | `roneneldan/TinyStories/default` | `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` | CDLA-Sharing-1.0 | train 2,119,719 + validation 21,990 | 1,000,775,442 B |

MATH의 7개 config 합은 12,500행이다. OpenR1-Math 저장소 전체의 세 config는
중복된 변형을 합쳐 8,435,506,438 B이므로 `default`만 명시적으로 선택한다.
TinyStories는 reasoning 기준 데이터가 아니라 lesson 01의 선택적 언어모델
예제로만 사용한다.

## 모델 재검증

| profile 역할 | model | revision | license | 파라미터 | BF16 weight 근사 |
|---|---|---|---|---:|---:|
| laptop student | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | Apache-2.0 | 751,632,384 | 1.50 GB |
| laptop teacher | `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | Apache-2.0 | 2,031,739,904 | 4.06 GB |
| server teacher | `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` | Apache-2.0 | 4,022,468,096 | 8.04 GB |

BF16 수치는 parameter count × 2 bytes의 근사이며 optimizer, activation,
KV cache와 framework overhead를 포함하지 않는다. laptop preset은 메모리
preflight를 통과해야만 실제 모델을 load한다.

## 승인 후 적용할 안전 규칙

1. Apache-2.0 upstream을 port하면 파일별 source commit, NOTICE, 변경점을 남긴다.
2. 라이선스가 없거나 코드가 공개되지 않은 방법은 논문 기반 clean-room으로
   작성하고 upstream 코드와 line-by-line 유사성을 만들지 않는다.
3. SOD는 v3 논문과 5월 코드의 의미 차이를 표와 테스트로 먼저 고정한다.
4. 논문 성능 수치와 이 저장소의 toy/laptop 결과를 같은 표에서 혼동하지 않는다.
5. 새 방법은 기존 12개 lesson에 통합하고 fast path의 필수 시간은 늘리지 않는다.

## 제외한 후보

- 추가 multi-turn intervention 논문: TCOD/SOD/SAGE-OPD와 학습 목표가 겹쳐
  현재 강좌를 늘리는 이득이 작다.
- 효율화-only 변형: core backend 자체의 top-k/chunking/QLoRA 실험으로 먼저
  가르칠 수 있어 별도 알고리즘으로 채택하지 않는다.
- black-box teacher 변형: teacher logits에 접근하는 필수 OPD 계열과 전제가
  달라 12개 lesson의 집중도를 해친다. 향후 별도 track 후보로 남긴다.

# OPD-study Goal 진행 기록

## 2026-08-19 — C0~C2

- A1~A4, A7~A8, Apache-2.0과 추가 논문 4개를 사용자 승인 범위로 고정했다.
- Kaggle과 MkDocs는 2단계로 보류했다.
- 빈 public 원격 `https://github.com/BangProx/OPD-study.git`, branch `main`,
  push/admin 권한, force-push 없음 상태를 확인했다.
- 문헌·코드·데이터·모델 revision/license 감사와 12강 교육 설계, 한영 용어,
  notebook 형식, API/config/fairness 계약, 15개 완료조건 추적표를 만들었다.
- 현재 작업 디렉터리는 sandbox write-root 때문에 일시적으로
  `/Users/bangbyeonghun/Documents/nlp/opd`이다. C9 마지막에 충돌 여부를 다시
  확인한 뒤 `OPD-study`로 옮기며, 그 전에는 경로 완료를 주장하지 않는다.

## 2026-08-19 — C3~C4 완료

- TinyArithmetic-OPD 4096/512/512, 고정 character tokenizer, student 2×64와
  teacher 4×128 Transformer를 구현했다.
- SFT, off-policy KD, GKD, full/sampled OPD, vOPD, OPD², TCOD F2B/B2F, SOD,
  SAGE-OPD 및 multi-turn calculator 환경과 실패 진단을 구현했다.
- 모든 필수 CLI 알고리즘의 train→checkpoint→resume→eval, 동일 초기 student,
  response-token 예산, teacher freeze를 테스트한다. 중단 없는 2-step SFT와
  1-step+resume 모델 SHA-256도 일치한다.
- `python -m opd_study.demo`는 weights-only-safe checkpoint, JSON/JSONL, loss와
  분포 진단 PNG, 응답 비교 HTML, TensorBoard를 만든다. `--prompt`/`--interactive`
  playground와 SAGE 별도 실행을 실제 수행했다. smoke의 0점 exact accuracy와
  무의미한 짧은 응답은 수렴 결과가 아니라 배관 검사로 그대로 노출한다.

## 2026-08-19 — C5~C6 완료

- 한국어 12개와 영어 12개 notebook을 생성하고 두 언어 모두 Jupyter kernel로
  top-to-bottom 재실행했다.
- code cell/source/check/exercise parity, concept map, 현재 위치, 오답노트,
  alt text, stored execution을 자동 검사했다.
- Colab quickstart도 로컬 headless 실행을 통과했다. GitHub에 공개된 commit이
  없으므로 새 hosted Colab runtime 검증은 아직 `UNVERIFIED`다.

## 2026-08-19 — C7 구현 및 로컬 검증

- 실제 pinned GSM8K shard 2개(총 2,725,633B)를 MIT 동의 뒤 다운로드해 SHA-256,
  7473/1319 official row와 6961/512/1319 derived split을 검증했다.
- 실제 GSM8K 16 train/8 validation row로 tiny SFT one-step을 수행했다.
  validation gold NLL 4.3741936684이며 official test는 사용하지 않았다. 이 값은
  Qwen 성능 또는 accuracy 주장이 아니다.
- Qwen3-0.6B/1.7B LoRA/QLoRA preflight와 opt-in `research-train`을 구현했다.
  모델·dataset 동의가 없으면 캐시/출력 생성 전에 실패한다. 실행 시 adapter,
  optimizer/RNG, shard checksum, config/dependency hash, 메모리, validation sample,
  experiment card를 남긴다.
- 현재 macOS arm64/Python 3.10.12/PyTorch 2.1.0 기본 환경에는 CUDA/MPS와
  research optional package가 없다. 별도 Python 3.12/PyTorch 2.13.0 검증 환경에서는
  Transformers 4.57.6, PEFT 0.20.0, Datasets 4.8.5, Accelerate 1.14.0의 실제
  import/API와 전체 테스트를 검증했다. 약 5.57GB Qwen 모델 다운로드와 실제 update,
  CUDA QLoRA, multi-GPU server는 사용자 승인/하드웨어 전까지 `UNVERIFIED`다.

## 2026-08-19 — C8 로컬 품질 게이트 완료

- 임시 venv editable 설치, wheel build(76,979B), wheel-only 설치와 CLI demo를
  실제 검증했다.
- `python scripts/quality_gate.py`: 35 pytest tests, Ruff, strict mypy 44 source
  modules, 24 executed notebooks/parity, 24 source records, 34 local links 통과.
- 24개 course notebook과 Colab notebook을 다시 실행했고 절대 로컬 경로,
  traceback, warning이 저장 출력에 남지 않음을 검사했다.
- 외부 README 링크와 9개 논문의 literature snapshot을 실제 조회했다. Hugging
  Face Papers에 미색인된 2개는 `not-indexed`로 명시했다.
- Linux/macOS/Windows Python 3.10/3.12 CI와 nightly notebook/source/link/snapshot
  workflow는 작성했지만 원격 실행 증거는 publish 전이므로 만들지 않았다.
- CUDA-only `gsm8k_qlora.yaml`과 Colab의 기본 비활성 QLoRA smoke cell을 추가했다.
  headless 검사는 동의 flag, CUDA guard, 기본 skip/no-download 상태까지 확인한다.
- 연구 의존성은 PyTorch 2.1과 최신 Transformers/PEFT의 실제 import 비호환을
  검출한 뒤 `torch>=2.2`로 분리했다. preflight는 설치 유무뿐 아니라 version과
  import 실패도 다운로드 전에 blocker로 보고한다. Python 3.12/PyTorch 2.13.0
  연구 환경에서도 35 tests, Ruff, mypy 44 modules와 전체 quality gate를 통과했다.
- 기본 PyTorch 2.1.0 환경에서도 35개 core/research-gate/CLI unittest가 통과했다.
- Markdown과 notebook cell을 합친 링크 148개(로컬 132, 외부 16)를 재검증했다.
- 최신 setuptools의 PEP 639 규약에 맞춰 `Apache-2.0` SPDX expression과
  LICENSE/NOTICE wheel metadata를 적용했다. 격리 wheel build는 경고 없이
  성공했고, wheel-only 설치 환경의 SFT/OPD demo도 artifact를 생성했다.
- 공개 대상은 123 files/약 1.3MB이며 1MB 초과 파일과 일반적인 secret pattern이
  없음을 검사했다. 실제 dataset, Colab 산출물, build/egg-info는 ignore된다.

## 2026-08-19 — C9 독립 재감사 보강

- 기존 40–50분 강의가 짧은 공통 template에 지나치게 의존한다는 품질 gap을
  발견했다. 24개 notebook 모두에 핵심 수식/상태분포, 실제 production 함수
  원문, 구현 이유, 대안과 trade-off, 강의별 exercise와 오답노트를 추가했다.
- checker는 언어별 최소 설명 깊이, source probe의 실제 저장 출력, 설계/대안
  section, 12개 고유 exercise/오답노트를 새로 강제한다. 보강본 24개를 새
  kernel에서 재실행했고 parity, 경고/traceback/절대 경로 감사를 통과했다.
- demo report는 held-out sample의 raw response를 방법별로 나란히 보여주며
  NLL/accuracy/agreement 외에 student/teacher entropy와 named FKL/RKL을 기록한다.
  distribution diagnostic PNG와 동일 metric의 TensorBoard scalar도 생성한다.
- 최신 PyTorch의 안전 기본값에서 dataclass checkpoint가 로드되지 않는 문제를
  발견했다. model config를 검증된 primitive mapping으로 직렬화하고 모든 제품
  checkpoint 로드를 `weights_only=True`로 고정했다. PyTorch 2.1과 2.13에서
  playground 및 resume/eval 경로를 확인했다.
- 등록된 10개 algorithm 모두 실제 CLI `train --smoke → eval` 경로로 artifact를
  만드는 통합 테스트를 추가했고, TCOD B2F teacher prefix가 context로만 남고
  student loss에서 제외되는지를 별도로 검증한다.
- 보강 후 전체 quality gate는 35 tests, Ruff, mypy 44 modules, 24 executed
  notebook/parity, Colab/source/local-link 검사를 통과했다.
- research runner도 config에 맞춰 loss/gradient/eval TensorBoard events를 남긴다.
  Colab optional QLoRA 검사는 실제 opt-in 시 adapter, safe optimizer checkpoint,
  metrics, experiment card와 TensorBoard artifact를 모두 요구한다.
- CI action major tag를 공식 저장소에서 2026-08-19 재확인한 immutable commit
  SHA로 고정했다.

## C9 남은 승인·외부 검증

- 약 5.57GB Qwen3 LoRA one-step 실실행 여부
- 첫 commit/push 후 3-OS GitHub Actions와 새 hosted Colab runtime 검증
- 마지막 로컬 디렉터리 `opd` → `OPD-study` 이동
- 사용자 별도 승인 전에는 commit/push/PR을 수행하지 않는다.
- C9 재확인에서 C0 당시 비어 있던 원격 `main`에 사용자 명의의 초기 commit
  `6f1e395`가 생긴 것을 발견했다. 내용은 2줄 README와 Python `.gitignore`뿐이나
  계약에 따라 임의 merge/reset하지 않았다. 권장안은 이 commit을 부모로 보존한
  뒤 현재 로컬 내용을 적용해 첫 구현 commit을 만드는 것이다.

## C9 승인 대기 상태

- 승인 없이 가능한 코드·강의·문서·패키징·보안·링크 감사는 완료했다.
- 원격 `main`은 여전히 `6f1e395`이며 로컬 123개 공개 대상 파일은 commit되지
  않은 상태다. 기존 commit을 부모로 보존한 fast-forward commit/push 승인이
  필요하다.
- Qwen3-0.6B/1.7B 실제 LoRA/QLoRA 검증에는 Apache-2.0 모델 약 5.57GB 다운로드와
  CUDA/Colab 사용 승인이 필요하다. 승인 전에는 다운로드·cloud 실행을 하지 않는다.
- 위 두 승인 없이는 3-OS 원격 CI, 새 hosted Colab, 실제 Qwen update와 최종
  `opd` → `OPD-study` 경로 인계를 증명할 수 없어 Goal을 승인 대기로 둔다.

## 2026-08-19 — C9 승인 재개 및 hermetic 검증

- 사용자가 원격 초기 commit `6f1e395`를 부모로 보존하는 fast-forward 게시와
  무료 Colab CUDA에서 pinned Qwen3-0.6B/1.7B LoRA·QLoRA smoke를 모두 승인했다.
- Colab quickstart는 LoRA와 QLoRA를 별도 opt-in flag로 실행하며 같은 model/data
  cache를 재사용하도록 보강했다. 기본 실행은 두 flag 모두 `False`라 CPU-safe이고
  모델을 다운로드하지 않는다.
- `/private/tmp/opd-study-hermetic-20260819`에
  `include-system-site-packages = false`인 Python 3.10 venv를 새로 만들었다. 해당
  환경의 유일한 site-packages와 PyTorch, Transformers, PEFT, Datasets 설치 경로가
  모두 venv 내부임을 확인해 Miniforge base 의존성을 배제했다.
- 이 hermetic 환경에서 35 tests와 20 subtests, Ruff, strict mypy 44 modules,
  한국어·영어 24개 notebook의 새 kernel top-to-bottom 실행, bilingual parity,
  Colab 기본/opt-in 계약, 24 source records, 132 local links를 통과했다.
- GitHub CLI는 Homebrew로 설치했지만 아직 GitHub 인증 세션이 없다. 게시 지침에
  따라 `gh auth login` 완료 전에는 commit/push를 진행하지 않는다. 인증 뒤 원격
  CI와 hosted Colab 검증을 계속한다.

## 2026-08-19 — C9 원격 CI와 hosted CUDA 검증

- GitHub 인증 뒤 원격 초기 commit `6f1e395`를 부모로 보존해 `main`에
  fast-forward 게시했다. core CI는 Linux/macOS/Windows × Python 3.10/3.12와
  research import job을 통과했고, nightly는 한국어·영어 24개 notebook 재실행,
  Colab/source/link/literature 검사와 artifact 업로드를 통과했다.
- 게시된 Colab을 새 Tesla T4 런타임에서 실행하며 editable install 직후 현재
  kernel이 `.pth`를 다시 읽지 않아 생기는 import 실패를 발견했다. clone의
  `src`를 즉시 `sys.path`에 넣도록 수정하고 회귀 검사와 `--language colab`
  실행 경로를 추가했다. 수정 commit `372c19f`의 hosted 기본 경로가 report,
  PNG, TensorBoard까지 통과했다.
- Colab 이미지의 PyTorch 2.11.0+cu128/Transformers 4.57.6 조합에 선택 설치된
  `torchao 0.10.0`이 호환되지 않는 문제를 실제 오류로 확인했다. OPD-study는
  torchao를 사용하지 않으므로 optional CUDA 셀에서만 `<0.16`을 감지·제거하는
  guard를 추가했다. 기본 CPU-safe 경로는 어떤 package도 제거하지 않는다.
- 같은 hosted T4/cache에서 pinned Qwen3-0.6B student와 Qwen3-1.7B teacher로
  LoRA와 QLoRA sampled reverse-KL OPD를 각각 한 스텝 실행했다. 두 실행 모두
  adapter, optimizer checkpoint, metrics, one-row validation, TensorBoard와
  experiment-card 계약을 통과했고 마지막 셀은 `Selected CUDA smoke paths
  completed.`를 출력했다.
- exact 값과 package/model/data revision, shard checksum, 한계는
  `docs/research/colab-cuda-smoke-2026-08-19.json`에 보존했다. 한 스텝의 0점
  exact accuracy는 성능 결과가 아니라 배관 검증임을 계속 명시한다.

## 2026-08-19 — C9 최종 인계

- 최종 CUDA 증거 commit `99a0866`에서 core CI `32255377916`은 Linux/macOS/
  Windows × Python 3.10/3.12와 research import를 모두 통과했다. 수동 실행한
  nightly `32255392705`도 24개 notebook 재실행, Colab/source/link/literature
  검사와 artifact 업로드를 통과했다.
- worktree와 `origin/main`이 같은 HEAD임을 확인하고, sibling target이 없음을
  검사한 뒤 로컬 디렉터리를 `/Users/bangbyeonghun/Documents/nlp/OPD-study`로
  이동했다. 최종 폴더명, GitHub repository 이름, origin URL이 모두
  `OPD-study`로 일치한다.

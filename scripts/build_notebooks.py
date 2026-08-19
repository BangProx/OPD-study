"""Generate the mirrored Korean/English tutorial notebooks from reviewed lesson data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nbformat
import yaml

REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = yaml.safe_load((REPOSITORY / "docs/sources.yml").read_text())
SOURCE_RECORDS = {
    item["id"]: (section, item)
    for section in ("papers", "code_sources", "datasets", "models")
    for item in SOURCE_MANIFEST[section]
}


@dataclass(frozen=True)
class Lesson:
    lesson_id: str
    slug: str
    title_ko: str
    title_en: str
    minutes: int
    track: tuple[str, ...]
    sources: tuple[str, ...]
    objectives_ko: tuple[str, ...]
    objectives_en: tuple[str, ...]
    position_ko: str
    position_en: str
    explanation_ko: str
    explanation_en: str
    prediction_ko: str
    prediction_en: str
    code_cells: tuple[str, ...]
    check_code: str
    exercise_ko: str
    exercise_en: str
    solution_ko: str
    solution_en: str
    mistakes_ko: str
    mistakes_en: str
    summary_ko: tuple[str, str, str]
    summary_en: tuple[str, str, str]


SETUP = """from pathlib import Path
import sys
import torch

repo_root = Path.cwd()
if not (repo_root / "src").exists():
    repo_root = Path.cwd().parents[1]
sys.path.insert(0, str(repo_root / "src"))

import opd_study
from opd_study.device import resolve_device
from opd_study.utils import seed_everything

seed_everything(42)
device_report = resolve_device("cpu")
print({"lesson": LESSON_ID, "opd_study": opd_study.__version__,
       "torch": torch.__version__, "device": device_report.selected,
       "profile": "toy", "network": "not required"})"""


LESSON_CODE: dict[str, tuple[tuple[str, ...], str]] = {
    "L00": (("""from opd_study.math import reverse_kl_from_logits

teacher_logits = torch.log(torch.tensor([0.70, 0.20, 0.10]))
initial_logits = torch.tensor([0.0, 0.0, 0.0])

def fit(objective: str) -> list[float]:
    logits = torch.nn.Parameter(initial_logits.clone())
    optimizer = torch.optim.SGD([logits], lr=0.8)
    history = []
    for _ in range(8):
        loss = (-torch.log_softmax(logits, -1)[0] if objective == "sft"
                else reverse_kl_from_logits(teacher_logits, logits))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        history.append(float(loss.detach()))
    return history

curves = {name: fit(name) for name in ("sft", "opd")}
print({name: [round(value, 3) for value in values] for name, values in curves.items()})""",
    """import matplotlib.pyplot as plt

for name, color in (("sft", "#0072B2"), ("opd", "#009E73")):
    plt.plot(curves[name], marker="o", label=name, color=color)
plt.xlabel("update"); plt.ylabel("training objective"); plt.legend(); plt.grid(alpha=.2)
plt.show()"""),
    """assert curves["sft"][-1] < curves["sft"][0]
assert curves["opd"][-1] < curves["opd"][0]
assert curves["sft"] != curves["opd"]
print("check passed: same initialization, different objectives, both decrease")"""),
    "L01": (("""logits = torch.tensor([2.0, 1.0, 0.0])
probabilities = torch.softmax(logits, dim=-1)
manual = logits.exp() / logits.exp().sum()
print("probabilities:", probabilities.tolist())
print("sum:", probabilities.sum().item(), "manual match:", torch.allclose(probabilities, manual))""",
    """from opd_study.models import TinyCausalLM, TinyTransformerConfig

config = TinyTransformerConfig(vocab_size=12, max_sequence_length=16,
                               number_of_layers=1, hidden_size=16,
                               number_of_heads=4, feed_forward_size=32)
model = TinyCausalLM(config).eval()
prefix_a = torch.tensor([[1, 4, 5, 6]])
prefix_b = prefix_a.clone(); prefix_b[0, -1] = 7
with torch.no_grad():
    logits_a, logits_b = model(prefix_a), model(prefix_b)
print("shape:", tuple(logits_a.shape),
      "past unchanged:", torch.allclose(logits_a[:, :-1], logits_b[:, :-1]))"""),
    """assert probabilities.shape == (3,)
assert torch.isclose(probabilities.sum(), torch.tensor(1.0))
assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1])
print("check passed: normalized next-token probabilities and causal prefix")"""),
    "L02": (("""from opd_study.math import (entropy_from_logits, forward_kl_from_logits,
                            generalized_jsd_from_logits, reverse_kl_from_logits)

teacher = torch.log(torch.tensor([[0.70, 0.20, 0.10]]))
student = torch.log(torch.tensor([[0.40, 0.35, 0.25]]))
values = {
    "H(teacher)": entropy_from_logits(teacher).item(),
    "KL(teacher||student)": forward_kl_from_logits(teacher, student).item(),
    "KL(student||teacher)": reverse_kl_from_logits(teacher, student).item(),
    "JSD_beta=.5": generalized_jsd_from_logits(teacher, student, beta=.5).item(),
}
print({name: round(value, 5) for name, value in values.items()})""",
    """boundaries = [generalized_jsd_from_logits(teacher, student, beta=beta).item()
              for beta in (0.0, 0.25, 0.5, 0.75, 1.0)]
print("beta sweep:", [round(value, 5) for value in boundaries])
print("Argument order is part of the definition; KL is not symmetric.")"""),
    """assert values["KL(teacher||student)"] >= 0
assert values["KL(student||teacher)"] >= 0
assert abs(boundaries[0] - values["KL(teacher||student)"]) < 1e-6
assert abs(boundaries[-1] - values["KL(student||teacher)"]) < 1e-6
print("check passed: non-negativity and named beta boundaries")"""),
    "L03": (("""import copy
from opd_study.algorithms import off_policy_kd_loss, score_teacher, supervised_fine_tuning_loss
from opd_study.data import CharacterTokenizer, collate_examples, generate_tiny_arithmetic
from opd_study.models import TinyCausalLM, TinyTransformerConfig

tokenizer = CharacterTokenizer(); splits = generate_tiny_arithmetic(train_rows=8, validation_rows=2, test_rows=2)
batch = collate_examples(splits.train[:2], tokenizer)
config = TinyTransformerConfig(vocab_size=tokenizer.vocab_size, number_of_layers=1,
    hidden_size=32, number_of_heads=4, feed_forward_size=64)
initial = TinyCausalLM(config); teacher = TinyCausalLM(config)
sft_student = copy.deepcopy(initial); kd_student = copy.deepcopy(initial)
teacher_signals = score_teacher(teacher, batch)
sft_output = supervised_fine_tuning_loss(sft_student(batch.token_ids, batch.attention_mask), batch)
kd_output = off_policy_kd_loss(kd_student(batch.token_ids, batch.attention_mask), batch, teacher_signals)
print("same response targets:", int(sft_output.effective_mask.sum()), int(kd_output.effective_mask.sum()))""",
    """from opd_study.utils import model_state_hash

print("initial hashes equal:", model_state_hash(sft_student) == model_state_hash(kd_student))
print("SFT reads hard target IDs; off-policy KD reads teacher distributions on the same fixed prefixes.")"""),
    """assert model_state_hash(sft_student) == model_state_hash(kd_student)
assert int(sft_output.effective_mask.sum()) == int(kd_output.effective_mask.sum())
assert not teacher_signals.logits.requires_grad
print("check passed: initialization, state source, token budget, and teacher detach are auditable")"""),
    "L04": (("""from opd_study.math import generalized_jsd_from_logits

teacher = torch.randn(2, 5, 7)
student = torch.randn(2, 5, 7, requires_grad=True)
for beta in (0.0, 0.5, 1.0):
    value = generalized_jsd_from_logits(teacher, student, beta=beta).mean()
    print(f"beta={beta}: {value.item():.5f}")
print("beta=0 is forward KL; beta=1 is reverse KL in this repository's convention.")""",
    """generator = torch.Generator().manual_seed(42)
lambda_on_policy = 0.5
state_sources = ["student" if torch.rand((), generator=generator) < lambda_on_policy
                 else "fixed" for _ in range(8)]
print("GKD state sources:", state_sources)
print("lambda chooses states; beta chooses the divergence. They are different knobs.")"""),
    """forward = generalized_jsd_from_logits(teacher, student, beta=0.0)
reverse = generalized_jsd_from_logits(teacher, student, beta=1.0)
assert forward.shape == reverse.shape == (2, 5)
assert set(state_sources) == {"fixed", "student"}
print("check passed: [B,T,V] -> [B,T], with separate lambda and beta")"""),
    "L05": (("""from opd_study.algorithms import (collect_student_trajectories,
    on_policy_distillation_loss, score_teacher)
from opd_study.data import CharacterTokenizer, generate_tiny_arithmetic
from opd_study.models import TinyCausalLM, TinyTransformerConfig

tokenizer = CharacterTokenizer(); splits = generate_tiny_arithmetic(train_rows=4, validation_rows=1, test_rows=1)
config = TinyTransformerConfig(vocab_size=tokenizer.vocab_size, number_of_layers=1,
    hidden_size=32, number_of_heads=4, feed_forward_size=64)
student, teacher = TinyCausalLM(config), TinyCausalLM(config)
trajectories = collect_student_trajectories(student, [row.prompt for row in splits.train[:2]],
    tokenizer, max_new_tokens=4, min_new_tokens=4, temperature=0.0)
signals = score_teacher(teacher, trajectories)
student_logits = student(trajectories.token_ids, trajectories.attention_mask)
output = on_policy_distillation_loss(student_logits, trajectories, signals)
print("shapes:", trajectories.token_ids.shape, student_logits.shape, output.token_loss.shape)
print("response tokens:", int(trajectories.response_mask.sum()), "loss:", float(output.loss.detach()))""",
    """optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
optimizer.zero_grad(); output.loss.backward(); optimizer.step()
print("teacher has gradients:", any(parameter.grad is not None for parameter in teacher.parameters()))
print("rollout snapshot detached:", not trajectories.student_logprobs.requires_grad)"""),
    """assert not trajectories.response_mask[:, :trajectories.prompt_lengths.min()].any()
assert output.loss.requires_grad
assert not any(parameter.grad is not None for parameter in teacher.parameters())
print("check passed: sample -> detached state -> teacher no_grad -> recomputed student update")"""),
    "L06": (("""from opd_study.config import load_config
from opd_study.device import require_qlora

toy = load_config(repo_root / "configs/toy/default.yaml")
laptop = load_config(repo_root / "configs/laptop/gsm8k_lora.yaml")
qlora = load_config(repo_root / "configs/laptop/gsm8k_qlora.yaml")
print("toy:", toy.profile, toy.backend, toy.data.id)
print("laptop pins:", laptop.model.student, laptop.model.student_revision)
print("CUDA preset:", qlora.model.finetuning, qlora.training.device, qlora.training.precision)
print("download estimate is documented before any network call:", laptop.data.expected_download_bytes)""",
    """try:
    require_qlora(device_report)
except RuntimeError as error:
    print("safe QLoRA block:", error)

RUN_OPTIONAL_NETWORK = False
print("Qwen/GSM8K smoke enabled:", RUN_OPTIONAL_NETWORK)
print("Exact command: opd-study research-train --config configs/laptop/gsm8k_lora.yaml --smoke --accept-dataset-license --accept-model-license")"""),
    """assert len(laptop.model.student_revision) == 40
assert laptop.model.trust_remote_code is False
assert qlora.model.finetuning == "qlora" and qlora.training.device == "cuda"
assert RUN_OPTIONAL_NETWORK is False
print("check passed: pinned assets, no remote code, unsupported QLoRA does not fall back")"""),
    "L07": (("""from opd_study.diagnostics import support_diagnostics

student = torch.tensor([[[4., 3., 1., 0.], [0., 1., 3., 4.]]])
compatible = student + torch.tensor([[[.1, 0., 0., 0.], [0., 0., 0., .1]]])
incompatible = student.flip(-1)
mask = torch.tensor([[True, True]])
for name, teacher in (("compatible", compatible), ("incompatible", incompatible)):
    result = support_diagnostics(student, teacher, mask, top_k=2)
    print(name, {"overlap": result.overlap_ratio, "entropy_gap": result.absolute_entropy_gap})""",
    """from opd_study.algorithms import vopd_loss
from opd_study.data import CharacterTokenizer, collate_examples, generate_tiny_arithmetic
from opd_study.types import TeacherSignals

tokenizer = CharacterTokenizer(); row = generate_tiny_arithmetic(train_rows=1, validation_rows=1, test_rows=1).train
batch = collate_examples(row, tokenizer)
shape = (*batch.token_ids.shape, tokenizer.vocab_size)
student_logits = torch.randn(shape, requires_grad=True); teacher_logits = torch.randn(shape)
vopd = vopd_loss(student_logits, batch, TeacherSignals(logits=teacher_logits), baseline_top_k=8)
print("vOPD reward/advantage:", vopd.metrics["vopd/mean_reward"], vopd.metrics["vopd/mean_advantage"])
print("The top-k KL is a detached baseline, not the optimized target.")"""),
    """good = support_diagnostics(student, compatible, mask, top_k=2)
bad = support_diagnostics(student, incompatible, mask, top_k=2)
assert good.overlap_ratio > bad.overlap_ratio
assert vopd.loss.requires_grad
print("check passed: overlap diagnoses state compatibility; vOPD keeps sampled-token gradients")"""),
    "L08": (("""from opd_study.envs import CalculatorEnvironment

environment = CalculatorEnvironment(2, (("+", 3), ("*", 4)))
print(environment.reset().observation)
after_error = environment.step(6)
print(after_error.observation)
final = environment.step(24)
print(final.observation)""",
    """environment.reset()
correct_first = environment.step(5)
correct_final = environment.step(20)
print("correct path:", correct_first.observation, "->", correct_final.observation)
print("The next observation depends on the student's previous action.")"""),
    """assert after_error.value == 6
assert final.target == 20 and not final.success
assert correct_final.success
print("check passed: an early action changes later states and final success")"""),
    "L09": (("""from opd_study.algorithms.tcod import curriculum_depth, temporal_curriculum_mask
from opd_study.data import CharacterTokenizer, collate_multiturn_text, generate_tiny_arithmetic

tokenizer = CharacterTokenizer(); rows = generate_tiny_arithmetic(train_rows=2, validation_rows=1, test_rows=1).train
batch = collate_multiturn_text([(row.prompt, tuple(row.response.splitlines())) for row in rows], tokenizer)
schedule = [curriculum_depth(step, start_depth=1, pacing_steps=2, maximum_depth=3)
            for step in range(6)]
print("depth schedule:", schedule)
for direction in ("f2b", "b2f"):
    mask = temporal_curriculum_mask(batch, depth=1, direction=direction)
    print(direction, "selected tokens:", int(mask.sum()))""",
    """early = temporal_curriculum_mask(batch, depth=1, direction="f2b")
late = temporal_curriculum_mask(batch, depth=1, direction="b2f")
print("F2B learns beginnings; B2F needs a successful/teacher prefix before this suffix mask.")
print("overlap between one-turn windows:", int((early & late).sum()))"""),
    """assert schedule == [1, 1, 2, 2, 3, 3]
assert not (early & late).any()
assert batch.turn_ids is not None
print("check passed: pacing and temporal slices are explicit")"""),
    "L10": (("""from opd_study.algorithms.sod import step_divergence_weights
from opd_study.data import CharacterTokenizer, collate_multiturn_text, generate_tiny_arithmetic

tokenizer = CharacterTokenizer(); rows = generate_tiny_arithmetic(train_rows=2, validation_rows=1, test_rows=1).train
batch = collate_multiturn_text([(row.prompt, tuple(row.response.splitlines())) for row in rows], tokenizer)
shape = (*batch.token_ids.shape, tokenizer.vocab_size)
student_logits, teacher_logits = torch.randn(shape), torch.randn(shape)
divergence, sod_weights = step_divergence_weights(student_logits, teacher_logits,
    batch.token_ids, batch.response_mask, batch.step_ids)
print("SOD mean divergence/weight:", float(divergence[divergence > 0].mean()),
      float(sod_weights[sod_weights > 0].mean()))""",
    """from opd_study.algorithms.sage_opd import sage_token_weights

number_of_turns = int(batch.turn_ids.max()) + 1
intervention = torch.tensor([[0.0, 0.5, 1.0]]).repeat(batch.token_ids.shape[0], 1)
sage_weights, confidence = sage_token_weights(teacher_logits, batch, intervention)
print("SAGE normalized token-weight sum:", float(sage_weights.sum()))
print("response token count:", int(batch.response_mask[:, 1:].sum()))
print("turn confidence row 0:", confidence[0].tolist())"""),
    """assert not divergence.requires_grad and not sod_weights.requires_grad
assert abs(float(sage_weights.sum()) - int(batch.response_mask[:, 1:].sum())) < 1e-4
assert (sage_weights >= 0).all()
print("check passed: SOD weights detach; SAGE preserves dense-OPD loss scale")"""),
    "L11": (("""from opd_study.algorithms import opd2_loss
from opd_study.data import CharacterTokenizer, collate_examples, generate_tiny_arithmetic
from opd_study.types import TeacherSignals

tokenizer = CharacterTokenizer(); rows = generate_tiny_arithmetic(train_rows=2, validation_rows=1, test_rows=1).train
batch = collate_examples(rows, tokenizer)
shape = (*batch.token_ids.shape, tokenizer.vocab_size)
student = torch.randn(shape, requires_grad=True)
teacher, teacher_base = torch.randn(shape), torch.randn(shape)
delta_output = opd2_loss(student, batch, TeacherSignals(logits=teacher),
    TeacherSignals(logits=teacher_base), centering_top_k=16)
print("OPD2 gate rate:", delta_output.metrics["opd2/gate_rate"])
print("Delta isolates teacher post-training change; multilingual retention must be evaluated separately.")""",
    """from opd_study.diagnostics.test_time_scaling import gained_and_lost_solvability, scaling_metrics

before = torch.tensor([[1,0,0,0], [0,0,1,0], [0,0,0,0]], dtype=torch.bool)
after = torch.tensor([[0,0,0,0], [1,1,0,0], [1,0,0,0]], dtype=torch.bool)
for k in (1, 2, 4):
    metric = scaling_metrics(after, k=k)
    print(k, "avg@K", metric.avg_at_k, "pass@K", metric.pass_at_k)
print("solvability:", gained_and_lost_solvability(before, after, k=4))"""),
    """metric = scaling_metrics(after, k=4)
assert metric.avg_at_k != metric.pass_at_k
changes = gained_and_lost_solvability(before, after, k=4)
assert sum(changes.values()) == before.shape[0]
assert 0 <= delta_output.metrics["opd2/gate_rate"] <= 1
print("check passed: delta gating and capability-boundary accounting are explicit")"""),
}


BASE = [
    ("L00", "opd_in_15_minutes", "15분 OPD 체험과 전체 지도", "OPD in 15 minutes", 20, ("fast", "full"), ("gkd",)),
    ("L01", "tokens_probabilities", "토큰·확률·자기회귀 LM", "Tokens, probabilities, autoregressive LMs", 30, ("full",), ("gkd",)),
    ("L02", "ce_kl_kd", "CE·엔트로피·KL·KD", "CE, entropy, KL, and KD", 40, ("fast", "full"), ("gkd",)),
    ("L03", "state_sources", "SFT·오프폴리시·온폴리시", "SFT, off-policy, and on-policy", 40, ("fast", "full"), ("gkd",)),
    ("L04", "original_gkd", "원 논문 GKD 해부", "Dissecting original GKD", 45, ("fast", "full"), ("gkd", "trl_gkd_reference")),
    ("L05", "modern_opd", "현대 OPD from scratch", "Modern OPD from scratch", 50, ("fast", "full"), ("gkd", "vopd")),
    ("L06", "real_training", "실제 train/eval", "Real training and evaluation", 50, ("full",), ("gkd", "openai/gsm8k", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B")),
    ("L07", "diagnostics_vopd", "왜 성공하거나 실패하는가", "Why OPD succeeds or fails", 40, ("full",), ("rethinking_opd", "vopd")),
    ("L08", "multiturn_agents", "멀티턴 agent OPD", "Multi-turn agent OPD", 35, ("full",), ("tcod", "sod", "sage_opd")),
    ("L09", "tcod", "TCOD", "Temporal Curriculum OPD", 40, ("full",), ("tcod", "tcod_official")),
    ("L10", "sod_sage", "SOD와 SAGE-OPD", "SOD and SAGE-OPD", 45, ("full",), ("sod", "sage_opd", "sod_official")),
    ("L11", "integration", "통합 비교와 다음 연구", "Integrated comparison and next research", 45, ("fast", "full"), ("opd2", "opd2_multilingual", "opd_test_time_scaling")),
]


OBJECTIVES_KO = [
    ("전체 학습 지도를 읽는다", "SFT와 OPD의 첫 차이를 실행한다", "빠른/전체 경로를 고른다"),
    ("logit을 확률로 바꾼다", "next-token shape를 읽는다", "causal prefix를 확인한다"),
    ("CE·entropy·KL을 계산한다", "KL 방향을 말로 설명한다", "JSD beta 경계를 확인한다"),
    ("학습 state의 생성자를 구분한다", "SFT와 KD를 공정 비교한다", "response budget을 감사한다"),
    ("GKD의 lambda와 beta를 분리한다", "일반화 JSD를 구현한다", "gradient 경로를 추적한다"),
    ("student rollout부터 업데이트까지 잇는다", "full/sampled estimator를 구분한다", "teacher gradient를 차단한다"),
    ("하드웨어를 사전 점검한다", "pinned preset을 읽는다", "안전한 실패를 해석한다"),
    ("support overlap을 측정한다", "실패 지표를 해석한다", "vOPD baseline을 설명한다"),
    ("turn과 환경 state를 표현한다", "오류 누적을 관찰한다", "single-turn 가정을 벗어난다"),
    ("시간 깊이 schedule을 만든다", "F2B/B2F를 구분한다", "선택된 turn을 감사한다"),
    ("SOD step weight를 계산한다", "SAGE intervention을 조합한다", "loss scale을 보존한다"),
    ("방법 선택 근거를 쓴다", "OPD2 delta를 계산한다", "avg@K와 pass@K를 분리한다"),
]

OBJECTIVES_EN = [
    ("read the complete course map", "run a first SFT-versus-OPD comparison", "choose the fast or full path"),
    ("turn logits into probabilities", "read next-token tensor shapes", "verify causal prefixes"),
    ("compute CE, entropy, and KL", "name the KL direction", "verify generalized-JSD boundaries"),
    ("identify who creates training states", "compare SFT and KD fairly", "audit response budgets"),
    ("separate GKD lambda from beta", "implement generalized JSD", "trace gradient paths"),
    ("connect student rollout to update", "distinguish full and sampled estimators", "block teacher gradients"),
    ("preflight hardware", "read pinned presets", "interpret safe failures"),
    ("measure support overlap", "interpret failure signals", "explain the vOPD baseline"),
    ("represent turns and environment state", "observe error compounding", "leave the single-turn assumption"),
    ("build a depth schedule", "distinguish F2B and B2F", "audit selected turns"),
    ("compute SOD step weights", "combine SAGE interventions", "preserve loss scale"),
    ("justify method selection", "compute the OPD2 delta", "separate avg@K from pass@K"),
]

EXPLANATIONS_KO = [
    "먼저 전체 루프를 한 번 본다. 아래 곡선은 대형 모델 성능이 아니라, 같은 초기 분포에서 hard-label SFT와 teacher 분포를 따르는 OPD 목적이 서로 다르다는 현미경 예제다.",
    "언어 모델은 매 위치에서 vocabulary 전체의 logit을 낸다. softmax는 이를 합이 1인 조건부 분포로 바꾸고, causal mask는 미래 토큰을 보지 못하게 한다.",
    "KL은 대칭 거리가 아니다. 이 강좌는 forward를 KL(teacher||student), reverse를 KL(student||teacher)로 고정하고 함수 인자에도 두 분포 이름을 쓴다.",
    "SFT와 off-policy KD는 target 형식은 다르지만 고정된 demonstration prefix에서 학습한다. 공정 비교는 같은 student 초기화와 response-token budget부터 시작한다.",
    "원 GKD에는 두 축이 있다. lambda는 student가 만든 state의 비율이고 beta는 그 state에서 비교할 divergence다. 둘을 한 hyperparameter처럼 설명하면 안 된다.",
    "현대 sampled-token OPD에서 sampling은 미분되지 않는다. rollout log-prob은 snapshot이고, update 시 student logits을 다시 계산해야 현재 parameter로 gradient가 흐른다.",
    "실제 모델 경로는 다운로드 전에 revision·라이선스·크기·device를 보여준다. 이 노트북은 offline toy fallback을 기본으로 하며 네트워크 결과를 꾸며내지 않는다.",
    "좋은 teacher가 항상 좋은 OPD를 만들지는 않는다. student가 방문한 state에서 top-k support가 겹치고, teacher가 새로운 유용한 신호를 주는지를 함께 본다.",
    "멀티턴에서는 action이 환경 observation을 바꾼다. 초반 한 번의 오류가 이후 teacher에게 낯선 state를 만들 수 있으므로 token loss만 보고 성공을 판단할 수 없다.",
    "TCOD는 처음부터 긴 trajectory 전체를 증류하지 않는다. F2B는 앞에서부터 깊이를 늘리고, B2F는 teacher/successful prefix 뒤의 마무리부터 student에게 넘긴다.",
    "SOD는 step divergence로 신뢰하기 어려운 구간을 낮추고, SAGE-OPD는 intervention과 teacher confidence를 곱한 뒤 dense OPD와 loss scale을 맞춘다.",
    "마지막에는 알고리즘 하나가 항상 최고라는 결론 대신 전제에 맞게 고른다. OPD2의 delta와 언어 유지, 작은 K 효율과 큰 K 능력 경계를 별도로 평가한다.",
]

EXPLANATIONS_EN = [
    "Start by seeing the whole loop once. The curves below are a microscope example, not large-model evidence: hard-label SFT and a teacher-distribution OPD objective follow different targets from the same initialization.",
    "A language model emits one vocabulary-wide logit vector at each position. Softmax turns it into a conditional distribution summing to one; the causal mask prevents future-token leakage.",
    "KL is not a symmetric distance. This course fixes forward as KL(teacher||student) and reverse as KL(student||teacher), and names both distributions in APIs.",
    "SFT and off-policy KD use different targets but learn on fixed demonstration prefixes. A fair comparison begins with the same student initialization and response-token budget.",
    "Original GKD has two axes. Lambda controls the fraction of student-generated states; beta selects the divergence at those states. They are not one hyperparameter.",
    "Sampling in modern sampled-token OPD is not differentiable. Rollout log-probabilities are snapshots; update-time student logits are recomputed so gradients use current parameters.",
    "The real-model path shows revisions, licenses, size, and device before download. This notebook defaults to an offline toy fallback and never fabricates network results.",
    "A stronger teacher does not guarantee successful OPD. Measure top-k support on student-visited states and ask whether the teacher supplies genuinely useful novelty.",
    "In multi-turn tasks, an action changes the next environment observation. One early error can create states unfamiliar to the teacher, so token loss alone cannot establish success.",
    "TCOD does not distill a full long trajectory from the start. F2B grows from early turns; B2F hands the ending to the student after a teacher/successful prefix.",
    "SOD attenuates unreliable regions using step divergence. SAGE-OPD multiplies intervention by teacher confidence, then normalizes to match dense-OPD loss scale.",
    "The final lesson chooses methods by assumptions instead of declaring one universal winner. Evaluate OPD2 delta and language retention separately from small-K efficiency and large-K capability boundaries.",
]


DEEP_DIVES_KO = {
    "L00": """OPD를 한 문장으로 줄이면 **현재 student가 만든 prefix에서 teacher 분포를 묻고 student만 업데이트하는 증류**다. SFT는 고정 정답 궤적, off-policy KD는 고정 prefix의 teacher 분포, OPD는 현재 student가 실제로 방문한 prefix를 학습 state로 쓴다.

OPD는 reward를 최대화하는 RL과 같지 않다. sampling은 학습할 state를 바꾸지만 supervision은 teacher의 조건부 token 분포다. 따라서 핵심 질문은 늘 네 가지다: *누가 state를 만들었나? teacher가 무엇을 반환했나? 어떤 token을 mask했나? gradient가 어디로 흐르나?*""",
    "L01": """자기회귀 모델은 `p(x_1:T)=Π_t p(x_t | x_<t)`로 sequence 확률을 분해한다. 구현에서 logits shape는 `[batch, time, vocabulary]`이고, 위치 `t`의 logits는 target `t+1`과 비교된다. 이 한 칸 shift를 빠뜨리면 미래 token을 그대로 맞히는 잘못된 학습이 된다.

padding mask는 존재하지 않는 위치를, response mask는 prompt를 loss에서 제외한다. causal mask는 attention 자체가 미래를 보지 못하게 한다. 세 mask는 목적이 다르므로 하나로 대체할 수 없다.""",
    "L02": """hard-label CE는 정답 token 하나만 남기지만 KD는 teacher의 전체 분포를 보존한다. `KL(teacher || student)`는 teacher가 확률을 둔 곳을 빠뜨릴 때 크게 벌하고, `KL(student || teacher)`는 student가 teacher의 저확률 영역에 질량을 둘 때 크게 벌한다. 그래서 전자는 coverage-seeking, 후자는 mode-seeking 경향으로 설명되지만 이는 보장된 행동 법칙이 아니다.

temperature `T`는 logits를 `T`로 나눠 분포를 부드럽게 한다. 고전 KD는 gradient 크기 변화를 보정하려 `T²`를 곱한다. 이 레포는 모든 log-softmax를 float32에서 계산하고 빈 mask·0 이하 temperature를 즉시 거부한다.""",
    "L03": """학습 state 분포를 기호로 쓰면 SFT/KD는 대체로 `s ~ d_data`, OPD는 `s ~ d_student`다. 모델이 추론 중 스스로 만든 오류 prefix는 `d_data`에 거의 없을 수 있다. OPD는 이 covariate shift를 직접 보지만, student가 너무 나쁜 state만 방문하면 오히려 teacher 신호가 불안정해질 수 있다.

공정 비교는 같은 초기 weight, prompt subset, optimizer, learning rate, response-token 수, 평가 split을 고정한다. optimizer step만 맞추면 길이가 긴 방법이 더 많은 token을 볼 수 있으므로 token budget도 별도로 기록한다.""",
    "L04": """GKD에는 서로 독립적인 두 축이 있다. `lambda`는 고정 state와 student state를 섞는 비율이고, `beta`는 그 state에서 generalized JSD의 모양을 정한다. 이 강좌의 convention에서 `beta=0`은 forward KL, `beta=1`은 reverse KL이다.

원 GKD는 선택한 prefix에서 **vocabulary 전체 분포**를 비교한다. 뒤의 현대 sampled OPD는 student가 뽑은 token만으로 score-function estimator를 만들 수 있다. 둘 다 on-policy state를 쓸 수 있지만 estimator와 분산이 같지 않다.""",
    "L05": """한 update는 `collect → freeze trajectory → teacher score → recompute student logits → masked loss → optimizer step` 순서다. rollout token은 discrete sample이라 그 선택을 통해 미분하지 않는다. rollout-time log-prob은 감사/importance 정보이고, 실제 gradient는 같은 token sequence를 현재 student에 다시 넣어 얻은 log-prob에서 나온다.

full reverse KL은 모든 vocabulary 항을 합해 낮은 분산의 정확한 token-state objective를 준다. sampled reverse KL은 `y ~ student`에서 `(log p_student(y)-log p_teacher(y)).detach() * log p_student(y)`를 사용한다. 메모리는 작지만 분산과 baseline 설계 문제가 생긴다.""",
    "L06": """실제 학습 경로는 실행보다 **preflight**가 먼저다. dataset/model revision, 예상 byte, license 동의, tokenizer·chat template, dtype, device와 fallback 정책을 확인한 뒤에만 다운로드한다. official test는 학습에 쓰지 않고 train에서 validation을 고정 seed로 분리한다.

LoRA는 frozen weight에 작은 low-rank update `B·A`를 학습한다. QLoRA는 base weight까지 4-bit로 양자화해 VRAM을 더 줄이지만 bitsandbytes/CUDA 조합을 실제 forward-backward-save-reload로 검증해야 한다. 이 레포는 실패 시 full fine-tuning으로 조용히 바꾸지 않는다.""",
    "L07": """OPD 실패는 최소 세 층으로 나눈다. (1) teacher 자체가 해당 state에서 틀림, (2) student와 teacher support가 너무 달라 유용한 token을 sample하지 못함, (3) estimator 분산·학습률·stale rollout 같은 최적화 문제다. loss 하나만으로 셋을 구분할 수 없다.

top-k overlap, entropy gap, forward/reverse KL, intervention rate를 함께 본다. vOPD는 top-k 기반 KL을 detached baseline으로 써 sampled estimator의 분산을 줄이려 한다. baseline은 기대 gradient를 바꾸지 않아야 하며 optimized target과 혼동하면 안 된다.""",
    "L08": """멀티턴에서는 trajectory가 단순 text가 아니라 `(observation_t, action_t, next observation, terminal)`의 연쇄다. student action이 transition을 바꾸므로 teacher가 원래 정답 trajectory에서만 잘해도 student state에서는 회복하지 못할 수 있다.

turn/step boundary를 token별로 보존하면 어떤 step이 실패했는지, environment token을 loss에서 제외했는지, terminal 뒤 token을 잘못 학습했는지 감사할 수 있다. sequence-level 성공과 token-level teacher agreement를 함께 기록해야 한다.""",
    "L09": """TCOD는 긴 multi-turn trajectory를 한 번에 맡기지 않고 student가 담당할 시간 구간을 점차 늘린다. F2B는 앞 turn부터 늘려 초기 의사결정을 연습한다. B2F는 성공한 teacher prefix 뒤의 마지막 turn부터 맡겨 쉬운 suffix에서 시작한다.

curriculum depth와 pacing은 별개다. depth는 현재 포함할 turn 수, pacing은 몇 optimizer step마다 depth를 늘릴지다. 잘린 trajectory에서도 observation/action 경계와 loss mask가 원래 turn과 정렬되어야 한다.""",
    "L10": """SOD는 step별 teacher/student divergence를 계산해 급격히 불일치하는 step의 distillation weight를 조절한다. weight를 detach할지 여부는 단순 구현 취향이 아니라 objective 의미를 바꾼다. 이 mini 구현은 안정적인 curriculum signal로 취급해 gradient를 끊는다.

SAGE-OPD는 teacher intervention 필요도와 teacher confidence를 곱해 token weight를 만들고, 평균 weight가 dense OPD와 같도록 정규화한다. intervention 0인 batch는 skip/fallback 정책이 필요하다. mini judge는 token-agreement proxy이며 semantic judge 성능을 주장하지 않는다.""",
    "L11": """OPD²는 post-trained teacher와 base teacher의 log-prob 차이를 이용해 post-training이 실제로 더 선호하게 만든 token에 집중한다. teacher 하나만 따라가는 OPD와 달리 `teacher - teacher_base` delta가 핵심이므로 두 model의 tokenizer/template 정합성이 필요하다.

test-time scaling에서는 `avg@K`(K개 sample의 평균 성공률)와 `pass@K`(하나라도 성공할 확률)를 분리한다. OPD가 작은 K 효율을 높이면서 큰 K에서 풀 수 있던 문제를 잃을 수도 있으므로 gained/lost solvability와 multilingual retention을 별도 평가한다.""",
}

DEEP_DIVES_EN = {
    "L00": """In one sentence, OPD **asks the teacher for a distribution at prefixes produced by the current student, then updates only the student**. SFT uses fixed answer trajectories, off-policy KD uses teacher distributions on fixed prefixes, and OPD uses prefixes the current student actually visits.

OPD is not simply RL. Sampling changes the training-state distribution, while supervision is still the teacher's conditional token distribution. Keep asking four questions: *who produced the state, what did the teacher return, which tokens are masked, and where do gradients flow?*""",
    "L01": """An autoregressive model factors sequence probability as `p(x_1:T)=Π_t p(x_t | x_<t)`. Logits have shape `[batch, time, vocabulary]`; logits at position `t` predict target `t+1`. Missing this one-position shift accidentally trains with future-token leakage.

The padding mask removes nonexistent positions, the response mask excludes prompts from the objective, and the causal mask prevents attention to the future. They solve different problems and cannot replace one another.""",
    "L02": """Hard-label CE retains one target token; KD retains the teacher's full distribution. `KL(teacher || student)` heavily penalizes missing teacher-supported regions, while `KL(student || teacher)` heavily penalizes student mass in teacher-low-probability regions. Coverage- versus mode-seeking is a useful tendency, not a guaranteed behavior law.

Temperature `T` divides logits to soften distributions. Classical KD multiplies by `T²` to compensate the gradient-scale change. This repository computes log-softmax in float32 and rejects empty masks or nonpositive temperatures.""",
    "L03": """In state-distribution notation, SFT/KD usually use `s ~ d_data`; OPD uses `s ~ d_student`. Error prefixes created at inference may be nearly absent from `d_data`. OPD exposes that covariate shift, but a very weak student may visit only states where teacher guidance is unstable.

A fair comparison fixes initial weights, prompt subset, optimizer, learning rate, response-token count, and evaluation split. Equal optimizer steps alone are insufficient because methods may process different numbers of tokens.""",
    "L04": """GKD has two independent axes. `lambda` mixes fixed and student-generated states; `beta` shapes generalized JSD at those states. Under this course's convention, `beta=0` is forward KL and `beta=1` is reverse KL.

Original GKD compares the **full vocabulary distribution** on a selected prefix. Modern sampled OPD can build a score-function estimator from only student-sampled tokens. Both may use on-policy states, but their estimators and variance differ.""",
    "L05": """One update follows `collect → freeze trajectory → teacher score → recompute student logits → masked loss → optimizer step`. A discrete rollout token is not differentiated through. Rollout-time log-probabilities are audit/importance snapshots; gradients come from current-student log-probabilities recomputed on the sampled sequence.

Full reverse KL sums every vocabulary term, giving an exact, lower-variance token-state objective. Sampled reverse KL uses `(log p_student(y)-log p_teacher(y)).detach() * log p_student(y)` for `y ~ student`. It saves memory but introduces variance and baseline-design questions.""",
    "L06": """The real path begins with **preflight**, not training. Check dataset/model revisions, expected bytes, license consent, tokenizer/chat template, dtype, device, and fallback policy before download. The official test split is never trained on; validation is deterministically carved from train.

LoRA learns a small low-rank update `B·A` over frozen weights. QLoRA also quantizes base weights to 4-bit, reducing VRAM further, but the bitsandbytes/CUDA combination needs a real forward-backward-save-reload probe. This repository never silently falls back to full fine-tuning.""",
    "L07": """Separate OPD failures into at least three layers: (1) the teacher is wrong on that state, (2) teacher/student support differs so much that useful tokens are never sampled, or (3) optimization fails through estimator variance, learning rate, or stale rollouts. One loss curve cannot distinguish them.

Read top-k overlap, entropy gap, both KL directions, and intervention rate together. vOPD uses a detached top-k KL baseline to reduce sampled-estimator variance. A baseline must not alter the expected gradient and is not the optimized target.""",
    "L08": """A multi-turn trajectory is not just text; it is a chain of `(observation_t, action_t, next observation, terminal)`. Student actions change transitions, so a teacher that succeeds only on the reference trajectory may not recover from student states.

Per-token turn/step boundaries reveal which step failed, whether environment tokens leaked into loss, and whether tokens after terminal were trained. Record sequence success alongside token-level teacher agreement.""",
    "L09": """TCOD does not hand the student an entire long trajectory immediately. F2B grows from early turns and practices initial decisions. B2F begins at an easier suffix after a successful/teacher prefix.

Curriculum depth and pacing are distinct: depth is the number of currently included turns; pacing controls optimizer steps between depth increases. Sliced trajectories must preserve original observation/action boundaries and aligned loss masks.""",
    "L10": """SOD computes per-step teacher/student divergence and adjusts distillation weight around abrupt disagreements. Whether weights are detached changes the objective, not just implementation style. The mini implementation treats them as a stable curriculum signal and stops their gradients.

SAGE-OPD multiplies teacher-intervention need by teacher confidence, then normalizes token weights to the dense-OPD scale. All-zero intervention batches need an explicit skip/fallback policy. The mini judge is a token-agreement proxy, not a semantic-judge claim.""",
    "L11": """OPD² uses the log-probability difference between a post-trained teacher and its base teacher, focusing on tokens made more preferred by post-training. Unlike single-teacher OPD, the `teacher - teacher_base` delta is central, so both models need aligned tokenizers and templates.

Test-time scaling separates `avg@K` (mean success among K samples) from `pass@K` (probability at least one succeeds). OPD may improve small-K efficiency while losing problems solvable at large K, so track gained/lost solvability and multilingual retention separately.""",
}

IMPLEMENTATION_KO = {
    "L00": """| 경로 | state 생성자 | target | 핵심 위험 |
|---|---|---|---|
| SFT | dataset/teacher trace | hard token | inference state 불일치 |
| off-policy KD | 고정 prefix | teacher full distribution | 오류 state 미관찰 |
| OPD | current student | teacher distribution/score | sampling 분산·나쁜 state |

실제 루프: [`losses.py`](../../src/opd_study/algorithms/losses.py), [`rollout.py`](../../src/opd_study/algorithms/rollout.py).""",
    "L01": """구현 순서는 `token IDs → token/position embedding → causal Transformer → layer norm → vocabulary logits`다. `TinyCausalLM.forward`는 입력 rank와 최대 길이를 검사하고 causal/padding mask를 분리한다. `generate`는 greedy(`temperature=0`)와 stochastic sampling을 명시적으로 나눈다.

실제 코드: [`tiny_transformer.py`](../../src/opd_study/models/tiny_transformer.py), [`tokenizer.py`](../../src/opd_study/data/tokenizer.py).""",
    "L02": """모든 KL API는 `(teacher_logits, student_logits)`라는 이름을 강제한다. reduction 전 shape는 `[B,T]`; 마지막에는 response mask로만 평균낸다. generalized JSD의 beta 경계는 별도 branch로 계산해 `log(0)`을 피한다.

실제 코드: [`math.py`](../../src/opd_study/math.py), [`losses.py`](../../src/opd_study/algorithms/losses.py).""",
    "L03": """`TrajectoryBatch`가 token IDs, attention/response mask, prompt length와 rollout snapshot을 한 계약으로 묶는다. SFT와 KD는 동일 batch의 동일 response target 수를 쓰되, SFT는 token ID, KD는 detached teacher logits을 읽는다.

실제 코드: [`types.py`](../../src/opd_study/types.py), [`core.py`](../../src/opd_study/training/core.py).""",
    "L04": """collection에서 lambda로 state source를 고르고 loss에서 beta로 divergence를 고른다. 두 선택을 함수 하나에 숨기지 않아 ablation이 가능하다. teacher는 `eval()`과 `no_grad()`로 score하고 student만 update한다.

실제 코드: [`core.py`](../../src/opd_study/training/core.py), [`losses.py`](../../src/opd_study/algorithms/losses.py).""",
    "L05": """`collect_student_trajectories`는 generation 동안 student의 기존 train/eval mode를 복원하고 log-prob snapshot을 detach한다. loss는 response target 위치만 한 칸 shift해 계산한다. teacher/student logits shape가 다르면 forward 전에 실패한다.

실제 코드: [`rollout.py`](../../src/opd_study/algorithms/rollout.py), [`losses.py`](../../src/opd_study/algorithms/losses.py).""",
    "L06": """research config는 `backend=research`, exact 40-char revision, download byte와 동의 flag를 가진다. preflight는 package 존재뿐 아니라 실제 import, PyTorch version, device와 QLoRA capability까지 검사한다. 모델 loader는 vocab와 chat template가 다르면 즉시 중단한다.

실제 코드: [`preflight.py`](../../src/opd_study/research/preflight.py), [`hf_backend.py`](../../src/opd_study/research/hf_backend.py).""",
    "L07": """support 진단은 response mask에만 top-k 집합 overlap과 entropy gap을 계산한다. vOPD baseline은 detach되고 sampled-token log-prob만 gradient를 가진다. threshold는 성능 보장이 아니라 비교 run 간 경보 기준으로만 사용한다.

실제 코드: [`support.py`](../../src/opd_study/diagnostics/support.py), [`vopd.py`](../../src/opd_study/algorithms/vopd.py).""",
    "L08": """환경 state는 immutable record이고 `step(action)`이 다음 observation과 terminal/success를 반환한다. multi-turn batch는 prompt token의 turn ID를 `-1`, response turn을 `0..N-1`로 둬 mask slicing을 안전하게 한다.

실제 코드: [`calculator.py`](../../src/opd_study/envs/calculator.py), [`tokenizer.py`](../../src/opd_study/data/tokenizer.py).""",
    "L09": """`curriculum_depth`는 step과 pacing을 정수식으로 계산한다. `temporal_curriculum_mask`는 F2B/B2F가 선택한 turn과 원 response mask의 교집합만 반환한다. B2F의 teacher-prefix 생성은 paper-scale 인프라 대신 mini backend에서 명시적 근사다.

실제 코드: [`tcod.py`](../../src/opd_study/algorithms/tcod.py), [`advanced.py`](../../src/opd_study/training/advanced.py).""",
    "L10": """SOD는 token KL을 step별 평균으로 모아 detached weight로 다시 token에 broadcast한다. SAGE는 intervention label, confidence, normalization을 각각 함수로 분리해 ablation할 수 있다. SOD 논문의 별도 GRPO 항은 구현하지 않았고 metric에 표시한다.

실제 코드: [`sod.py`](../../src/opd_study/algorithms/sod.py), [`sage_opd.py`](../../src/opd_study/algorithms/sage_opd.py).""",
    "L11": """OPD²는 teacher/base delta를 center하고 양의 개선 영역을 gate한다. test-time scaling helper는 boolean `[problem, sample]` 행렬을 받아 metric 정의를 코드에 고정한다. 이 둘은 같은 loss가 아니라 학습 선택과 평가 관점이다.

실제 코드: [`opd2.py`](../../src/opd_study/algorithms/opd2.py), [`test_time_scaling.py`](../../src/opd_study/diagnostics/test_time_scaling.py).""",
}

IMPLEMENTATION_EN = {
    "L00": """| path | state producer | target | main risk |
|---|---|---|---|
| SFT | dataset/teacher trace | hard token | inference-state mismatch |
| off-policy KD | fixed prefix | teacher full distribution | no error states |
| OPD | current student | teacher distribution/score | variance/bad states |

Production loop: [`losses.py`](../../src/opd_study/algorithms/losses.py), [`rollout.py`](../../src/opd_study/algorithms/rollout.py).""",
    "L01": """The implementation flows through `token IDs → token/position embedding → causal Transformer → layer norm → vocabulary logits`. `TinyCausalLM.forward` validates rank/length and separates causal from padding masks. `generate` explicitly separates greedy (`temperature=0`) and stochastic sampling.

Production code: [`tiny_transformer.py`](../../src/opd_study/models/tiny_transformer.py), [`tokenizer.py`](../../src/opd_study/data/tokenizer.py).""",
    "L02": """Every KL API names its arguments `(teacher_logits, student_logits)`. Before reduction the shape is `[B,T]`; only response positions enter the final mean. Generalized-JSD beta boundaries use explicit branches to avoid `log(0)`.

Production code: [`math.py`](../../src/opd_study/math.py), [`losses.py`](../../src/opd_study/algorithms/losses.py).""",
    "L03": """`TrajectoryBatch` places token IDs, attention/response masks, prompt lengths, and rollout snapshots under one contract. SFT and KD use the same batch and response-target count; SFT reads token IDs while KD reads detached teacher logits.

Production code: [`types.py`](../../src/opd_study/types.py), [`core.py`](../../src/opd_study/training/core.py).""",
    "L04": """Collection chooses the state source with lambda; loss chooses divergence with beta. Keeping them in separate functions makes ablations auditable. The teacher scores under `eval()` and `no_grad()` while only the student updates.

Production code: [`core.py`](../../src/opd_study/training/core.py), [`losses.py`](../../src/opd_study/algorithms/losses.py).""",
    "L05": """`collect_student_trajectories` restores the student's previous train/eval mode and detaches rollout log-probability snapshots. Loss shifts one position and selects response targets only. Teacher/student logit shape mismatches fail before update.

Production code: [`rollout.py`](../../src/opd_study/algorithms/rollout.py), [`losses.py`](../../src/opd_study/algorithms/losses.py).""",
    "L06": """Research configs carry `backend=research`, exact 40-character revisions, expected bytes, and consent flags. Preflight checks real imports, PyTorch version, device, and QLoRA capability—not merely package presence. The loader fails on vocabulary or chat-template mismatch.

Production code: [`preflight.py`](../../src/opd_study/research/preflight.py), [`hf_backend.py`](../../src/opd_study/research/hf_backend.py).""",
    "L07": """Support diagnostics compute top-k set overlap and entropy gap only on response targets. The vOPD baseline is detached; only sampled-token log-probabilities carry gradient. Thresholds are alert criteria across comparable runs, not performance guarantees.

Production code: [`support.py`](../../src/opd_study/diagnostics/support.py), [`vopd.py`](../../src/opd_study/algorithms/vopd.py).""",
    "L08": """Environment state is immutable; `step(action)` returns the next observation plus terminal/success. Multi-turn batches give prompt tokens turn ID `-1` and response turns `0..N-1`, making mask slicing auditable.

Production code: [`calculator.py`](../../src/opd_study/envs/calculator.py), [`tokenizer.py`](../../src/opd_study/data/tokenizer.py).""",
    "L09": """`curriculum_depth` derives depth from step and pacing with integer arithmetic. `temporal_curriculum_mask` returns only the intersection of F2B/B2F-selected turns and the response mask. Teacher-prefix generation for B2F is an explicit mini-backend approximation.

Production code: [`tcod.py`](../../src/opd_study/algorithms/tcod.py), [`advanced.py`](../../src/opd_study/training/advanced.py).""",
    "L10": """SOD averages token KL by step, then broadcasts detached weights back to tokens. SAGE separates intervention labels, confidence, and normalization for ablation. The paper's separate GRPO term is omitted and labeled in metrics.

Production code: [`sod.py`](../../src/opd_study/algorithms/sod.py), [`sage_opd.py`](../../src/opd_study/algorithms/sage_opd.py).""",
    "L11": """OPD² centers the teacher/base delta and gates positive improvement regions. Test-time-scaling helpers accept a boolean `[problem, sample]` matrix, fixing metric definitions in code. These are a training-selection mechanism and an evaluation lens, not one loss.

Production code: [`opd2.py`](../../src/opd_study/algorithms/opd2.py), [`test_time_scaling.py`](../../src/opd_study/diagnostics/test_time_scaling.py).""",
}

ALTERNATIVES_KO = {
    "L00": """**언제 무엇을 쓰나?** 정답 trace가 충분하면 SFT가 가장 단순하다. 고정 prefix에서 teacher의 dark knowledge가 필요하면 off-policy KD, inference 중 오류 state까지 교정하려면 OPD를 고려한다. OPD가 언제나 우월하다는 전제는 두지 않는다.""",
    "L01": """실제 LLM은 subword tokenizer, RoPE, tied embeddings, RMSNorm, FlashAttention 등을 쓸 수 있다. 이 강좌는 character tokenizer와 표준 Transformer를 택해 token 경계와 mask를 눈으로 검사하게 한다. 이 선택은 교육용이며 Qwen backend가 같은 구조라는 뜻이 아니다.""",
    "L02": """full logits가 비싸면 top-k logits나 sampled token만 저장할 수 있다. top-k는 tail mass를 버리므로 retained probability mass와 근사 오차를 같이 보고한다. API teacher가 log-prob을 일부만 주면 full-KL 구현으로 가장하지 않는다.""",
    "L03": """state source를 batch 단위, example 단위, token/turn 단위로 섞을 수 있다. 이 mini runtime은 재현성과 설명을 위해 batch 단위로 고른다. 더 세밀한 혼합은 분산을 줄일 수 있지만 trajectory provenance가 복잡해진다.""",
    "L04": """lambda는 고정값, warm-up schedule, 성능 기반 adaptive schedule이 가능하다. beta도 FKL/RKL/JSD 사이를 고를 수 있다. 기본은 논문 의미를 먼저 재현하는 고정값이며 adaptive 정책은 별도 실험 변수로 둔다.""",
    "L05": """full estimator는 작은 vocab/저장 가능한 teacher logits에 적합하다. sampled estimator는 큰 vocab·API log-prob 환경에 유리하지만 여러 sample, control variate, clipping이 필요할 수 있다. old-policy rollout을 재사용하면 importance correction과 policy-version 감사가 필요하다.""",
    "L06": """full fine-tuning은 가장 직접적이나 메모리가 크다. LoRA는 넓은 환경에서 안정적이고, QLoRA는 VRAM을 줄이는 대신 CUDA/bitsandbytes 의존성이 커진다. macOS에서는 QLoRA를 full-FT로 대체하지 말고 LoRA/CPU toy 또는 검증된 CUDA를 선택한다.""",
    "L07": """overlap을 높이는 처방은 teacher 교체, temperature 조정, off-policy/on-policy mixture, curriculum, multiple samples 등이다. 지표 하나가 나쁘다고 모두 적용하지 말고 실패 층을 먼저 분류한 뒤 한 변수씩 ablation한다.""",
    "L08": """환경을 text transcript로만 저장할 수도 있지만 observation/action 구조를 잃는다. 반대로 완전한 simulator snapshot은 정확하지만 크다. 최소한 turn ID, terminal, environment observation hash와 action을 보존하는 절충이 필요하다.""",
    "L09": """F2B는 초기 decision 오류가 핵심일 때, B2F는 긴 horizon 때문에 성공 경험이 희소할 때 자연스럽다. random window나 difficulty curriculum도 대안이지만 TCOD 결과로 부르려면 논문의 방향·prefix 조건을 보존해야 한다.""",
    "L10": """SOD와 SAGE는 경쟁하는 단일 recipe라기보다 다른 실패 신호를 쓴다. step divergence가 신뢰도 proxy이면 SOD, recoverability/teacher 판단을 직접 물을 수 있으면 SAGE가 자연스럽다. 둘을 합치려면 weight scale과 double-counting을 새로 검증한다.""",
    "L11": """선택 규칙: single-turn이고 support가 좋으면 vanilla OPD부터, sampled variance가 크면 vOPD, long-horizon이면 TCOD/SOD/SAGE, post-training delta 보존이 목적이면 OPD²를 검토한다. 최종 선택은 같은 budget의 SFT/KD baseline과 held-out 평가로 결정한다.""",
}

ALTERNATIVES_EN = {
    "L00": """**When should you use each path?** Use SFT when reliable answer traces are sufficient, off-policy KD for teacher dark knowledge on fixed prefixes, and consider OPD when inference-time error states matter. Do not assume OPD must always win.""",
    "L01": """Production LLMs may use subword tokenizers, RoPE, tied embeddings, RMSNorm, and FlashAttention. This course chooses a character tokenizer and standard Transformer so token boundaries and masks stay visible. That is an educational choice, not a claim that Qwen has the same architecture.""",
    "L02": """When full logits are expensive, store top-k logits or sampled tokens. Top-k discards tail mass, so report retained probability mass and approximation error. If an API teacher exposes only partial log-probabilities, do not label the result full KL.""",
    "L03": """State sources can mix per batch, example, token, or turn. The mini runtime chooses per batch for reproducibility and clarity. Finer mixing may reduce variance but complicates trajectory provenance.""",
    "L04": """Lambda may be fixed, warmed up, or adapted to performance; beta can select FKL, RKL, or intermediate JSD. Defaults first preserve paper semantics. Adaptive policies remain separate experimental variables.""",
    "L05": """Full estimators suit small vocabularies or available teacher logits. Sampled estimators suit large vocabularies/API log-probs but may need multiple samples, control variates, or clipping. Reusing old-policy rollouts requires importance correction and policy-version audits.""",
    "L06": """Full fine-tuning is direct but memory-heavy. LoRA is broadly stable; QLoRA saves VRAM but adds CUDA/bitsandbytes constraints. On macOS, do not substitute full FT for failed QLoRA—choose LoRA/CPU toy or validated CUDA.""",
    "L07": """Possible overlap remedies include a different teacher, temperature, off/on-policy mixture, curriculum, or multiple samples. Classify the failure layer first, then ablate one change at a time instead of applying every remedy to one bad metric.""",
    "L08": """A text-only transcript is compact but loses observation/action structure; full simulator snapshots are exact but large. A practical minimum preserves turn IDs, terminal state, an observation hash, and actions.""",
    "L09": """F2B is natural when early decisions dominate errors; B2F helps when long horizons make successful experience sparse. Random windows or difficulty curricula are alternatives, but calling them TCOD requires preserving direction and prefix conditions.""",
    "L10": """SOD and SAGE use different failure signals rather than being one competing recipe. Choose SOD when step divergence is a useful reliability proxy, SAGE when recoverability/teacher judgment can be queried. Combining them needs a new scale and double-counting audit.""",
    "L11": """A practical rule: start with vanilla OPD for aligned single-turn support; consider vOPD for sampled variance, TCOD/SOD/SAGE for long horizons, and OPD² for post-training deltas. Decide against equal-budget SFT/KD baselines on held-out data.""",
}


def bounded_source_probe(imports: str, object_names: tuple[str, ...]) -> str:
    objects = ", ".join(object_names) + ","
    return f"""import inspect
{imports}

objects_to_show = ({objects})
for object_to_show in objects_to_show:
    source_lines = inspect.getsource(object_to_show).splitlines()
    print(f"\\n# {{object_to_show.__module__}}.{{object_to_show.__qualname__}}")
    print("\\n".join(source_lines[:80]))
    if len(source_lines) > 80:
        print(f"... {{len(source_lines) - 80}} more lines; open the linked source file")"""


SOURCE_PROBES = {
    "L00": bounded_source_probe(
        "from opd_study.algorithms import on_policy_distillation_loss",
        ("on_policy_distillation_loss",),
    ),
    "L01": bounded_source_probe(
        "from opd_study.models import TinyCausalLM",
        ("TinyCausalLM.forward", "TinyCausalLM.generate"),
    ),
    "L02": bounded_source_probe(
        "from opd_study.math import forward_kl_from_logits, reverse_kl_from_logits",
        ("forward_kl_from_logits", "reverse_kl_from_logits"),
    ),
    "L03": bounded_source_probe(
        "from opd_study.algorithms import supervised_fine_tuning_loss, off_policy_kd_loss",
        ("supervised_fine_tuning_loss", "off_policy_kd_loss"),
    ),
    "L04": bounded_source_probe(
        "from opd_study.algorithms import generalized_kd_loss",
        ("generalized_kd_loss",),
    ),
    "L05": bounded_source_probe(
        "from opd_study.algorithms import collect_student_trajectories, sampled_reverse_kl_loss",
        ("collect_student_trajectories", "sampled_reverse_kl_loss"),
    ),
    "L06": bounded_source_probe(
        "from opd_study.research import research_preflight",
        ("research_preflight",),
    ),
    "L07": bounded_source_probe(
        "from opd_study.diagnostics import support_diagnostics\nfrom opd_study.algorithms import vopd_loss",
        ("support_diagnostics", "vopd_loss"),
    ),
    "L08": bounded_source_probe(
        "from opd_study.envs import CalculatorEnvironment",
        ("CalculatorEnvironment.step",),
    ),
    "L09": bounded_source_probe(
        "from opd_study.algorithms.tcod import temporal_curriculum_mask, tcod_loss",
        ("temporal_curriculum_mask", "tcod_loss"),
    ),
    "L10": bounded_source_probe(
        "from opd_study.algorithms.sod import step_divergence_weights\nfrom opd_study.algorithms.sage_opd import sage_opd_loss",
        ("step_divergence_weights", "sage_opd_loss"),
    ),
    "L11": bounded_source_probe(
        "from opd_study.algorithms import opd2_loss\nfrom opd_study.diagnostics import scaling_metrics",
        ("opd2_loss", "scaling_metrics"),
    ),
}


EXERCISES_KO = {
    "L00": ("**연습 (5분):** 같은 초기 logits에서 teacher가 `[0.45, 0.45, 0.10]`일 때 hard-label SFT와 reverse-KL OPD가 첫 두 token을 어떻게 다르게 다룰지 실행 전에 적고 확인하라.", "<details><summary>확인 기준</summary>SFT는 선택한 hard target 하나만 직접 올리지만 OPD는 첫 두 token에 teacher가 둔 질량을 함께 반영한다고 설명하면 된다.</details>"),
    "L01": ("**연습 (5분):** `prefix_b`의 마지막 token이 아니라 첫 token을 바꾸면 어느 위치 logits부터 달라져야 하는지 예측하고 assertion을 작성하라.", "<details><summary>확인 기준</summary>바뀐 첫 token 이후 위치는 달라질 수 있지만 그보다 과거 위치는 없다. 미래 token이 과거 logits를 바꾸지 않는 방향을 검사한다.</details>"),
    "L02": ("**연습 (7분):** teacher `[0.99,0.01]`, student `[0.5,0.5]`에서 FKL과 RKL을 손으로 계산하고 두 인자를 바꿨을 때 값이 왜 달라지는지 설명하라.", "<details><summary>확인 기준</summary>`sum teacher*(log teacher-log student)`와 `sum student*(log student-log teacher)`를 따로 계산하고 KL 비대칭을 언급한다.</details>"),
    "L03": ("**연습 (7분):** SFT batch와 KD batch의 response mask를 하나씩 출력하고 prompt 길이를 두 배로 늘려도 effective token budget이 변하지 않는 assertion을 추가하라.", "<details><summary>확인 기준</summary>prompt 위치는 모두 false이고 response target 수만 budget에 들어가야 한다.</details>"),
    "L04": ("**연습 (8분):** seed를 고정한 채 lambda를 0, 0.5, 1로 바꾸고 state-source trace를 비교하라. beta는 그대로 두고 두 knob가 독립임을 기록하라.", "<details><summary>확인 기준</summary>lambda 0은 fixed, 1은 student state만 고르며 beta/JSD 수식은 변하지 않는다.</details>"),
    "L05": ("**연습 (10분):** sampled reverse-KL의 advantage에서 `.detach()`를 제거했을 때 gradient 식에 생기는 추가 항을 적어보고, production 코드는 수정하지 말고 작은 복제 식으로 gradient 차이를 확인하라.", "<details><summary>확인 기준</summary>detach가 없으면 advantage 자체의 student log-prob에도 gradient가 생겨 의도한 score-function estimator와 달라진다.</details>"),
    "L06": ("**연습 (8분):** laptop config에서 device를 `mps`, finetuning을 `qlora`로 가정한 실패 보고서를 작성하라. fallback으로 full FT를 제안하면 안 된다.", "<details><summary>확인 기준</summary>bitsandbytes/CUDA 제약, 예상 다운로드, 동의 flag와 LoRA/CPU toy 대안을 명시한다.</details>"),
    "L07": ("**연습 (8분):** overlap은 낮지만 entropy gap은 작은 synthetic logits를 만들고, 이것만으로 teacher 품질 불량을 결론내릴 수 없는 이유를 쓰라.", "<details><summary>확인 기준</summary>support 순위 불일치와 분포 sharpness는 다른 축이며 정답/환경 성공 근거가 추가로 필요하다.</details>"),
    "L08": ("**연습 (8분):** 같은 최종 숫자라도 첫 action이 다른 두 trajectory를 만들고 observation trace가 같은지 비교하라.", "<details><summary>확인 기준</summary>environment transition 때문에 중간 state가 다르면 token sequence만 같은지와 별개로 trajectory provenance가 다르다.</details>"),
    "L09": ("**연습 (8분):** 3-turn batch에서 depth 1/2/3의 F2B와 B2F mask를 표로 만들고 각 turn이 처음 포함되는 step을 적어라.", "<details><summary>확인 기준</summary>F2B는 작은 turn ID부터, B2F는 큰 turn ID부터 포함하며 pacing과 depth를 구분한다.</details>"),
    "L10": ("**연습 (10분):** SAGE intervention을 전부 0과 전부 1로 바꿔 skip/normalization 동작을 비교하고, SOD weight에 gradient가 없는지 재확인하라.", "<details><summary>확인 기준</summary>빈 intervention은 명시적으로 처리되고, dense case의 weight 합은 response token 수에 맞으며 SOD weight는 detached다.</details>"),
    "L11": ("**연습 (10분):** 방법 선택 memo를 5문장으로 쓴다: task horizon, teacher/student support, logit 접근성, hardware, SFT baseline을 반드시 포함하라.", "<details><summary>확인 기준</summary>알고리즘 이름보다 전제와 측정 계획이 먼저 나오고 avg@K/pass@K 또는 언어 유지 중 관련 guardrail을 포함한다.</details>"),
}

EXERCISES_EN = {
    "L00": ("**Exercise (5 min):** with the same initial logits and teacher `[0.45, 0.45, 0.10]`, predict how hard-label SFT and reverse-KL OPD treat the first two tokens, then verify.", "<details><summary>Check</summary>SFT directly raises one chosen hard target; OPD reflects teacher mass on both leading tokens.</details>"),
    "L01": ("**Exercise (5 min):** change the first rather than last token of `prefix_b`. Predict which logits may change and write an assertion.", "<details><summary>Check</summary>Positions after the changed token may differ; no future token may alter an earlier logit.</details>"),
    "L02": ("**Exercise (7 min):** hand-compute FKL and RKL for teacher `[0.99,0.01]`, student `[0.5,0.5]`; explain why swapping arguments changes the value.", "<details><summary>Check</summary>Compute the teacher-weighted and student-weighted log-ratios separately and name KL asymmetry.</details>"),
    "L03": ("**Exercise (7 min):** print one SFT and KD response mask, then assert that doubling prompt length does not change the effective token budget.", "<details><summary>Check</summary>Prompt positions remain false; only response targets count toward budget.</details>"),
    "L04": ("**Exercise (8 min):** with a fixed seed, compare state-source traces at lambda 0, .5, and 1 while keeping beta fixed. Record why the knobs are independent.", "<details><summary>Check</summary>Lambda 0 selects fixed states and 1 student states; the beta/JSD formula does not change.</details>"),
    "L05": ("**Exercise (10 min):** remove `.detach()` from a copied sampled-RKL advantage, derive the extra gradient term, and compare gradients without editing production code.", "<details><summary>Check</summary>Without detach, the advantage's student log-probability also differentiates, changing the intended score-function estimator.</details>"),
    "L06": ("**Exercise (8 min):** write a failure report for `device=mps` plus `finetuning=qlora`. Do not suggest silent full-FT fallback.", "<details><summary>Check</summary>Name bitsandbytes/CUDA constraints, expected download and consent, plus LoRA/CPU-toy alternatives.</details>"),
    "L07": ("**Exercise (8 min):** create logits with low overlap but a small entropy gap. Explain why that alone cannot establish a bad teacher.", "<details><summary>Check</summary>Support ranking and sharpness are different axes; correctness/environment-success evidence is still needed.</details>"),
    "L08": ("**Exercise (8 min):** construct two trajectories with the same final number but different first actions; compare observation traces.", "<details><summary>Check</summary>Different environment transitions mean different provenance even if some text/final values coincide.</details>"),
    "L09": ("**Exercise (8 min):** tabulate F2B/B2F masks at depths 1, 2, and 3 for a three-turn batch, including the first step each turn appears.", "<details><summary>Check</summary>F2B grows from low turn IDs, B2F from high IDs; distinguish pacing from depth.</details>"),
    "L10": ("**Exercise (10 min):** set all SAGE interventions to zero and one; compare skip/normalization and reconfirm SOD weights have no gradient.", "<details><summary>Check</summary>Zero intervention is explicit, dense weights sum to response-token count, and SOD weights stay detached.</details>"),
    "L11": ("**Exercise (10 min):** write a five-sentence method memo covering task horizon, support, logit access, hardware, and the SFT baseline.", "<details><summary>Check</summary>Assumptions and measurement precede the algorithm name; include an avg@K/pass@K or language-retention guardrail.</details>"),
}

MISTAKES_KO = {
    "L00": """### M1 — OPD를 reward optimization으로 부르기

- 틀린 형태: student sample에 점수를 주니 곧바로 RL이라고 한다.
- 왜 틀렸나: 여기 supervision은 teacher token 분포이며 reward objective가 아니다.
- 고친 형태: state source와 target source를 분리해 말한다.
- 관련 검사: `test_teacher_is_frozen_and_student_updates`

### M2 — toy loss 감소를 benchmark 승리로 해석하기

- 틀린 형태: 8번 update 곡선으로 OPD가 SFT보다 낫다고 결론낸다.
- 왜 틀렸나: 서로 다른 objective의 배관만 확인한다.
- 고친 형태: 같은 budget의 held-out accuracy·agreement와 한계를 함께 본다.
- 관련 검사: `test_demo_writes_all_learner_facing_artifacts`""",
    "L01": """### M1 — logits를 이미 확률이라고 생각하기

- 틀린 형태: logits 합이 1이라고 가정한다.
- 왜 틀렸나: logits는 정규화되지 않은 실수 score다.
- 고친 형태: vocabulary 축 softmax와 합을 확인한다.
- 관련 검사: `test_future_token_does_not_change_past_logits`

### M2 — causal mask와 response mask를 합치기

- 틀린 형태: 미래 차단 mask로 prompt loss까지 제거됐다고 생각한다.
- 왜 틀렸나: attention 접근성과 objective 포함 여부는 다르다.
- 고친 형태: causal/attention/response mask를 따로 감사한다.
- 관련 검사: `test_sft_counts_only_response_targets`""",
    "L02": """### M1 — `KL(p,q)`의 방향을 암기만 하기

- 틀린 형태: 함수 인자 이름 없이 첫/둘째 인자를 추측한다.
- 왜 틀렸나: 논문·라이브러리 convention이 다르다.
- 고친 형태: `KL(teacher || student)`처럼 분포 역할을 쓴다.
- 관련 검사: `test_forward_kl_matches_hand_calculation`

### M2 — temperature만 바꾸고 gradient scale을 비교하기

- 틀린 형태: `T²` 보정 없이 loss 크기 차이를 objective 우열로 읽는다.
- 왜 틀렸나: softmax derivative scale도 변한다.
- 고친 형태: 동일 convention과 보정 여부를 run card에 기록한다.
- 관련 검사: `test_temperature_and_empty_masks_fail_loudly`""",
    "L03": """### M1 — target이 같으면 state도 같다고 보기

- 틀린 형태: SFT와 KD가 같은 answer를 쓰니 같은 학습이라고 한다.
- 왜 틀렸나: hard ID와 teacher distribution은 정보량이 다르다.
- 고친 형태: state source와 target representation을 각각 표시한다.
- 관련 검사: `test_teacher_is_frozen_and_student_updates`

### M2 — optimizer step만 공정성으로 보고하기

- 틀린 형태: 길이가 달라도 step 수만 맞춘다.
- 왜 틀렸나: 처리한 response token 수가 달라질 수 있다.
- 고친 형태: initial hash, split, step과 token budget을 모두 비교한다.
- 관련 검사: `test_demo_writes_all_learner_facing_artifacts`""",
    "L04": """### M1 — lambda와 beta를 같은 knob로 설명하기

- 틀린 형태: lambda를 올리면 reverse KL이 된다고 말한다.
- 왜 틀렸나: lambda는 state mixture, beta는 divergence다.
- 고친 형태: collection과 loss config를 별도 열로 기록한다.
- 관련 검사: `test_gjsd_boundaries_have_named_kl_direction`

### M2 — GKD와 sampled policy-gradient OPD를 동일시하기

- 틀린 형태: on-policy prefix라는 이유만으로 estimator도 같다고 한다.
- 왜 틀렸나: full vocabulary와 sampled token estimator는 분산·메모리가 다르다.
- 고친 형태: state source와 estimator를 두 축으로 분류한다.
- 관련 검사: `test_rollout_snapshots_are_detached_and_mode_is_restored`""",
    "L05": """### M1 — rollout graph를 optimizer까지 유지하기

- 틀린 형태: discrete sampling을 통해 gradient가 흐른다고 기대한다.
- 왜 틀렸나: sampled token 선택은 미분 가능하지 않다.
- 고친 형태: trajectory를 detach하고 current logits를 재계산한다.
- 관련 검사: `test_rollout_snapshots_are_detached_and_mode_is_restored`

### M2 — prompt·padding까지 KL 평균에 넣기

- 틀린 형태: `[B,T]` loss를 그대로 mean한다.
- 왜 틀렸나: prompt 길이와 padding이 budget을 왜곡한다.
- 고친 형태: shifted response mask로 `masked_mean`한다.
- 관련 검사: `test_sft_counts_only_response_targets`""",
    "L06": """### M1 — 모델 이름만 pin하고 revision은 최신으로 두기

- 틀린 형태: Hub ID만 기록해 재실행 때 weight가 바뀐다.
- 왜 틀렸나: code/config/weight가 이동할 수 있다.
- 고친 형태: 40-char revision, license, bytes와 checksum을 기록한다.
- 관련 검사: `test_all_checked_in_presets_parse`

### M2 — QLoRA 실패를 full FT로 숨기기

- 틀린 형태: bitsandbytes가 안 되면 더 큰 메모리 경로로 자동 전환한다.
- 왜 틀렸나: OOM과 결과 의미 변경을 숨긴다.
- 고친 형태: 명시적으로 차단하고 LoRA 또는 CUDA 환경을 선택하게 한다.
- 관련 검사: `test_qlora_preset_is_explicit_cuda_and_opt_in`""",
    "L07": """### M1 — 낮은 overlap을 teacher 오답으로 단정하기

- 틀린 형태: top-k가 다르면 teacher가 나쁘다고 한다.
- 왜 틀렸나: student가 유용한 teacher mode를 아직 못 본 것일 수 있다.
- 고친 형태: correctness, entropy, KL과 environment 성공을 함께 본다.
- 관련 검사: `test_identical_support_is_perfectly_aligned`

### M2 — vOPD baseline에 gradient를 흘리기

- 틀린 형태: baseline까지 optimization target처럼 미분한다.
- 왜 틀렸나: score-function 기대 gradient가 바뀔 수 있다.
- 고친 형태: baseline을 detach하고 sampled log-prob만 미분한다.
- 관련 검사: `test_vopd_is_zero_when_teacher_equals_student`""",
    "L08": """### M1 — 멀티턴을 긴 single-turn text로만 보기

- 틀린 형태: turn/observation 경계를 버린다.
- 왜 틀렸나: action이 다음 state를 바꾸는 인과를 감사할 수 없다.
- 고친 형태: turn IDs, terminal과 observation을 보존한다.
- 관련 검사: `test_an_early_error_changes_later_state`

### M2 — token agreement를 task success로 부르기

- 틀린 형태: teacher token과 많이 같으면 환경 성공이라고 한다.
- 왜 틀렸나: 한 핵심 action 오류가 전체 task를 실패시킬 수 있다.
- 고친 형태: sequence success와 token 지표를 함께 기록한다.
- 관련 검사: `test_an_early_error_changes_later_state`""",
    "L09": """### M1 — B2F를 단순 reverse mask로 구현하기

- 틀린 형태: 실패한 student prefix 뒤 suffix만 선택한다.
- 왜 틀렸나: B2F는 성공/teacher prefix 조건이 핵심이다.
- 고친 형태: prefix provenance와 선택 turn을 함께 기록한다.
- 관련 검사: `test_tcod_curriculum_and_directions`

### M2 — depth와 global step을 같은 값으로 쓰기

- 틀린 형태: 매 step마다 무조건 turn 하나를 늘린다.
- 왜 틀렸나: pacing schedule이 사라진다.
- 고친 형태: start depth, pacing steps, max depth를 명시한다.
- 관련 검사: `test_tcod_curriculum_and_directions`""",
    "L10": """### M1 — SOD weight gradient 정책을 생략하기

- 틀린 형태: detach 여부를 프레임워크 기본값에 맡긴다.
- 왜 틀렸나: objective와 second-order 경로가 달라진다.
- 고친 형태: curriculum weight로 detach한다고 코드·문서·테스트에 고정한다.
- 관련 검사: `test_sod_downweights_a_divergence_jump`

### M2 — mini SAGE proxy를 semantic judge로 부르기

- 틀린 형태: token agreement proxy를 teacher 판단 성능으로 보고한다.
- 왜 틀렸나: 의미적 recoverability를 측정하지 않는다.
- 고친 형태: proxy label과 research judge 미검증 상태를 표시한다.
- 관련 검사: `test_sage_weights_normalize_and_skip`""",
    "L11": """### M1 — avg@K와 pass@K를 바꿔 쓰기

- 틀린 형태: sample 평균 성공과 하나 이상 성공을 같은 수치로 쓴다.
- 왜 틀렸나: K가 커질 때 의미가 크게 갈린다.
- 고친 형태: sampling matrix에서 두 정의를 별도로 계산한다.
- 관련 검사: `test_avg_and_pass_at_k_are_not_interchangeable`

### M2 — 하나의 benchmark로 방법을 확정하기

- 틀린 형태: math accuracy만 보고 language retention과 큰-K 능력을 무시한다.
- 왜 틀렸나: post-training capability가 이동할 수 있다.
- 고친 형태: task, retention, support와 scaling guardrail을 함께 둔다.
- 관련 검사: `test_opd2_gate_closes_when_teacher_equals_base`""",
}

MISTAKES_EN = {
    "L00": """### M1 — Calling OPD reward optimization

- Wrong: sampling plus a score must mean RL.
- Why: supervision here is a teacher token distribution, not a reward objective.
- Fix: name state source and target source separately.
- Related check: `test_teacher_is_frozen_and_student_updates`

### M2 — Reading toy loss decrease as benchmark victory

- Wrong: conclude OPD beats SFT from eight updates.
- Why: the run only checks two objective pipelines.
- Fix: use equal-budget held-out accuracy/agreement and limitations.
- Related check: `test_demo_writes_all_learner_facing_artifacts`""",
    "L01": """### M1 — Treating logits as probabilities

- Wrong: assume logits already sum to one.
- Why: logits are unnormalized real scores.
- Fix: apply vocabulary-axis softmax and check the sum.
- Related check: `test_future_token_does_not_change_past_logits`

### M2 — Merging causal and response masks

- Wrong: assume blocking the future also removes prompt loss.
- Why: attention visibility and objective inclusion differ.
- Fix: audit causal, attention, and response masks separately.
- Related check: `test_sft_counts_only_response_targets`""",
    "L02": """### M1 — Memorizing unnamed `KL(p,q)` direction

- Wrong: guess roles from argument position.
- Why: paper/library conventions vary.
- Fix: write `KL(teacher || student)` with named arguments.
- Related check: `test_forward_kl_matches_hand_calculation`

### M2 — Comparing temperatures without gradient scaling

- Wrong: interpret loss-size changes without noting `T²` correction.
- Why: softmax derivative scale also changes.
- Fix: record temperature and correction convention in the run card.
- Related check: `test_temperature_and_empty_masks_fail_loudly`""",
    "L03": """### M1 — Assuming equal targets imply equal learning

- Wrong: SFT and KD share an answer, so they are identical.
- Why: hard IDs and teacher distributions carry different information.
- Fix: label state source and target representation separately.
- Related check: `test_teacher_is_frozen_and_student_updates`

### M2 — Reporting only optimizer steps for fairness

- Wrong: equalize steps despite different sequence lengths.
- Why: processed response-token counts can differ.
- Fix: compare initial hash, split, steps, and token budget.
- Related check: `test_demo_writes_all_learner_facing_artifacts`""",
    "L04": """### M1 — Conflating lambda and beta

- Wrong: raising lambda makes the objective reverse KL.
- Why: lambda mixes states; beta selects divergence.
- Fix: log collection and loss configs separately.
- Related check: `test_gjsd_boundaries_have_named_kl_direction`

### M2 — Equating GKD with sampled policy-gradient OPD

- Wrong: on-policy prefixes imply identical estimators.
- Why: full-vocabulary and sampled-token estimators differ in variance/memory.
- Fix: classify state source and estimator on two axes.
- Related check: `test_rollout_snapshots_are_detached_and_mode_is_restored`""",
    "L05": """### M1 — Retaining the rollout graph through update

- Wrong: expect gradient through a discrete sampled token.
- Why: token selection is not differentiable.
- Fix: detach trajectories and recompute current logits.
- Related check: `test_rollout_snapshots_are_detached_and_mode_is_restored`

### M2 — Averaging KL over prompts and padding

- Wrong: directly mean the `[B,T]` loss.
- Why: prompt length and padding distort the budget.
- Fix: apply the shifted response mask with `masked_mean`.
- Related check: `test_sft_counts_only_response_targets`""",
    "L06": """### M1 — Pinning only a model name

- Wrong: leave revision at latest.
- Why: code, config, and weights can move.
- Fix: record a 40-char revision, license, bytes, and checksums.
- Related check: `test_all_checked_in_presets_parse`

### M2 — Hiding QLoRA failure with full FT

- Wrong: silently switch to a larger-memory path when bitsandbytes fails.
- Why: this hides OOM risk and changes experiment meaning.
- Fix: block explicitly and offer LoRA or validated CUDA.
- Related check: `test_qlora_preset_is_explicit_cuda_and_opt_in`""",
    "L07": """### M1 — Declaring a bad teacher from low overlap

- Wrong: different top-k sets mean the teacher is wrong.
- Why: the student may not yet visit a useful teacher mode.
- Fix: combine correctness, entropy, KL, and environment success.
- Related check: `test_identical_support_is_perfectly_aligned`

### M2 — Differentiating the vOPD baseline

- Wrong: optimize the baseline as a target.
- Why: it can alter the expected score-function gradient.
- Fix: detach the baseline; differentiate sampled log-probability only.
- Related check: `test_vopd_is_zero_when_teacher_equals_student`""",
    "L08": """### M1 — Treating multi-turn as one long text

- Wrong: discard turn and observation boundaries.
- Why: action-to-next-state causality becomes unauditable.
- Fix: preserve turn IDs, terminal state, and observations.
- Related check: `test_an_early_error_changes_later_state`

### M2 — Calling token agreement task success

- Wrong: high teacher-token agreement implies environment success.
- Why: one critical action can fail the whole task.
- Fix: report sequence success with token metrics.
- Related check: `test_an_early_error_changes_later_state`""",
    "L09": """### M1 — Implementing B2F as a reversed mask

- Wrong: select a suffix after a failed student prefix.
- Why: a successful/teacher prefix is central to B2F.
- Fix: record prefix provenance and selected turns.
- Related check: `test_tcod_curriculum_and_directions`

### M2 — Equating depth with global step

- Wrong: add one turn every optimizer step.
- Why: pacing disappears.
- Fix: name start depth, pacing steps, and maximum depth.
- Related check: `test_tcod_curriculum_and_directions`""",
    "L10": """### M1 — Omitting the SOD weight-gradient policy

- Wrong: leave detach behavior to incidental framework operations.
- Why: the objective and higher-order path change.
- Fix: specify and test detached curriculum weights.
- Related check: `test_sod_downweights_a_divergence_jump`

### M2 — Calling the mini SAGE proxy a semantic judge

- Wrong: report token agreement as teacher-judgment performance.
- Why: it does not measure semantic recoverability.
- Fix: label the proxy and unverified research judge.
- Related check: `test_sage_weights_normalize_and_skip`""",
    "L11": """### M1 — Swapping avg@K and pass@K

- Wrong: use mean sample success and any-success probability interchangeably.
- Why: their meanings diverge as K grows.
- Fix: compute both from the same sampling matrix.
- Related check: `test_avg_and_pass_at_k_are_not_interchangeable`

### M2 — Choosing a method from one benchmark

- Wrong: use math accuracy alone and ignore retention/large-K ability.
- Why: post-training capability can move.
- Fix: combine task, retention, support, and scaling guardrails.
- Related check: `test_opd2_gate_closes_when_teacher_equals_base`""",
}


def make_lessons() -> tuple[Lesson, ...]:
    lessons: list[Lesson] = []
    for index, base in enumerate(BASE):
        lesson_id, slug, title_ko, title_en, minutes, track, sources = base
        previous_id = "start" if index == 0 else BASE[index - 1][0]
        next_id = "finish" if index == len(BASE) - 1 else BASE[index + 1][0]
        prediction_ko = f"실행 전 예측: {lesson_id}의 첫 출력에서 가장 먼저 확인해야 할 invariant는 무엇일까? 한 문장으로 적고 실행한다."
        prediction_en = f"Predict before running: which invariant should you inspect first in {lesson_id}'s output? Write one sentence, then run."
        exercise_ko, solution_ko = EXERCISES_KO[lesson_id]
        exercise_en, solution_en = EXERCISES_EN[lesson_id]
        mistakes_ko = MISTAKES_KO[lesson_id]
        mistakes_en = MISTAKES_EN[lesson_id]
        code_cells, check_code = LESSON_CODE[lesson_id]
        lessons.append(
            Lesson(
                lesson_id, slug, title_ko, title_en, minutes, track, sources,
                OBJECTIVES_KO[index], OBJECTIVES_EN[index],
                f"{previous_id} → **{lesson_id}** → {next_id}",
                f"{previous_id} → **{lesson_id}** → {next_id}",
                EXPLANATIONS_KO[index], EXPLANATIONS_EN[index],
                prediction_ko, prediction_en, code_cells, check_code,
                exercise_ko, exercise_en, solution_ko, solution_en,
                mistakes_ko, mistakes_en,
                (OBJECTIVES_KO[index][0], OBJECTIVES_KO[index][1], OBJECTIVES_KO[index][2]),
                (OBJECTIVES_EN[index][0], OBJECTIVES_EN[index][1], OBJECTIVES_EN[index][2]),
            )
        )
    return tuple(lessons)


def cell_metadata(cell_id: str, role: str) -> dict[str, Any]:
    return {"opd_study": {"cell_id": cell_id, "role": role}}


def markdown(lesson_id: str, suffix: str, role: str, text: str) -> Any:
    return nbformat.v4.new_markdown_cell(
        text, metadata=cell_metadata(f"{lesson_id}-{suffix}", role)
    )


def code(lesson_id: str, suffix: str, role: str, source: str) -> Any:
    return nbformat.v4.new_code_cell(
        source, metadata=cell_metadata(f"{lesson_id}-{suffix}", role)
    )


def source_reference(source_id: str) -> str:
    section, record = SOURCE_RECORDS[source_id]
    if section == "papers":
        url = record["url"]
        version = record["arxiv"]
        license_name = record["paper_license"]
    elif section == "code_sources":
        url = record["repository"]
        version = record["revision"] or "no-public-revision"
        license_name = record["license"] or "no-code-reuse"
    elif section == "datasets":
        url = f"https://huggingface.co/datasets/{source_id}"
        version = record["revision"]
        license_name = record["license"]
    else:
        url = f"https://huggingface.co/{source_id}"
        version = record["revision"]
        license_name = record["license"]
    return (
        f"- [`{source_id}`]({url}) · `{version}` · license `{license_name}` · "
        "[audited manifest](../../docs/sources.yml)"
    )


def concept_map(lesson: Lesson, language: str) -> str:
    if lesson.lesson_id == "L00":
        intro = "전체 지도" if language == "ko" else "Full map"
        return f"""### {intro}

```mermaid
flowchart LR
  D[Prompt/Data] --> R{{State source}}
  R --> SFT[SFT]; R --> KD[Off-policy KD]; R --> OPD[Student rollout / OPD]
  OPD --> T[Teacher score] --> L[Masked loss] --> U[Student update] --> E[Evaluation]
  OPD --> G[GKD / vOPD / OPD2]; OPD --> M[Multi-turn: TCOD / SOD / SAGE]
```

```text
Prompt/Data -> fixed hard labels: SFT
            -> fixed teacher traces: off-policy KD
            -> current student rollout: OPD -> teacher score -> masked loss -> update
               -> GKD / vOPD / OPD2 / multi-turn TCOD-SOD-SAGE
All routes -> fair evaluation: accuracy, agreement, KL, overlap, avg@K, pass@K
```

Alt text: Three state sources lead to SFT, off-policy KD, or a student-rollout OPD loop; OPD branches into GKD, variance/delta methods, and multi-turn stabilization before fair evaluation."""
    label = "현재 위치" if language == "ko" else "Current position"
    return f"""### {label}: {lesson.position_ko if language == 'ko' else lesson.position_en}

```text
Prompt/Data -> state source -> ... -> {lesson.lesson_id} -> ... -> fair evaluation
```

Alt text: The course map highlights {lesson.lesson_id} between its prerequisite and next lesson; every method remains connected to the same evaluation stage."""


def build_notebook(lesson: Lesson, language: str) -> Any:
    korean = language == "ko"
    title = lesson.title_ko if korean else lesson.title_en
    objectives = lesson.objectives_ko if korean else lesson.objectives_en
    goal_lines = "\n".join(f"- {item}" for item in objectives)
    path_label = ", ".join(lesson.track)
    cells = [
        markdown(lesson.lesson_id, "T00", "objective", f"# {lesson.lesson_id} · {title}"),
        markdown(
            lesson.lesson_id,
            "G01",
            "objective",
            (f"## Goal\n\n**예상 시간:** {lesson.minutes}분 · **경로:** {path_label}\n\n{goal_lines}"
             if korean else f"## Goal\n\n**Estimated time:** {lesson.minutes} min · **Path:** {path_label}\n\n{goal_lines}"),
        ),
        markdown(lesson.lesson_id, "G02", "map", concept_map(lesson, language)),
        markdown(lesson.lesson_id, "S01", "explain", "## Setup"),
        code(lesson.lesson_id, "S02", "demo", f'LESSON_ID = "{lesson.lesson_id}"\n' + SETUP),
        markdown(lesson.lesson_id, "P01", "explain", "## Steps"),
        markdown(
            lesson.lesson_id,
            "P02",
            "explain",
            f"### 1/3 · 8–12 min\n\n{lesson.explanation_ko if korean else lesson.explanation_en}\n\n"
            + ("그림 대체 설명: 출력의 label과 숫자는 색 없이도 읽을 수 있다."
               if korean else "Figure alt: labels and numbers remain readable without color."),
        ),
        markdown(
            lesson.lesson_id,
            "P02A",
            "explain",
            ("### 핵심 원리\n\n" + DEEP_DIVES_KO[lesson.lesson_id]
             if korean else "### Core mechanics\n\n" + DEEP_DIVES_EN[lesson.lesson_id]),
        ),
        markdown(
            lesson.lesson_id,
            "P02B",
            "explain",
            ("### 실제 구현: 왜 이렇게 만들었나\n\n" + IMPLEMENTATION_KO[lesson.lesson_id]
             if korean else "### Production implementation: why this design\n\n" + IMPLEMENTATION_EN[lesson.lesson_id]),
        ),
        code(
            lesson.lesson_id,
            "P02C",
            "demo",
            SOURCE_PROBES[lesson.lesson_id],
        ),
        markdown(
            lesson.lesson_id,
            "P02D",
            "explain",
            ("### 다른 선택지는 없나?\n\n" + ALTERNATIVES_KO[lesson.lesson_id]
             if korean else "### Alternatives and trade-offs\n\n" + ALTERNATIVES_EN[lesson.lesson_id]),
        ),
        markdown(
            lesson.lesson_id,
            "P02E",
            "explain",
            "### 2/3 · 실행하고 관찰하기" if korean else "### 2/3 · Run and observe",
        ),
        markdown(
            lesson.lesson_id,
            "P03",
            "predict",
            lesson.prediction_ko if korean else lesson.prediction_en,
        ),
    ]
    for index, source in enumerate(lesson.code_cells, start=1):
        cells.append(code(lesson.lesson_id, f"P{index + 3:02d}", "demo", source))
    cells.extend(
        [
            markdown(lesson.lesson_id, "C01", "check", "## Checks"),
            code(lesson.lesson_id, "C02", "check", lesson.check_code),
            markdown(
                lesson.lesson_id,
                "C03",
                "exercise",
                lesson.exercise_ko if korean else lesson.exercise_en,
            ),
            markdown(
                lesson.lesson_id,
                "C04",
                "solution",
                lesson.solution_ko if korean else lesson.solution_en,
            ),
            markdown(
                lesson.lesson_id,
                "M01",
                "mistake-note",
                "## 내가 자주 틀리는 것" if korean else "## My recurring mistakes",
            ),
            markdown(
                lesson.lesson_id,
                "M02",
                "mistake-note",
                lesson.mistakes_ko if korean else lesson.mistakes_en,
            ),
            markdown(
                lesson.lesson_id,
                "R01",
                "summary",
                "## 60초 요약" if korean else "## 60-second summary",
            ),
            markdown(
                lesson.lesson_id,
                "R02",
                "summary",
                "\n".join(
                    f"{index}. {item}"
                    for index, item in enumerate(
                        lesson.summary_ko if korean else lesson.summary_en, start=1
                    )
                ),
            ),
            markdown(lesson.lesson_id, "N01", "next", "## Next Steps"),
            markdown(
                lesson.lesson_id,
                "N02",
                "next",
                ("다음 노트북으로 가기 전, 위 assertion을 다시 실행하고 틀린 예측 한 줄을 남긴다."
                 if korean else "Before the next notebook, rerun the assertions and record one prediction you revised."),
            ),
            markdown(
                lesson.lesson_id,
                "N03",
                "source",
                "### Sources\n\n" + "\n".join(
                    source_reference(source_id) for source_id in lesson.sources
                ),
            ),
        ]
    )
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.update(
        {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "opd_study": {
                "lesson_id": lesson.lesson_id,
                "language": language,
                "track": list(lesson.track),
                "profile": "toy",
                "source_ids": list(lesson.sources),
                "schema_version": 1,
            },
        }
    )
    return notebook


def main() -> None:
    for lesson in make_lessons():
        for language in ("ko", "en"):
            output = REPOSITORY / "notebooks" / language / f"{lesson.lesson_id[1:]}_{lesson.slug}.ipynb"
            output.parent.mkdir(parents=True, exist_ok=True)
            nbformat.write(build_notebook(lesson, language), output)
    print("generated 12 Korean and 12 English notebooks")


if __name__ == "__main__":
    main()

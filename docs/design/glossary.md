# Korean–English glossary

Use the Korean term with the English term in parentheses on first appearance in each
lesson. Identifiers and docstrings remain English.

| Stable term | 한국어 | English | Usage note |
|---|---|---|---|
| autoregressive_lm | 자기회귀 언어 모델 | autoregressive language model | Avoid translating as merely “sequential model.” |
| teacher | 교사 모델 | teacher model | A scorer need not generate the rollout. |
| student | 학생 모델 | student model | The optimized/current policy. |
| distillation | 지식 증류 | knowledge distillation | Use “증류” after first definition. |
| on_policy | 온폴리시 | on-policy | Data/state distribution comes from the current student. |
| off_policy | 오프폴리시 | off-policy | Fixed/other-policy states; do not equate with teacher labels alone. |
| rollout | 롤아웃 | rollout | A sampled response or environment trajectory. |
| trajectory | 궤적 | trajectory | Ordered states/actions/tokens, possibly multi-turn. |
| exposure_bias | 노출 편향 | exposure bias | Qualify rather than claiming it explains every mismatch. |
| distribution_shift | 분포 이동 | distribution shift | State/prefix distribution shift in this course. |
| entropy | 엔트로피 | entropy | State which distribution it belongs to. |
| cross_entropy | 교차 엔트로피 | cross-entropy | Name target and predicted distribution. |
| forward_kl | 정방향 KL | forward KL | Course convention: `KL(teacher || student)`. |
| reverse_kl | 역방향 KL | reverse KL | Course convention: `KL(student || teacher)`. |
| generalized_jsd | 일반화 JSD | generalized Jensen–Shannon divergence | Always show beta convention beside the formula. |
| support | 지지집합 | support | Tokens/states with non-negligible mass. |
| support_overlap | 지지집합 중첩 | support overlap | Report top-k definition and mass. |
| response_mask | 응답 마스크 | response mask | Excludes prompt, padding and optionally environment tokens. |
| stop_gradient | 그래디언트 차단 | stop-gradient / detach | Distinguish sampling non-differentiability from detach. |
| sampled_estimator | 표본 기반 추정량 | sampled estimator | Report bias/variance and sampling policy. |
| control_variate | 제어 변량 | control variate | vOPD baseline is detached and action-independent. |
| intervention | 개입 | intervention | SAGE teacher decision after environment feedback. |
| curriculum_depth | 커리큘럼 깊이 | curriculum depth | Number/range of supervised turns in TCOD. |
| exact_match | 완전 일치 정확도 | exact match | Normalization rule belongs with the metric. |
| teacher_agreement | 교사 일치율 | teacher agreement | Not a substitute for task correctness. |
| avg_at_k | K회 평균 정확도 | avg@K | Expected success among K samples. |
| pass_at_k | K회 중 최소 1회 성공률 | pass@K | Capability-boundary proxy depends on sampling and K. |
| capability_boundary | 능력 경계 | capability boundary | Do not infer solely from avg@1. |
| clean_room | 클린룸 재구현 | clean-room reimplementation | Based on paper description, not unlicensed source code. |

## Symbol convention

- `p_T`: teacher token distribution.
- `p_theta` or `p_S`: current student distribution.
- `x`: prompt/input; `y_<t`: generated prefix; `y_t`: token at position `t`.
- `lambda_on_policy`: probability/fraction of student-generated training states.
- `beta_jsd`: interpolation coefficient, with its exact formula always shown.
- Shapes use `[B, T, V]` for logits/log-probabilities and `[B, T]` for token IDs/masks.

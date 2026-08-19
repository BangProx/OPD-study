# OPD math conventions

Let `p_T(v|c_t)` be the teacher distribution and `p_θ(v|c_t)` the current student
distribution at prefix/state `c_t`. All masks below select response tokens only.

## Foundations

- Hard-label SFT: `-log p_θ(y_t|c_t)` on fixed demonstrations.
- Forward KL: `KL(p_T || p_θ) = Σ_v p_T(v)(log p_T(v)-log p_θ(v))`.
- Reverse KL: `KL(p_θ || p_T) = Σ_v p_θ(v)(log p_θ(v)-log p_T(v))`.
- GKD generalized JSD uses `m=βp_T+(1-β)p_θ` and the repository convention
  `β KL(p_T||m)+(1-β) KL(p_θ||m)`, with explicit boundary definitions β=0 as
  forward KL and β=1 as reverse KL.

The implementation computes log-softmax and reductions in float32, then divides by the
effective mask sum. Empty masks fail instead of returning NaN. See
[`math.py`](../src/opd_study/math.py) and analytic tests in
[`test_math.py`](../tests/test_math.py).

## Original GKD versus modern OPD

The ICLR 2024 GKD algorithm has two independent controls:

1. `lambda_on_policy` chooses whether a training prefix comes from fixed data or the
   current student.
2. `beta_jsd` chooses the divergence on that prefix.

Modern reasoning OPD commonly uses a student-sampled response and the sampled-token
reward `r_t = log p_T(y_t|c_t)-log p_θ(y_t|c_t)`. Maximizing
`stopgrad(r_t) log p_θ(y_t|c_t)` is a policy-gradient estimate of minimizing reverse KL.
The full-vocabulary form computes the conditional reverse KL exactly at every visited
state. These methods share student states but not estimator cost or variance.

## vOPD

vOPD observes that the value of the sampled OPD reward is
`E_{v~p_θ}[r(v)] = -KL(p_θ||p_T)`. Its advantage is therefore
`r(y_t) + KL(p_θ||p_T)`. The KL is detached and action-independent: it reduces sampled
gradient variance without becoming the optimized full-vocabulary target. A student
top-k renormalized KL may approximate the baseline while preserving action independence.

## OPD²

OPD² defines `RΔ(v)=log p_T(v)-log p_Tbase(v)`, centers both delta and conventional OPD
rewards under the student distribution, and uses the delta advantage only where
`AΔ × AOPD > 0`. The gate prevents a delta update that moves against ordinary teacher
alignment. The teacher-base must be the corresponding pre-reasoning checkpoint; an
arbitrary third model does not satisfy the method's premise.

## Multi-turn weights

- TCOD-F2B selects the first `k(n)` turns. TCOD-B2F requires a teacher/successful prefix
  and selects the last `k(n)` student turns. A suffix mask alone is not B2F.
- SOD computes per-step sampled-logprob gap `d_k`, then cumulative ratio weights. Core
  implements the distillation term; paper-scale SOD also adds GRPO.
- SAGE-OPD maps Skip/Weak/Strong to `0/α/1`, multiplies by mean teacher top-1
  confidence per turn, then normalizes weights so their total matches the dense
  response-token count.

Primary paper/version IDs and licenses are in [`sources.yml`](sources.yml).

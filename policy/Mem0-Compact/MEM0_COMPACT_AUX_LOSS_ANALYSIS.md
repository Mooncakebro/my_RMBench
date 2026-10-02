# Mem0-Compact Aux-Loss Analysis & Memory-Module Fixes

Context: training run with Mem-0 warm start + frozen base model. Observed log
(step ~24.7k/30k):

```
[train] step 24700/30000 loss 5.0991 (action 0.0030 obs 483.8772 nll 23.2188 mem_l2 22.3696 total 5.0963) lr 8.29e-06
```

## 1. Decoding the log

`total = action + λ_obs·obs + λ_nll·nll + λ_mem·mem_l2` with
λ_obs=0.01, λ_nll=0.01, λ_mem=0.001 (`vendor/jamel_compact/config.py`,
assembled in `source/models/execution_module/mem0_compact_executor.py`):

| term    | raw    | weighted | share of total |
|---------|--------|----------|----------------|
| action  | 0.004  | 0.004    | 0.08 %         |
| obs     | ~487   | ~4.87    | **~95 %**      |
| nll     | ~23.3  | 0.233    | 4.5 %          |
| mem_l2  | ~23    | 0.023    | 0.4 %          |

Notes:
- "action" is the DiT action loss (the CompactMoDE wrapper never computes text
  CE). It is already near-zero because the warm-started frozen base predicts
  actions well — so the aux losses are effectively the *only* memory training
  signal, and 95 % of it is the obs-prediction MSE.
- `mem_l2` is a *sum* of squares over all elements (`loss.py`), so 23 over
  28 layers × 16 slots × 512 dims × batch is tiny per element — not exploding.

## 2. Findings

1. **Kalman filter ~45× miscalibrated.** NLL = 0.5·(log R + e/R) with surprise
   e ≈ 490 ⇒ learned observation noise R ≈ 11, ~45× below the actual surprise.
   If R were calibrated (R ≈ e), NLL would be ≈ 0.5·(log 490 + 1) ≈ 3.6, not
   23.3. The softplus-MLP parameterization of R_psi cannot grow output scale
   fast enough at λ_nll = 0.01 under global grad clipping.

2. **Surprise mechanism permanently saturated.** `surprise_clip = 10` was
   designed for O(1) residuals; with e ≈ 490 the inflation term is pegged at
   the clip for every sample/layer/step — the adaptive variance inflation (U2)
   has zero dynamic range left.

3. **All U2/U3 hyperparameters mis-scaled for this error regime.**
   `r_min=0.01`, `init_variance=0.5`, `surprise_clip=10` assume O(1) errors,
   but z is an *unnormalized* 512-d projection of frozen Qwen3-VL hidden
   states whose scale grows with layer depth. The whole predict–correct loop
   operated outside its design range.

4. **Real structural bottleneck in the obs model.** `z_pred =
   obs_model(m_hat.mean(dim=1))`: the observation model saw only the *mean
   over 16 memory slots* and predicted only the *mean over k=4 observation
   tokens*. Mean-pool → MLP → predict-a-mean is a narrow channel, and it is
   the main supervised signal shaping what memory stores.

5. **Part of L_obs is irreducible.** z_target pools the *current* image
   tokens; after an action the scene changes in ways (memory, action) cannot
   predict. The absolute MSE value therefore cannot by itself prove the memory
   is weak — an explained-variance (R²) diagnostic is needed (not yet added).

## 3. Fixes applied (v3, `vendor/jamel_compact/model.py`)

1. **Normalized observation space.** `obs_norm = LayerNorm(d_mem, affine-off)`
   applied per token in `extract_observation()`; `z_pred` is LayerNorm'ed the
   same way before the MSE. Surprise e is now bounded (≈ [0, 4]) regardless of
   backbone hidden-state scale, so `surprise_clip`, `r_min`, `init_variance`
   are back in their intended O(1) regime and the clip no longer saturates.

2. **Log-parameterized observation noise.** R_psi output is now interpreted as
   log R: `R = exp(log_r) + r_min` with log_r clamped to [-10, 10]. The NLL
   gradient becomes scale-free (multiplicative), so R can traverse orders of
   magnitude in a few hundred steps instead of stalling 45× low.

3. **Widened obs-model channel.** k learned `obs_pred_queries` cross-attend to
   **all 16 memory slots** (`obs_pred_attn`), then the obs MLP predicts each of
   the k observation tokens **individually** (target = per-token z_down, not
   the token mean). Slot-specific information now reaches the prediction.

Supporting changes:
- `config.py`: `model_version` bumped 2 → 3 (side-memory state dicts changed:
  new `obs_pred_queries` / `obs_pred_attn`; `R_psi` weights now mean log R).
- `from_pretrained` now prints *missing* keys as well, so warm-starting from
  a v2 checkpoint explicitly shows which v3 modules are freshly initialized.

## 4. Checkpoint compatibility

The frozen base LLM and `action_embed` load unchanged. Side-memory weights
from v2 checkpoints load with `strict=False`; the new v3 parameters
(`obs_pred_queries`, `obs_pred_attn`) start randomly initialized and old
`R_psi` weights change semantics (softplus → log-R), so **memory modules
should be retrained** (base warm start unaffected). Old side-memory
checkpoints are not recommended for warm start.

## 5. Deferred recommendations (not yet applied)

- Rebalance λ_obs (obs term is 95 % of the objective; action is 0.08 %) or
  adaptively weight by a detached running mean of the obs loss.
- Log per-layer means of e / R / K and obs R² (1 − MSE/Var(z)) — the
  diagnostics plumbing in `correct(return_diagnostics=True)` already exists.
- Only if still needed after 1–3: scale capacity (num_mem_tokens 16→32/64,
  mem_dim 512→1024, deeper obs model).
- Note: `obs_loss: "infonce"` in config.py is a dead option (MSE is
  hardcoded); with normalized embeddings, InfoNCE remains a viable alternative.
- `policy/Pi05-Compact/vendor/jamel_compact` is a separate copy and was NOT
  modified; port the fixes there if parity is wanted.

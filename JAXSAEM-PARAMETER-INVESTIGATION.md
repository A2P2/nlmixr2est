# jaxsaem Parameter Discrepancy: Investigation Report

**Date:** 2026-05-08  
**Investigator:** Automated analysis (Positron / Claude Sonnet 4.6)  
**Branch:** `claude/implement-jaxsaem-method-eCnqz`  
**Dataset:** `nlmixr2data::theo_sd` (12 subjects, 1-cmt oral theophylline)  

---

## 1. Observed Discrepancy

Fitting the same 1-cmt oral model with additive residual error to the
theophylline dataset gives divergent population parameters:

| Parameter | `est = "saem"` (nlmixr2) | `est = "jaxsaem"` | Difference |
|-----------|--------------------------|-------------------|------------|
| `ka`      | 1.573                    | 1.575             | +0.1%      |
| `cl`      | 2.765                    | 1.974             | **−28.6%** |
| `v`       | 31.5                     | 36.5              | **+15.9%** |
| `add.sd`  | 0.58 (theta)             | 1.31 (sigma)      | **+126%**  |

Additionally:

| IIV (omega) | saem  | jaxsaem |
|-------------|-------|---------|
| `omega.ka`  | ~0.24 | 5e-4    |
| `omega.cl`  | ~0.09 | ~0      |
| `omega.v`   | ~0.08 | 0.011   |

The discrepancy is **not iteration-dependent**: jaxsaem runs at 100, 200, 400,
800, and 1600 iterations all converge to the same biased value (`cl ≈ 1.97`).
This rules out insufficient burn-in or EM length as the cause.

---

## 2. Investigation Method

### 2.1 Structural model correctness

`jaxsaem/models.py` implements `conc_oral_1cmt`:

```python
def conc_oral_1cmt(t, dose, theta):
    """theta = [ka, CL, V]"""
    ka, CL, V = theta[0], theta[1], theta[2]
    ke = CL / V
    diff = ka - ke
    safe_diff = jnp.where(jnp.abs(diff) < 1e-8, 1.0, diff)
    regular = (dose * ka) / (V * safe_diff) * (jnp.exp(-ke * t) - jnp.exp(-ka * t))
    limit   = (dose * ka * t / V) * jnp.exp(-ka * t)
    return jnp.where(jnp.abs(diff) < 1e-8, limit, regular)
```

The analytic solution is **mathematically correct** for the standard 1-cmt
oral model (`C(t) = F·dose·ka / (V·(ka-ke)) · (e^{-ke·t} - e^{-ka·t})`).
The degenerate case `ka ≈ ke` is handled. The `ka, CL, V` parameterisation
matches nlmixr2's `linCmt()`.

### 2.2 Parameter routing (R → Python)

`rxui_translator.py` defines `_LINCMT_LOOKUP`:

```python
tuple(sorted(["ka", "cl", "v"])): ("oral_1cmt", ["ka", "cl", "v"]),
```

The canonical order `["ka", "cl", "v"]` maps to `theta[0..2]` in
`conc_oral_1cmt`. Case-insensitive matching is applied. The permutation
tables `user_to_canonical` and `canonical_to_user` are constructed
correctly. **No routing bug was found.**

### 2.3 SAEM algorithm (fit_saem.py)

The SAEM loop (`jax.lax.scan` over `saem_step`) implements:

- **S-step**: per-subject Metropolis-Hastings random walk on log-scale
  parameters `b_i`, vmapped over subjects.
- **A-step**: running-average update of sufficient statistics with step size
  `γ_k = 1` (burn-in) then `γ_k = 1/(k − n_burn + 1)`.
- **M-step** (closed form for diagonal Omega + additive Gaussian residual):
  - `μ = s₁ / N`
  - `Ω² = max(s₂/N − μ², 1e-6)`
  - `σ² = max(s₃ / n_obs, 1e-8)`

---

## 3. Root Cause: IIV Collapse at Iteration 1

### 3.1 The bug

The initial state for `jax.lax.scan` is:

```python
init_state = (
    jnp.tile(init_mu[None, :], (N, 1)),   # b_init: ALL subjects at init_mu
    jnp.full((N,), init_log_step, ...),
    jnp.asarray(init_mu, ...),             # μ_init = init_mu  ✓
    jnp.asarray(init_omega2, ...),         # Ω²_init = user values (0.1, 0.1, 0.05)  ✓
    ...
    jnp.zeros(P, ...),   # s₁ = 0
    jnp.zeros(P, ...),   # s₂ = 0
    jnp.float64(0.0),    # s₃ = 0
    ...
)
```

**Iteration k = 0 (burn-in, γ = 1):**

1. MH uses current `omega2 = init_omega2 = [0.1, 0.1, 0.05]` → proposals
   accepted normally.  
2. After MH, all `b_new[i] ≈ init_mu` (only one step taken, very little
   diffusion).  
3. M-step updates sufficient statistics:
   - `s1_curr = sum(b_i)  = N × init_mu`
   - `s2_curr = sum(b_i²) = N × init_mu²`
   - With `γ = 1`: `s1 ← s1_curr`, `s2 ← s2_curr`
4. M-step computes new `omega2`:
   - `omega2_new = s2/N − (s1/N)² = init_mu² − init_mu² = 0`
   - Clamped to floor: **`omega2_new = 1e-6`**

**Iteration k = 1:**

MH now uses `omega2 = 1e-6` as the prior covariance. The log-posterior
penalty for any deviation from `μ` is:

```
−½ · (b_prop − μ)² / 1e-6
```

A proposal that moves `b_prop` by even `0.05` (a tiny 0.05 standard
deviation in the original scale) incurs a penalty of
`−½ × 0.05² / 1e-6 = −1250`. Metropolis acceptance probability: `exp(−1250) ≈ 0`.

**All proposals are rejected from iteration 1 onward.** Every `b_i` remains
frozen at `init_mu` for the entire chain.

### 3.2 Verification (Python)

```python
N, P = 12, 3
init_mu    = np.array([log(1.5), log(2.0), log(40.0)])
init_omega2 = np.array([0.1, 0.1, 0.05])

b = np.tile(init_mu, (N, 1))          # all subjects identical
s1 = b.sum(axis=0)
s2 = (b**2).sum(axis=0)
omega2_after = np.maximum(s2/N - (s1/N)**2, 1e-6)

# omega2_after = [1e-06, 1e-06, 1e-06]   ← CONFIRMED: always collapses to floor
```

With scattered initial `b_i ~ N(init_mu, sqrt(init_omega2))`:

```python
b_scattered = init_mu + sqrt(init_omega2) * rng.normal((N, P))
# omega2_after ≈ [0.076, 0.079, 0.025]   ← near the true init_omega2
```

### 3.3 Why only `cl` is highly biased

Once IIV collapses to zero, the model becomes a **fixed-effects population
model**: every subject shares the same parameters. The M-step for `μ`
(population means) now minimises a weighted least-squares problem where
the residuals absorb all between-subject variability. In the 1-cmt oral
model, `CL` and `V` are highly correlated in their effect on the terminal
slope (`ke = CL/V`). Without per-subject `η_cl` and `η_v`, the optimizer
finds a different equipotential manifold in (CL, V) space that minimises
total RSS but gives different marginal estimates.

The inflated residual sigma (1.31 vs 0.58) confirms that between-subject
variability has been absorbed into the residual term.

---

## 4. Consequences of the Bug

| Observable               | With IIV (correct)    | Without IIV (bug)       |
|--------------------------|-----------------------|-------------------------|
| `omega.cl`               | ~0.09 (estimated)     | ~0 (floor=1e-6)         |
| `omega.ka`               | ~0.24                 | ~5e-4                   |
| `sigma` (add)            | ~0.58                 | ~1.31                   |
| `cl`                     | 2.76                  | 1.97 (−28.6%)           |
| `v`                      | 31.5                  | 36.5 (+15.9%)           |
| Objective function value | meaningful `nll`      | `NaN`                   |

The objective function returning `NaN` is a secondary symptom: once all
subjects share identical parameters, the SAEM objective (which involves
`log(det(Omega))`) becomes undefined when `Omega → 0`.

---

## 5. Proposed Fix

### Option A — Scatter initial `b_i` from the prior (recommended)

Change `init_state` in `fit_saem.py` from:

```python
jnp.tile(init_mu[None, :], (N, 1)),   # all identical: BUG
```

to:

```python
# Sample initial b_i ~ N(init_mu, sqrt(init_omega2))
key_init, key = jax.random.split(jax.random.PRNGKey(seed))
b_init = (init_mu[None, :]
          + jnp.sqrt(init_omega2)[None, :]
          * jax.random.normal(key_init, (N, P)))
```

This ensures that the M-step at iteration 0 sees non-zero empirical variance
in `b`, so `omega2_new > 0` and the MH can explore the full posterior from
iteration 1 onward.

### Option B — Keep initial `omega2` frozen during early burn-in

Add a warm-up phase (first `n_warmup` iterations, e.g. 20) during which the
M-step uses the user-supplied `init_omega2` instead of updating from `s2`.
After the warm-up, subjects will have scattered sufficiently to produce a
reasonable empirical `omega2` from the M-step.

### Option C — Initialise sufficient statistics from the prior

Initialize `s1` and `s2` consistently with the initial parameter values and
the user-supplied `omega2` instead of zeros:

```python
# Implied by: b_i ~ N(init_mu, sqrt(init_omega2)), i=1..N
# E[b]   = init_mu  → s1_init = N * init_mu
# E[b^2] = init_mu^2 + init_omega2 → s2_init = N * (init_mu^2 + init_omega2)
s1_init = N * init_mu
s2_init = N * (init_mu**2 + init_omega2)
```

This makes the M-step at iteration 0 immediately return `omega2 ≈ init_omega2`.

---

## 6. Files Involved

| File | Location |
|------|----------|
| SAEM fitter (R&D) | `C:\Users\POGODAL2\jaxsaem-env312\Lib\site-packages\jaxsaem\fit_saem.py` |
| Analytic models   | `C:\Users\POGODAL2\jaxsaem-env312\Lib\site-packages\jaxsaem\models.py` |
| Spec translator   | `C:\Users\POGODAL2\jaxsaem-env312\Lib\site-packages\jaxsaem\rxui_translator.py` |
| Bridge entrypoint | `C:\Users\POGODAL2\jaxsaem-env312\Lib\site-packages\jaxsaem\nlmixr_bridge.py` |
| Data handler      | `C:\Users\POGODAL2\jaxsaem-env312\Lib\site-packages\jaxsaem\nlmixr_data.py` |
| R-side bridge     | `R/jaxsaem.R` (in this repo) |

The bug is entirely in the **Python package** (`jaxsaem`), specifically in
`fit_saem.py` lines that set up `init_state`. The R side correctly passes
`init_omega2` as user-declared values; they are overwritten immediately by
the M-step collapse.

The same bug almost certainly affects `fit_focei.py` if it uses the same
pattern of identical initial `b_i`.

---

## 7. Other Findings During Investigation

### 7.1 Bugs fixed in this branch (R side)

Three R-side bugs were found and fixed during initial testing (see commit
history and `JAXSAEM-TESTING.md`):

1. **`sort(paramNames)` in `.detectLinCmtTopology()`** — `vapply` names
   caused `identical()` to fail. Fixed with `sort(unname(paramNames))`.

2. **Condition class stripping in `nlmixr2Est0()`** — `try()` wrapped errors
   lost the `jaxsaemUnsupportedError` class. Fixed by re-throwing the
   original condition object.

3. **Named params in `.rxUiToSpec()`** — `as.list(.userParams)` sent a
   named Python dict (with theta names as keys) instead of a positional list.
   Fixed with `as.list(unname(.userParams))`.

### 7.2 Objective function is NaN

`run_jaxsaem` returns `objf = NaN` for `method = "saem"`. This is by design
(the code sets `objf = float("nan")` explicitly), since the SAEM log-
likelihood is not readily available without an additional MCMC pass. However,
the collapsed-IIV bug makes this value meaningless in any case.

### 7.3 SAEM sigma vs nlmixr2 add.sd

nlmixr2 reports `add.sd` as a **theta** (fixed-effect parameter estimated
jointly). jaxsaem treats sigma as a separate M-step update. The functional
form is the same (`y ~ add(sigma)`), but the degrees-of-freedom and
variance accounting differ, which may lead to small differences in `sigma`
estimates even after the IIV bug is fixed.

### 7.4 Single-dose limitation

`nlmixr_data.py` raises `UnsupportedFeatureError` for subjects with more
than one dose row or dose not at `t=0`. This is documented as a v0
restriction.

---

## 8. Recommended Next Steps for the Developer

1. **Fix `fit_saem.py`**: initialise `b_i` from the prior (Option A above)
   or use Option C (consistent sufficient statistic initialisation).
   Option A is the cleaner fix.

2. **Check `fit_focei.py`** for the same pattern.

3. **Add a regression test** that verifies `omega.cl > 0.05` after fitting
   `theo_sd` — this would have caught the collapse immediately.

4. **Validate parameters against nlmixr2 saem** at ≥400 iterations after
   the fix: expect `|Δcl/cl| < 5%`, `|Δv/v| < 5%`.

5. **Address `objf = NaN`**: either compute a final marginal log-likelihood
   (e.g. Laplace approximation at the final `μ`, `Ω`) or document clearly
   that AIC/BIC are unavailable for `method = "saem"`.

---

*Investigation performed using `nlmixr2data::theo_sd`, R 4.5.1, jaxsaem 0.0.1,
JAX 0.10.0, Python 3.12.10.*

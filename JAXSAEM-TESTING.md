# Testing `est = "jaxsaem"`

This guide walks through verifying the `est = "jaxsaem"` integration end
to end. It is intentionally ordered shortest -> longest so you can stop
as soon as something breaks.

The fitting algorithm itself lives in the Python package `jaxsaem`
(https://github.com/a2p2/fastsaem); this package only contains the R-side
bridge dispatched via `reticulate`.

## 1. Get the code

```bash
git clone --branch claude/implement-jaxsaem-method-eCnqz \
    https://github.com/a2p2/nlmixr2est.git
cd nlmixr2est
```

## 2. R-only smoke (no Python yet)

This catches namespace, dispatch, and preflight issues independently of
the Python side. `reticulate` lives in `Suggests`, so pull it
explicitly.

```r
install.packages(c("reticulate", "checkmate", "remotes"))
remotes::install_local(".", dependencies = TRUE)

library(nlmixr2est)

## control object exists and validates
str(jaxsaemControl())
jaxsaemControl(method = "foce", nIter = 100)   # should succeed
jaxsaemControl(method = "nope")                # should error

## dispatch reaches the right method (will fail at the bridge import,
## but that proves est = "jaxsaem" routes to nlmixr2Est.jaxsaem)
one.cmt <- function() {
  ini({
    tka <- log(1.5); tcl <- log(2); tv <- log(40)
    eta.ka ~ 0.1;    eta.cl ~ 0.1;  eta.v ~ 0.05
    add.sd <- 0.5
  })
  model({
    ka <- exp(tka + eta.ka)
    cl <- exp(tcl + eta.cl)
    v  <- exp(tv  + eta.v)
    linCmt() ~ add(add.sd)
  })
}
nlmixr2(one.cmt, nlmixr2data::theo_sd, est = "jaxsaem")
## expected without Python: an error mentioning
## "could not import 'jaxsaem.nlmixr_bridge'"
```

Verify the negative paths trip without ever touching Python:

```r
bad <- function() {
  ini({
    tka <- log(1.5); tcl <- log(2); tv <- log(40)
    eta.ka ~ 0.1;    eta.cl ~ 0.1;  eta.v ~ 0.05
    add.sd  <- 0.5
    prop.sd <- 0.1
  })
  model({
    ka <- exp(tka + eta.ka)
    cl <- exp(tcl + eta.cl)
    v  <- exp(tv  + eta.v)
    linCmt() ~ add(add.sd) + prop(prop.sd)
  })
}
tryCatch(
  nlmixr2(bad, nlmixr2data::theo_sd, est = "jaxsaem"),
  jaxsaemUnsupportedError = function(e) cat("OK:", conditionMessage(e), "\n")
)
```

## 3. Install the Python side

```bash
python3.10 -m venv ~/jaxsaem-env
source ~/jaxsaem-env/bin/activate
pip install git+https://github.com/a2p2/fastsaem.git
python -c "import jaxsaem; from jaxsaem.nlmixr_bridge import run_jaxsaem; print('ok')"
```

If `pip install` fails (the package may not be on PyPI), install from a
local clone instead: `pip install -e /path/to/fastsaem`.

## 4. End-to-end fit

```r
useJaxsaem(python = "~/jaxsaem-env/bin/python")
## or simply useJaxsaem() if RETICULATE_PYTHON is already set

fit <- nlmixr2(one.cmt, nlmixr2data::theo_sd, est = "jaxsaem",
               control = jaxsaemControl(nIter = 100, nBurn = 50, seed = 1))
print(fit)
fit$theta
diag(fit$omega)
fit$sigma
```

Sanity check: refit the same model with `est = "saem"` and compare the
thetas. They should land in the same neighborhood (within roughly 5-10%
on the log scale).

## 5. Run the test suite

```r
## from the repo root
devtools::test(filter = "jaxsaem")
```

Tests skip cleanly when `reticulate::py_module_available("jaxsaem")` is
`FALSE`, so they are safe to run with or without the Python env active.

## 6. Re-verify after the IIV-collapse fix (runbook for another agent)

The 2026-05-08 investigation
(`JAXSAEM-PARAMETER-INVESTIGATION.md`) traced the parameter-bias
symptoms (CL ~28% low, Omega ~0) to an iteration-0 IIV-collapse bug in
the Python package `jaxsaem`. The fix lives on a separate fastsaem
branch and has not yet been merged. This section is a self-contained
runbook for a follow-up agent (or human) to verify the fix end to end
and update the investigation report. Run from a clean shell.

### 6.1 Inputs

- **R repo / branch**: `a2p2/nlmixr2est` @ `claude/implement-jaxsaem-method-eCnqz` (this branch).
- **Python repo / branch**: `a2p2/fastsaem` @ `claude/fix-saem-iiv-collapse-eCnqz`.
- **Reference dataset**: `nlmixr2data::theo_sd` (12 subjects, 1-cmt oral).

The fastsaem branch contains two commits on top of `claude/python-pk-solver-TXiOg`:
1. `cc063a3` -- scatter `b_init ~ N(init_mu, diag(init_omega2))` in
   `jaxsaem/fit_saem.py`.
2. `c048f14` -- regression tests in `tests/test_iiv_init.py`.

### 6.2 Reinstall the patched Python side

```bash
source ~/jaxsaem-env/bin/activate   # whichever env was used in section 3
pip install --force-reinstall --no-deps \
    git+https://github.com/a2p2/fastsaem.git@claude/fix-saem-iiv-collapse-eCnqz

python -c "from jaxsaem.fit_saem import run_saem; \
           import inspect; \
           src = inspect.getsource(run_saem); \
           assert 'b_init = ' in src and 'jax.random.split' in src, \
               'fix not present'; \
           print('fix present')"
```

The assertion is the cheapest way to confirm `pip` actually pulled the
patched branch and not a cached wheel. If it fails, force-reinstall
again with `--no-cache-dir`.

### 6.3 Run the Python regression tests

```bash
cd ~/checkouts                            # pick a working dir
git clone --branch claude/fix-saem-iiv-collapse-eCnqz \
    https://github.com/a2p2/fastsaem.git
cd fastsaem
pytest tests/test_iiv_init.py -v
```

**Pass criteria**: all three tests pass:
- `test_omega2_does_not_collapse_to_floor`
- `test_omega2_independent_of_iter_count`
- `test_initial_b_is_scattered`

If any fail, **do not** continue -- inspect the failure and report
back. Pre-fix, all three should fail; post-fix, all three should pass.

### 6.4 Re-run the R regression test

```r
## from the nlmixr2est repo root, branch claude/implement-jaxsaem-method-eCnqz
useJaxsaem(python = "~/jaxsaem-env/bin/python")  # adjust path
devtools::test(filter = "jaxsaem")
```

**Pass criteria**: all six `test-jaxsaem.R` tests pass, including the
new `"recovers non-collapsed Omega (regression)"` test which asserts
`diag(omega) > 1e-3` componentwise and at least one entry above 0.05.

### 6.5 Parameter-recovery check vs nlmixr2 saem

This is the qualitative end-to-end check that validated the original
regression. Run both fitters on the same data and compare.

```r
library(nlmixr2est)
useJaxsaem(python = "~/jaxsaem-env/bin/python")

one.cmt <- function() {
  ini({
    tka <- log(1.5); tcl <- log(2); tv <- log(40)
    eta.ka ~ 0.1;    eta.cl ~ 0.1;  eta.v ~ 0.05
    add.sd <- 0.5
  })
  model({
    ka <- exp(tka + eta.ka)
    cl <- exp(tcl + eta.cl)
    v  <- exp(tv  + eta.v)
    linCmt() ~ add(add.sd)
  })
}

dat <- nlmixr2data::theo_sd

fit_saem <- suppressMessages(suppressWarnings(
  nlmixr2(one.cmt, dat, est = "saem",
          control = saemControl(nBurn = 200, nEm = 200, seed = 1, print = 0))
))
fit_jax  <- suppressMessages(suppressWarnings(
  nlmixr2(one.cmt, dat, est = "jaxsaem",
          control = jaxsaemControl(nIter = 400, nBurn = 200, seed = 1))
))

theta_saem <- exp(fit_saem$theta[c("tka", "tcl", "tv")])
theta_jax  <- fit_jax$theta[c("ka", "cl", "v")]
pct_diff   <- 100 * (theta_jax - theta_saem) / theta_saem

print(round(rbind(saem = theta_saem, jax = theta_jax,
                  pct_diff = pct_diff), 3))
print(diag(fit_jax$omega))
```

**Pass criteria** (all must hold):
1. `abs(pct_diff)` < 10% for `ka`, `cl`, and `v`.
2. `diag(fit_jax$omega)` has at least one entry above 0.05 (matches
   the saem omega magnitudes).
3. `fit_jax$sigma` is in the same neighborhood as the saem add.sd
   (within ~30%, not 2x larger).

If criterion 1 holds for `ka` and `v` but `cl` is still ~28% low,
the Python fix did not actually load -- repeat 6.2.

### 6.6 Report results

Append a new section to `JAXSAEM-PARAMETER-INVESTIGATION.md` titled
**"9. Post-fix verification (YYYY-MM-DD)"** with:

- The pytest output from 6.3.
- The R `devtools::test()` output from 6.4.
- The `pct_diff` and `diag(omega)` output from 6.5.
- A one-line verdict: **PASS** (all criteria met) or **FAIL** (which
  criterion, with values).

Commit it as `docs: post-fix verification of IIV collapse` on this
nlmixr2est branch. Do not modify the original investigation sections
(1-8); they are the historical record.

If verification passes, the fastsaem fix is ready for merge into
`claude/python-pk-solver-TXiOg`; if it fails, the failure section is
the new starting point for the next iteration.

## Likely first-time failure modes

- **`could not import 'jaxsaem.nlmixr_bridge'`** -- Python env not on
  reticulate's path. Call `useJaxsaem(python = "/abs/path/to/python")`
  once, or set `RETICULATE_PYTHON` before loading the package.

- **`UnsupportedFeatureError` from a model you expected to work** -- the
  v0 Python translator is strict about parameter names (must be
  lowercased `cl`, `v`, `ka`, `v1`, `q`, `v2`) and requires every theta
  to be log-normal mu-referenced. The R preflight uses the same table,
  so mismatches surface as a classed `jaxsaemUnsupportedError`.

- **NAMESPACE diff after `devtools::document()`** -- the in-tree
  NAMESPACE is hand-edited to match the roxygen tags in the new files.
  Regenerating should produce the same lines.

- **CL still ~28% biased after the fix** -- pip installed the wrong
  branch, or a stale wheel is being used. Re-run section 6.2 with
  `--no-cache-dir` and re-check the `assert 'b_init = '` probe.

## Supported subset (v0)

- linCmt() topology only, with one of: `(cl,v)`, `(ka,cl,v)`,
  `(cl,v1,q,v2)`, `(ka,cl,v1,q,v2)`.
- Diagonal Omega.
- Single additive residual error.
- Single dose at t = 0.
- No standard errors / covariance step (`covPresent = FALSE`).

Anything outside that subset hard-errors with a `jaxsaemUnsupportedError`
before the reticulate boundary.

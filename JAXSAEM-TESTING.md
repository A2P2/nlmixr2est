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

## Supported subset (v0)

- linCmt() topology only, with one of: `(cl,v)`, `(ka,cl,v)`,
  `(cl,v1,q,v2)`, `(ka,cl,v1,q,v2)`.
- Diagonal Omega.
- Single additive residual error.
- Single dose at t = 0.
- No standard errors / covariance step (`covPresent = FALSE`).

Anything outside that subset hard-errors with a `jaxsaemUnsupportedError`
before the reticulate boundary.

## Speed comparison: est = "saem" vs est = "jaxsaem"
##
## Run from the repo root:
##   Rscript benchmark_jaxsaem.R
##
## Requires:
##   - nlmixr2est built from this branch (jaxsaem-test-lib)
##   - jaxsaem-env312 Python venv with jaxsaem + JAX installed
## -------------------------------------------------------------------------

lib    <- "C:/Users/POGODAL2/AppData/Local/Temp/jaxsaem-test-lib"
python <- file.path(Sys.getenv("USERPROFILE"), "jaxsaem-env312/Scripts/python.exe")
python <- normalizePath(python, winslash = "/")

# Remove WindowsApps Python stub from PATH so reticulate loads the right DLL
.path <- strsplit(Sys.getenv("PATH"), ";")[[1]]
Sys.setenv(PATH = paste(.path[!grepl("WindowsApps", .path, ignore.case = TRUE)],
                        collapse = ";"))
Sys.setenv(RETICULATE_PYTHON = python)

.libPaths(c(lib, .libPaths()))
library(reticulate)
use_python(python, required = TRUE)
library(nlmixr2est)

# ---- models -----------------------------------------------------------------

one_cmt_oral <- function() {
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

two_cmt_oral <- function() {
  ini({
    tka  <- log(1.5); tcl  <- log(2);  tv1  <- log(20)
    tq   <- log(0.5); tv2  <- log(30)
    eta.ka ~ 0.1; eta.cl ~ 0.1; eta.v1 ~ 0.05
    eta.q  ~ 0.05; eta.v2 ~ 0.05
    add.sd <- 0.5
  })
  model({
    ka <- exp(tka + eta.ka)
    cl <- exp(tcl + eta.cl)
    v1 <- exp(tv1 + eta.v1)
    q  <- exp(tq  + eta.q)
    v2 <- exp(tv2 + eta.v2)
    linCmt() ~ add(add.sd)
  })
}

data <- nlmixr2data::theo_sd

# ---- benchmark helper -------------------------------------------------------

run_bench <- function(model, data, est, control, label, n_rep = 3L) {
  times <- numeric(n_rep)
  for (i in seq_len(n_rep)) {
    t0 <- proc.time()[["elapsed"]]
    suppressMessages(suppressWarnings(
      fit <- nlmixr2(model, data, est = est, control = control)
    ))
    times[i] <- proc.time()[["elapsed"]] - t0
  }
  converged <- if (est == "jaxsaem") isTRUE(fit$converged) else TRUE
  list(label = label, est = est, times = times,
       mean_s = mean(times), sd_s = sd(times),
       converged = converged)
}

# ---- configurations to compare ----------------------------------------------

scenarios <- list(
  list(
    label   = "1-cmt oral | nIter=200",
    model   = one_cmt_oral,
    saem_ctl = saemControl(nBurn = 100, nEm = 100, seed = 1L, print = 0L),
    jax_ctl  = jaxsaemControl(nIter = 200L, nBurn = 100L, seed = 1L)
  ),
  list(
    label   = "1-cmt oral | nIter=400",
    model   = one_cmt_oral,
    saem_ctl = saemControl(nBurn = 200, nEm = 200, seed = 1L, print = 0L),
    jax_ctl  = jaxsaemControl(nIter = 400L, nBurn = 200L, seed = 1L)
  ),
  list(
    label   = "2-cmt oral | nIter=200",
    model   = two_cmt_oral,
    saem_ctl = saemControl(nBurn = 100, nEm = 100, seed = 1L, print = 0L),
    jax_ctl  = jaxsaemControl(nIter = 200L, nBurn = 100L, seed = 1L)
  )
)

# ---- run --------------------------------------------------------------------

cat(sprintf("\n%-35s  %-8s  %6s  %6s  %6s  %s\n",
            "Scenario", "est", "mean_s", "sd_s", "speedup", "converged"))
cat(strrep("-", 80), "\n")

results <- list()
for (sc in scenarios) {
  r_saem <- run_bench(sc$model, data, "saem",    sc$saem_ctl, sc$label)
  r_jax  <- run_bench(sc$model, data, "jaxsaem", sc$jax_ctl,  sc$label)
  speedup <- r_saem$mean_s / r_jax$mean_s

  cat(sprintf("%-35s  %-8s  %6.1f  %6.2f  %6s  %s\n",
              sc$label, "saem",
              r_saem$mean_s, r_saem$sd_s, "", ""))
  cat(sprintf("%-35s  %-8s  %6.1f  %6.2f  %5.1fx  %s\n",
              "", "jaxsaem",
              r_jax$mean_s, r_jax$sd_s,
              speedup,
              if (r_jax$converged) "converged" else "NOT CONVERGED"))
  cat(strrep("-", 80), "\n")

  results[[length(results) + 1]] <- list(saem = r_saem, jax = r_jax,
                                          speedup = speedup)
}

# ---- parameter comparison (last scenario only) ------------------------------

cat("\nParameter comparison for last scenario (1-cmt oral, nIter=400):\n\n")

sc <- scenarios[[2]]
suppressMessages(suppressWarnings({
  fit_saem <- nlmixr2(sc$model, data, est = "saem",    control = sc$saem_ctl)
  fit_jax  <- nlmixr2(sc$model, data, est = "jaxsaem", control = sc$jax_ctl)
}))

theta_saem <- setNames(
  exp(fit_saem$theta[c("tka", "tcl", "tv")]),
  c("ka", "cl", "v")
)
theta_jax  <- fit_jax$theta

pct_diff <- 100 * (theta_jax - theta_saem) / theta_saem

cmp <- data.frame(
  param    = names(theta_saem),
  saem     = round(theta_saem, 4),
  jaxsaem  = round(theta_jax,  4),
  pct_diff = round(pct_diff,   2),
  row.names = NULL
)
print(cmp)

invisible(results)

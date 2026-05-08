## Speed comparison: est = "saem" vs est = "jaxsaem" -- N-subject scaling
##
## Tests N = 12 (theo_sd), 100, 1000 subjects.
## Simulates data from the same 1-cmt oral model for larger N.
##
## Run: Rscript benchmark_jaxsaem_nscale.R
## -------------------------------------------------------------------------

lib    <- "C:/Users/POGODAL2/AppData/Local/Temp/jaxsaem-test-lib"
python <- normalizePath(file.path(Sys.getenv("USERPROFILE"),
                                  "jaxsaem-env312/Scripts/python.exe"),
                        winslash = "/")
.path <- strsplit(Sys.getenv("PATH"), ";")[[1]]
Sys.setenv(PATH  = paste(.path[!grepl("WindowsApps", .path, ignore.case = TRUE)],
                         collapse = ";"))
Sys.setenv(RETICULATE_PYTHON = python)
.libPaths(c(lib, .libPaths()))
library(reticulate); use_python(python, required = TRUE)
library(nlmixr2est)

# ---- 1-cmt oral model -------------------------------------------------------

one_cmt <- function() {
  ini({
    tka <- log(1.5); tcl <- log(2); tv <- log(40)
    eta.ka ~ 0.25; eta.cl ~ 0.09; eta.v ~ 0.04
    add.sd <- 0.7
  })
  model({
    ka <- exp(tka + eta.ka)
    cl <- exp(tcl + eta.cl)
    v  <- exp(tv  + eta.v)
    linCmt() ~ add(add.sd)
  })
}

# ---- simulate N-subject dataset from theo_sd template ----------------------

sim_dataset <- function(N, seed = 42L) {
  set.seed(seed)
  base <- nlmixr2data::theo_sd          # 12 subjects x 11 obs (EVID=101 for dose)
  obs_times <- sort(unique(base$TIME[base$EVID == 0]))  # observation times
  obs_times <- obs_times[obs_times > 0]                 # drop t=0 obs
  dose <- mean(base$AMT[base$EVID > 0], na.rm = TRUE)  # ~320 mg (EVID=101)

  # Parameter draws
  ka_pop <- 1.5; cl_pop <- 2.0; v_pop <- 40.0
  ka_i <- ka_pop * exp(rnorm(N, 0, sqrt(0.25)))
  cl_i <- cl_pop * exp(rnorm(N, 0, sqrt(0.09)))
  v_i  <- v_pop  * exp(rnorm(N, 0, sqrt(0.04)))
  sigma <- 0.7

  rows <- vector("list", N * (length(obs_times) + 1))
  idx <- 1L
  for (i in seq_len(N)) {
    rows[[idx]] <- data.frame(ID = i, TIME = 0, AMT = dose, DV = 0,
                               EVID = 1, MDV = 1)
    idx <- idx + 1L
    ke   <- cl_i[i] / v_i[i]
    diff <- ka_i[i] - ke
    for (t in obs_times) {
      if (abs(diff) < 1e-6) {
        # Bateman limit: C(t) = dose*ka*t/V * exp(-ka*t)
        conc <- (dose * ka_i[i] * t / v_i[i]) * exp(-ka_i[i] * t)
      } else {
        conc <- (dose * ka_i[i]) / (v_i[i] * diff) *
                (exp(-ke * t) - exp(-ka_i[i] * t))
      }
      dv <- max(conc + rnorm(1, 0, sigma), 0.001)
      rows[[idx]] <- data.frame(ID = i, TIME = t, AMT = 0, DV = dv,
                                 EVID = 0, MDV = 0)
      idx <- idx + 1L
    }
  }
  do.call(rbind, rows)
}

# ---- benchmark helper -------------------------------------------------------

run_bench <- function(model, data, est, control, n_rep = 2L) {
  times <- numeric(n_rep)
  fit <- NULL
  err <- NULL
  for (i in seq_len(n_rep)) {
    t0 <- proc.time()[["elapsed"]]
    res <- tryCatch(
      suppressMessages(suppressWarnings(
        nlmixr2(model, data, est = est, control = control)
      )),
      error = function(e) e
    )
    times[i] <- proc.time()[["elapsed"]] - t0
    if (inherits(res, "error")) { err <- conditionMessage(res); break }
    fit <- res
  }
  if (!is.null(err))
    return(list(fit = NULL, times = times, mean_s = NA_real_, sd_s = NA_real_,
                converged = FALSE, error = err))
  converged <- if (est == "jaxsaem") isTRUE(fit$converged) else TRUE
  list(fit = fit, times = times, mean_s = mean(times), sd_s = sd(times),
       converged = converged, error = NULL)
}

# ---- datasets ---------------------------------------------------------------

N_vals <- c(12, 100, 1000)
datasets <- list(
  "12"   = nlmixr2data::theo_sd,
  "100"  = sim_dataset(100,  seed = 1L),
  "1000" = sim_dataset(1000, seed = 1L)
)

nIter <- 200L
nBurn <- 100L
saem_ctl  <- saemControl(nBurn = nBurn, nEm = nIter - nBurn,
                          seed = 1L, print = 0L)
jax_ctl   <- jaxsaemControl(nIter = nIter, nBurn = nBurn, seed = 1L)

# ---- run --------------------------------------------------------------------

cat(sprintf("\n%-6s  %-9s  %8s  %6s  %7s  %s\n",
            "N", "est", "mean_s", "sd_s", "speedup", "converged"))
cat(strrep("-", 60), "\n")

all_res <- list()
for (N in N_vals) {
  dat <- datasets[[as.character(N)]]
  r_saem <- run_bench(one_cmt, dat, "saem",    saem_ctl)
  r_jax  <- run_bench(one_cmt, dat, "jaxsaem", jax_ctl)
  speedup <- if (!is.na(r_saem$mean_s) && !is.na(r_jax$mean_s))
               r_saem$mean_s / r_jax$mean_s else NA_real_

  saem_label <- if (!is.null(r_saem$error))
    sprintf("ERROR: %s", substr(r_saem$error, 1, 50)) else ""
  cat(sprintf("%-6d  %-9s  %8s  %6s  %7s  %s\n",
              N, "saem",
              if (is.na(r_saem$mean_s)) "  OOM" else sprintf("%6.1f", r_saem$mean_s),
              if (is.na(r_saem$sd_s))  "  ---" else sprintf("%6.2f", r_saem$sd_s),
              "", saem_label))
  cat(sprintf("%-6s  %-9s  %8.1f  %6.2f  %s  %s\n",
              "", "jaxsaem", r_jax$mean_s, r_jax$sd_s,
              if (is.na(speedup)) "     N/A" else sprintf("%6.1fx", speedup),
              if (r_jax$converged) "converged" else "NOT CONVERGED"))
  cat(strrep("-", 60), "\n")

  # parameter comparison (skip if saem failed)
  if (!is.null(r_saem$fit) && !is.null(r_jax$fit)) {
    theta_saem <- setNames(exp(r_saem$fit$theta[c("tka","tcl","tv")]),
                           c("ka","cl","v"))
    theta_jax  <- r_jax$fit$theta[c("ka","cl","v")]
    pct_diff   <- round(100 * (theta_jax - theta_saem) / theta_saem, 1)
    cat(sprintf("  params saem: ka=%.3f cl=%.3f v=%.3f  add.sd=%.3f\n",
                theta_saem["ka"], theta_saem["cl"], theta_saem["v"],
                r_saem$fit$theta["add.sd"]))
    cat(sprintf("  params  jax: ka=%.3f cl=%.3f v=%.3f  sigma=%.3f\n",
                theta_jax["ka"],  theta_jax["cl"],  theta_jax["v"],
                r_jax$fit$sigma))
    cat(sprintf("  pct_diff:    ka=%+.1f%% cl=%+.1f%% v=%+.1f%%\n\n",
                pct_diff["ka"], pct_diff["cl"], pct_diff["v"]))
  } else if (!is.null(r_jax$fit)) {
    cat(sprintf("  jaxsaem: ka=%.3f cl=%.3f v=%.3f  sigma=%.3f  (saem OOM, no comparison)\n\n",
                r_jax$fit$theta["ka"], r_jax$fit$theta["cl"],
                r_jax$fit$theta["v"],  r_jax$fit$sigma))
  }

  all_res[[as.character(N)]] <- list(saem = r_saem, jax = r_jax,
                                      speedup = speedup)
}

cat("\n=== Summary: speedup by N ===\n")
cat(sprintf("%-6s  %8s  %8s  %7s\n", "N", "saem_s", "jax_s", "speedup"))
for (N in N_vals) {
  r <- all_res[[as.character(N)]]
  saem_s <- if (is.na(r$saem$mean_s)) "     OOM" else sprintf("%8.1f", r$saem$mean_s)
  sp_s   <- if (is.na(r$speedup))     "    N/A " else sprintf("%6.1fx", r$speedup)
  cat(sprintf("%-6d  %8s  %8.1f  %7s\n", N, saem_s, r$jax$mean_s, sp_s))
}

invisible(all_res)

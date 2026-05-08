## Speed comparison: est = "saem" vs est = "jaxsaem"
## All four supported linCmt() topologies x multiple N values
##
## Models:  1-cmt IV, 1-cmt oral, 2-cmt IV, 2-cmt oral
## N values: 12, 50, 100, 250, 500, 1000
## nIter = 200, nBurn = 100, n_rep = 2 per cell
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

# ============================================================
# Model definitions
# ============================================================

mdl_1cmt_iv <- function() {
  ini({
    tcl <- log(2); tv <- log(40)
    eta.cl ~ 0.09; eta.v ~ 0.04
    add.sd <- 0.7
  })
  model({
    cl <- exp(tcl + eta.cl)
    v  <- exp(tv  + eta.v)
    linCmt() ~ add(add.sd)
  })
}

mdl_1cmt_oral <- function() {
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

mdl_2cmt_iv <- function() {
  ini({
    tcl <- log(2);  tv1 <- log(20)
    tq  <- log(0.5); tv2 <- log(30)
    eta.cl ~ 0.09; eta.v1 ~ 0.04; eta.q ~ 0.04; eta.v2 ~ 0.04
    add.sd <- 0.7
  })
  model({
    cl <- exp(tcl + eta.cl)
    v1 <- exp(tv1 + eta.v1)
    q  <- exp(tq  + eta.q)
    v2 <- exp(tv2 + eta.v2)
    linCmt() ~ add(add.sd)
  })
}

mdl_2cmt_oral <- function() {
  ini({
    tka <- log(1.5); tcl <- log(2);  tv1 <- log(20)
    tq  <- log(0.5); tv2 <- log(30)
    eta.ka ~ 0.25; eta.cl ~ 0.09; eta.v1 ~ 0.04
    eta.q  ~ 0.04;  eta.v2 ~ 0.04
    add.sd <- 0.7
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

# ============================================================
# Analytic simulation helpers (match jaxsaem models.py exactly)
# ============================================================

OBS_TIMES <- c(0.25, 0.5, 1, 2, 4, 6, 8, 12, 18, 24)
DOSE      <- 320
SIGMA     <- 0.7

.conc_1cmt_iv <- function(t, dose, cl, v) {
  (dose / v) * exp(-(cl / v) * t)
}

.conc_1cmt_oral <- function(t, dose, ka, cl, v) {
  ke <- cl / v; d <- ka - ke
  if (abs(d) < 1e-6) (dose * ka * t / v) * exp(-ka * t)
  else (dose * ka) / (v * d) * (exp(-ke * t) - exp(-ka * t))
}

.conc_2cmt_iv <- function(t, dose, cl, v1, q, v2) {
  k <- cl/v1; k12 <- q/v1; k21 <- q/v2
  s <- k + k12 + k21
  disc  <- sqrt(max(s^2 - 4*k*k21, 0))
  alpha <- 0.5*(s+disc); beta <- 0.5*(s-disc)
  A <- (alpha-k21)/(v1*(alpha-beta)); B <- (k21-beta)/(v1*(alpha-beta))
  dose*(A*exp(-alpha*t) + B*exp(-beta*t))
}

.conc_2cmt_oral <- function(t, dose, ka, cl, v1, q, v2) {
  k <- cl/v1; k12 <- q/v1; k21 <- q/v2
  s <- k + k12 + k21
  disc  <- sqrt(max(s^2 - 4*k*k21, 0))
  alpha <- 0.5*(s+disc); beta <- 0.5*(s-disc)
  eps <- 1e-10
  da  <- if (abs(ka-alpha) < eps) eps else ka-alpha
  db  <- if (abs(ka-beta)  < eps) eps else ka-beta
  dab <- if (abs(alpha-beta) < eps) eps else alpha-beta
  A <- (k21-alpha)/(-da * dab)
  B <- (k21-beta) /(-db * (-dab))
  C <- (k21-ka)   /(da * db)
  (dose*ka/v1)*(A*exp(-alpha*t) + B*exp(-beta*t) + C*exp(-ka*t))
}

.add_noise <- function(conc) max(conc + rnorm(1, 0, SIGMA), 0.001)

sim_dataset <- function(topology, N, seed = 1L) {
  set.seed(seed)
  nobs <- length(OBS_TIMES)
  rows <- vector("list", N * (nobs + 1L))
  idx  <- 1L

  dose_row <- function(i)
    data.frame(ID=i, TIME=0, AMT=DOSE, DV=0, EVID=1, MDV=1)
  obs_row  <- function(i, t, dv)
    data.frame(ID=i, TIME=t, AMT=0, DV=dv, EVID=0, MDV=0)

  if (topology == "iv_1cmt") {
    cl_i <- 2.0 * exp(rnorm(N, 0, sqrt(0.09)))
    v_i  <- 40  * exp(rnorm(N, 0, sqrt(0.04)))
    for (i in seq_len(N)) {
      rows[[idx]] <- dose_row(i); idx <- idx + 1L
      for (t in OBS_TIMES) {
        rows[[idx]] <- obs_row(i, t, .add_noise(.conc_1cmt_iv(t,DOSE,cl_i[i],v_i[i])))
        idx <- idx + 1L
      }
    }

  } else if (topology == "oral_1cmt") {
    ka_i <- 1.5 * exp(rnorm(N, 0, sqrt(0.25)))
    cl_i <- 2.0 * exp(rnorm(N, 0, sqrt(0.09)))
    v_i  <- 40  * exp(rnorm(N, 0, sqrt(0.04)))
    for (i in seq_len(N)) {
      rows[[idx]] <- dose_row(i); idx <- idx + 1L
      for (t in OBS_TIMES) {
        rows[[idx]] <- obs_row(i, t, .add_noise(.conc_1cmt_oral(t,DOSE,ka_i[i],cl_i[i],v_i[i])))
        idx <- idx + 1L
      }
    }

  } else if (topology == "iv_2cmt") {
    cl_i <- 2.0 * exp(rnorm(N, 0, sqrt(0.09)))
    v1_i <- 20  * exp(rnorm(N, 0, sqrt(0.04)))
    q_i  <- 0.5 * exp(rnorm(N, 0, sqrt(0.04)))
    v2_i <- 30  * exp(rnorm(N, 0, sqrt(0.04)))
    for (i in seq_len(N)) {
      rows[[idx]] <- dose_row(i); idx <- idx + 1L
      for (t in OBS_TIMES) {
        rows[[idx]] <- obs_row(i, t, .add_noise(.conc_2cmt_iv(t,DOSE,cl_i[i],v1_i[i],q_i[i],v2_i[i])))
        idx <- idx + 1L
      }
    }

  } else {  # oral_2cmt
    ka_i <- 1.5 * exp(rnorm(N, 0, sqrt(0.25)))
    cl_i <- 2.0 * exp(rnorm(N, 0, sqrt(0.09)))
    v1_i <- 20  * exp(rnorm(N, 0, sqrt(0.04)))
    q_i  <- 0.5 * exp(rnorm(N, 0, sqrt(0.04)))
    v2_i <- 30  * exp(rnorm(N, 0, sqrt(0.04)))
    for (i in seq_len(N)) {
      rows[[idx]] <- dose_row(i); idx <- idx + 1L
      for (t in OBS_TIMES) {
        rows[[idx]] <- obs_row(i, t, .add_noise(.conc_2cmt_oral(t,DOSE,ka_i[i],cl_i[i],v1_i[i],q_i[i],v2_i[i])))
        idx <- idx + 1L
      }
    }
  }
  do.call(rbind, rows)
}

# ============================================================
# Benchmark helper
# ============================================================

run_bench <- function(model, data, est, control, n_rep = 2L) {
  fit <- NULL; err <- NULL; times <- numeric(n_rep)
  for (i in seq_len(n_rep)) {
    t0  <- proc.time()[["elapsed"]]
    res <- tryCatch(
      suppressMessages(suppressWarnings(nlmixr2(model, data, est = est, control = control))),
      error = function(e) e
    )
    times[i] <- proc.time()[["elapsed"]] - t0
    if (inherits(res, "error")) { err <- conditionMessage(res); break }
    fit <- res
  }
  if (!is.null(err))
    return(list(fit = NULL, times = times, mean_s = NA_real_, sd_s = NA_real_,
                converged = FALSE, error = err))
  list(fit = fit, times = times, mean_s = mean(times), sd_s = sd(times),
       converged = if (est == "jaxsaem") isTRUE(fit$converged) else TRUE,
       error = NULL)
}

fmt_s <- function(x) if (is.na(x)) "    OOM" else sprintf("%7.1f", x)
fmt_x <- function(x) if (is.na(x)) "   N/A" else sprintf("%5.1fx", x)

# ============================================================
# Scenario definitions
# ============================================================

N_VALS   <- c(12, 50, 100, 250, 500, 1000)
NITER    <- 200L; NBURN <- 100L
SAEM_CTL <- saemControl(nBurn = NBURN, nEm = NITER - NBURN, seed = 1L, print = 0L)
JAX_CTL  <- jaxsaemControl(nIter = NITER, nBurn = NBURN, seed = 1L)

scenarios <- list(
  list(id = "1cmt_iv",   label = "1-cmt IV",   topo = "iv_1cmt",
       model = mdl_1cmt_iv,
       saem_names = c("tcl","tv"),          jax_names = c("cl","v")),
  list(id = "1cmt_oral", label = "1-cmt oral", topo = "oral_1cmt",
       model = mdl_1cmt_oral,
       saem_names = c("tka","tcl","tv"),    jax_names = c("ka","cl","v")),
  list(id = "2cmt_iv",   label = "2-cmt IV",   topo = "iv_2cmt",
       model = mdl_2cmt_iv,
       saem_names = c("tcl","tv1","tq","tv2"), jax_names = c("cl","v1","q","v2")),
  list(id = "2cmt_oral", label = "2-cmt oral", topo = "oral_2cmt",
       model = mdl_2cmt_oral,
       saem_names = c("tka","tcl","tv1","tq","tv2"), jax_names = c("ka","cl","v1","q","v2"))
)

# ============================================================
# Pre-simulate all datasets
# ============================================================

cat("Simulating datasets...\n")
datasets <- list()
for (sc in scenarios) {
  datasets[[sc$id]] <- lapply(
    setNames(N_VALS, as.character(N_VALS)),
    function(n) sim_dataset(sc$topo, n, seed = 1L)
  )
}
cat("Done. Starting benchmark.\n\n")

# ============================================================
# Run all (model x N) cells
# ============================================================

all_res <- list()

for (sc in scenarios) {
  cat(sprintf("══════════════════════════════════════════════════════════════\n"))
  cat(sprintf("Model: %-10s  (nIter=%d, nBurn=%d, n_rep=2)\n",
              sc$label, NITER, NBURN))
  cat(sprintf("══════════════════════════════════════════════════════════════\n"))
  cat(sprintf("%-6s  %-9s  %7s  %6s  %6s  %s\n",
              "N", "est", "mean_s", "sd_s", "speed", "status"))
  cat(strrep("-", 64), "\n")

  sc_res <- list()
  for (N in N_VALS) {
    dat    <- datasets[[sc$id]][[as.character(N)]]
    r_saem <- run_bench(sc$model, dat, "saem",    SAEM_CTL)
    r_jax  <- run_bench(sc$model, dat, "jaxsaem", JAX_CTL)
    speedup <- if (!is.na(r_saem$mean_s) && !is.na(r_jax$mean_s))
                 r_saem$mean_s / r_jax$mean_s else NA_real_

    saem_status <- if (!is.null(r_saem$error))
      paste0("ERROR: ", substr(r_saem$error, 1, 35)) else "ok"
    jax_status  <- if (r_jax$converged) "converged" else "NOT CONVERGED"

    cat(sprintf("%-6d  %-9s  %7s  %6s  %6s  %s\n",
                N, "saem",
                fmt_s(r_saem$mean_s),
                if (is.na(r_saem$sd_s)) "   ---" else sprintf("%6.2f", r_saem$sd_s),
                "", saem_status))
    cat(sprintf("%-6s  %-9s  %7s  %6.2f  %6s  %s\n",
                "", "jaxsaem", fmt_s(r_jax$mean_s), r_jax$sd_s,
                fmt_x(speedup), jax_status))

    if (!is.null(r_saem$fit) && !is.null(r_jax$fit)) {
      th_s <- exp(r_saem$fit$theta[sc$saem_names])
      th_j <- r_jax$fit$theta[sc$jax_names]
      pd   <- round(100 * (th_j - th_s) / th_s, 1)
      cat(sprintf("         pct_diff: %s\n",
                  paste(sprintf("%s:%+.1f%%", sc$jax_names, pd), collapse = "  ")))
    } else if (!is.null(r_jax$fit)) {
      th_j <- r_jax$fit$theta[sc$jax_names]
      cat(sprintf("         jaxsaem (saem OOM): %s  sigma=%.3f\n",
                  paste(sprintf("%s=%.3f", sc$jax_names, th_j), collapse = "  "),
                  r_jax$fit$sigma))
    }
    cat(strrep("-", 64), "\n")

    sc_res[[as.character(N)]] <- list(saem = r_saem, jax = r_jax, speedup = speedup)
  }
  all_res[[sc$id]] <- sc_res
  cat("\n")
}

# ============================================================
# Summary tables
# ============================================================

W <- 10  # column width per N value
pad <- sprintf("%-14s", "")

# Speedup table
cat(strrep("=", 14 + W * length(N_VALS)), "\n")
cat("SPEEDUP (saem / jaxsaem mean_s)\n")
cat(strrep("=", 14 + W * length(N_VALS)), "\n")
cat(sprintf("%-14s", "N →"))
for (N in N_VALS) cat(sprintf("%*d", W, N))
cat("\n", strrep("-", 14 + W * length(N_VALS)), "\n", sep = "")
for (sc in scenarios) {
  cat(sprintf("%-14s", sc$label))
  for (N in N_VALS)
    cat(sprintf("%*s", W, fmt_x(all_res[[sc$id]][[as.character(N)]]$speedup)))
  cat("\n")
}
cat(strrep("-", 14 + W * length(N_VALS)), "\n\n")

# jaxsaem absolute time table
cat(strrep("=", 14 + W * length(N_VALS)), "\n")
cat("jaxsaem time (s)\n")
cat(strrep("=", 14 + W * length(N_VALS)), "\n")
cat(sprintf("%-14s", "N →"))
for (N in N_VALS) cat(sprintf("%*d", W, N))
cat("\n", strrep("-", 14 + W * length(N_VALS)), "\n", sep = "")
for (sc in scenarios) {
  cat(sprintf("%-14s", sc$label))
  for (N in N_VALS)
    cat(sprintf("%*s", W, fmt_s(all_res[[sc$id]][[as.character(N)]]$jax$mean_s)))
  cat("\n")
}
cat(strrep("-", 14 + W * length(N_VALS)), "\n\n")

# saem absolute time table
cat(strrep("=", 14 + W * length(N_VALS)), "\n")
cat("saem time (s)  [OOM = memory allocation failure]\n")
cat(strrep("=", 14 + W * length(N_VALS)), "\n")
cat(sprintf("%-14s", "N →"))
for (N in N_VALS) cat(sprintf("%*d", W, N))
cat("\n", strrep("-", 14 + W * length(N_VALS)), "\n", sep = "")
for (sc in scenarios) {
  cat(sprintf("%-14s", sc$label))
  for (N in N_VALS)
    cat(sprintf("%*s", W, fmt_s(all_res[[sc$id]][[as.character(N)]]$saem$mean_s)))
  cat("\n")
}
cat(strrep("-", 14 + W * length(N_VALS)), "\n")

invisible(all_res)

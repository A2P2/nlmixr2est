## Tests for est = "jaxsaem". The whole file is skipped when the Python
## side is unavailable, so contributors without a JAX environment do not
## see false failures.

skip_if_not_installed("reticulate")
skip_if_not(reticulate::py_module_available("jaxsaem"))

test_that("useJaxsaem imports the Python module", {
  expect_silent(useJaxsaem())
})

test_that("jaxsaemControl validates and structures its arguments", {
  .c <- jaxsaemControl()
  expect_s3_class(.c, "jaxsaemControl")
  expect_identical(.c$method, "saem")

  .c2 <- jaxsaemControl(method = "foce", nIter = 100, nBurn = 50,
                        seed = 7, foceMaxiter = 50)
  expect_identical(.c2$method, "foce")
  expect_identical(.c2$nIter, 100L)

  expect_error(jaxsaemControl(nIter = -1), "nIter")
  expect_error(jaxsaemControl(method = "nope"))
  expect_error(jaxsaemControl(unknownArg = TRUE), "unused argument")
})

test_that("est = 'jaxsaem' fits a 1-cmt oral linCmt model", {
  one.cmt <- function() {
    ini({
      tka <- log(1.5)
      tcl <- log(2.0)
      tv  <- log(40.0)
      eta.ka ~ 0.1
      eta.cl ~ 0.1
      eta.v  ~ 0.05
      add.sd <- 0.5
    })
    model({
      ka <- exp(tka + eta.ka)
      cl <- exp(tcl + eta.cl)
      v  <- exp(tv  + eta.v)
      linCmt() ~ add(add.sd)
    })
  }

  .dat <- nlmixr2data::theo_sd

  .fit <- nlmixr2(one.cmt, .dat, est = "jaxsaem",
                  control = jaxsaemControl(nIter = 50, nBurn = 25, seed = 1))
  expect_s3_class(.fit, "jaxsaemFit")
  expect_true(all(c("ka", "cl", "v") %in% names(.fit$theta)))
  expect_true(all(.fit$theta > 0))
  expect_identical(dim(.fit$omega), c(3L, 3L))
})

test_that("est = 'jaxsaem' recovers non-collapsed Omega (regression)", {
  ## Regression test for the iteration-0 IIV-collapse bug fixed in
  ## fastsaem (see JAXSAEM-PARAMETER-INVESTIGATION.md). Pre-fix, every
  ## diag(omega) entry would land near the 1e-6 floor regardless of
  ## init_omega2 or n_iter, and the fit would mask the missing IIV by
  ## absorbing it into the residual sigma. With a non-degenerate true
  ## Omega in the data (theo_sd) and a reasonable iter count, at least
  ## one diag(omega) entry should be well above the floor.

  one.cmt <- function() {
    ini({
      tka <- log(1.5)
      tcl <- log(2.0)
      tv  <- log(40.0)
      eta.ka ~ 0.1
      eta.cl ~ 0.1
      eta.v  ~ 0.05
      add.sd <- 0.5
    })
    model({
      ka <- exp(tka + eta.ka)
      cl <- exp(tcl + eta.cl)
      v  <- exp(tv  + eta.v)
      linCmt() ~ add(add.sd)
    })
  }

  .fit <- nlmixr2(one.cmt, nlmixr2data::theo_sd, est = "jaxsaem",
                  control = jaxsaemControl(nIter = 200, nBurn = 100,
                                           seed = 1))
  .omegaDiag <- diag(.fit$omega)
  expect_true(all(.omegaDiag > 1e-3),
              info = paste0("diag(omega) collapsed toward floor: ",
                            paste(signif(.omegaDiag, 3), collapse = ", ")))
  expect_true(any(.omegaDiag > 0.05),
              info = paste0("no diag(omega) entry > 0.05; got ",
                            paste(signif(.omegaDiag, 3), collapse = ", ")))
})

test_that("non-additive residual errors out before the Python boundary", {
  bad.model <- function() {
    ini({
      tka <- log(1.5)
      tcl <- log(2.0)
      tv  <- log(40.0)
      eta.ka ~ 0.1
      eta.cl ~ 0.1
      eta.v  ~ 0.05
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

  expect_error(
    nlmixr2(bad.model, nlmixr2data::theo_sd, est = "jaxsaem"),
    class = "jaxsaemUnsupportedError"
  )
})

test_that("unsupported parameter sets error out before the Python boundary", {
  weird.model <- function() {
    ini({
      tka <- log(1.5)
      tcl <- log(2.0)
      tv  <- log(40.0)
      tlag <- log(0.5)
      eta.ka  ~ 0.1
      eta.cl  ~ 0.1
      eta.v   ~ 0.05
      eta.lag ~ 0.05
      add.sd <- 0.5
    })
    model({
      ka  <- exp(tka  + eta.ka)
      cl  <- exp(tcl  + eta.cl)
      v   <- exp(tv   + eta.v)
      lag <- exp(tlag + eta.lag)
      linCmt() ~ add(add.sd)
    })
  }

  expect_error(
    nlmixr2(weird.model, nlmixr2data::theo_sd, est = "jaxsaem"),
    class = "jaxsaemUnsupportedError"
  )
})

test_that("print.jaxsaemFit emits the expected header", {
  .fake <- list(
    theta = c(ka = 1, cl = 2, v = 30),
    omega = diag(c(0.1, 0.1, 0.05)),
    sigma = c(add_err = 0.5),
    objf = NA_real_,
    converged = TRUE,
    method = "saem",
    time = list(total = 0.42, fit = 0.42)
  )
  class(.fake) <- c("nlmixr2FitData", "jaxsaemFit", "list")
  expect_output(print(.fake), "est = 'jaxsaem'")
  expect_output(print(.fake), "inner method:")
  expect_output(print(.fake), "OBJF not calculated for SAEM")
})

#' Control Options for the JAX-backed SAEM/FOCE method
#'
#' Configure a fit dispatched to the Python `jaxsaem` package via
#' `reticulate`. The Python side exposes both stochastic SAEM and
#' deterministic FOCE-Laplace algorithms; pick one with `method`. The
#' v0 bridge supports a fixed subset of population PK models (the four
#' linCmt() topologies with a log-normal mu-referenced theta and a
#' diagonal Omega); models outside that subset will error before the
#' Python boundary.
#'
#' @param method Either `"saem"` (default; stochastic, faster
#'   exploration) or `"foce"` (deterministic Laplace approximation).
#'
#' @param nIter Number of SAEM iterations (ignored when
#'   `method = "foce"`). Default 400.
#'
#' @param nBurn Number of burn-in iterations before averaging in SAEM
#'   (ignored when `method = "foce"`). Default 200.
#'
#' @param seed RNG seed forwarded to the JAX side (threefry). Default 1.
#'
#' @param foceMaxiter Outer-loop iteration cap for FOCE-Laplace. Only
#'   used when `method = "foce"`. Default 200.
#'
#' @param python Optional path to a Python interpreter. When non-`NULL`,
#'   `reticulate::use_python(python, required = TRUE)` is called once at
#'   the start of the fit; equivalent to a manual call to
#'   [useJaxsaem()] beforehand.
#'
#' @param covMethod Covariance method. The v0 jaxsaem bridge does not
#'   produce standard errors, so only `""` (no covariance) is accepted.
#'   The argument is kept for API parity with [saemControl()].
#'
#' @param print The number of iterations between progress prints.
#'   Default 1.
#'
#' @param compress Logical; passed through to downstream nlmixr2 output
#'   handling for parity with other control objects. Default `TRUE`.
#'
#' @param ... Reserved for future control options. Unknown names raise
#'   an error.
#'
#' @return A list of class `"jaxsaemControl"` for use as the `control`
#'   argument of [nlmixr2()] when `est = "jaxsaem"`.
#'
#' @author Matthew L. Fidler
#' @family Estimation control
#' @seealso [useJaxsaem()], [saemControl()]
#' @export
jaxsaemControl <- function(method = c("saem", "foce"),
                           nIter = 400L,
                           nBurn = 200L,
                           seed = 1L,
                           foceMaxiter = 200L,
                           python = NULL,
                           covMethod = "",
                           print = 1L,
                           compress = TRUE,
                           ...) {
  .xtra <- list(...)
  .bad <- names(.xtra)
  if (length(.bad) > 0) {
    stop("unused argument: ", paste(paste0("'", .bad, "'"), collapse = ", "),
         call. = FALSE)
  }

  method <- match.arg(method)
  checkmate::assertIntegerish(nIter, any.missing = FALSE, len = 1, lower = 1)
  checkmate::assertIntegerish(nBurn, any.missing = FALSE, len = 1, lower = 0)
  checkmate::assertIntegerish(seed, any.missing = FALSE, len = 1)
  checkmate::assertIntegerish(foceMaxiter, any.missing = FALSE, len = 1, lower = 1)
  checkmate::assertIntegerish(print, any.missing = FALSE, len = 1, lower = 0)
  checkmate::assertLogical(compress, any.missing = FALSE, len = 1)
  if (!is.null(python)) {
    checkmate::assertString(python, min.chars = 1)
  }
  covMethod <- match.arg(covMethod, choices = "")

  .ret <- list(
    method = method,
    nIter = as.integer(nIter),
    nBurn = as.integer(nBurn),
    seed = as.integer(seed),
    foceMaxiter = as.integer(foceMaxiter),
    python = python,
    covMethod = covMethod,
    print = as.integer(print),
    compress = compress
  )
  class(.ret) <- "jaxsaemControl"
  .ret
}

#' @rdname nmObjHandleControlObject
#' @export
nmObjHandleControlObject.jaxsaemControl <- function(control, env) {
  assign("jaxsaemControl", control, envir = env)
}

#' @rdname getValidNlmixrCtl
#' @export
getValidNlmixrCtl.jaxsaem <- function(control) {
  .ctl <- control[[1]]
  if (is.null(.ctl)) .ctl <- jaxsaemControl()
  if (is.null(attr(.ctl, "class")) && is(.ctl, "list")) {
    .ctl <- do.call("jaxsaemControl", .ctl)
  }
  if (!inherits(.ctl, "jaxsaemControl")) {
    .minfo("invalid control for `est=\"jaxsaem\"`, using default")
    .ctl <- jaxsaemControl()
  } else {
    .ctl <- do.call(jaxsaemControl, .ctl)
  }
  .ctl
}

#' @rdname nmObjGetControl
#' @export
nmObjGetControl.jaxsaem <- function(x, ...) {
  .env <- x[[1]]
  if (exists("jaxsaemControl", envir = .env)) {
    .control <- get("jaxsaemControl", envir = .env)
    if (inherits(.control, "jaxsaemControl")) return(.control)
  }
  if (exists("control", envir = .env)) {
    .control <- get("control", envir = .env)
    if (inherits(.control, "jaxsaemControl")) return(.control)
  }
  stop("cannot find a 'jaxsaemControl' object", call. = FALSE)
}

#' @rdname rxUiDeparse
#' @export
rxUiDeparse.jaxsaemControl <- function(object, var) {
  .default <- jaxsaemControl()
  .w <- .deparseDifferent(.default, object, "python")
  .deparseFinal(.default, object, .w, var)
}

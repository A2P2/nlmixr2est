## ---------------------------------------------------------------------------
## est = "jaxsaem": bridge to the Python `jaxsaem` package via reticulate.
##
## The fitting algorithm itself lives in Python/JAX (see
## https://github.com/a2p2/fastsaem). This file is the R-side glue: it
## validates a model + data against the v0 supported subset, flattens the
## rxUi to a plain spec dict, calls run_jaxsaem(), and wraps the result.
## ---------------------------------------------------------------------------

#' Configure reticulate for the jaxsaem bridge
#'
#' One-time helper that points reticulate at a Python interpreter and
#' imports the `jaxsaem` package, surfacing import errors immediately
#' rather than at fit time. Calling this before [nlmixr2()] is optional
#' if reticulate is already configured.
#'
#' @param python Optional path to a Python interpreter. When non-`NULL`,
#'   passed to [reticulate::use_python()] with `required = TRUE`.
#'
#' @return Invisibly returns the imported `jaxsaem` Python module.
#'
#' @author Matthew L. Fidler
#' @seealso [jaxsaemControl()]
#' @export
useJaxsaem <- function(python = NULL) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("est = 'jaxsaem' requires the 'reticulate' package; ",
         "install it with install.packages('reticulate')",
         call. = FALSE)
  }
  if (!is.null(python)) {
    reticulate::use_python(python, required = TRUE)
  }
  .js <- tryCatch(reticulate::import("jaxsaem", convert = FALSE),
                  error = function(e) {
                    stop("could not import the 'jaxsaem' Python package: ",
                         conditionMessage(e),
                         "\n  install it on the active Python with: ",
                         "pip install jaxsaem",
                         call. = FALSE)
                  })
  invisible(.js)
}

#' @rdname nlmixr2Est
#' @export
nlmixr2Est.jaxsaem <- function(env, ...) {
  .ui <- env$ui
  rxode2::assertRxUiTransformNormal(.ui, " for the estimation routine 'jaxsaem'",
                                    .var.name = .ui$modelName)
  rxode2::assertRxUiMixedOnly(.ui, " for the estimation routine 'jaxsaem'",
                              .var.name = .ui$modelName)
  rxode2::warnRxBounded(.ui, " which are ignored in 'jaxsaem'",
                        .var.name = .ui$modelName)
  .jaxsaemFamilyControl(env, ...)
  on.exit({
    if (exists("control", envir = .ui)) {
      rm("control", envir = .ui)
    }
  }, add = TRUE)
  .jaxsaemFamilyFit(env, ...)
}
attr(nlmixr2Est.jaxsaem, "covPresent") <- FALSE
attr(nlmixr2Est.jaxsaem, "unbounded") <- TRUE
attr(nlmixr2Est.jaxsaem, "mu") <- TRUE
attr(nlmixr2Est.jaxsaem, "iov") <- FALSE

#' Get the jaxsaem control statement and install it into the ui
#'
#' @param env Environment with `ui` in it
#' @param ... Other arguments
#' @return Nothing, called for side effects
#' @author Matthew L. Fidler
#' @noRd
.jaxsaemFamilyControl <- function(env, ...) {
  .ui <- env$ui
  .control <- env$control
  if (is.null(.control)) {
    .control <- jaxsaemControl()
  }
  if (!inherits(.control, "jaxsaemControl")) {
    .control <- do.call(nlmixr2est::jaxsaemControl, .control)
  }
  assign("control", .control, envir = .ui)
}

#' Fit the jaxsaem family of models
#'
#' @param env Environment from `nlmixr2Est()`
#' @param ... Other arguments
#' @return `nlmixr2FitData` wrapping the Python-side fit
#' @author Matthew L. Fidler
#' @noRd
.jaxsaemFamilyFit <- function(env, ...) {
  .ui <- env$ui
  .control <- .ui$control
  .data <- env$data

  if (!is.null(.control$python)) {
    useJaxsaem(.control$python)
  }

  .spec <- .rxUiToSpec(.ui)
  .pdata <- .nlmixrDataToDict(.data)
  .bridge <- .jaxsaemBridge()

  .t0 <- proc.time()
  .fitDictPy <- tryCatch(
    .bridge$run_jaxsaem(
      spec         = reticulate::r_to_py(.spec),
      data         = reticulate::r_to_py(.pdata),
      method       = .control$method,
      n_iter       = .control$nIter,
      n_burn       = .control$nBurn,
      seed         = .control$seed,
      foce_maxiter = .control$foceMaxiter
    ),
    error = function(e) {
      .msg <- conditionMessage(e)
      if (grepl("UnsupportedFeatureError", .msg, fixed = TRUE)) {
        .clean <- sub(".*UnsupportedFeatureError:[[:space:]]*", "", .msg)
        .stopUnsupported(.clean)
      }
      stop("jaxsaem (Python) failed: ", .msg, call. = FALSE)
    }
  )
  .fitDict <- reticulate::py_to_r(.fitDictPy)
  .elapsed <- (proc.time() - .t0)[["elapsed"]]

  .wrapAsNlmixrFit(.fitDict, env, .ui, .control, .elapsed)
}

#' Translate an rxUi to the Python `jaxsaem` spec dict
#'
#' Mirrors the schema in `jaxsaem/nlmixr_bridge.py` and the
#' linCmt() topology table in `jaxsaem/rxui_translator.py`. Performs
#' R-side preflight so the user gets an R-language error before the
#' reticulate boundary.
#'
#' @param ui rxode2 UI object
#' @return Plain list ready for [reticulate::r_to_py()]
#' @noRd
.rxUiToSpec <- function(ui) {
  .iniDf <- ui$iniDf
  .predDf <- ui$predDf

  .thetaRows <- !is.na(.iniDf$ntheta)
  .etaRows <- !is.na(.iniDf$neta1)
  .resRows <- !is.na(.iniDf$err)

  .thetaParams <- .iniDf$name[.thetaRows & !.resRows]
  if (length(.thetaParams) == 0L) {
    stop("no population (theta) parameters found in the model",
         call. = FALSE)
  }

  .muRef <- ui$muRefDataFrame
  if (is.null(.muRef) || !all(.thetaParams %in% .muRef$theta)) {
    .stopUnsupported(
      "every theta must be log-normal mu-referenced for est = 'jaxsaem'; ",
      "express each parameter as exp(theta + eta) in the model block"
    )
  }
  .userParams <- tolower(vapply(.thetaParams,
                                function(.t) .muRef$eta[.muRef$theta == .t][1L],
                                character(1L)))
  .userParams <- sub("^eta\\.", "", .userParams)

  .topology <- .detectLinCmtTopology(.userParams)
  if (is.null(.topology)) {
    .stopUnsupported(
      "linCmt() parameter set ", .deparseList(.userParams),
      " is not supported by est = 'jaxsaem'; v0 supports ",
      "(cl,v), (ka,cl,v), (cl,v1,q,v2), (ka,cl,v1,q,v2)"
    )
  }

  .etaInfo <- .iniDf[.etaRows, , drop = FALSE]
  .diagOnly <- all(.etaInfo$neta1 == .etaInfo$neta2)
  if (!.diagOnly) {
    .stopUnsupported(
      "est = 'jaxsaem' v0 only supports a diagonal Omega; ",
      "off-diagonal correlations were declared in ini({...})"
    )
  }

  .ini <- list()
  for (.i in seq_along(.thetaParams)) {
    .tName <- .thetaParams[.i]
    .uName <- .userParams[.i]
    .thetaVal <- .iniDf$est[.iniDf$name == .tName][1L]
    .etaName <- .muRef$eta[.muRef$theta == .tName][1L]
    .omegaVal <- 0
    .erows <- which(.etaInfo$name == .etaName)
    if (length(.erows) >= 1L) {
      .omegaVal <- .etaInfo$est[.erows[1L]]
    }
    .ini[[.uName]] <- list(theta = exp(.thetaVal), omega = .omegaVal)
  }

  .residual <- .extractResidual(.iniDf, .predDf)

  list(
    kind = "lincmt",
    lincmt_topology = .topology,
    params = as.list(.userParams),
    ini = .ini,
    residual = .residual,
    jax_model = NULL
  )
}

#' Detect the linCmt() topology from a parameter name set
#'
#' Reused verbatim from `jaxsaem/rxui_translator.py` so the Python and R
#' sides agree on which parameter sets are supported.
#'
#' @param paramNames Character vector of (lowercased) user parameter
#'   names
#' @return One of `"iv_1cmt"`, `"oral_1cmt"`, `"iv_2cmt"`,
#'   `"oral_2cmt"`, or `NULL`
#' @noRd
.detectLinCmtTopology <- function(paramNames) {
  .s <- sort(paramNames)
  if (identical(.s, sort(c("cl", "v")))) return("iv_1cmt")
  if (identical(.s, sort(c("ka", "cl", "v")))) return("oral_1cmt")
  if (identical(.s, sort(c("cl", "v1", "q", "v2")))) return("iv_2cmt")
  if (identical(.s, sort(c("ka", "cl", "v1", "q", "v2")))) return("oral_2cmt")
  NULL
}

#' Extract the residual specification from iniDf and predDf
#'
#' v0 supports a single additive residual only; combined or
#' proportional models error out cleanly.
#'
#' @param iniDf `ui$iniDf` data.frame
#' @param predDf `ui$predDf` data.frame
#' @return Plain list `list(kind = "add", value = <numeric>)`
#' @noRd
.extractResidual <- function(iniDf, predDf) {
  if (is.null(predDf) || nrow(predDf) == 0L) {
    .stopUnsupported("model has no residual error specification")
  }
  if (nrow(predDf) > 1L) {
    .stopUnsupported(
      "est = 'jaxsaem' v0 only supports a single endpoint; ",
      nrow(predDf), " endpoints were declared"
    )
  }
  .resRows <- iniDf[!is.na(iniDf$err), , drop = FALSE]
  if (nrow(.resRows) == 0L) {
    .stopUnsupported("model has no residual error parameter")
  }
  .errs <- as.character(.resRows$err)
  .addLike <- c("add", "norm", "dnorm", "lnorm", "dlnorm", "logn", "dlogn")
  .propLike <- c("prop", "propT", "pow", "powT", "pow2", "powT2")
  .isAdd <- .errs %in% .addLike
  .isProp <- .errs %in% .propLike
  if (any(.isProp)) {
    .stopUnsupported(
      "residual kind 'prop' is declared but not implemented in ",
      "est = 'jaxsaem' v0; use est = 'saem' for combined or ",
      "proportional residuals"
    )
  }
  if (!all(.isAdd) || nrow(.resRows) > 1L) {
    .stopUnsupported(
      "residual kind ", .deparseList(.errs),
      " is not supported by est = 'jaxsaem' v0; only a single ",
      "additive residual is supported"
    )
  }
  list(kind = "add", value = .resRows$est[1L])
}

#' Convert a long-format nlmixr data.frame to the Python data dict
#'
#' Lower-cases column names; required columns are id/time/dv/evid/amt
#' (case-insensitive). `mdv` is optional.
#'
#' @param df Data frame
#' @return Plain list ready for [reticulate::r_to_py()]
#' @noRd
.nlmixrDataToDict <- function(df) {
  checkmate::assertDataFrame(df, min.rows = 1)
  .lower <- tolower(names(df))
  .required <- c("id", "time", "dv", "evid", "amt")
  .miss <- setdiff(.required, .lower)
  if (length(.miss) > 0L) {
    stop("dataset missing required columns for est = 'jaxsaem': ",
         paste(.miss, collapse = ", "), call. = FALSE)
  }
  .pick <- function(.col) df[[which(.lower == .col)[1L]]]
  .out <- list(
    id = .pick("id"),
    time = as.numeric(.pick("time")),
    dv = as.numeric(.pick("dv")),
    evid = as.integer(.pick("evid")),
    amt = as.numeric(.pick("amt"))
  )
  if ("mdv" %in% .lower) {
    .out$mdv <- as.integer(.pick("mdv"))
  }
  .out
}

#' Cached importer for the Python bridge module
#'
#' @return Reference to `jaxsaem.nlmixr_bridge` (convert=FALSE)
#' @noRd
.jaxsaemBridge <- local({
  .cache <- NULL
  function() {
    if (!is.null(.cache)) return(.cache)
    if (!requireNamespace("reticulate", quietly = TRUE)) {
      stop("est = 'jaxsaem' requires the 'reticulate' package; ",
           "install it with install.packages('reticulate')",
           call. = FALSE)
    }
    .cache <<- tryCatch(
      reticulate::import("jaxsaem.nlmixr_bridge", convert = FALSE),
      error = function(e) {
        stop("could not import 'jaxsaem.nlmixr_bridge': ",
             conditionMessage(e),
             "\n  install the Python package with: pip install jaxsaem",
             "\n  or call useJaxsaem(python = '/path/to/python') first",
             call. = FALSE)
      })
    .cache
  }
})

#' Wrap the Python fit dict as an nlmixr2FitData-shaped object
#'
#' v0 jaxsaem does not produce a covariance matrix or residual tables,
#' so we cannot route through `nlmixr2CreateOutputFromUi()` end-to-end
#' (which expects FOCEi-style internals). Instead we build a minimal
#' fit list and class it so that `est = "jaxsaem"` plays well with
#' `print()` and basic accessors.
#'
#' @param fitDict R-side conversion of the Python return dict
#' @param env nlmixr2Est environment
#' @param ui rxode2 UI object
#' @param ctl `jaxsaemControl` object
#' @param elapsedSec Wall-clock time spent in `run_jaxsaem()` (s)
#' @return Object of class
#'   `c("nlmixr2FitData", "jaxsaemFit", "list")`
#' @noRd
.wrapAsNlmixrFit <- function(fitDict, env, ui, ctl, elapsedSec) {
  .paramNames <- as.character(unlist(fitDict$param_names))
  .canonical <- as.character(unlist(fitDict$canonical_order))
  .theta <- vapply(.paramNames,
                   function(.n) as.numeric(fitDict$theta[[.n]]),
                   numeric(1L))
  .omegaDiag <- vapply(.canonical,
                       function(.n) {
                         .v <- fitDict$omega[[paste0("omega.", .n)]]
                         if (is.null(.v)) 0 else as.numeric(.v)
                       },
                       numeric(1L))
  .omega <- diag(.omegaDiag, nrow = length(.omegaDiag))
  dimnames(.omega) <- list(paste0("eta.", .canonical),
                           paste0("eta.", .canonical))

  .sigma <- unlist(fitDict$sigma)

  .etas <- as.matrix(fitDict$etas)
  if (length(.etas) > 0L) {
    colnames(.etas) <- paste0("eta.", .paramNames)
    .ranef <- data.frame(ID = unlist(fitDict$ids), .etas,
                         check.names = FALSE, stringsAsFactors = FALSE)
  } else {
    .ranef <- data.frame(ID = unlist(fitDict$ids))
  }

  .objf <- as.numeric(fitDict$objf)
  .converged <- isTRUE(as.logical(fitDict$converged))
  .method <- as.character(fitDict$method)

  .ret <- list(
    theta = .theta,
    omega = .omega,
    sigma = .sigma,
    objf = .objf,
    converged = .converged,
    method = .method,
    time = list(total = elapsedSec, fit = elapsedSec),
    etas = .ranef,
    ranef = .ranef,
    ui = ui,
    control = ctl,
    est = "jaxsaem",
    canonicalOrder = .canonical,
    paramOrder = .paramNames
  )
  class(.ret) <- c("nlmixr2FitData", "jaxsaemFit", "list")
  .ret
}

#' Print a jaxsaem fit
#'
#' @param x Object of class `jaxsaemFit`
#' @param ... Ignored
#' @return Invisibly returns `x`
#' @export
print.jaxsaemFit <- function(x, ...) {
  cat("nlmixr2 fit (est = 'jaxsaem')\n")
  cat("  inner method: ", x$method, "\n", sep = "")
  cat("  converged:    ", x$converged, "\n", sep = "")
  cat("  objf:         ",
      if (is.na(x$objf)) "n/a (SAEM)" else format(x$objf, digits = 6),
      "\n", sep = "")
  cat("  fit time:     ", format(x$time$total, digits = 4), " s\n", sep = "")
  cat("\nFixed effects (theta):\n")
  print(x$theta)
  cat("\nResidual error (sigma):\n")
  print(x$sigma)
  cat("\nRandom effects (Omega diagonal):\n")
  print(diag(x$omega))
  invisible(x)
}

## ---------------------------------------------------------------------------
## Internal: error helpers
## ---------------------------------------------------------------------------

#' Raise a classed `jaxsaemUnsupportedError`
#'
#' @param ... Pieces of the error message; concatenated with `paste0()`
#' @return Never returns; calls `stop()`
#' @noRd
.stopUnsupported <- function(...) {
  .msg <- paste0(...)
  .cond <- structure(
    class = c("jaxsaemUnsupportedError", "error", "condition"),
    list(message = .msg, call = sys.call(-1L))
  )
  stop(.cond)
}

#' Pretty-print a character vector for use inside an error message
#'
#' @param x Character vector
#' @return A single string like `"(a, b, c)"`
#' @noRd
.deparseList <- function(x) {
  paste0("(", paste(x, collapse = ", "), ")")
}

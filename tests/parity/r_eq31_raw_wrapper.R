#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(glmnet))

format_number <- function(value) {
  if (is.na(value)) {
    return("NA")
  }
  formatC(value, digits = 17, format = "fg", flag = "#")
}

emit_vector <- function(tag, values) {
  cat(tag, paste(vapply(as.numeric(values), format_number, character(1)), collapse = " "), "\n")
}

emit_matrix <- function(tag, values) {
  matrix_values <- as.matrix(values)
  flattened <- c(t(matrix_values))
  cat(
    tag,
    nrow(matrix_values),
    ncol(matrix_values),
    paste(vapply(flattened, format_number, character(1)), collapse = " "),
    "\n"
  )
}

parse_args <- function(argv) {
  parsed <- list()
  for (argument in argv) {
    if (!startsWith(argument, "--")) {
      stop(sprintf("Unexpected argument: %s", argument))
    }
    key_value <- strsplit(sub("^--", "", argument), "=", fixed = TRUE)[[1]]
    key <- gsub("-", "_", key_value[[1]])
    value <- if (length(key_value) > 1) {
      paste(key_value[-1], collapse = "=")
    } else {
      "TRUE"
    }
    parsed[[key]] <- value
  }
  parsed
}

parse_numeric_vector <- function(text) {
  if (is.null(text) || identical(text, "")) {
    return(numeric(0))
  }
  values <- strsplit(text, ",", fixed = TRUE)[[1]]
  as.numeric(values)
}

parse_numeric_matrix <- function(text) {
  row_tokens <- strsplit(text, ";", fixed = TRUE)[[1]]
  rows <- lapply(row_tokens, parse_numeric_vector)
  width <- unique(vapply(rows, length, integer(1)))
  if (length(width) != 1) {
    stop("Matrix rows must all have the same width")
  }
  matrix(unlist(rows), ncol = width, byrow = TRUE)
}

normalize_scalar_string <- function(text) {
  if (is.null(text)) {
    return(NULL)
  }
  trimws(tolower(text))
}

script_args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", script_args[grep("^--file=", script_args)])
script_dir <- dirname(normalizePath(script_path[[1]]))
repo_root <- normalizePath(file.path(script_dir, "..", "..", ".."))
source(file.path(repo_root, "hddid-r", "R", "Examplehighdimdiffindiff.R"))

args <- parse_args(commandArgs(trailingOnly = TRUE))

required_args <- c(
  "delta_y_valid",
  "pi_hat_valid",
  "phi1_hat_valid",
  "phi0_hat_valid",
  "rho_hat_valid",
  "x_valid",
  "z_valid",
  "z0",
  "basis_family",
  "basis_degree",
  "oracle_lane"
)
missing_args <- required_args[!vapply(required_args, function(name) !is.null(args[[name]]), logical(1))]
if (length(missing_args) > 0) {
  stop(sprintf("Missing required arguments: %s", paste(missing_args, collapse = ", ")))
}

delta_y_valid <- parse_numeric_vector(args$delta_y_valid)
pi_hat_valid <- parse_numeric_vector(args$pi_hat_valid)
phi1_hat_valid <- parse_numeric_vector(args$phi1_hat_valid)
phi0_hat_valid <- parse_numeric_vector(args$phi0_hat_valid)
rho_hat_valid <- parse_numeric_vector(args$rho_hat_valid)
x_valid <- parse_numeric_matrix(args$x_valid)
z_valid <- parse_numeric_vector(args$z_valid)
z0 <- parse_numeric_vector(args$z0)
basis_family <- normalize_scalar_string(args$basis_family)
basis_degree <- as.integer(args$basis_degree)
oracle_lane <- normalize_scalar_string(args$oracle_lane)

n_valid <- length(delta_y_valid)
aligned_lengths <- c(
  length(pi_hat_valid),
  length(phi1_hat_valid),
  length(phi0_hat_valid),
  length(rho_hat_valid),
  nrow(x_valid),
  length(z_valid)
)
if (any(aligned_lengths != n_valid)) {
  stop("All valid-sample arrays must align")
}

if (!basis_family %in% c("polynomial", "trigonometric")) {
  stop("basis_family must be polynomial or trigonometric")
}
if (!oracle_lane %in% c("r-parity-polynomial", "paper-trigonometric")) {
  stop("oracle_lane is not recognized")
}
if (basis_family == "polynomial" && oracle_lane != "r-parity-polynomial") {
  stop("oracle_lane must be r-parity-polynomial for polynomial basis")
}
if (basis_family == "trigonometric" && oracle_lane != "paper-trigonometric") {
  stop("oracle_lane must be paper-trigonometric for trigonometric basis")
}

if (basis_family == "polynomial") {
  q <- basis_degree
  basis_valid_full <- sieve.Pol(z_valid, q)
  evaluation_basis <- sieve.Pol(z0, q)
} else {
  q <- 2L * basis_degree
  basis_valid_full <- sieve.TriPol(z_valid, basis_degree)
  evaluation_basis <- sieve.TriPol(z0, basis_degree)
}

basis_design_valid <- basis_valid_full[, -1, drop = FALSE]
w_valid <- cbind(x_valid, basis_design_valid)
newy <- rho_hat_valid * (
  delta_y_valid
  - (1 - pi_hat_valid) * phi1_hat_valid
  - pi_hat_valid * phi0_hat_valid
)

p <- ncol(x_valid)
penalty_factor <- c(rep(1, p), rep(0, ncol(basis_design_valid)))
lambda_override <- if (!is.null(args$lambda_override)) as.numeric(args$lambda_override) else NA_real_
lambda_used <- NA_real_
betahat <- rep(NA_real_, ncol(w_valid))

if (!is.na(lambda_override)) {
  lambda_used <- lambda_override
  if (abs(lambda_used) <= sqrt(.Machine$double.eps)) {
    betahat <- as.numeric(qr.solve(w_valid, newy))
  } else {
    fit <- glmnet(
      x = w_valid,
      y = newy,
      alpha = 1,
      lambda = lambda_used,
      penalty.factor = penalty_factor,
      intercept = FALSE,
      standardize = FALSE
    )
    betahat <- as.numeric(as.matrix(fit$beta))
  }
} else if (!is.null(args$foldid)) {
  foldid <- as.integer(parse_numeric_vector(args$foldid))
  if (length(foldid) != n_valid) {
    stop("foldid must align with valid observations")
  }
  if (!is.null(args$seed)) {
    set.seed(as.integer(args$seed))
  }
  cv_fit <- cv.glmnet(
    x = w_valid,
    y = newy,
    alpha = 1,
    foldid = foldid,
    nfolds = length(unique(foldid)),
    penalty.factor = penalty_factor,
    intercept = FALSE,
    standardize = FALSE
  )
  lambda_used <- cv_fit$lambda.min
  fit <- glmnet(
    x = w_valid,
    y = newy,
    alpha = 1,
    lambda = lambda_used,
    penalty.factor = penalty_factor,
    intercept = FALSE,
    standardize = FALSE
  )
  betahat <- as.numeric(as.matrix(fit$beta))
}

meta <- c(
  n_valid = n_valid,
  rows_w = nrow(w_valid),
  cols_w = ncol(w_valid),
  rows_basis_full = nrow(basis_valid_full),
  cols_basis_full = ncol(basis_valid_full),
  rows_basis_design = nrow(basis_design_valid),
  cols_basis_design = ncol(basis_design_valid),
  lambda_used = lambda_used,
  basis_family = basis_family,
  basis_degree = basis_degree,
  q = q,
  oracle_lane = oracle_lane
)

meta_tokens <- mapply(
  function(name, value) sprintf("%s=%s", name, value),
  names(meta),
  vapply(meta, function(value) {
    if (length(value) != 1) {
      stop("META values must be scalar")
    }
    if (is.numeric(value)) {
      return(format_number(as.numeric(value)))
    }
    as.character(value)
  }, character(1)),
  USE.NAMES = FALSE
)
cat("META", paste(meta_tokens, collapse = " "), "\n")
emit_vector("NEWY", newy)
emit_vector("BETAHAT", betahat)
emit_matrix("WVALID", w_valid)
emit_matrix("BASISFULL", basis_valid_full)
emit_matrix("BASISDESIGN", basis_design_valid)
emit_matrix("EVALBASIS", evaluation_basis)

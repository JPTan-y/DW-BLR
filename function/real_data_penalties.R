args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
root <- normalizePath(args[1], mustWork = TRUE)
library(jsonlite)
cfg <- fromJSON(file.path(root, "parameters.json"), simplifyVector = FALSE)
for (name in names(cfg)) {
  package <- if (name == "SCAD") "ncvreg" else "glmnet"
  if (!requireNamespace(package, quietly = TRUE)) stop(paste("Missing R package:", package))
  if (packageVersion(package) != package_version(cfg[[name]]$package_version)) {
    stop(paste(package, "version must be", cfg[[name]]$package_version))
  }
}
X <- as.matrix(read.csv(file.path(root, "X.csv"), check.names = FALSE))
y <- read.csv(file.path(root, "y.csv"))$pCR
stopifnot(nrow(X) == length(y), all(is.finite(X)), identical(sort(unique(y)), c(0L, 1L)))
warnings_seen <- character()
run_fit <- function(expr) withCallingHandlers(expr, warning = function(w) {
  warnings_seen <<- c(warnings_seen, conditionMessage(w))
  invokeRestart("muffleWarning")
})
results <- list()
for (name in names(cfg)) {
  p <- cfg[[name]]
  lam <- p$lambda_
  grid <- exp(seq(log(p$path$start), log(p$path$stop), length.out = p$path$log_points))
  if (name == "SCAD") {
    grid <- sort(unique(c(grid[grid > lam], lam)), decreasing = TRUE)
    fit_one <- function(tol) ncvreg::ncvreg(X, y, family = p$family, penalty = p$penalty,
      gamma = p$gamma, alpha = p$alpha, lambda = grid, eps = tol,
      max.iter = p$max_iter_path_total, warn = TRUE, returnX = TRUE)
    loose <- run_fit(fit_one(p$crosscheck_eps))
    tight <- run_fit(fit_one(p$eps))
    stopifnot(length(tight$lambda) == length(grid), sum(tight$iter) < p$max_iter_path_total)
    j <- which.min(abs(tight$lambda - lam))
    b <- as.numeric(coef(tight)[, j])
    b0 <- as.numeric(coef(loose)[, which.min(abs(loose$lambda - lam))])
    iterations <- sum(tight$iter)
  } else {
    glmnet::glmnet.control(fdev = p$glmnet_control$fdev, devmax = p$glmnet_control$devmax)
    grid <- sort(unique(c(grid, lam)), decreasing = TRUE)
    fit_one <- function(tol) glmnet::glmnet(X, y, family = p$family, alpha = p$alpha,
      lambda = grid, intercept = p$intercept, standardize = p$standardize,
      thresh = tol, maxit = p$maxit)
    loose <- run_fit(fit_one(p$crosscheck_thresh))
    tight <- run_fit(fit_one(p$thresh))
    stopifnot(tight$jerr == 0L, length(tight$lambda) == length(grid))
    j <- which.min(abs(tight$lambda - lam))
    b <- as.numeric(coef(tight)[, j])
    b0 <- as.numeric(coef(loose)[, which.min(abs(loose$lambda - lam))])
    iterations <- tight$npasses
  }
  stopifnot(abs(tight$lambda[j] - lam) < 1e-14, all(is.finite(b)))
  delta <- max(abs(plogis(b[1] + X %*% b[-1]) - plogis(b0[1] + X %*% b0[-1])))
  same_selection <- identical(abs(b[-1]) > p$nonzero_threshold, abs(b0[-1]) > p$nonzero_threshold)
  stopifnot(delta < 1e-5, same_selection)
  results[[name]] <- list(intercept = b[1], beta = b[-1],
    selected = abs(b[-1]) > p$nonzero_threshold,
    diagnostics = list(converged = TRUE, backend = p$backend, lambda = lam,
      iterations = iterations, package_version = as.character(packageVersion(if (name == "SCAD") "ncvreg" else "glmnet")),
      crosscheck_prediction_max_difference = delta, same_selection = same_selection))
  saveRDS(tight, file.path(root, paste0(gsub(" ", "_", name), ".rds")))
}
write_json(results, file.path(root, "results.json"), auto_unbox = TRUE, digits = NA, pretty = TRUE)
writeLines(capture.output(sessionInfo()), file.path(root, "sessionInfo.txt"))
writeLines(unique(warnings_seen), file.path(root, "warnings.txt"))

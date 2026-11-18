required <- c(glmnet = "4.1-10", ncvreg = "3.16.0")
user_library <- Sys.getenv("R_LIBS_USER")
if (!nzchar(user_library)) stop("Set R_LIBS_USER to a writable R package library")
dir.create(user_library, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(user_library, .libPaths()))
repos <- c(CRAN = "https://cloud.r-project.org")
for (package in c("jsonlite", "remotes")) {
  if (!requireNamespace(package, quietly = TRUE)) {
    install.packages(package, repos = repos, lib = user_library)
  }
}
for (package in names(required)) {
  available <- requireNamespace(package, quietly = TRUE)
  if (!available || packageVersion(package) != package_version(required[[package]])) {
    remotes::install_version(package, version = required[[package]], repos = repos,
                             lib = user_library, upgrade = "never")
  }
  stopifnot(packageVersion(package) == package_version(required[[package]]))
}
sessionInfo()

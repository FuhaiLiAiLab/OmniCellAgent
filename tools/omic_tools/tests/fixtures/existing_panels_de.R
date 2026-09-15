# Test oracle: execute the unchanged DE chunks in casestudy_panels.Rmd.
# Deliberately exclude plotting so this checks the original statistical code.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3L)
suppressPackageStartupMessages({library(limma); library(data.table)})
params <- list(session_dir = args[2], ref_csv = "reference.csv",
               alt_csv = "alternate.csv", ref_label = "Healthy",
               alt_label = "Diseased", input_scale = "linear_cp10k")
dir.create(args[3], recursive = TRUE, showWarnings = FALSE)
out_stem <- file.path(args[3], "DE_results")
fdr_thresh <- 0.05
lines <- readLines(args[1], warn = FALSE)
for (chunk in c("load-data", "scale-check", "combine", "de")) {
  start <- which(lines == paste0("```{r ", chunk, "}"))
  stopifnot(length(start) == 1L)
  end <- which(seq_along(lines) > start & lines == "```")[1]
  stopifnot(!is.na(end))
  eval(parse(text = lines[seq.int(start + 1L, end - 1L)]))
}

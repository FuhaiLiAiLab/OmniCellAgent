#!/usr/bin/env Rscript
# sex_stratified_de.R - disease-vs-normal DE within each sex for one finished cohort folder of the
# omic workflow (omic_<cell type>/), for the female-vs-male 4-quadrant figure
# (plot_sex_stratified_scatter.py).
#
# Reuses the cohort the workflow sampled: the two foranalysis matrices (linear CP10K), each
# metacell's source dataset (de_covariates_*.csv) and its sex and donor (labels.csv). Scale and
# genes match run_casestudy.R (log2(CP10K + 1); genes finite in >= 80% of metacells with nonzero
# variance, filtered on the whole cohort). Metacells without a female/male label are then dropped
# and one limma model is fitted to the four groups:
#   ~ 0 + group + dataset,  group = status x sex,  donor as random block (duplicateCorrelation)
# with robust trended eBayes. There is no chrY score: sex is in the group term. Contrasts: the
# disease effect in each sex (the figure's two axes), the sex effect within each status, and
#   interaction = (disease effect in males) - (disease effect in females),
# the formal test of a sex-dependent disease response.
#
# Usage: Rscript sex_stratified_de.R <cohort_dir> <out_dir>
# Writes merged_de.csv (the plot script's input), sex_disease_interaction.csv (all contrasts)
# and summary.json.

suppressPackageStartupMessages({ library(data.table); library(limma) })

LFC_THR <- 0.5   # 4-quadrant figure cutoffs
FDR_THR <- 0.05
GROUPS <- c("normal.female", "normal.male", "disease.female", "disease.male")

find_one <- function(dir, pattern) {
  f <- list.files(dir, pattern, full.names = TRUE)
  if (length(f) != 1L) stop(sprintf("Expected one file matching %s in %s, found %d.", pattern, dir, length(f)))
  f
}

read_expr <- function(path) {
  dt <- fread(path)
  genes <- as.character(dt[[1]])
  dt[, 1 := NULL]
  mat <- as.matrix(dt)
  rownames(mat) <- genes
  storage.mode(mat) <- "numeric"
  mat
}

read_cohort <- function(cohort_dir) {
  ref <- read_expr(find_one(cohort_dir, "^foranalysis_combined_normal_df_.*\\.csv$"))
  alt <- read_expr(find_one(cohort_dir, "^foranalysis_combined_disease_df_.*\\.csv$"))
  cov <- fread(find_one(cohort_dir, "^de_covariates_.*\\.csv$"), colClasses = "character")
  lab <- fread(file.path(cohort_dir, "labels.csv"), colClasses = "character")

  genes <- intersect(rownames(ref), rownames(alt))
  expr <- log2(cbind(ref[genes, ], alt[genes, ]) + 1)
  expr <- expr[rowSums(is.finite(expr)) >= ncol(expr) * 0.8, , drop = FALSE]
  v <- apply(expr, 1, stats::var, na.rm = TRUE)
  expr <- expr[is.finite(v) & v > 0, , drop = FALSE]

  # labels.csv rows are [disease block; normal block], each in ds_/ns_sample_<i> order. Check the
  # join on status and dataset instead of trusting the order.
  if (nrow(lab) != ncol(expr)) stop(sprintf("labels.csv has %d rows for %d metacells.", nrow(lab), ncol(expr)))
  is_alt <- startsWith(colnames(expr), "ds_sample_")
  row <- ifelse(is_alt, 0L, ncol(alt)) + as.integer(sub("^[dn]s_sample_", "", colnames(expr))) + 1L
  if (anyNA(row) || anyDuplicated(row) || any(row > nrow(lab))) stop("Unexpected matrix column names.")
  lab <- lab[row]
  lab_normal <- tolower(trimws(lab$disease_BMG_name)) == "normal"   # raw `disease` keeps source spellings
  if (any(lab_normal == is_alt)) stop("labels.csv disease status does not match the matrix columns.")
  dataset <- cov$dataset_id[match(colnames(expr), cov$sample)]
  if (anyNA(dataset) || any(dataset != lab$dataset_id)) stop("labels.csv datasets do not match de_covariates.")

  meta <- data.frame(sample = colnames(expr), status = ifelse(is_alt, "disease", "normal"),
                     sex = tolower(trimws(lab$sex_normalized)), dataset = dataset,
                     donor = paste(lab$source, lab$dataset_id, lab$donor_id, sep = "|"),
                     stringsAsFactors = FALSE)
  list(expr = expr, meta = meta)
}

fit_joint_model <- function(expr, meta) {
  group <- factor(paste(meta$status, meta$sex, sep = "."), levels = GROUPS)
  dataset <- droplevels(factor(meta$dataset))
  formula <- "~ 0 + group + dataset"
  design <- model.matrix(~ 0 + group + dataset)
  if (nlevels(dataset) < 2L || qr(design)$rank < ncol(design)) {
    message("Dataset term omitted: single dataset or confounded with group.")
    formula <- "~ 0 + group"
    design <- model.matrix(~ 0 + group)
  }
  colnames(design)[seq_along(GROUPS)] <- GROUPS
  colnames(design) <- make.names(colnames(design), unique = TRUE)
  corfit <- duplicateCorrelation(expr, design, block = meta$donor)
  rho <- corfit$consensus.correlation
  if (!is.finite(rho)) stop("duplicateCorrelation returned a non-finite consensus.")
  message(sprintf("DE model: %s, donor block, consensus within-donor correlation = %.3f", formula, rho))
  fit <- lmFit(expr, design, block = meta$donor, correlation = rho)
  cm <- makeContrasts(
    disease_in_female = disease.female - normal.female,
    disease_in_male = disease.male - normal.male,
    sex_in_disease = disease.male - disease.female,
    sex_in_normal = normal.male - normal.female,
    interaction = (disease.male - normal.male) - (disease.female - normal.female),
    levels = design)
  fit2 <- eBayes(contrasts.fit(fit, cm), robust = TRUE, trend = TRUE)
  tables <- lapply(colnames(cm), function(k) {
    t <- topTable(fit2, coef = k, number = Inf, sort.by = "none")
    setNames(t[, c("logFC", "P.Value", "adj.P.Val")], paste(k, c("logFC", "P", "FDR"), sep = "_"))
  })
  list(table = data.frame(Gene = rownames(expr), do.call(cbind, tables), check.names = FALSE),
       formula = formula, rho = rho)
}

main <- function(args) {
  if (length(args) != 2L) stop("Usage: Rscript sex_stratified_de.R <cohort_dir> <out_dir>")
  dir.create(args[2], showWarnings = FALSE, recursive = TRUE)
  cohort <- read_cohort(args[1])
  keep <- cohort$meta$sex %in% c("female", "male")
  expr <- cohort$expr[, keep, drop = FALSE]
  meta <- cohort$meta[keep, ]
  message(sprintf("%d genes; %d of %d metacells with a female/male label", nrow(expr), sum(keep), length(keep)))
  key <- factor(paste(meta$status, meta$sex, sep = "."), levels = GROUPS)
  groups <- data.frame(group = GROUPS, metacells = as.integer(table(key)),
                       donors = as.integer(tapply(meta$donor, key, function(d) length(unique(d)))))
  print(groups, row.names = FALSE)
  if (any(groups$metacells < 2L)) stop("Each status x sex group needs at least two metacells.")

  model <- fit_joint_model(expr, meta)
  out <- model$table
  fwrite(out[order(out$interaction_P), ], file.path(args[2], "sex_disease_interaction.csv"))
  merged <- data.frame(gene = out$Gene,
                       log2FC_female = out$disease_in_female_logFC, pvalue_female = out$disease_in_female_P,
                       FDR_female = out$disease_in_female_FDR,
                       log2FC_male = out$disease_in_male_logFC, pvalue_male = out$disease_in_male_P,
                       FDR_male = out$disease_in_male_FDR,
                       interaction_logFC = out$interaction_logFC, interaction_P = out$interaction_P,
                       interaction_FDR = out$interaction_FDR)
  fwrite(merged, file.path(args[2], "merged_de.csv"))

  # The figure's categories, and how many one-sex genes the interaction test supports
  f_sig <- merged$FDR_female < FDR_THR & abs(merged$log2FC_female) >= LFC_THR
  m_sig <- merged$FDR_male < FDR_THR & abs(merged$log2FC_male) >= LFC_THR
  one_sex <- xor(f_sig, m_sig)
  same <- sign(merged$log2FC_female) == sign(merged$log2FC_male)
  int_sig <- merged$interaction_FDR < FDR_THR
  summary <- list(
    cohort_dir = normalizePath(args[1]), model = model$formula, donor_block = "source|dataset_id|donor_id",
    within_donor_correlation = model$rho, genes = nrow(expr),
    metacells_dropped_unknown_sex = sum(!keep), groups = groups,
    cutoffs = list(abs_log2FC = LFC_THR, FDR = FDR_THR),
    pearson_r_female_male = cor(merged$log2FC_female, merged$log2FC_male),
    both_same_direction = sum(f_sig & m_sig & same), discordant = sum(f_sig & m_sig & !same),
    one_sex_only = sum(one_sex), one_sex_same_sign_in_other_sex = sum(one_sex & same),
    one_sex_interaction_FDR_below_cutoff = sum(one_sex & int_sig),
    interaction_FDR_below_cutoff = sum(int_sig),
    top_interaction_genes = head(merged[order(merged$interaction_P), c("gene", "log2FC_female", "log2FC_male",
                                                                     "interaction_logFC", "interaction_FDR")], 15))
  jsonlite::write_json(summary, file.path(args[2], "summary.json"), auto_unbox = TRUE, pretty = TRUE, digits = 4)
  message(sprintf(paste0("r(female, male) = %.2f; both same direction %d, discordant %d, one sex only %d ",
                         "(%d with interaction FDR < %.2f); interaction FDR < %.2f: %d genes"),
                  summary$pearson_r_female_male, summary$both_same_direction, summary$discordant,
                  summary$one_sex_only, summary$one_sex_interaction_FDR_below_cutoff, FDR_THR, FDR_THR,
                  summary$interaction_FDR_below_cutoff))
}

if (sys.nframe() == 0L) main(commandArgs(trailingOnly = TRUE))

#!/usr/bin/env Rscript
# Controlled donor sensitivity analysis. Source counts/results are read only.
# CLI: original_run_dir output_dir alpha min_count min_gene_donors min_group_donors

fail <- function(code, message, details = NULL) {
  stop(structure(list(message = message, call = NULL, reason_code = code, details = details),
                 class = c("sensitivity_error", "error", "condition")))
}

write_json <- function(value, path) {
  jsonlite::write_json(value, path, auto_unbox = TRUE, pretty = TRUE,
                       na = "null", null = "null", digits = NA)
}

write_csv <- function(value, path) {
  utils::write.csv(value, path, row.names = FALSE, na = "")
}

fixed_family_bh <- function(pvalue, valid) {
  if (length(pvalue) != length(valid)) stop("pvalue and valid must have equal lengths")
  valid <- !is.na(valid) & valid & is.finite(pvalue) & pvalue >= 0 & pvalue <= 1
  family <- rep(1, length(pvalue))
  family[valid] <- pvalue[valid]
  adjusted <- stats::p.adjust(family, method = "BH", n = length(family))
  adjusted[!valid] <- NA_real_
  adjusted
}

compare_source_effects <- function(exported, original, source_path, tolerance = 1e-6) {
  original_index <- match(exported$gene, original$gene)
  old_effect <- original$log2FoldChange[original_index]
  comparable <- is.finite(old_effect) & is.finite(exported$log2FoldChange)
  converged <- comparable & !is.na(exported$beta_converged) & exported$beta_converged
  if ("filter_reason" %in% names(original)) {
    reason <- original$filter_reason[original_index]
    converged <- converged & (is.na(reason) | reason != "nonconverged")
  }
  if ("beta_converged" %in% names(original)) {
    flag <- original$beta_converged[original_index]
    converged <- converged & !is.na(flag) & flag
  }
  maximum <- function(mask) {
    if (any(mask)) max(abs(old_effect[mask] - exported$log2FoldChange[mask])) else NA_real_
  }
  all_difference <- maximum(comparable)
  converged_difference <- maximum(converged)
  list(n_compared_genes = sum(comparable), max_abs_lfc_difference = all_difference,
       within_tolerance = is.finite(all_difference) && all_difference <= tolerance,
       n_converged_compared_genes = sum(converged),
       max_abs_lfc_difference_converged = converged_difference,
       within_tolerance_converged = is.finite(converged_difference) && converged_difference <= tolerance,
       tolerance = tolerance, source = source_path)
}

independent_ratio_factors <- function(counts) {
  positive <- rowSums(counts > 0L) == ncol(counts)
  if (!ncol(counts) || !any(positive)) {
    fail("ratio_normalization_unavailable", "No retained gene is positive in every included donor.")
  }
  logs <- log(counts[positive, , drop = FALSE])
  ratios <- sweep(logs, 1L, rowMeans(logs), "-")
  list(factors = exp(apply(ratios, 2L, stats::median)), n_reference_genes = sum(positive))
}

read_source <- function(source, settings) {
  original_comparison <- file.path(source, "comparisons", "ad_vs_control")
  count_path <- file.path(source, "inputs", "pseudobulk_counts.csv")
  metadata_path <- file.path(original_comparison, "donor_metadata.csv")
  factor_path <- file.path(original_comparison, "size_factors.csv")
  metadata <- utils::read.csv(metadata_path, colClasses = "character", check.names = FALSE,
                              na.strings = c("", "NA"))
  required <- c("sample_id", "source", "dataset_id", "donor_id", "study", "age", "sex", "disease")
  if (anyDuplicated(names(metadata)) || !all(required %in% names(metadata)) || !nrow(metadata)) {
    fail("invalid_metadata", "Source comparison metadata needs unique required columns and donor rows.")
  }
  if (anyNA(metadata$sample_id) || anyDuplicated(metadata$sample_id) ||
      any(!nzchar(trimws(metadata$sample_id)))) {
    fail("invalid_metadata", "Source comparison sample IDs must be nonempty and unique.")
  }
  identity <- metadata[c("source", "dataset_id", "donor_id")]
  if (anyNA(identity) || any(!nzchar(trimws(as.matrix(identity)))) || anyDuplicated(identity)) {
    fail("invalid_metadata", "Source donor identities must be unique (source, dataset_id, donor_id) keys.")
  }
  reasons <- lapply(seq_len(nrow(metadata)), function(i) character())
  add_reason <- function(mask, reason) {
    for (i in which(mask)) reasons[[i]] <<- c(reasons[[i]], reason)
  }
  empty <- function(x) is.na(x) | !nzchar(trimws(x))
  for (term in c("study", "age", "sex", "disease")) {
    add_reason(empty(metadata[[term]]), paste0("missing_", term))
  }
  age <- suppressWarnings(as.numeric(metadata$age))
  add_reason(!empty(metadata$age) & (!is.finite(age) | age < 0), "invalid_age")
  add_reason(!empty(metadata$sex) & !metadata$sex %in% c("female", "male"), "invalid_sex")
  add_reason(!empty(metadata$disease) & !metadata$disease %in% c("control", "AD"), "invalid_disease")
  for (field in c("age_conflict", "sex_conflict", "disease_conflict")) {
    if (field %in% names(metadata)) {
      flag <- tolower(trimws(metadata[[field]]))
      add_reason(is.na(flag) | !flag %in% c("false", "0"), field)
    }
  }
  if (any(lengths(reasons) > 0L)) {
    details <- lapply(which(lengths(reasons) > 0L), function(i) {
      list(sample_id = metadata$sample_id[i], reasons = I(reasons[[i]]))
    })
    fail("fixed_cohort_not_complete", "The fixed source cohort has unresolved covariates; no model-specific donor removal is allowed.", details)
  }
  metadata$age <- age
  rownames(metadata) <- metadata$sample_id
  raw <- utils::read.csv(count_path, colClasses = "character", check.names = FALSE,
                         na.strings = c("", "NA"))
  if (ncol(raw) < 2L || names(raw)[1L] != "gene" || !nrow(raw) ||
      anyDuplicated(names(raw)) || anyNA(raw[[1L]]) || anyDuplicated(raw[[1L]]) ||
      any(!nzchar(trimws(raw[[1L]])))) {
    fail("invalid_counts", "Source raw counts require unique gene rows and sample columns.")
  }
  if (!all(metadata$sample_id %in% names(raw)[-1L])) {
    fail("sample_mismatch", "Source counts do not contain every fixed comparison donor.")
  }
  values <- suppressWarnings(as.numeric(as.matrix(raw[, metadata$sample_id, drop = FALSE])))
  if (any(!is.finite(values)) || any(values < 0) || any(values != floor(values)) ||
      any(values > .Machine$integer.max)) {
    fail("invalid_counts", "Fixed-cohort counts must be finite nonnegative integers within R's integer range.")
  }
  counts <- matrix(as.integer(values), nrow = nrow(raw), dimnames = list(raw[[1L]], metadata$sample_id))
  totals <- colSums(counts)
  if (any(totals <= 0)) fail("zero_library_in_fixed_cohort", "The fixed source comparison contains a zero-library donor.")
  if ("library_size" %in% names(metadata)) {
    reported <- suppressWarnings(as.numeric(metadata$library_size))
    if (any(!is.finite(reported)) || any(reported != totals)) {
      fail("library_size_mismatch", "Source donor library sizes differ from original gene-count sums.")
    }
  }
  metadata$library_size <- unname(totals)
  factors <- utils::read.csv(factor_path, check.names = FALSE, colClasses = "character")
  if (!all(c("sample_id", "size_factor") %in% names(factors)) ||
      anyNA(factors$sample_id) || anyDuplicated(factors$sample_id) ||
      !setequal(factors$sample_id, metadata$sample_id)) {
    fail("size_factor_sample_mismatch", "Original size factors must match every fixed donor exactly once.")
  }
  original_factors <- suppressWarnings(as.numeric(factors$size_factor[match(metadata$sample_id, factors$sample_id)]))
  if (any(!is.finite(original_factors)) || any(original_factors <= 0)) {
    fail("invalid_original_size_factors", "Original size factors must be finite and positive.")
  }
  keep <- rowSums(counts >= settings$min_count) >= settings$min_gene_donors
  if (!any(keep)) fail("no_genes_pass_count_filter", "No genes pass the fixed full-cohort count filter.")
  expected <- independent_ratio_factors(counts[keep, , drop = FALSE])
  relative_error <- abs(original_factors / expected$factors - 1)
  tolerance <- 1e-8 # Original R CSV exports retain approximately 15 significant digits.
  if (any(relative_error > tolerance)) {
    fail("original_size_factor_mismatch", "Original factors do not equal independently estimated full-cohort ratio factors.",
         list(max_relative_error = max(relative_error), tolerance = tolerance,
              donors = I(metadata$sample_id[relative_error > tolerance])))
  }
  status_path <- file.path(original_comparison, "status.json")
  original_status <- if (file.exists(status_path)) jsonlite::read_json(status_path, simplifyVector = TRUE) else NULL
  if (!is.null(original_status) &&
      (!identical(original_status$status, "success") ||
       !identical(original_status$contrast$column, "disease") ||
       !identical(original_status$contrast$numerator, "AD") ||
       !identical(original_status$contrast$reference, "control"))) {
    fail("invalid_original_comparison", "The source comparison must be a successful AD-versus-control fit.")
  }
  list(counts = counts, metadata = metadata, keep = keep, original_factors = original_factors,
       original_status = original_status, original_comparison = original_comparison,
       factor_validation = list(max_relative_error = max(relative_error), tolerance = tolerance,
                                n_reference_genes = expected$n_reference_genes),
       source_files = list(counts = count_path, metadata = metadata_path, size_factors = factor_path))
}

model_specs <- function(metadata) {
  studies <- sort(unique(metadata$study))
  support <- do.call(rbind, lapply(studies, function(study) {
    groups <- table(factor(metadata$disease[metadata$study == study], levels = c("control", "AD")))
    data.frame(study = study, disease = names(groups), n_donors = as.integer(groups),
                shared_study = all(groups > 0), stringsAsFactors = FALSE)
  }))
  shared <- studies[vapply(studies, function(study) {
    all(c("control", "AD") %in% metadata$disease[metadata$study == study])
  }, logical(1L))]
  spec <- function(name, nuisance, cohort, omitted = NULL) {
    list(name = name, nuisance = nuisance, cohort = cohort, omitted_study = omitted)
  }
  models <- list(spec("full_A", character(), "full"), spec("full_B", "study", "full"),
                 spec("full_C", c("study", "age", "sex"), "full"),
                 spec("shared_C", c("study", "age", "sex"), "shared"))
  for (i in seq_along(shared)) {
    models[[length(models) + 1L]] <- spec(sprintf("leave_one_out_%02d", i), c("study", "age", "sex"), "leave_one_out", shared[i])
  }
  list(models = models, shared_studies = shared, support = support)
}

fit_model <- function(spec, source, shared_studies, settings, directory, export_expression = FALSE) {
  dir.create(directory, recursive = TRUE)
  audit <- new.env(parent = emptyenv())
  audit$status <- list(
    model = spec$name, status = "pending", cohort = spec$cohort, fitted_model_count = 0L,
    omitted_study = spec$omitted_study, alpha = settings$alpha,
    contrast = list(column = "disease", numerator = "AD", reference = "control"),
    requested_formula = paste("~", paste(c(spec$nuisance, "disease"), collapse = " + ")),
    dropped_constant_terms = I(character()), constant_term_audit = list(),
    filter = list(min_count = settings$min_count, min_gene_donors = settings$min_gene_donors,
                  independent_filtering = FALSE, family = "all count-filter retained genes",
                  invalid_tests_for_bh = "p=1; reported padj remains missing",
                  n_input_genes = nrow(source$counts)),
    normalization = list(type = "ratio", fallback = "none",
                         source = if (spec$cohort == "full") {
                           if (is.null(source$factor_source)) "original_full_cohort_factors" else source$factor_source
                         } else "reestimated_subset_ratio_factors"),
    test = list(method = "DESeq2 Wald", beta_prior = FALSE, automatic_outlier_replacement = FALSE,
                cooks_cutoff = "DESeq2 default", independent_filtering = FALSE,
                lfc_shrinkage = "none", adjustment = "fixed-family BH",
                confidence_interval = "95% unshrunk Wald interval for finite, converged estimates"),
    versions = list(R = R.version.string, DESeq2 = as.character(utils::packageVersion("DESeq2"))),
    warnings = I(character()), started_at = format(Sys.time(), tz = "UTC", usetz = TRUE)
  )
  finish <- function(state, code = NULL, reason = NULL) {
    audit$status$status <- state
    audit$status$reason_code <- code
    audit$status$reason <- reason
    audit$status$completed_at <- format(Sys.time(), tz = "UTC", usetz = TRUE)
    cat("\nModel status:", state, if (!is.null(code)) code else "", "\n")
    if (!is.null(reason)) cat(reason, "\n")
    audit$status
  }
  log <- file(file.path(directory, "fit.log"), "wt")
  sink(log)
  sink(log, type = "message")
  on.exit({ sink(type = "message"); sink(); close(log)
            write_json(audit$status, file.path(directory, "status.json")) }, add = TRUE)
  tryCatch(withCallingHandlers({
    cat("Model:", spec$name, "\n", R.version.string, "\nDESeq2", as.character(utils::packageVersion("DESeq2")), "\n")
    included <- if (spec$cohort == "full") rep(TRUE, nrow(source$metadata)) else source$metadata$study %in% shared_studies
    if (!is.null(spec$omitted_study)) included <- included & source$metadata$study != spec$omitted_study
    metadata <- source$metadata[included, , drop = FALSE]
    counts <- source$counts[, metadata$sample_id, drop = FALSE]
    keep <- if (spec$cohort == "full") source$keep else rowSums(counts >= settings$min_count) >= settings$min_gene_donors
    audit$status$included_donors <- I(metadata$sample_id)
    audit$status$excluded_donors <- I(source$metadata$sample_id[!included])
    audit$status$included_studies <- I(sort(unique(metadata$study)))
    audit$status$n_donors <- nrow(metadata)
    audit$status$group_counts <- as.list(table(factor(metadata$disease, levels = c("control", "AD"))))
    audit$status$filter$n_retained_genes <- sum(keep)
    audit$status$filter$n_bh_genes <- sum(keep)
    audit$status$filter$n_filtered_genes <- sum(!keep)
    write_csv(metadata, file.path(directory, "donor_metadata.csv"))
    if (!nrow(metadata)) return(finish("skipped", "no_donors", "No donors meet this study subset definition."))
    dropped <- spec$nuisance[vapply(spec$nuisance, function(term) length(unique(metadata[[term]])) == 1L, logical(1L))]
    audit$status$dropped_constant_terms <- I(dropped)
    audit$status$constant_term_audit <- lapply(dropped, function(term) {
      list(term = term, value = as.character(metadata[[term]][1L]), reason = "constant in this fixed donor subset")
    })
    formula <- stats::reformulate(c(setdiff(spec$nuisance, dropped), "disease"))
    audit$status$formula <- paste(deparse(formula), collapse = " ")
    cat("Requested formula:", audit$status$requested_formula, "\nFitted formula:", audit$status$formula, "\n")
    if (any(unlist(audit$status$group_counts) < settings$min_group_donors)) {
      return(finish("skipped", "insufficient_group_donors", paste("Each level requires at least", settings$min_group_donors, "donors.")))
    }
    metadata$study <- factor(metadata$study, levels = sort(unique(metadata$study)))
    metadata$sex <- factor(metadata$sex, levels = c("female", "male"))
    metadata$disease <- factor(metadata$disease, levels = c("control", "AD"))
    design <- stats::model.matrix(formula, metadata)
    decomposition <- qr(design)
    rank <- decomposition$rank
    aliased <- if (rank < ncol(design)) colnames(design)[decomposition$pivot[seq.int(rank + 1L, ncol(design))]] else character()
    audit$status$design <- list(n_rows = nrow(design), n_columns = ncol(design), rank = rank,
                                residual_df = nrow(design) - rank, aliased_columns = I(aliased),
                                columns = I(colnames(design)))
    write_csv(data.frame(sample_id = rownames(design), design, check.names = FALSE), file.path(directory, "design_matrix.csv"))
    if (rank < ncol(design)) return(finish("skipped", "design_not_full_rank", "Confounded adjusted model; nonconstant covariates were retained."))
    if (nrow(design) <= rank) return(finish("skipped", "no_residual_degrees_of_freedom", "No residual degrees of freedom for dispersion estimation."))
    if (!"diseaseAD" %in% colnames(design) ||
        !all(design[, "diseaseAD"] == as.integer(metadata$disease == "AD"))) {
      return(finish("error", "contrast_direction_invalid", "Design does not code AD versus control as requested."))
    }
    if (!any(keep)) return(finish("skipped", "no_genes_pass_count_filter", "No genes pass this cohort's raw-count filter."))
    if (!any(rowSums(counts[keep, , drop = FALSE] > 0L) == ncol(counts))) {
      return(finish("skipped", "ratio_normalization_unavailable", "No retained gene is positive in every donor."))
    }
    dds <- DESeq2::DESeqDataSetFromMatrix(counts[keep, , drop = FALSE], colData = metadata, design = formula)
    if (spec$cohort == "full") {
      DESeq2::sizeFactors(dds) <- source$original_factors
      if (is.null(source$factor_source)) audit$status$normalization$original_factors_validated <- TRUE
      else audit$status$normalization$estimated_factors_validated <- TRUE
      audit$status$normalization$validation <- source$factor_validation
    } else {
      dds <- DESeq2::estimateSizeFactors(dds, type = "ratio")
    }
    factors <- DESeq2::sizeFactors(dds)
    if (any(!is.finite(factors)) || any(factors <= 0)) return(finish("error", "invalid_size_factors", "Ratio normalization yielded invalid factors."))
    audit$status$normalization$n_reference_genes <- sum(rowSums(counts[keep, , drop = FALSE] > 0L) == ncol(counts))
    write_csv(data.frame(sample_id = metadata$sample_id, size_factor = factors), file.path(directory, "size_factors.csv"))
    audit$status$fitted_model_count <- 1L
    dds <- DESeq2::DESeq(dds, test = "Wald", fitType = "parametric", sfType = "ratio",
                         betaPrior = FALSE, minReplicatesForReplace = Inf,
                         parallel = FALSE, quiet = FALSE)
    if (!"disease_AD_vs_control" %in% DESeq2::resultsNames(dds)) {
      return(finish("error", "contrast_direction_invalid", "Missing DESeq2 AD-versus-control coefficient."))
    }
    audit$status$contrast$coefficient <- "disease_AD_vs_control"
    audit$status$test$actual_dispersion_fit <- attr(DESeq2::dispersionFunction(dds), "fitType")
    result <- as.data.frame(DESeq2::results(dds, contrast = c("disease", "AD", "control"),
                                            alpha = settings$alpha, independentFiltering = FALSE,
                                            pAdjustMethod = "BH"))
    fit_metadata <- S4Vectors::mcols(dds)
    converged <- fit_metadata$betaConv
    if (is.null(converged)) return(finish("error", "missing_convergence_flags", "DESeq2 did not expose beta convergence flags."))
    valid_estimate <- !is.na(converged) & converged & is.finite(result$log2FoldChange) &
      is.finite(result$lfcSE) & result$lfcSE > 0
    valid_test <- valid_estimate & is.finite(result$pvalue) & result$pvalue >= 0 & result$pvalue <= 1
    reasons <- rep("", nrow(result))
    reasons[!is.finite(result$pvalue)] <- "pvalue_unavailable"
    if (!is.null(fit_metadata$maxCooks)) {
      cooks <- !is.na(fit_metadata$maxCooks) & fit_metadata$maxCooks > stats::qf(0.99, ncol(design), nrow(design) - ncol(design))
      reasons[!is.finite(result$pvalue) & cooks] <- "cooks_outlier"
    }
    reasons[!valid_estimate] <- "invalid_effect_or_se"
    reasons[is.na(converged) | !converged] <- "nonconverged"
    normalized_mean <- rowMeans(sweep(counts, 2L, factors, "/"))
    exported <- data.frame(gene = rownames(counts), baseMean = normalized_mean,
                            log2FoldChange = NA_real_, lfcSE = NA_real_, stat = NA_real_,
                            pvalue = NA_real_, padj = NA_real_, padj_deseq2 = NA_real_,
                            pvalue_for_bh = NA_real_, in_count_filter = keep, tested = FALSE,
                            beta_converged = NA, lfc_ci_low = NA_real_, lfc_ci_high = NA_real_,
                            filter_reason = ifelse(rowSums(counts) == 0, "all_zero", "low_count"))
    exported[keep, names(result)] <- result
    exported$padj_deseq2[keep] <- result$padj
    exported$padj[keep] <- fixed_family_bh(result$pvalue, valid_test)
    exported$pvalue_for_bh[keep] <- ifelse(valid_test, result$pvalue, 1)
    exported$tested[keep] <- valid_test
    exported$beta_converged[keep] <- converged
    exported$filter_reason[keep] <- reasons
    lower <- upper <- rep(NA_real_, nrow(result))
    lower[valid_estimate] <- result$log2FoldChange[valid_estimate] - stats::qnorm(0.975) * result$lfcSE[valid_estimate]
    upper[valid_estimate] <- result$log2FoldChange[valid_estimate] + stats::qnorm(0.975) * result$lfcSE[valid_estimate]
    exported$lfc_ci_low[keep] <- lower
    exported$lfc_ci_high[keep] <- upper
    audit$status$filter$n_valid_tests <- sum(valid_test)
    audit$status$filter$n_nonconverged <- sum(is.na(converged) | !converged)
    audit$status$filter$n_unavailable_pvalues <- sum(!is.finite(result$pvalue))
    audit$status$n_significant <- sum(exported$padj < settings$alpha, na.rm = TRUE)
    original_result_path <- file.path(source$original_comparison, "results.csv")
    if (spec$name == "full_C" && file.exists(original_result_path)) {
      original <- utils::read.csv(original_result_path, check.names = FALSE)
      if (all(c("gene", "log2FoldChange") %in% names(original)) && !anyDuplicated(original$gene)) {
        audit$status$source_effect_validation <- compare_source_effects(exported, original, original_result_path)
      }
    }
    write_csv(exported, file.path(directory, "results.csv"))
    if (isTRUE(export_expression)) {
      # Reuse this fitted object's factors and dispersion trend; never refit DE.
      transformed <- DESeq2::varianceStabilizingTransformation(dds, blind = FALSE)
      expression <- SummarizedExperiment::assay(transformed)
      if (any(!is.finite(expression))) return(finish("error", "invalid_vst", "VST produced nonfinite donor expression values."))
      normalized <- sweep(counts, 2L, factors, "/")
      write_csv(data.frame(gene = rownames(counts), normalized, check.names = FALSE),
                file.path(directory, "normalized_counts.csv"))
      write_csv(data.frame(gene = rownames(expression), expression, check.names = FALSE),
                file.path(directory, "vst_expression.csv"))
      audit$status$vst <- list(method = "varianceStabilizingTransformation", blind = FALSE,
                               source = "same fitted DESeqDataSet", n_genes = nrow(expression),
                               n_donors = ncol(expression), used_for_differential_expression = FALSE)
      audit$status$normalization$matrix_gene_scope <- "all input genes"
    }
    finish("success")
  }, warning = function(w) {
    audit$status$warnings <- I(c(audit$status$warnings, conditionMessage(w)))
    cat("WARNING:", conditionMessage(w), "\n")
    invokeRestart("muffleWarning")
  }), error = function(e) finish("error", if (!is.null(e$reason_code)) e$reason_code else "deseq2_error", conditionMessage(e)))
}

main <- function(args = commandArgs(trailingOnly = TRUE)) {
  if (length(args) != 6L) stop("Usage: models.R original_run_dir output_dir alpha min_count min_gene_donors min_group_donors")
  if (!requireNamespace("jsonlite", quietly = TRUE) || !requireNamespace("DESeq2", quietly = TRUE)) {
    stop("R packages DESeq2 and jsonlite are required.")
  }
  parameters <- suppressWarnings(as.numeric(args[3L:6L]))
  if (any(!is.finite(parameters)) || parameters[1L] <= 0 || parameters[1L] >= 1 ||
      any(parameters[2L:4L] != floor(parameters[2L:4L])) || any(parameters[2L:3L] < 1) ||
      parameters[4L] < 3 || any(parameters[2L:4L] > .Machine$integer.max)) {
    stop("Require 0 < alpha < 1, positive integer count/gene thresholds, and min_group_donors >= 3.")
  }
  source_path <- normalizePath(args[1L], mustWork = TRUE)
  output <- normalizePath(args[2L], mustWork = FALSE)
  if (output == source_path || startsWith(output, paste0(source_path, "/"))) stop("Output must be separate from the original run.")
  protected <- file.path(output, c("manifest.json", "cohort_support.csv", "models"))
  if (any(file.exists(protected))) stop("Refusing to overwrite existing sensitivity outputs.")
  if (!dir.exists(output) && !dir.create(output, recursive = TRUE)) stop("Cannot create sensitivity output directory.")
  settings <- list(alpha = parameters[1L], min_count = parameters[2L],
                   min_gene_donors = parameters[3L], min_group_donors = parameters[4L],
                   independent_filtering = FALSE, p_adjust_method = "fixed-family BH",
                   full_models_controlled = I(c("donors", "gene filter", "original ratio size factors")))
  manifest_path <- file.path(output, "manifest.json")
  manifest <- list(status = "running", source_run = source_path, output_dir = output,
                   settings = settings, model_order = I(character()), models = list(),
                   shared_studies = I(character()), started_at = format(Sys.time(), tz = "UTC", usetz = TRUE),
                   interpretation = "Sensitivity audit: adaptive independent filtering is disabled; main padj uses all count-filter retained genes, with invalid tests contributing p=1 and reported padj missing.")
  write_json(manifest, manifest_path)
  tryCatch({
    source <- read_source(source_path, settings)
    definitions <- model_specs(source$metadata)
    write_csv(definitions$support, file.path(output, "cohort_support.csv"))
    manifest$model_order <- I(vapply(definitions$models, `[[`, character(1L), "name"))
    manifest$shared_studies <- I(definitions$shared_studies)
    counts_by_group <- function(rows) as.list(table(factor(source$metadata$disease[rows], levels = c("control", "AD"))))
    manifest$full_cohort <- list(n_donors = nrow(source$metadata), group_counts = counts_by_group(rep(TRUE, nrow(source$metadata))))
    shared_rows <- source$metadata$study %in% definitions$shared_studies
    manifest$shared_cohort <- list(n_donors = sum(shared_rows), group_counts = counts_by_group(shared_rows), n_studies = length(definitions$shared_studies))
    manifest$original_factor_validation <- source$factor_validation
    manifest$source_files <- source$source_files
    manifest$versions <- list(R = R.version.string, DESeq2 = as.character(utils::packageVersion("DESeq2")), jsonlite = as.character(utils::packageVersion("jsonlite")))
    manifest$library_paths <- I(.libPaths())
    write_json(manifest, manifest_path)
    for (spec in definitions$models) {
      cat("[Sensitivity]", spec$name, "\n")
      flush.console()
      status <- fit_model(spec, source, definitions$shared_studies, settings, file.path(output, "models", spec$name))
      manifest$models[[spec$name]] <- status
      write_json(manifest, manifest_path)
      cat("[Sensitivity]", spec$name, status$status, "\n")
      flush.console()
    }
    states <- vapply(manifest$models, `[[`, character(1L), "status")
    manifest$status <- if (all(states == "success")) "success" else "partial"
    manifest$completed_at <- format(Sys.time(), tz = "UTC", usetz = TRUE)
    write_json(manifest, manifest_path)
    if (any(states == "error")) 1L else 0L
  }, error = function(e) {
    manifest$status <- "error"
    manifest$reason_code <- if (!is.null(e$reason_code)) e$reason_code else "sensitivity_error"
    manifest$reason <- conditionMessage(e)
    manifest$details <- e$details
    manifest$completed_at <- format(Sys.time(), tz = "UTC", usetz = TRUE)
    write_json(manifest, manifest_path)
    message(conditionMessage(e))
    1L
  })
}

if (sys.nframe() == 0L) {
  exit_code <- tryCatch(main(), error = function(e) { message(conditionMessage(e)); 1L })
  quit(save = "no", status = exit_code)
}

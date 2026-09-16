#!/usr/bin/env Rscript
# Standalone limma DE and panels migrated from casestudy_panels.Rmd.
# Rscript run_casestudy.R [--stage de|plots|all] <session_dir> [out_dir]
# COMPOSITE, WHICH_PC and PERMUTATIONS control plotting; PSEUDOBULK defaults off
# and can only enable the legacy donor contrast with an explicit --stage all.

parse_de_config <- function(args) {
  allowed <- c("stage", "ref-csv", "alt-csv", "ref-label", "alt-label",
               "input-scale", "de-dir", "state-file", "diagnostics-file",
               "de-fdr", "top-n", "final-out-dir", "final-de-dir", "volcano-dir")
  options <- list()
  positional <- character()
  i <- 1L
  while (i <= length(args)) {
    if (startsWith(args[i], "--")) {
      key <- substring(args[i], 3L)
      if (!key %in% allowed || i == length(args) || startsWith(args[i + 1L], "--")) {
        stop("Unknown option or missing value: ", args[i], call. = FALSE)
      }
      if (!is.null(options[[key]])) stop("Duplicate option: --", key, call. = FALSE)
      options[[key]] <- args[i + 1L]
      i <- i + 2L
    } else {
      positional <- c(positional, args[i])
      i <- i + 1L
    }
  }
  if (length(positional) < 1L || length(positional) > 2L) {
    stop("Usage: Rscript run_casestudy.R [--stage de|plots|all] <session_dir> [out_dir]", call. = FALSE)
  }
  get_option <- function(name, default) {
    if (is.null(options[[name]])) default else options[[name]]
  }
  session_dir <- normalizePath(positional[1], mustWork = TRUE)
  pick_csv <- function(option, pattern) {
    path <- options[[option]]
    if (is.null(path)) {
      hits <- list.files(session_dir, pattern = pattern, full.names = TRUE)
      if (length(hits) != 1L) stop("Expected one input for --", option,
                                  "; found ", length(hits), call. = FALSE)
      path <- hits
    } else if (!startsWith(path, "/")) {
      path <- file.path(session_dir, path)
    }
    normalizePath(path, mustWork = TRUE)
  }
  out_dir <- if (length(positional) == 2L) positional[2] else file.path(session_dir, "casestudy_R")
  config <- list(
    stage = get_option("stage", "all"), session_dir = session_dir,
    ref_csv = pick_csv("ref-csv", "^foranalysis_combined_normal_df_.*\\.csv$"),
    alt_csv = pick_csv("alt-csv", "^foranalysis_combined_disease_df_.*\\.csv$"),
    ref_label = get_option("ref-label", "Healthy"),
    alt_label = get_option("alt-label", "Diseased"),
    input_scale = get_option("input-scale", "linear_cp10k"),
    out_dir = out_dir,
    volcano_dir = get_option("volcano-dir", file.path(session_dir, "volcano_plots")),
    de_dir = get_option("de-dir", file.path(session_dir, "differential_expression")),
    final_out_dir = get_option("final-out-dir", out_dir),
    final_de_dir = get_option("final-de-dir", get_option("de-dir", file.path(session_dir, "differential_expression"))),
    state_file = get_option("state-file", file.path(out_dir, "analysis_state.rds")),
    diagnostics_file = get_option("diagnostics-file", NULL),
    de_fdr = suppressWarnings(as.numeric(get_option("de-fdr", "0.05"))),
    top_n = suppressWarnings(as.numeric(get_option("top-n", "1000")))
  )
  if (!config$stage %in% c("de", "plots", "all")) stop("--stage must be de, plots or all.", call. = FALSE)
  if (!is.finite(config$de_fdr) || config$de_fdr < 0 || config$de_fdr > 1) {
    stop("--de-fdr must be between 0 and 1.", call. = FALSE)
  }
  if (!is.finite(config$top_n) || config$top_n < 1 || config$top_n != floor(config$top_n)) {
    stop("--top-n must be a positive integer.", call. = FALSE)
  }
  labels <- c(config$ref_label, config$alt_label)
  if (any(!nzchar(trimws(labels))) || anyDuplicated(make.names(labels))) {
    stop("Reference and alternate labels must be nonempty and distinct R model names.", call. = FALSE)
  }
  if (!config$input_scale %in% c("linear_cp10k", "log2_already")) {
    stop("--input-scale must be linear_cp10k or log2_already.", call. = FALSE)
  }
  if (!is.null(config$diagnostics_file) && !file.exists(config$diagnostics_file)) {
    stop("Diagnostics file not found: ", config$diagnostics_file, call. = FALSE)
  }
  config
}

read_inputs <- function(config) {
  if (!requireNamespace("data.table", quietly = TRUE)) stop("R package data.table is required.")
  read_expr <- function(path) {
    dt <- data.table::fread(path)
    if (ncol(dt) < 3L) stop("Each group needs a gene column and at least two cells: ", path)
    if (!names(dt)[1] %in% c("Name", "Gene") && is.numeric(dt[[1]])) {
      stop("First column must contain gene identifiers: ", path)
    }
    genes <- as.character(dt[[1]])
    if (anyNA(genes) || any(!nzchar(trimws(genes))) || anyDuplicated(genes)) {
      stop("Gene identifiers must be nonempty and unique: ", path)
    }
    dt[, 1 := NULL]
    if (!all(vapply(dt, is.numeric, logical(1)))) stop("Expression columns must be numeric: ", path)
    mat <- as.matrix(dt)
    rownames(mat) <- genes
    storage.mode(mat) <- "numeric"
    if (any(is.infinite(mat)) || any(mat < 0, na.rm = TRUE)) {
      stop("Expression values must be nonnegative, without infinities: ", path)
    }
    mat
  }
  mat_ref <- read_expr(config$ref_csv)
  mat_alt <- read_expr(config$alt_csv)

  # casestudy_panels.Rmd: scale-check. Retain its median-column-sum check.
  if (identical(config$input_scale, "linear_cp10k")) {
    med_ref <- median(colSums(mat_ref, na.rm = TRUE))
    med_alt <- median(colSums(mat_alt, na.rm = TRUE))
    ok <- abs(med_ref - 1e4) / 1e4 < 0.01 && abs(med_alt - 1e4) / 1e4 < 0.01
    if (!ok) stop(sprintf("Expected linear CP10K; median column sums are %.1f / %.1f.", med_ref, med_alt))
  }

  # casestudy_panels.Rmd: combine. Reference columns precede alternate columns.
  common_genes <- intersect(rownames(mat_ref), rownames(mat_alt))
  if (length(common_genes) < 10L) stop("Too few overlapping genes between files.")
  mat_ref <- mat_ref[common_genes, , drop = FALSE]
  mat_alt <- mat_alt[common_genes, , drop = FALSE]
  expr <- cbind(mat_ref, mat_alt)
  group <- factor(c(rep(config$ref_label, ncol(mat_ref)), rep(config$alt_label, ncol(mat_alt))),
                  levels = c(config$ref_label, config$alt_label))
  colnames(expr) <- make.unique(colnames(expr), sep = "_")
  if (identical(config$input_scale, "linear_cp10k")) expr <- log2(expr + 1)
  expr <- expr[rowSums(is.finite(expr)) >= ncol(expr) * 0.8, , drop = FALSE]
  v <- apply(expr, 1, stats::var, na.rm = TRUE)
  expr <- expr[is.finite(v) & v > 0, , drop = FALSE]
  if (nrow(expr) < 2L) stop("Too few variable genes for limma.")
  if (any(rowSums(is.finite(expr[, group == config$ref_label, drop = FALSE])) < 2L) ||
      any(rowSums(is.finite(expr[, group == config$alt_label, drop = FALSE])) < 2L)) {
    stop("Retained genes need at least two finite observations in each group.")
  }
  message(sprintf("DE input: %d tested genes, %s n=%d, %s n=%d",
                  nrow(expr), config$ref_label, ncol(mat_ref), config$alt_label, ncol(mat_alt)))
  list(expr = expr, group = group, mat_ref = mat_ref, mat_alt = mat_alt,
       input_md5 = tools::md5sum(c(config$ref_csv, config$alt_csv)))
}

run_de <- function(inputs, config) {
  if (!requireNamespace("limma", quietly = TRUE)) stop("R package limma is required.")
  if (!requireNamespace("statmod", quietly = TRUE)) stop("R package statmod is required for robust limma.")
  expr <- inputs$expr
  group <- inputs$group
  # Verbatim statistical sequence from casestudy_panels.Rmd's de chunk.
  design <- model.matrix(~ 0 + group)
  colnames(design) <- make.names(levels(group))
  contrast_str <- paste(make.names(config$alt_label), "-", make.names(config$ref_label))
  cm <- limma::makeContrasts(contrasts = contrast_str, levels = design)
  fit <- limma::lmFit(expr, design)
  fit2 <- limma::contrasts.fit(fit, cm)
  fit2 <- limma::eBayes(fit2, robust = TRUE, trend = TRUE)
  res <- limma::topTable(fit2, coef = 1, number = Inf, sort.by = "P", adjust.method = "BH")
  res$Gene <- rownames(res)
  res <- res[, c("Gene", "logFC", "AveExpr", "t", "P.Value", "adj.P.Val", "B")]
  if (any(!is.finite(as.matrix(res[, c("logFC", "P.Value", "adj.P.Val")])))) {
    stop("limma returned non-finite DE statistics.")
  }
  list(schema_version = 1L, config = config, inputs = inputs, native = res,
       contrast = contrast_str, limma_version = as.character(utils::packageVersion("limma")))
}

export_results <- function(state, config) {
  res <- state$native
  # Preserve the original Python input order, restricted to R-tested genes.
  genes <- rownames(state$inputs$expr)
  res <- res[match(genes, res$Gene), , drop = FALSE]
  ref <- state$inputs$mat_ref[genes, , drop = FALSE]
  alt <- state$inputs$mat_alt[genes, , drop = FALSE]
  if (identical(config$input_scale, "log2_already")) {
    ref <- 2^ref - 1
    alt <- 2^alt - 1
    if (any(is.infinite(ref)) || any(is.infinite(alt))) stop("Cannot recover finite linear expression for effect_size.")
  }
  # Compatibility-only Cohen's d; does not affect limma, filtering, or FDR.
  n_ref <- rowSums(is.finite(ref))
  n_alt <- rowSums(is.finite(alt))
  pooled_sd <- sqrt(((n_ref - 1) * apply(ref, 1, stats::var, na.rm = TRUE) +
                    (n_alt - 1) * apply(alt, 1, stats::var, na.rm = TRUE)) /
                   (n_ref + n_alt - 2))
  cohens_d <- (rowMeans(alt, na.rm = TRUE) - rowMeans(ref, na.rm = TRUE)) / (pooled_sd + 1e-8)
  compat <- data.frame(Name = res$Gene, log2_fold_change = res$logFC,
                       effect_size = unname(cohens_d), p_value = res$P.Value,
                       FDR = res$adj.P.Val, is_significant = res$adj.P.Val < config$de_fdr,
                       abs_log2_fc = abs(res$logFC), check.names = FALSE)
  if (any(!is.finite(compat$effect_size))) stop("Cannot export non-finite effect_size.")
  significant <- compat[compat$is_significant, , drop = FALSE]
  top <- function(table, field, decreasing = FALSE) {
    head(table[order(table[[field]], decreasing = decreasing), , drop = FALSE], config$top_n)
  }
  tables <- list(
    unpaired_differential_expression_results = compat,
    significant_genes_by_fdr = top(significant, "FDR"),
    significant_genes_by_fc = top(significant, "abs_log2_fc", TRUE),
    significant_upregulated_genes = top(significant[significant$log2_fold_change > 0, , drop = FALSE], "p_value"),
    significant_downregulated_genes = top(significant[significant$log2_fold_change < 0, , drop = FALSE], "p_value")
  )
  diagnostics <- if (is.null(config$diagnostics_file)) character() else
    readLines(config$diagnostics_file, warn = FALSE, encoding = "UTF-8")
  for (directory in unique(c(config$out_dir, config$de_dir, dirname(config$state_file)))) {
    if (!dir.exists(directory) && !dir.create(directory, recursive = TRUE)) stop("Cannot create output directory: ", directory)
  }
  write_csv <- function(table, path, comments = character()) {
    temporary <- tempfile(pattern = ".de-", tmpdir = dirname(path))
    on.exit(unlink(temporary), add = TRUE)
    # Binary mode avoids a second locale-dependent conversion after UTF-8
    # encoding; in LC_CTYPE=C a text connection can corrupt diagnostic bytes.
    connection <- file(temporary, open = "wb")
    tryCatch({
      if (length(comments)) writeLines(enc2utf8(paste0("# ", comments)), connection, useBytes = TRUE)
      utils::write.table(table, connection, sep = ",", row.names = FALSE, col.names = TRUE,
                         quote = TRUE, na = "", qmethod = "double")
    }, finally = close(connection))
    if (!file.rename(temporary, path)) stop("Cannot write output: ", path)
  }
  for (name in names(tables)) write_csv(tables[[name]], file.path(config$de_dir, paste0(name, ".csv")), diagnostics)
  write_csv(state$native, file.path(config$out_dir, "DE_results_table.csv"))
  state$compat <- compat
  # Python validates files in a staging directory before moving them. Keep
  # reusable state pointed at the published paths, not deleted temp locations.
  state$config$out_dir <- config$final_out_dir
  state$config$de_dir <- config$final_de_dir
  if (!identical(config$final_out_dir, config$out_dir)) {
    state$config$state_file <- file.path(config$final_out_dir, basename(config$state_file))
  }
  state$diagnostics_text <- diagnostics
  state$config$diagnostics_file <- NULL
  temporary <- tempfile(pattern = ".state-", tmpdir = dirname(config$state_file))
  on.exit(unlink(temporary), add = TRUE)
  saveRDS(state, temporary)
  if (!file.rename(temporary, config$state_file)) stop("Cannot write analysis state.")
  message(sprintf("DE complete: %d tested genes, %d significant; contrast %s",
                  nrow(compat), nrow(significant), state$contrast))
  invisible(state)
}

env_flag <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (!nzchar(value)) default else tolower(value) %in% c("1", "true", "yes", "t")
}

validate_plot_state <- function(state, config) {
  if (!identical(state$schema_version, 1L) || is.null(state$compat)) stop("Unsupported analysis state.")
  for (key in c("ref_label", "alt_label", "input_scale")) {
    if (!identical(state$config[[key]], config[[key]])) stop("Analysis state mismatch: ", key)
  }
  current <- tools::md5sum(c(config$ref_csv, config$alt_csv))
  if (!identical(unname(current), unname(state$inputs$input_md5))) {
    stop("Analysis state mismatch: source matrix hashes. Run --stage de again.")
  }
  if (!identical(levels(state$inputs$group), c(config$ref_label, config$alt_label))) {
    stop("Analysis state mismatch: group labels.")
  }
  invisible(state)
}

run_plots <- function(state, config) {
  manifest <- file.path(config$out_dir, "plot_manifest.json")
  unlink(manifest)
  validate_plot_state(state, config)
  if (identical(config$stage, "plots") && env_flag("PSEUDOBULK", FALSE)) {
    stop("PSEUDOBULK=true cannot run during --stage plots; use --stage all explicitly.")
  }
  permutations <- suppressWarnings(as.numeric(Sys.getenv("PERMUTATIONS", "999")))
  if (!is.finite(permutations) || permutations < 1 || permutations != floor(permutations)) {
    stop("PERMUTATIONS must be a positive integer.")
  }
  which_pc <- Sys.getenv("WHICH_PC", "PC1")
  if (!which_pc %in% c("PC1", "PC2")) stop("WHICH_PC must be PC1 or PC2.")
  for (path in config$out_dir) dir.create(path, recursive = TRUE, showWarnings = FALSE)
  config$out_dir <- normalizePath(config$out_dir, mustWork = TRUE)
  params <- list(out_dir = config$out_dir, out_prefix = "DE_results", ref_label = config$ref_label,
                 alt_label = config$alt_label, fdr_thresh = 0.05, fc_thresh = 1,
                 n_corrgram = 20, n_pca = 100, permutations = permutations,
                 which_pc = which_pc, composite = env_flag("COMPOSITE", TRUE))
  suppressPackageStartupMessages({
    library(ggplot2); library(ggrepel); library(patchwork); library(RColorBrewer)
    library(data.table); library(corrgram); library(vegan); library(multcompView); library(ggpubr)
  })
  out_stem <- file.path(config$out_dir, params$out_prefix)
  fdr_thresh <- params$fdr_thresh
  fc_thresh <- params$fc_thresh
  expr <- state$inputs$expr
  group <- state$inputs$group
  res <- state$native
  transform_applied <- identical(config$input_scale, "linear_cp10k")
  # The following calculations and panels retain the original Rmd code.

  # pca-permanova
top_genes_pca <- head(res$Gene[order(res$adj.P.Val)], n = min(params$n_pca, nrow(res)))

pca <- prcomp(t(expr[top_genes_pca, , drop = FALSE]), scale. = TRUE)
pve <- (pca$sdev^2) / sum(pca$sdev^2)

pcoadata <- data.frame(
  sample  = colnames(expr),
  PC1     = pca$x[, 1],
  PC2     = pca$x[, 2],
  Subtype = group
)

pal2 <- setNames(colorRampPalette(brewer.pal(12, "Paired"))(2),
                 c(params$ref_label, params$alt_label))

# Bray-Curtis needs non-negative input; log2(CP10K + 1) satisfies that.
dist_mat   <- vegdist(t(expr[top_genes_pca, , drop = FALSE]), method = "bray")
otu.adonis <- adonis2(dist_mat ~ Subtype, data = pcoadata,
                      permutations = params$permutations)

# A permutation test cannot resolve p below 1/(permutations + 1). Report the
# floor as an inequality rather than as a measured probability.
perm_floor <- 1 / (params$permutations + 1)
p_obs <- otu.adonis$`Pr(>F)`[1]
perm_p_label <- if (!is.na(p_obs) && p_obs <= perm_floor + 1e-12) {
  paste0("p <= ", format(perm_floor, scientific = FALSE))
} else {
  paste0("p = ", signif(p_obs, 3))
}
message("PERMANOVA (", params$permutations, " permutations): R2 = ",
        round(otu.adonis$R2[1], 5), ", ", perm_p_label)

  # panel-a-volcano
res$neglog10AdjP <- -log10(pmax(res$adj.P.Val, .Machine$double.xmin))
res$status <- "NS"
res$status[res$adj.P.Val < fdr_thresh & res$logFC >=  fc_thresh] <- "Up"
res$status[res$adj.P.Val < fdr_thresh & res$logFC <= -fc_thresh] <- "Down"
res$status <- factor(res$status, levels = c("Down", "NS", "Up"))
res$is_sig <- with(res, adj.P.Val < fdr_thresh & abs(logFC) >= fc_thresh)

up_n   <- sum(res$status == "Up", na.rm = TRUE)
down_n <- sum(res$status == "Down", na.rm = TRUE)

lab_df <- res[order(res$adj.P.Val), ][seq_len(min(20, nrow(res))), ]
x_min <- min(res$logFC, na.rm = TRUE); x_max <- max(res$logFC, na.rm = TRUE)
y_max <- max(res$neglog10AdjP, na.rm = TRUE); x_span <- x_max - x_min

fc_axis_lab <- if (transform_applied) {
  paste0("log2 Fold Change (", params$alt_label, " / ", params$ref_label, ")")
} else {
  paste0("Mean difference, input scale (", params$alt_label, " - ", params$ref_label, ")")
}

volcano <- ggplot(res, aes(x = logFC, y = neglog10AdjP)) +
  geom_vline(xintercept = -fc_thresh, color = "steelblue3", linetype = "dashed") +
  geom_vline(xintercept =  fc_thresh, color = "red3",       linetype = "dashed") +
  geom_point(aes(color = status, shape = is_sig), alpha = 0.85, size = 2) +
  scale_color_manual(values = c(Down = "steelblue3", NS = "grey70", Up = "red3")) +
  scale_shape_manual(values = c(`FALSE` = 16, `TRUE` = 17),
                     labels = c("Not sig", "Significant")) +
  # ylim keeps repelled labels out of the band the two count annotations occupy;
  # without it they collide as soon as a top gene sits near the y maximum.
  geom_text_repel(data = lab_df, aes(label = Gene), size = 3, max.overlaps = 200,
                  min.segment.length = 0, box.padding = 0.3, point.padding = 0.2,
                  ylim = c(NA, y_max * 0.88)) +
  annotate("text", x = x_min + 0.02 * x_span, y = y_max * 0.98,
           label = paste0("Up (FC >= ", fc_thresh, ", FDR < ", fdr_thresh, "): ", up_n),
           hjust = 0, vjust = 1, color = "red3", fontface = "bold") +
  annotate("text", x = x_min + 0.02 * x_span, y = y_max * 0.92,
           label = paste0("Down (FC <= -", fc_thresh, ", FDR < ", fdr_thresh, "): ", down_n),
           hjust = 0, vjust = 1, color = "steelblue3", fontface = "bold") +
  labs(title = paste0("Volcano plot: ", params$alt_label, " vs ", params$ref_label),
       x = fc_axis_lab, y = expression(-log[10]("BH-adjusted P"))) +
  theme_bw() +
  theme(plot.title = element_text(face = "bold"), legend.position = "top",
        aspect.ratio = 1) +
  coord_cartesian(clip = "off")

ggsave(paste0(out_stem, "_volcano.png"), volcano, width = 5.5, height = 5.5, dpi = 300,
       device = ragg::agg_png)
ggsave(paste0(out_stem, "_volcano.pdf"), volcano, width = 5.5, height = 5.5)
volcano

  # panel-b-corrgram
top_genes_cg <- head(res$Gene[order(res$adj.P.Val)], n = min(params$n_corrgram, nrow(res)))
df_cells <- as.data.frame(t(expr[top_genes_cg, , drop = FALSE]))

# Reverse corrgram's default palette so negative correlations are blue and
# positive correlations are red; white remains the midpoint at zero.
corr_palette <- colorRampPalette(c("#2166AC", "#92C5DE", "#F7F7F7",
                                   "#F4A582", "#B2182B"))

draw_corrgram_key <- function() {
  op <- par(no.readonly = TRUE)
  on.exit(par(op), add = TRUE)

  # This strip sits wholly inside the enlarged bottom outer margin. Keeping a
  # white gutter between it and the matrix prevents the key from competing with
  # the lowest row of cells or the diagonal gene labels.
  par(fig = c(0.08, 0.92, 0.005, 0.115), mar = rep(0, 4),
      new = TRUE, xpd = NA, pty = "m")
  plot.new()
  plot.window(xlim = c(-1, 1), ylim = c(0, 1), xaxs = "i", yaxs = "i")

  key_breaks <- seq(-1, 1, length.out = 201)
  rect(key_breaks[-length(key_breaks)], 0.44,
       key_breaks[-1], 0.70,
       col = corr_palette(length(key_breaks) - 1), border = NA)
  rect(-1, 0.44, 1, 0.70, border = "grey35", lwd = 0.5)

  key_ticks <- c(-1, -0.5, 0, 0.5, 1)
  segments(key_ticks, 0.39, key_ticks, 0.44, col = "grey25", lwd = 0.5)
  text(key_ticks, 0.30, labels = c("-1", "-0.5", "0", "0.5", "1"),
       cex = 0.60)
  text(0, 0.84, "Pearson correlation (r)", font = 2, cex = 0.68)
  text(0, 0.12,
       "Square: color/intensity = r   |   Circle: fill = |r|   |   Diagonal: gene symbol",
       cex = 0.62)
}

# Wrapped in a function so section 9 can re-draw it into a captured device.
# par(pty = "s") keeps the correlation grid square; on.exit restores it even if
# corrgram errors partway through.
draw_corrgram <- function() {
  op <- par(no.readonly = TRUE)
  on.exit(par(op), add = TRUE)
  par(pty = "s")
  corrgram(df_cells, order = TRUE,
           lower.panel = panel.shade, upper.panel = panel.pie, text.panel = panel.txt,
           col.regions = corr_palette,
           cor.method = "pearson", label.srt = -45, cex.labels = 0.65,
           oma = c(7.5, 4, 4.5, 4), cex.main = 1,
           main = "Gene-expression correlations")
  draw_corrgram_key()
}

ragg::agg_png(paste0(out_stem, "_corrgram_genes.png"), width = 1000, height = 1000, res = 150)
draw_corrgram()
dev.off()

pdf(paste0(out_stem, "_corrgram_genes.pdf"), width = 6.7, height = 6.7)
draw_corrgram()
dev.off()


  # panel-source-gene-csvs
# Artifact-only export: read existing panel inputs, create new tables, and write
# CSV files without modifying any plot object or source data.
local({
gene_full_names <- function(genes) {
  if (!requireNamespace("AnnotationDbi", quietly = TRUE) ||
      !requireNamespace("org.Hs.eg.db", quietly = TRUE)) {
    return(rep(NA_character_, length(genes)))
  }
  # Synthetic identifiers and non-human symbols may have no annotation keys.
  mapped <- tryCatch(suppressMessages(AnnotationDbi::mapIds(
    org.Hs.eg.db::org.Hs.eg.db, keys = unique(genes), keytype = "SYMBOL",
    column = "GENENAME", multiVals = "first"
  )), error = function(e) setNames(rep(NA_character_, length(genes)), genes))
  unname(mapped[genes])
}

gene_source_rows <- function(genes) {
  rows <- as.data.table(copy(res[match(genes, res$Gene), , drop = FALSE]))
  data.table(
    gene_symbol = genes,
    gene_full_name = gene_full_names(genes),
    logFC = rows$logFC,
    AveExpr = rows$AveExpr,
    P.Value = rows$P.Value,
    adj.P.Val = rows$adj.P.Val,
    status = as.character(rows$status),
    is_significant = rows$is_sig
  )
}

panel_a_genes <- gene_source_rows(lab_df$Gene)
panel_a_genes[, label_rank := seq_len(.N)]
setcolorder(panel_a_genes, c("label_rank", "gene_symbol", "gene_full_name"))
fwrite(panel_a_genes, paste0(out_stem, "_panel_A_labeled_genes.csv"))

corr_for_order <- cor(df_cells, use = "pairwise.complete.obs", method = "pearson")
corr_eigen <- eigen(corr_for_order)$vectors[, 1:2, drop = FALSE]
corr_angle <- ifelse(corr_eigen[, 1] > 0,
                     atan(corr_eigen[, 2] / corr_eigen[, 1]),
                     atan(corr_eigen[, 2] / corr_eigen[, 1]) + pi)
corr_display_genes <- colnames(corr_for_order)[order(corr_angle)]

panel_b_genes <- gene_source_rows(corr_display_genes)
panel_b_genes[, `:=`(
  display_order = seq_len(.N),
  selection_rank = match(gene_symbol, top_genes_cg)
)]
setcolorder(panel_b_genes,
            c("display_order", "selection_rank", "gene_symbol", "gene_full_name"))
fwrite(panel_b_genes, paste0(out_stem, "_panel_B_corrgram_genes.csv"))
})

  # panel-c-violin
rt <- data.frame(
  id         = pcoadata$sample,
  Type       = factor(pcoadata$Subtype,
                      levels = c(params$ref_label, params$alt_label)),
  Expression = pcoadata[[params$which_pc]]
)

wilcox_fit <- wilcox.test(Expression ~ Type, data = rt, exact = FALSE)
wilcox_p <- wilcox_fit$p.value
wilcox_p_label <- if (is.finite(wilcox_p) && wilcox_p > 0) {
  formatC(wilcox_p, format = "e", digits = 2)
} else {
  paste0("< ", formatC(.Machine$double.xmin, format = "e", digits = 2))
}

y_range <- range(rt$Expression, na.rm = TRUE)
y_span <- diff(y_range)
if (!is.finite(y_span) || y_span == 0) y_span <- max(abs(y_range), 1)

comparison_df <- data.frame(
  group1 = params$ref_label,
  group2 = params$alt_label,
  y.position = y_range[2] + 0.08 * y_span,
  label = paste0("Cell-level Wilcoxon p = ", wilcox_p_label)
)

group_counts <- table(rt$Type)
group_axis_labels <- setNames(
  paste0(names(group_counts), "\n(n cells = ",
         format(as.integer(group_counts), big.mark = ",", trim = TRUE), ")"),
  names(group_counts)
)

violin <- ggviolin(rt, x = "Type", y = "Expression", fill = "Type",
                   xlab = "Disease status", ylab = paste0(params$which_pc, " score"),
                   trim = TRUE, add = "boxplot",
                   add.params = list(fill = "white", width = 0.15, color = "black")) +
  stat_pvalue_manual(comparison_df, label = "label", tip.length = 0.01,
                     bracket.size = 0.5, size = 3.5) +
  scale_fill_manual(values = pal2[levels(rt$Type)], drop = FALSE) +
  scale_x_discrete(labels = group_axis_labels) +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.18))) +
  theme_bw() +
  theme(axis.text = element_text(color = "black", face = "bold"),
        legend.position = "none")

out_violin <- paste0(out_stem, "_violin_", params$which_pc, "_2groups.pdf")
ggsave(out_violin, violin, width = 6, height = 4)
message("Saved: ", out_violin)
violin

  # panel-d-pca
p_main <- ggplot(pcoadata, aes(PC1, PC2)) +
  geom_point(aes(colour = Subtype, fill = Subtype), size = 4) +
  scale_color_manual(values = pal2) +
  labs(x = paste0("(PC1: ", round(pve[1] * 100, 2), "%)"),
       y = paste0("(PC2: ", round(pve[2] * 100, 2), "%)")) +
  geom_vline(xintercept = 0, linetype = "dotted") +
  geom_hline(yintercept = 0, linetype = "dotted") +
  theme_bw() +
  theme(
    panel.background = element_rect(fill = "white", colour = "black"),
    axis.title.x = element_text(colour = "black", size = 12, margin = margin(t = 5), face = "bold"),
    axis.title.y = element_text(colour = "black", size = 12, margin = margin(r = 5), face = "bold"),
    axis.text = element_text(color = "black", face = "bold"),
    plot.title = element_blank(),
    legend.title = element_blank(),
    legend.key = element_blank(),
    legend.text = element_text(color = "black", size = 9, face = "bold"),
    legend.spacing.x = unit(0.1, "cm"),
    legend.key.width = unit(0.5, "cm"),
    legend.key.height = unit(0.5, "cm"),
    legend.background = element_blank(),
    legend.box.background = element_rect(colour = "black"),
    legend.position = "inside",
    legend.position.inside = c(0.001, 0.999),
    legend.justification = c(0.0001, 1),
    aspect.ratio = 1
  )

cld_table <- function(pc) {
  # Tukey comparison names use '-' as a separator. Keep display labels out
  # of that encoding so disease names containing '-' remain unambiguous.
  display_groups <- levels(factor(pcoadata$Subtype))
  group_codes <- paste0("group", seq_along(display_groups))
  coded <- pcoadata
  coded$Subtype <- factor(group_codes[match(as.character(coded$Subtype), display_groups)],
                          levels = group_codes)
  f   <- aov(as.formula(paste(pc, "~ Subtype")), data = coded)
  cld <- multcompLetters4(f, TukeyHSD(f))
  dt  <- aggregate(as.formula(paste(pc, "~ Subtype")), data = pcoadata, FUN = max)
  names(dt)[2] <- "value_max"
  letters_df <- as.data.frame.list(cld$Subtype)
  letters_df$Subtype <- display_groups[match(rownames(letters_df), group_codes)]
  merge(letters_df, dt, by = "Subtype", all.x = TRUE)
}
text1 <- cld_table("PC1")
text2 <- cld_table("PC2")

box_theme <- theme_bw() +
  theme(panel.background = element_rect(fill = "white", colour = "black"),
        axis.text = element_blank(), axis.ticks = element_blank(),
        plot.title = element_blank(), legend.position = "none")

# after_stat() replaces the retired ..lower.. / ..ymin.. syntax; linewidth
# replaces size for line-based geoms (both deprecated since ggplot2 3.4).
p_top <- ggplot(pcoadata, aes(Subtype, PC1, fill = Subtype)) +
  geom_boxplot(outlier.shape = NA, width = 0.5, color = "black", linetype = "dotted") +
  stat_boxplot(aes(ymin = after_stat(lower), ymax = after_stat(upper)),
               outlier.shape = NA, width = 0.5) +
  stat_boxplot(geom = "errorbar", aes(ymin = after_stat(ymax)), width = 0.2, linewidth = 0.35) +
  stat_boxplot(geom = "errorbar", aes(ymax = after_stat(ymin)), width = 0.2, linewidth = 0.35) +
  geom_text(data = text1, aes(label = Letters, y = value_max),
            angle = -90, color = "black", size = 4, vjust = -0.4) +
  scale_fill_manual(values = pal2) + labs(x = NULL, y = NULL) +
  box_theme + coord_flip()

p_right <- ggplot(pcoadata, aes(Subtype, PC2, fill = Subtype)) +
  geom_boxplot(outlier.shape = NA, width = 0.5, color = "black", linetype = "dotted") +
  stat_boxplot(aes(ymin = after_stat(lower), ymax = after_stat(upper)),
               outlier.shape = NA, width = 0.5) +
  stat_boxplot(geom = "errorbar", aes(ymin = after_stat(ymax)), width = 0.2, linewidth = 0.35) +
  stat_boxplot(geom = "errorbar", aes(ymax = after_stat(ymin)), width = 0.2, linewidth = 0.35) +
  geom_text(data = text2, aes(label = Letters, y = value_max),
            angle = 0, color = "black", size = 4, vjust = -0.4) +
  scale_fill_manual(values = pal2) + labs(x = NULL, y = NULL) +
  box_theme

p_stat <- ggplot() +
  geom_text(aes(x = 0, y = 0.1,
                label = paste("PERMANOVA",
                              paste0("df = ", otu.adonis$Df[1]),
                              paste0("R2 = ", round(otu.adonis$R2[1], 3)),
                              perm_p_label,
                              paste0("perms = ", params$permutations),
                              sep = "\n")),
            size = 3.5, color = "black", fontface = "bold") +
  theme_bw() +
  theme(panel.background = element_rect(fill = "white", colour = "black"),
        axis.title = element_blank(), axis.ticks = element_blank(),
        axis.text = element_blank(), plot.title = element_blank(),
        legend.position = "none")

final_plot <- p_top + p_stat + p_main + p_right +
  plot_layout(heights = c(1, 4), widths = c(4, 1), ncol = 2, nrow = 2)

ggsave(paste0(out_stem, "_PCA_panel.png"), final_plot, width = 6, height = 6, dpi = 300,
       device = ragg::agg_png)
ggsave(paste0(out_stem, "_PCA_panel.pdf"), final_plot, width = 6, height = 6)
final_plot

if (params$composite) {
  # ggplotify reads base par() before opening its capture device. Supply a
  # non-file device so standalone use does not implicitly create Rplots.pdf.
  pdf(NULL)
  capture_device <- dev.cur()
  on.exit(if (capture_device %in% dev.list()) dev.off(capture_device), add = TRUE)
panel_b <- ggplotify::as.ggplot(function() draw_corrgram())
  dev.off(capture_device)
panel_d <- wrap_elements(final_plot)

figure_abcd <- (volcano | panel_b | violin | panel_d) +
  plot_annotation(tag_levels = "A") &
  theme(plot.tag = element_text(face = "bold", size = 18))

ggsave(paste0(out_stem, "_figure_ABCD.png"), figure_abcd,
       width = 22, height = 5.5, dpi = 300, limitsize = FALSE,
       device = ragg::agg_png)
ggsave(paste0(out_stem, "_figure_ABCD.pdf"), figure_abcd,
       width = 22, height = 5.5, limitsize = FALSE)
message("Composite written to: ", paste0(out_stem, "_figure_ABCD.{png,pdf}"))

}

  files <- c(paste0(out_stem, "_volcano.", c("png", "pdf")),
             paste0(out_stem, "_corrgram_genes.", c("png", "pdf")),
             paste0(out_stem, "_PCA_panel.", c("png", "pdf")), out_violin,
             paste0(out_stem, "_panel_A_labeled_genes.csv"),
             paste0(out_stem, "_panel_B_corrgram_genes.csv"))
  if (params$composite) files <- c(files, paste0(out_stem, "_figure_ABCD.", c("png", "pdf")))
  files <- unique(normalizePath(files, mustWork = TRUE))
  if (any(file.info(files)$size <= 0)) stop("Empty plot output.")
  temporary <- tempfile(pattern = ".manifest-", tmpdir = config$out_dir)
  on.exit(unlink(temporary), add = TRUE)
  jsonlite::write_json(list(status = "success", files = as.list(files)),
                      temporary, auto_unbox = TRUE, pretty = TRUE)
  if (!file.rename(temporary, manifest)) stop("Cannot publish plot manifest.")
  message("Plots complete: ", length(files), " files")
  invisible(files)
}


run_optional_pseudobulk <- function(state, config) {
  # Preserve the old explicitly enabled donor analysis only for --stage all.
  suppressPackageStartupMessages({ library(data.table); library(limma) })
  params <- list(session_dir = config$session_dir, labels_csv = "labels.csv",
                 ref_label = config$ref_label, alt_label = config$alt_label)
  expr <- state$inputs$expr
  group <- state$inputs$group
  mat_ref <- state$inputs$mat_ref
  mat_alt <- state$inputs$mat_alt
  contrast_str <- state$contrast
  fdr_thresh <- 0.05
  sig_count <- sum(state$native$adj.P.Val < fdr_thresh)
  out_stem <- file.path(config$out_dir, "DE_results")
lab <- fread(file.path(params$session_dir, params$labels_csv))

n_alt <- ncol(mat_alt); n_ref <- ncol(mat_ref)
if (nrow(lab) != n_alt + n_ref) {
  stop(sprintf(paste0(
    "labels.csv has %d rows but the two matrices hold %d cells. The positional ",
    "join is invalid -- most likely select_contrast() excluded one or more ",
    "classes. Re-export with the excluded cells, or add an explicit cell id."),
    nrow(lab), n_alt + n_ref))
}

# Verify the assumed [alt block; ref block] ordering instead of trusting it.
frac_norm <- function(v) mean(grepl("^(normal|healthy)$", tolower(trimws(v))))
f_first <- frac_norm(head(lab$disease, n_alt))
f_last  <- frac_norm(tail(lab$disease, n_ref))
if (!(f_first < 0.05 && f_last > 0.95)) {
  stop(sprintf(paste0(
    "labels.csv row order does not match [%s block; %s block] ",
    "(normal fraction: first block %.2f, second block %.2f). Refusing to join."),
    params$alt_label, params$ref_label, f_first, f_last))
}

meta <- data.frame(
  cell  = colnames(expr),
  group = group,
  donor = c(tail(lab$donor_id, n_ref), head(lab$donor_id, n_alt)),
  stringsAsFactors = FALSE
)
# expr columns are [ref block; alt block]; lab rows are [alt block; ref block].

keep <- !is.na(meta$donor) & tolower(trimws(meta$donor)) != "unknown"
message(sprintf("Cells with a usable donor_id: %d of %d (%.0f%% dropped)",
                sum(keep), nrow(meta), 100 * mean(!keep)))

meta_k <- meta[keep, ]
expr_k <- expr[, keep, drop = FALSE]

donor_key <- paste(meta_k$group, meta_k$donor, sep = "|")
pb <- sapply(split(seq_len(ncol(expr_k)), donor_key),
             function(idx) rowMeans(expr_k[, idx, drop = FALSE]))

pb_group <- factor(sub("\\|.*$", "", colnames(pb)),
                   levels = c(params$ref_label, params$alt_label))
message(sprintf("Pseudobulk units: %d (%s n=%d, %s n=%d)",
                ncol(pb),
                params$ref_label, sum(pb_group == params$ref_label),
                params$alt_label, sum(pb_group == params$alt_label)))

pb_design <- model.matrix(~ 0 + pb_group)
colnames(pb_design) <- make.names(levels(pb_group))
pb_cm  <- makeContrasts(contrasts = contrast_str, levels = pb_design)
pb_fit <- eBayes(contrasts.fit(lmFit(pb, pb_design), pb_cm), robust = TRUE, trend = TRUE)
pb_res <- topTable(pb_fit, coef = 1, number = Inf, sort.by = "P")
pb_res$Gene <- rownames(pb_res)
pb_res <- pb_res[, c("Gene", "logFC", "AveExpr", "t", "P.Value", "adj.P.Val", "B")]

fwrite(pb_res, paste0(out_stem, "_table_pseudobulk_donor.csv"))
message("Donor-level FDR < ", fdr_thresh, ": ",
        sum(pb_res$adj.P.Val < fdr_thresh, na.rm = TRUE),
        "  (cell-level was ", sig_count, ")")

}
main <- function(args) {
  config <- parse_de_config(args)
  if (config$stage %in% c("plots", "all")) unlink(file.path(config$out_dir, "plot_manifest.json"))
  if (config$stage %in% c("de", "all")) {
    inputs <- read_inputs(config)
    state <- run_de(inputs, config)
    state <- export_results(state, config)
  } else {
    state <- readRDS(config$state_file)
  }
  if (identical(config$stage, "all") && env_flag("PSEUDOBULK", FALSE)) run_optional_pseudobulk(state, config)
  if (config$stage %in% c("plots", "all")) run_plots(state, config)
}

if (sys.nframe() == 0L) main(commandArgs(trailingOnly = TRUE))

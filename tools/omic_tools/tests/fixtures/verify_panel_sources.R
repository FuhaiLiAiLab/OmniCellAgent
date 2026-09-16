# Test-only observer: run the real plotting stage, never DE, and compare
# renderer inputs/labels with exported CSVs and the saved native DE table.
args <- commandArgs(TRUE)
source(args[1])
config <- parse_de_config(args[-1])
stopifnot(identical(config$stage, "plots"))
state <- readRDS(config$state_file)
observed <- new.env(parent = emptyenv())
observed$a <- NULL
observed$b <- list()
observed$pca_alpha <- NULL
observed$pca_size <- NULL

ggsave <- function(filename, plot, ...) {
  if (endsWith(filename, "_volcano.png")) {
    built <- ggplot2::ggplot_build(plot)
    layers <- which(vapply(plot$layers, function(layer)
      inherits(layer$geom, "GeomTextRepel"), logical(1)))
    stopifnot(length(layers) == 1L)
    observed$a <- built$data[[layers]]
    annotations <- unlist(lapply(built$data, function(layer) layer$label))
    stopifnot(any(grepl("Up (logFC >=", annotations, fixed = TRUE)),
              any(grepl("Down (logFC <=", annotations, fixed = TRUE)))
  }
  if (endsWith(filename, "_PCA_panel.png")) {
    main <- get("p_main", envir = parent.frame())
    points <- ggplot2::ggplot_build(main)$data[[1]]
    observed$pca_alpha <- unique(points$alpha)
    observed$pca_size <- unique(points$size)
    original <- get("pcoadata", envir = parent.frame())
    displayed <- main$data
    aligned <- displayed[match(original$sample, displayed$sample), , drop = FALSE]
    stopifnot(identical(observed$pca_alpha, 0.25),
              identical(observed$pca_size, 1.7),
              nrow(displayed) == nrow(original),
              isTRUE(all.equal(original, aligned, check.attributes = FALSE)),
              !identical(displayed$sample, original$sample))
  }
  ggplot2::ggsave(filename = filename, plot = plot, ...)
}

corrgram <- function(...) {
  call_args <- list(...)
  input <- call_args[[1]]
  draw_text <- call_args$text.panel
  labels <- character()
  call_args$text.panel <- function(...) {
    text_args <- list(...)
    label <- if ("txt" %in% names(text_args)) text_args$txt else text_args[[3]]
    labels <<- c(labels, as.character(label))
    do.call(draw_text, text_args)
  }
  result <- do.call(corrgram::corrgram, call_args)
  observed$b[[length(observed$b) + 1L]] <- list(input = input, labels = labels)
  invisible(result)
}

run_plots(state, config)
a <- read.csv(file.path(config$out_dir, "DE_results_panel_A_labeled_genes.csv"))
b <- read.csv(file.path(config$out_dir, "DE_results_panel_B_corrgram_genes.csv"))
native <- state$native
selected <- head(native$Gene[order(native$adj.P.Val)], 20)
ranked <- native[order(native$adj.P.Val), , drop = FALSE]
expected_a <- c(head(ranked$Gene[ranked$adj.P.Val < .05 & ranked$logFC >= 1], 10),
                ranked$Gene[ranked$adj.P.Val < .05 & ranked$logFC <= -1])
stopifnot(!is.null(observed$a), length(observed$b) >= 2L,
          identical(as.character(observed$a$label), as.character(a$gene_symbol)),
          identical(as.character(a$gene_symbol), as.character(expected_a)),
          all(a$is_significant), !any(a$status == "NS"),
          identical(observed$pca_alpha, 0.25),
          identical(a$label_rank, seq_len(nrow(a))),
          identical(b$display_order, seq_len(nrow(b))),
          identical(b$selection_rank, match(b$gene_symbol, selected)))

max_errors <- list()
for (panel in c("A", "B")) {
  csv <- if (panel == "A") a else b
  rows <- native[match(csv$gene_symbol, native$Gene), , drop = FALSE]
  stopifnot(!anyNA(rows$Gene), !anyDuplicated(csv$gene_symbol))
  errors <- list()
  for (column in c("logFC", "AveExpr", "P.Value", "adj.P.Val")) {
    stopifnot(isTRUE(all.equal(csv[[column]], rows[[column]], tolerance = 1e-12)))
    errors[[column]] <- max(abs(csv[[column]] - rows[[column]]))
  }
  expected_status <- ifelse(rows$adj.P.Val < .05 & rows$logFC >= 1, "Up",
                     ifelse(rows$adj.P.Val < .05 & rows$logFC <= -1, "Down", "NS"))
  stopifnot(identical(csv$status, expected_status),
            identical(csv$is_significant, rows$adj.P.Val < .05 & abs(rows$logFC) >= 1))
  max_errors[[panel]] <- errors
}
stopifnot(isTRUE(all.equal(observed$a$x, a$logFC, tolerance = 1e-12)),
          isTRUE(all.equal(observed$a$y,
            -log10(pmax(a$adj.P.Val, .Machine$double.xmin)), tolerance = 1e-12)))
for (draw in observed$b) {
  stopifnot(identical(draw$labels, as.character(b$gene_symbol)),
            identical(colnames(draw$input), as.character(selected)),
            isTRUE(all.equal(as.matrix(draw$input),
              t(state$inputs$expr[selected, , drop = FALSE]), check.attributes = FALSE)))
}
reference_dir <- Sys.getenv("OMIC_PANEL_REFERENCE_DIR", "")
historical_compared <- FALSE
if (nzchar(reference_dir)) {
  # A's annotation policy intentionally changed; only B must match old exports.
  for (panel in "B") {
    filename <- if (panel == "A") "DE_results_panel_A_labeled_genes.csv" else
      "DE_results_panel_B_corrgram_genes.csv"
    historical <- read.csv(file.path(reference_dir, filename))
    current <- if (panel == "A") a else b
    stopifnot(identical(names(historical), names(current)),
              isTRUE(all.equal(historical, current, tolerance = 1e-12)))
  }
  historical_compared <- TRUE
}
jsonlite::write_json(list(
  panel_A = list(genes = a$gene_symbol, label_rank = a$label_rank,
                 renderer_labels_match = TRUE),
  panel_B = list(genes = b$gene_symbol, selection_rank = b$selection_rank,
                 renderer_order_matches = TRUE, observed_draws = length(observed$b)),
  historical_csvs_match = historical_compared,
  historical_comparison_panels = if (historical_compared) "B" else character(),
  pca_point_alpha = observed$pca_alpha,
  pca_point_size = observed$pca_size,
  pca_coordinates_preserved = TRUE,
  pca_draw_order_shuffled = TRUE,
  max_absolute_errors = max_errors),
  file.path(config$out_dir, "panel_source_verification.json"),
  pretty = TRUE, auto_unbox = TRUE)
cat("Panel A/B renderer genes, ordering, expression input and DE values verified\n")

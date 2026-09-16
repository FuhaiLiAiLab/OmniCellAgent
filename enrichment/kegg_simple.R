# kegg_simple.R - Simplified enrichment visualization (no Bioconductor dependencies)
#
# USAGE:
# Rscript kegg_simple.R [enrichment_directory] [output_directory]
#
# Examples:
#   Rscript kegg_simple.R /path/to/enrichment/results /path/to/output
#

# Optional: --enrichment-root PATH --comparison-name TEXT --bar-output-dir PATH
# Only reads existing enrichment results; never performs enrichment or DE.
# Load required libraries (no Bioconductor)
library(ggplot2)
library(dplyr)
library(stringr)
library(forcats)
library(plotly)
library(htmlwidgets)
library(jsonlite)
library(ragg)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
extra <- list()
positional <- character()
while (length(args)) {
  if (startsWith(args[1], "--")) {
    if (!args[1] %in% c("--enrichment-root", "--comparison-name", "--bar-output-dir") || length(args) < 2)
      stop("Unknown option or missing option value: ", args[1])
    extra[[substring(args[1], 3)]] <- args[2]
    args <- args[-c(1, 2)]
  } else {
    positional <- c(positional, args[1])
    args <- args[-1]
  }
}
args <- positional
if (length(args) > 2) stop("Expected at most two positional paths")

# Configuration - File paths
if (length(args) > 0) {
  BASE_PATH <- args[1]
} else {
  BASE_PATH <- "dataset_outputs/enrichment_results/Alzheimer's_disease_all_regulated"
}

# Output directory
if (length(args) > 1) {
  OUTPUT_DIR <- args[2]
} else {
  OUTPUT_DIR <- "."
}

# Create output directory if it doesn't exist
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
OUTPUT_DIR <- normalizePath(OUTPUT_DIR, mustWork = TRUE)
manifest_path <- file.path(OUTPUT_DIR, "plot_manifest.json")
unlink(manifest_path)
generated_files <- character()
write_manifest <- function(status, error = NULL) {
  jsonlite::write_json(list(status = status, files = I(unname(generated_files)),
                           error = error), manifest_path, auto_unbox = TRUE, pretty = TRUE)
}
options(error = function() {
  generated_files <<- character()
  write_manifest("failed", geterrmessage())
  quit(save = "no", status = 1, runLast = FALSE)
})
if (length(extra) && !all(c("enrichment-root", "comparison-name", "bar-output-dir") %in% names(extra)))
  stop("Supply all three optional enrichment bar plot arguments")
record_file <- function(path) {
  generated_files <<- unique(c(generated_files, normalizePath(path, mustWork = TRUE)))
}
save_png <- function(path, plot, width = 12, height = 10) {
  ggsave(path, plot = plot, device = ragg::agg_png, width = width, height = height,
         units = "in", dpi = 300)
  record_file(path)
}
save_widget <- function(widget, path) {
  # Relative dependencies travel with the HTML when staging is published.
  previous <- setwd(OUTPUT_DIR)
  on.exit(setwd(previous))
  libdir <- paste0(tools::file_path_sans_ext(basename(path)), "_files")
  htmlwidgets::saveWidget(widget, file = basename(path), selfcontained = FALSE, libdir = libdir)
  record_file(path)
  record_file(list.files(libdir, recursive = TRUE, full.names = TRUE))
}
safe_logp <- function(p) -log10(pmax(p, .Machine$double.xmin))
read_status <- function(directory) {
  path <- file.path(directory, "enrichment_status.json")
  if (!file.exists(path)) return(NULL) # legacy CSV-only directories remain supported
  status <- jsonlite::read_json(path, simplifyVector = FALSE)
  if (is.null(status$status) || length(status$status) != 1 ||
      !status$status %in% c("success", "empty", "skipped"))
    stop("Refusing incomplete or failed enrichment input: ", path)
  if (length(status$libraries)) for (entry in status$libraries) {
    if (is.null(entry$status) || !entry$status %in% c("success", "empty", "skipped"))
      stop("Refusing incomplete or failed enrichment library: ", path)
  }
  status
}
# Validate every participating group before rendering any output.
root_status <- read_status(if (length(extra)) extra[["enrichment-root"]] else dirname(BASE_PATH))
read_status(BASE_PATH)
if (length(extra)) for (direction in c("all", "up", "down")) {
  read_status(file.path(extra[["enrichment-root"]], paste0(
    gsub(" ", "_", extra[["comparison-name"]], fixed = TRUE), "_", direction, "_regulated")))
}

cat("Using enrichment results path:", BASE_PATH, "\n")
cat("Using output directory:", OUTPUT_DIR, "\n")

# Input CSV file paths
GO_BP_FILE <- file.path(BASE_PATH, "GO_Biological_Process_2021_results.csv")
GO_CC_FILE <- file.path(BASE_PATH, "GO_Cellular_Component_2021_results.csv")
GO_MF_FILE <- file.path(BASE_PATH, "GO_Molecular_Function_2021_results.csv")
DISGENET_FILE <- file.path(BASE_PATH, "DisGeNET_results.csv")
KEGG_FILE <- file.path(BASE_PATH, "KEGG_2021_Human_results.csv")

# Output file paths
KEGG_DOTPLOT_FILE <- file.path(OUTPUT_DIR, "kegg_dotplot.png")
KEGG_DOTPLOT_HTML <- file.path(OUTPUT_DIR, "kegg_dotplot.html")
COMBINED_PLOT_FILE <- file.path(OUTPUT_DIR, "pathway_combined_plot.png")
COMBINED_PLOT_HTML <- file.path(OUTPUT_DIR, "pathway_combined_plot.html")

# Set options for headless server
options(browser = FALSE)

MAX_GENE_SHOW <- 9

# Function to safely read and convert enrichment CSV files
convert_enrichment_file <- function(file, ontology = NULL) {
  status <- read_status(dirname(file))
  database <- sub("_results.csv$", "", basename(file))
  if ((!is.null(root_status) && root_status$status %in% c("empty", "skipped")) ||
      (!is.null(status) && (status$status %in% c("empty", "skipped") ||
       (!is.null(status$libraries[[database]]) &&
        status$libraries[[database]]$status %in% c("empty", "skipped"))))) return(NULL)
  if (!file.exists(file)) {
    cat("File not found:", file, "\n")
    return(NULL)
  }
  
  df <- read.csv(file, stringsAsFactors = FALSE)
  colnames(df) <- gsub("\\.", "_", colnames(df))
  required <- c("Term", "Genes", "P_value", "Adjusted_P_value")
  if (!all(required %in% names(df))) stop("Malformed enrichment columns: ", file)
  for (column in c("P_value", "Adjusted_P_value")) {
    value <- suppressWarnings(as.numeric(df[[column]]))
    if (any(!is.finite(value) | value < 0 | value > 1)) stop("Invalid p-values: ", file)
    df[[column]] <- value
  }
  
  if (nrow(df) == 0) {
    cat("Empty file:", file, "\n")
    return(NULL)
  }
  
  # Handle different column name formats (with dots or spaces)
  colnames(df) <- gsub("\\.", "_", colnames(df))
  
  df <- df %>%
    mutate(
      Description = str_trim(str_replace(Term, "\\s*\\([^)]+\\)$", "")),
      geneID = str_remove_all(Genes, "\\[|\\]|'|\""),
      geneID = str_replace_all(geneID, "\\s+", ""),
      geneID = str_replace_all(geneID, ",", "/"),
      Count = ifelse(is.na(geneID) | geneID == "", 0L, lengths(str_split(geneID, "/"))),
      pvalue = P_value,
      p_adjust = Adjusted_P_value
    )
  
  if (!is.null(ontology)) {
    df$ONTOLOGY <- ontology
  }
  
  df <- df %>%
    select(any_of(c("Term", "ONTOLOGY", "Description", "geneID", "Count", "pvalue", "p_adjust", "Odds_Ratio")))
  
  return(df)
}

# Function to truncate gene lists for display
truncate_genes <- function(gene_string, max_genes = MAX_GENE_SHOW) {
  genes <- unlist(strsplit(gene_string, "/"))
  if (length(genes) <= max_genes) {
    return(gene_string)
  }
  truncated <- paste(genes[1:max_genes], collapse = "/")
  return(paste0(truncated, "/..."))
}

# ============================================================================
# LOAD DATA
# ============================================================================
cat("\nLoading enrichment data...\n")

kegg_df <- convert_enrichment_file(KEGG_FILE)
go_bp <- convert_enrichment_file(GO_BP_FILE, "BP")
go_cc <- convert_enrichment_file(GO_CC_FILE, "CC")
go_mf <- convert_enrichment_file(GO_MF_FILE, "MF")
disgenet_df <- convert_enrichment_file(DISGENET_FILE, "DisGeNET")

# ============================================================================
# KEGG DOTPLOT
# ============================================================================
if (!is.null(kegg_df) && nrow(kegg_df) > 0) {
  cat("\nCreating KEGG dotplot...\n")
  
  top_n <- 20
  plot_df <- kegg_df %>%
    arrange(p_adjust) %>%
    head(top_n) %>%
    mutate(
      Description_factor = fct_reorder(Description, Count),
      log10_padj = safe_logp(p_adjust)
    )
  
  # Static ggplot
  kegg_dot_plot <- ggplot(plot_df, aes(x = Count, y = Description_factor)) +
    geom_point(aes(size = Count, color = log10_padj)) +
    scale_color_gradient(low = "blue", high = "red", name = "-log10(p.adjust)") +
    scale_size(range = c(3, 8)) +
    labs(
      title = "KEGG Enrichment Dotplot",
      x = "Gene Count",
      y = "KEGG Pathway"
    ) +
    theme_minimal() +
    theme(
      axis.text.y = element_text(size = 10),
      plot.title = element_text(hjust = 0.5, face = "bold")
    )
  
  # Save static plot
  save_png(KEGG_DOTPLOT_FILE, kegg_dot_plot)
  cat("KEGG dotplot (PNG) saved to:", KEGG_DOTPLOT_FILE, "\n")
  
  # Create interactive plotly version
  tryCatch({
    # Use character Description and numeric sorting for plotly
    plot_df_plotly <- plot_df %>%
      arrange(Count) %>%
      mutate(
        Description_char = as.character(Description),
        y_order = row_number()
      )
    
    interactive_kegg <- plot_ly(
      data = plot_df_plotly,
      x = ~Count,
      y = ~Description_char,
      size = ~Count,
      color = ~log10_padj,
      colors = c("blue", "red"),
      type = "scatter",
      mode = "markers",
      marker = list(sizemode = 'diameter', sizeref = 0.5, sizemin = 5),
      text = ~paste("Pathway:", Description_char, 
                    "<br>Gene Count:", Count, 
                    "<br>-log10(p.adj):", round(log10_padj, 2),
                    "<br>Genes:", substr(geneID, 1, 100)),
      hovertemplate = "%{text}<extra></extra>",
      height = 800,
      width = 1000
    ) %>%
    layout(
      title = list(text = "KEGG Enrichment Dotplot (Interactive)", x = 0.5),
      xaxis = list(title = "Gene Count"),
      yaxis = list(title = "KEGG Pathway", categoryorder = "array", categoryarray = plot_df_plotly$Description_char),
      showlegend = TRUE
    )
    
    save_widget(interactive_kegg, KEGG_DOTPLOT_HTML)
    cat("KEGG dotplot (HTML) saved to:", KEGG_DOTPLOT_HTML, "\n")
    
  }, error = function(e) {
    stop("Interactive KEGG plot failed: ", conditionMessage(e))
  })
  
} else {
  cat("No KEGG data available for plotting\n")
}

# ============================================================================
# COMBINED PATHWAY PLOT
# ============================================================================
# Combine all available data
all_data <- list(go_bp, go_cc, go_mf, disgenet_df)
all_data <- all_data[!sapply(all_data, is.null)]

if (!is.null(kegg_df) && nrow(kegg_df) > 0) {
  kegg_with_ont <- kegg_df %>% mutate(ONTOLOGY = "KEGG")
  all_data <- c(all_data, list(kegg_with_ont))
}

if (length(all_data) > 0) {
  cat("\nCreating combined pathway plot...\n")
  
  ALL <- bind_rows(all_data)
  
  # Select top 5 per ontology
  use_pathway <- ALL %>%
    group_by(ONTOLOGY) %>%
    arrange(p_adjust, desc(Count)) %>%
    slice_head(n = 5) %>%
    ungroup() %>%
    mutate(
      ONTOLOGY = factor(ONTOLOGY, levels = c("BP", "CC", "MF", "KEGG", "DisGeNET")),
      geneID_limited = sapply(geneID, truncate_genes)
    ) %>%
    arrange(ONTOLOGY, p_adjust) %>%
    mutate(Description = factor(Description, levels = rev(unique(Description))))
  
  # Color palette
  pal <- c(
    "BP" = "#4DBBD5",
    "CC" = "#00A087", 
    "MF" = "#E64B35",
    "KEGG" = "#7E6148",
    "DisGeNET" = "#3C5488"
  )
  
  # Static bar plot
  combined_plot <- ggplot(use_pathway, aes(x = safe_logp(p_adjust), y = Description, fill = ONTOLOGY)) +
    geom_col(width = 0.7) +
    geom_text(aes(x = 0.1, label = Description), hjust = 0, size = 3.5) +
    scale_fill_manual(name = 'Category', values = pal) +
    labs(
      title = "Combined Pathway Enrichment",
      x = "-log10(Adjusted P-value)",
      y = NULL
    ) +
    theme_minimal() +
    theme(
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      plot.title = element_text(hjust = 0.5, face = "bold"),
      legend.position = "bottom"
    )
  
  # Save static plot
  save_png(COMBINED_PLOT_FILE, combined_plot)
  cat("Combined plot (PNG) saved to:", COMBINED_PLOT_FILE, "\n")
  
  # Create interactive plotly version
  tryCatch({
    interactive_combined <- plot_ly(
      data = use_pathway,
      x = ~safe_logp(p_adjust),
      y = ~reorder(Description, safe_logp(p_adjust)),
      color = ~ONTOLOGY,
      colors = pal,
      type = "bar",
      orientation = 'h',
      text = ~paste("Pathway:", Description,
                    "<br>Category:", ONTOLOGY,
                    "<br>Gene Count:", Count,
                    "<br>-log10(p.adj):", round(safe_logp(p_adjust), 2),
                    "<br>Genes:", geneID_limited),
      hovertemplate = "%{text}<extra></extra>",
      height = 800,
      width = 1200
    ) %>%
    layout(
      title = list(text = "Combined Pathway Enrichment (Interactive)", x = 0.5),
      xaxis = list(title = "-log10(Adjusted P-value)"),
      yaxis = list(title = "Pathway", tickfont = list(size = 10)),
      showlegend = TRUE,
      legend = list(orientation = "h", y = -0.1),
      margin = list(l = 300)
    )
    
    save_widget(interactive_combined, COMBINED_PLOT_HTML)
    cat("Combined plot (HTML) saved to:", COMBINED_PLOT_HTML, "\n")
    
  }, error = function(e) {
    stop("Interactive combined plot failed: ", conditionMessage(e))
  })
  
} else {
  cat("No data available for combined plot\n")
}

if (length(extra)) {
  bar_dir <- extra[["bar-output-dir"]]
  dir.create(bar_dir, recursive = TRUE, showWarnings = FALSE)
  database_names <- c(Reactome_2022 = "Reactome Pathways", KEGG_2021_Human = "KEGG Pathways")
  direction_names <- c(all = "All", up = "Upregulated", down = "Downregulated")
  for (direction in names(direction_names)) for (database in names(database_names)) {
    input <- file.path(extra[["enrichment-root"]], paste0(
      gsub(" ", "_", extra[["comparison-name"]], fixed = TRUE), "_", direction, "_regulated"),
      paste0(database, "_results.csv"))
    df <- convert_enrichment_file(input)
    if (is.null(df)) next
    df <- df %>% arrange(p_adjust) %>% head(10) %>% mutate(
      score = safe_logp(p_adjust), row = rev(seq_len(n())))
    labels <- ifelse(nchar(df$Term) > 60, paste0(substr(df$Term, 1, 57), "..."), df$Term)
    plot <- ggplot(df, aes(x = score, y = row)) +
      geom_col(width = 0.65, fill = "#9e9ac8", colour = "#6a51a3", alpha = 0.7) +
      geom_text(aes(x = score / 2, label = paste("Count:", Count)), colour = "#3f007d", fontface = "bold") +
      geom_text(aes(x = score * 1.05, label = sprintf("p=%.2e", p_adjust)), hjust = 0, colour = "#4a4a4a") +
      scale_y_continuous(breaks = df$row, labels = labels) +
      scale_x_continuous(expand = expansion(mult = c(0, 0.25))) +
      labs(title = paste0("Top 10 Enriched ", database_names[[database]], " Terms\n",
                          direction_names[[direction]], " Genes - ", extra[["comparison-name"]]),
           x = "-log10(Adjusted P-Value)", y = "Pathway") + theme_classic()
    stem <- file.path(bar_dir, paste0(database, "_", direction, "_regulated"))
    height <- min(12, 2 + 0.4 * nrow(df))
    save_png(paste0(stem, ".png"), plot, height = height)
    ggsave(paste0(stem, ".pdf"), plot, device = grDevices::pdf, width = 12, height = height)
    record_file(paste0(stem, ".pdf"))
  }
}
write_manifest(if (length(generated_files)) "success" else "empty")
cat("\n=== KEGG visualization complete ===\n")

# kegg_simple.R - Simplified enrichment visualization (no Bioconductor dependencies)
#
# USAGE:
# Rscript kegg_simple.R [enrichment_directory] [output_directory]
#
# Examples:
#   Rscript kegg_simple.R /path/to/enrichment/results /path/to/output
#

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

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

# Load required libraries (no Bioconductor)
library(ggplot2)
library(dplyr)
library(stringr)
library(forcats)
library(plotly)
library(htmlwidgets)

# Set options for headless server
options(browser = FALSE)

MAX_GENE_SHOW <- 9

# Function to safely read and convert enrichment CSV files
convert_enrichment_file <- function(file, ontology = NULL) {
  if (!file.exists(file)) {
    cat("File not found:", file, "\n")
    return(NULL)
  }
  
  df <- read.csv(file, stringsAsFactors = FALSE)
  
  if (nrow(df) == 0) {
    cat("Empty file:", file, "\n")
    return(NULL)
  }
  
  # Handle different column name formats (with dots or spaces)
  colnames(df) <- gsub("\\.", "_", colnames(df))
  
  df <- df %>%
    mutate(
      Description = str_trim(str_replace(Term, "\\s*\\([^)]+\\)$", "")),
      geneID = str_remove_all(Genes, "\\[|\\]|'"),
      geneID = str_replace_all(geneID, "\\s+", ""),
      geneID = str_replace_all(geneID, ",", "/"),
      Count = lengths(str_split(geneID, "/")),
      pvalue = P_value,
      p_adjust = Adjusted_P_value
    )
  
  if (!is.null(ontology)) {
    df$ONTOLOGY <- ontology
  }
  
  df <- df %>%
    select(any_of(c("ONTOLOGY", "Description", "geneID", "Count", "pvalue", "p_adjust", "Odds_Ratio")))
  
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
      log10_padj = -log10(p_adjust)
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
  ggsave(KEGG_DOTPLOT_FILE, plot = kegg_dot_plot, width = 12, height = 10, units = "in", dpi = 300)
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
    
    htmlwidgets::saveWidget(interactive_kegg, file = KEGG_DOTPLOT_HTML, selfcontained = TRUE)
    cat("KEGG dotplot (HTML) saved to:", KEGG_DOTPLOT_HTML, "\n")
    
  }, error = function(e) {
    cat("Interactive KEGG plot failed:", conditionMessage(e), "\n")
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
    mutate(Description = factor(Description, levels = rev(Description)))
  
  # Color palette
  pal <- c(
    "BP" = "#4DBBD5",
    "CC" = "#00A087", 
    "MF" = "#E64B35",
    "KEGG" = "#7E6148",
    "DisGeNET" = "#3C5488"
  )
  
  # Static bar plot
  combined_plot <- ggplot(use_pathway, aes(x = -log10(p_adjust), y = Description, fill = ONTOLOGY)) +
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
  ggsave(COMBINED_PLOT_FILE, plot = combined_plot, width = 12, height = 10, units = "in", dpi = 300)
  cat("Combined plot (PNG) saved to:", COMBINED_PLOT_FILE, "\n")
  
  # Create interactive plotly version
  tryCatch({
    interactive_combined <- plot_ly(
      data = use_pathway,
      x = ~(-log10(p_adjust)),
      y = ~reorder(Description, -log10(p_adjust)),
      color = ~ONTOLOGY,
      colors = pal,
      type = "bar",
      orientation = 'h',
      text = ~paste("Pathway:", Description,
                    "<br>Category:", ONTOLOGY,
                    "<br>Gene Count:", Count,
                    "<br>-log10(p.adj):", round(-log10(p_adjust), 2),
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
    
    htmlwidgets::saveWidget(interactive_combined, file = COMBINED_PLOT_HTML, selfcontained = TRUE)
    cat("Combined plot (HTML) saved to:", COMBINED_PLOT_HTML, "\n")
    
  }, error = function(e) {
    cat("Interactive combined plot failed:", conditionMessage(e), "\n")
  })
  
} else {
  cat("No data available for combined plot\n")
}

cat("\n=== KEGG visualization complete ===\n")

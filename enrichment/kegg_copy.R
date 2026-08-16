# kegg_analysis.R
#
# USAGE:
# Rscript kegg.R [enrichment_directory] [output_directory]
#
# Arguments:
#   enrichment_directory - Path to directory containing CSV enrichment results (optional)
#   output_directory     - Path where PNG plots will be saved (optional, default: current directory)
#
# Examples:
#   Rscript kegg.R  # Uses default paths
#   Rscript kegg.R /path/to/enrichment/results  # Custom enrichment path, output to current dir
#   Rscript kegg.R /path/to/enrichment/results /path/to/output  # Custom enrichment and output paths
#

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Configuration - File paths
if (length(args) > 0) {
  # Use user-provided enrichment results path
  BASE_PATH <- args[1]
} else {
  # Default path if no argument provided - use local relative path
  BASE_PATH <- "dataset_outputs/enrichment_results/Alzheimer's_disease_all_regulated"
}

# Output directory
if (length(args) > 1) {
  OUTPUT_DIR <- args[2]
} else {
  OUTPUT_DIR <- "."  # Current directory
}

# Print the paths being used
cat("Using enrichment results path:", BASE_PATH, "\n")
cat("Using output directory:", OUTPUT_DIR, "\n")

# Input CSV file paths
GO_BP_FILE <- file.path(BASE_PATH, "GO_Biological_Process_2021_results.csv")
GO_CC_FILE <- file.path(BASE_PATH, "GO_Cellular_Component_2021_results.csv")
GO_MF_FILE <- file.path(BASE_PATH, "GO_Molecular_Function_2021_results.csv")
DISGENET_FILE <- file.path(BASE_PATH, "DisGeNET_results.csv")
KEGG_FILE <- file.path(BASE_PATH, "KEGG_2021_Human_results.csv")

# Output file paths (can also be customized)
KEGG_DOTPLOT_FILE <- file.path(OUTPUT_DIR, "kegg_dotplot.png")
KEGG_DOTPLOT_HTML <- file.path(OUTPUT_DIR, "kegg_dotplot.html")
COMBINED_PLOT_FILE <- file.path(OUTPUT_DIR, "pathway_combined_plot.png")
COMBINED_PLOT_HTML <- file.path(OUTPUT_DIR, "pathway_combined_plot.html")

# Load libraries
library(ggprism)
library(tidyverse)
library(org.Hs.eg.db)
library(clusterProfiler)
library(dplyr)
library(ragg)
library(plotly)

# Set options for headless server
options(plotly.engine = "kaleido")
options(browser = FALSE)
Sys.setenv("DISPLAY" = ":99")  # Virtual display

# Override capabilities to trick plotly
local({
  old_capabilities <- capabilities
  assign("capabilities", function(...) {
    caps <- old_capabilities(...)
    caps["cairo"] <- TRUE
    caps["png"] <- TRUE
    caps
  }, envir = .GlobalEnv)
})

library(htmlwidgets)
# Color palette
pal <- c("#7bc4e2", "#acd372", "#fbb05b", "#ed6ca4")

set.seed(1234)
MAX_GENE_SHOW = 9

# GO enrichment
library(dplyr)
library(stringr)
# library(readr)  # Comment out problematic readr

# Function to convert a single GO file
convert_GO_file <- function(file, ontology) {
  df <- read.csv(file)  # Use base R instead of read_csv
  
  # Extract GO ID and Description
  df <- df %>%
    mutate(
      ID = str_extract(Term, "GO:\\d+"),
      Description = str_trim(str_replace(Term, "\\s*\\(GO:\\d+\\)", "")),
      geneID = str_remove_all(Genes, "\\[|\\]|'"),
      geneID = str_replace_all(geneID, "\\s+", ""),
      geneID = str_replace_all(geneID, ",", "/"),  # unify format
      Count = lengths(str_split(geneID, "/")),
      ONTOLOGY = ontology
    )
  
  # Placeholder columns for GeneRatio, BgRatio, RichFactor, FoldEnrichment, zScore
  df <- df %>%
    mutate(
      GeneRatio = paste0(Count, "/", Count + 100),   # Example ratio placeholder
      BgRatio = "1000/20000",                        # Placeholder
      RichFactor = Count / 100,                      # Placeholder
      FoldEnrichment = `Odds.Ratio`,                   # Use Odds.Ratio (dot instead of space)
      zScore = -log10(`P.value`) * sign(`Odds.Ratio` - 1), # Rough z-score approximation
      pvalue = `P.value`,
      p.adjust = `Adjusted.P.value`,
      qvalue = `Adjusted.P.value`                              # Approximate qvalue = p.adjust
    ) %>%
    select(ONTOLOGY, ID, Description, GeneRatio, BgRatio, RichFactor,
           FoldEnrichment, zScore, pvalue, p.adjust, qvalue, geneID, Count)
  
  return(df)
}

# Apply the function to your three GO files
go_bp <- convert_GO_file(GO_BP_FILE, "BP")
go_cc <- convert_GO_file(GO_CC_FILE, "CC")
go_mf <- convert_GO_file(GO_MF_FILE, "MF")
disgenet_df <- convert_GO_file(DISGENET_FILE, "DisGeNET")

# Combine into one data frame
GO <- bind_rows(go_bp, go_cc, go_mf, disgenet_df)

# KEGG enrichment
convert_kegg_csv <- function(file) {
  df <- read.csv(file)  # Use base R instead of read_csv
  
  df <- df %>%
    mutate(
      ID = str_extract(Term, "K\\d+"),
      Description = str_trim(str_replace(Term, "\\s*K\\d+", "")),
      geneID = str_remove_all(Genes, "\\[|\\]|'"),
      geneID = str_replace_all(geneID, "\\s+", ""),
      geneID = str_replace_all(geneID, ",", "/"),
      Count = lengths(str_split(geneID, "/")),
      GeneRatio = paste0(Count, "/", Count + 100),  # placeholder
      BgRatio = "1000/20000",                       # placeholder
      pvalue = `P.value`,
      p.adjust = `Adjusted.P.value`,
      qvalue = `Adjusted.P.value`
    ) %>%
    select(ID, Description, GeneRatio, BgRatio, pvalue, p.adjust,
           qvalue, geneID, Count)
  
  return(df)
}

# Example usage
kegg_df <- convert_kegg_csv(KEGG_FILE)

# Dotplot for KEGG results
agg_png(KEGG_DOTPLOT_FILE, width = 15, height = 10, units = "in", res = 300)

# Select top N categories (e.g., 20)
top_n <- 20
plot_df <- kegg_df %>%
  arrange(p.adjust) %>%
  head(top_n) %>%
  mutate(
    Description = fct_reorder(Description, Count),  # order by Count or -log10(p.adjust)
    log10_padj = -log10(p.adjust)
  )

# Create custom dotplot
kegg_dot_plot = ggplot(plot_df, aes(x = Count, y = Description)) +
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
    plot.title = element_text(
      hjust = 0.5,
     face = "bold")
  )

# Create simple interactive HTML using plot_ly directly (no ggplotly)
tryCatch({
  # Debug: check data structure
  cat("plot_df structure:\n")
  str(plot_df)
  cat("Description class:", class(plot_df$Description), "\n")
  
  # Convert factors to character to avoid issues
  plot_df_clean <- plot_df %>%
    mutate(Description = as.character(Description))
  
  # Create basic plotly scatter plot
  interactive_plot <- plot_ly(
    data = plot_df_clean,
    x = ~Count,
    y = ~Description,
    size = ~Count,
    color = ~log10_padj,
    colors = "RdYlBu",
    type = "scatter",
    mode = "markers",
    marker = list(sizemode = 'diameter', sizeref = 2, sizemin = 4),
    text = ~paste("Pathway:", Description, "<br>Count:", Count, "<br>-log10(p.adj):", round(log10_padj, 2)),
    hovertemplate = "%{text}<extra></extra>"
  ) %>%
  layout(
    title = "KEGG Enrichment Dotplot",
    xaxis = list(title = "Gene Count"),
    yaxis = list(title = "KEGG Pathway"),
    showlegend = TRUE
  )
  
  # Save as HTML
  htmlwidgets::saveWidget(
    interactive_plot, 
    file = KEGG_DOTPLOT_HTML, 
    selfcontained = TRUE
  )
  
  cat("Interactive KEGG dotplot HTML created successfully using native plot_ly\n")
  cat("Saved to:", KEGG_DOTPLOT_HTML, "\n")
  
}, error = function(e) {
  cat("Interactive plot creation failed:", conditionMessage(e), "\n")
  cat("Error details:", e$message, "\n")
  cat("Proceeding with static plot only\n")
})

# Always save static plot as backup
ggsave(KEGG_DOTPLOT_FILE, plot = kegg_dot_plot, width = 15, height = 10, units = "in", dpi = 300)

cat("KEGG dotplot saved to:", KEGG_DOTPLOT_FILE, "\n")

# Prepare data for combined plot
ALL <- bind_rows(go_bp, go_cc, go_mf, disgenet_df, kegg_df %>% mutate(ONTOLOGY = "KEGG"))

# Select top N per ONTOLOGY
use_pathway <- ALL %>%
  group_by(ONTOLOGY) %>%
  arrange(p.adjust, desc(Count)) %>%
  slice_head(n = 5) %>%
  ungroup() %>%
  mutate(ONTOLOGY = factor(ONTOLOGY, levels = rev(c("BP", "CC", "MF", "KEGG", "DisGeNET")))) %>%
  arrange(ONTOLOGY, p.adjust) %>%
  mutate(Description = factor(Description, levels = Description)) %>%
  rowid_to_column("index")

# Plotting parameters
width <- 0.5
# xaxis_max <- max(-log10(use_pathway$p.adjust)) + 1
xaxis_max <- max(-log10(use_pathway$p.adjust)) + 1.5  # or +2 for longer labels


# Create rectangular backgrounds for each ONTOLOGY
rect.data <- use_pathway %>%
  group_by(ONTOLOGY) %>%
  summarise(n = n(), .groups = "drop")

# Define color palette for ONTOLOGY
pal <- c(
  "BP" = "#4DBBD5",
  "CC" = "#00A087",
  "MF" = "#E64B35",
  "KEGG" = "#7E6148",
  "DisGeNET" = "#3C5488"
)

count_x <- -0.2
desc_x  <- 0.05
gene_x  <- 0.05
rect_xmin <- -0.8
rect_xmax <- -0.4

rect.data <- rect.data %>%
  mutate(
    xmin = rect_xmin,
    xmax = rect_xmax,
    ymax = cumsum(n),
    ymin = lag(ymax, default = 0) + 0.6,
    ymax = ymax + 0.4
  )
# Plot



truncate_genes <- function(gene_string, max_genes = MAX_GENE_SHOW) {
  genes <- unlist(strsplit(gene_string, "/"))
  if (length(genes) <= max_genes) {
    return(gene_string)
  }
  truncated <- paste(genes[1:max_genes], collapse = "/")
  return(paste0(truncated, "/..."))
}

# Apply to geneID column
use_pathway$geneID_limited <- sapply(use_pathway$geneID, truncate_genes)



agg_png(COMBINED_PLOT_FILE, width = 10, height = 10, units = "in", res = 300)

kegg_combine_plot <- ggplot(use_pathway, aes(x = -log10(p.adjust), y = index, fill = ONTOLOGY)) +
  geom_col(
    aes(y = Description), width = 0.6
  ) +
  geom_text(
    aes(x = desc_x, label = Description),
    hjust = 0,
     size = 5
  ) +
  geom_text(
    aes(x = gene_x, label = geneID_limited, colour = ONTOLOGY),
    hjust = 0, vjust = 2.6, 
    size = 3.5, fontface = 'italic', lineheight= 1.2, 
    show.legend = FALSE
  ) +

  # 
  # geom_text(
  #   aes(x = 0.15 * xaxis_max, label = geneID_wrapped, colour = ONTOLOGY), 
  #   hjust = 0, vjust = 2.6, size = 3.5, fontface = 'italic', 
  #   lineheight = 1.1, show.legend = FALSE
  # )

  geom_point(
    aes(x = count_x, size = Count),
    shape = 21
  ) +
  geom_text(
    aes(x = count_x, label = Count)
  ) +
  scale_size_continuous(name = 'Count', range = c(5, 16)) +
  geom_rect(
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = ONTOLOGY),
    data = rect.data,
    radius = unit(2, 'mm'),
    inherit.aes = FALSE
  ) +
  geom_text(
    aes(x = (xmin + xmax) / 2, y = (ymin + ymax) / 2, label = ONTOLOGY),
    data = rect.data,
    inherit.aes = FALSE
  ) +
  geom_segment(
    aes(x = 0, y = 0, xend = xaxis_max, yend = 0),
    linewidth = 1.5,
    inherit.aes = FALSE
  ) +
  labs(y = NULL) +
  scale_fill_manual(name = 'Category', values = pal) +
  scale_colour_manual(values = pal) +
  
  scale_x_continuous(
    # limits = c(rect_xmin, xaxis_max + 6),  # increase to give more right padding
    breaks = seq(0, xaxis_max, 2),
    expand = expansion(c(0, 0))
  ) + 
  
  theme_prism() +
  theme(
    axis.text.y = element_blank(),
    axis.line = element_blank(),
    axis.ticks.y = element_blank(),
    legend.title = element_text()
  )

# Create simple interactive HTML for combined plot
tryCatch({
  # Create basic plotly bar chart
  interactive_plot2 <- plot_ly(
    data = use_pathway,
    x = ~(-log10(p.adjust)),
    y = ~reorder(Description, -log10(p.adjust)),
    color = ~ONTOLOGY,
    colors = c("#4DBBD5", "#00A087", "#E64B35", "#7E6148", "#3C5488"),
    type = "bar",
    orientation = 'h',
    text = ~paste("Pathway:", Description, "<br>Ontology:", ONTOLOGY, "<br>Count:", Count, "<br>-log10(p.adj):", round(-log10(p.adjust), 2)),
    hovertemplate = "%{text}<extra></extra>"
  ) %>%
  layout(
    title = "Combined Pathway Enrichment",
    xaxis = list(title = "-log10(p.adjust)"),
    yaxis = list(title = "Pathway"),
    showlegend = TRUE,
    height = 800
  )
  
  # Save as HTML
  htmlwidgets::saveWidget(
    interactive_plot2, 
    file = COMBINED_PLOT_HTML, 
    selfcontained = TRUE
  )
  
  cat("Interactive combined plot HTML created successfully using native plot_ly\n")
  cat("Saved to:", COMBINED_PLOT_HTML, "\n")
  
}, error = function(e) {
  cat("Interactive combined plot creation failed:", conditionMessage(e), "\n")
  cat("Proceeding with static plot only\n")
})

# Always save static plot as backup
ggsave(COMBINED_PLOT_FILE, plot = kegg_combine_plot, width = 10, height = 10, units = "in", dpi = 300)

cat("Combined pathway plot saved to:", COMBINED_PLOT_FILE, "\n")

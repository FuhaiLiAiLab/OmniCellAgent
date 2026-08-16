# kegg_analysis.R

# Load libraries
library(ggprism)
library(tidyverse)
library(org.Hs.eg.db)
library(clusterProfiler)
library(dplyr)
library(ragg)
# Color palette
pal <- c("#7bc4e2", "#acd372", "#fbb05b", "#ed6ca4")

set.seed(1234)
MAX_GENE_SHOW = 9

# GO enrichment
library(dplyr)
library(stringr)
library(readr)

# Function to convert a single GO file
convert_GO_file <- function(file, ontology) {
  df <- read_csv(file)
  
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
      FoldEnrichment = `Odds Ratio`,                   # Use Odds Ratio if available
      zScore = -log10(`P-value`) * sign(`Odds Ratio` - 1), # Rough z-score approximation
      pvalue = `P-value`,
      p.adjust = `Adjusted P-value`,
      qvalue = `Adjusted P-value`                              # Approximate qvalue = p.adjust
    ) %>%
    select(ONTOLOGY, ID, Description, GeneRatio, BgRatio, RichFactor,
           FoldEnrichment, zScore, pvalue, p.adjust, qvalue, geneID, Count)
  
  return(df)
}

# Apply the function to your three GO files
go_bp <- convert_GO_file("/storage1/fs1/fuhai.li/Active/di.huang/Research/LLM/RAG-MLLM/bioRAGUI/bioRAG/tools/dataset_outputs/enrichment_results/Alzheimer's_Disease_all_regulated/GO_Biological_Process_2021_results.csv", "BP")
go_cc <- convert_GO_file("/storage1/fs1/fuhai.li/Active/di.huang/Research/LLM/RAG-MLLM/bioRAGUI/bioRAG/tools/dataset_outputs/enrichment_results/Alzheimer's_Disease_all_regulated/GO_Cellular_Component_2021_results.csv", "CC")
go_mf <- convert_GO_file("/storage1/fs1/fuhai.li/Active/di.huang/Research/LLM/RAG-MLLM/bioRAGUI/bioRAG/tools/dataset_outputs/enrichment_results/Alzheimer's_Disease_all_regulated/GO_Molecular_Function_2021_results.csv", "MF")
disgenet_df <- convert_GO_file("/storage1/fs1/fuhai.li/Active/di.huang/Research/LLM/RAG-MLLM/bioRAGUI/bioRAG/tools/dataset_outputs/enrichment_results/Alzheimer's_Disease_all_regulated/DisGeNET_results.csv", "DisGeNET")

# Combine into one data frame
GO <- bind_rows(go_bp, go_cc, go_mf, disgenet_df)

# KEGG enrichment
convert_kegg_csv <- function(file) {
  df <- read_csv(file)
  
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
      pvalue = `P-value`,
      p.adjust = `Adjusted P-value`,
      qvalue = `Adjusted P-value`
    ) %>%
    select(ID, Description, GeneRatio, BgRatio, pvalue, p.adjust,
           qvalue, geneID, Count)
  
  return(df)
}

# Example usage
kegg_df <- convert_kegg_csv("/storage1/fs1/fuhai.li/Active/di.huang/Research/LLM/RAG-MLLM/bioRAGUI/bioRAG/tools/dataset_outputs/enrichment_results/Alzheimer's_Disease_all_regulated/KEGG_2021_Human_results.csv")

# Dotplot for KEGG results
agg_png("kegg_dotplot.png", width = 15, height = 10, units = "in", res = 300)

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
ggplot(plot_df, aes(x = Count, y = Description)) +
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
dev.off()

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



agg_png("pathway_combined_plot.png", width = 10, height = 10, units = "in", res = 300)

ggplot(use_pathway, aes(x = -log10(p.adjust), y = index, fill = ONTOLOGY)) +
  geom_col(
    aes(y = Description), width = 0.6, alpha = 0.7
  ) +
  geom_text(
    aes(x = desc_x, label = Description),
    hjust = 0, size = 5
  ) +
  geom_text(
    aes(x = gene_x, label = geneID_limited, colour = ONTOLOGY),
    hjust = 0, vjust = 2.6, size = 3.5, fontface = 'italic', lineheight= 1.2, 
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
  
  # scale_x_continuous(
  #   breaks = seq(0, xaxis_max, 2), 
  #   expand = expansion(c(0, 0))
  # ) +
  
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

dev.off()
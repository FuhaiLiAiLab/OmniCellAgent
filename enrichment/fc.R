# fc.R - Converted from fc(1).Rmd

# ---- Libraries ----
library(pheatmap)
library(data.table)
library(gridExtra)
library(ggplot2)

# ---- 1. Read the CSV and keep what we need ----
df <- fread("/storage1/fs1/fuhai.li/Active/di.huang/Research/LLM/RAG-MLLM/bioRAGUI/bioRAG/tools/dataset_outputs/differential_expression/significant_genes_by_fdr.csv")
rownames(df) <- df[[1]]      # first column = gene names
df[[1]]      <- NULL

# ---- 2. Pick genes to show (edit ‘n_show’ or supply your own list) ----
n_show <- 50
ord    <- order(abs(df$log2_fold_change), decreasing = TRUE)
plot_df <- data.frame(
  gene  = factor(rownames(df)[ord][seq_len(n_show)],
                 levels = rev(rownames(df)[ord][seq_len(n_show)])),  # keep order
  log2FC = df$log2_fold_change[ord][seq_len(n_show)],
  pval   = df$p_value[ord][seq_len(n_show)]
)

plot_df$neglogP <- -log10(plot_df$pval)    # size aesthetic

brks  <- pretty(plot_df$neglogP, n = 3)          # e.g. 1, 2, 3
labs  <- format(signif(10^(-brks), digits = 2), scientific = TRUE)
        # converts back to p‑values:  10^-1 = 0.1, etc.

# ---- 3. Lollipop plot ----
p <- ggplot(plot_df, aes(x = log2FC, y = gene)) +
  geom_segment(aes(x = 0, xend = log2FC, y = gene, yend = gene),
               colour = "grey50") +
  geom_point(aes(size = neglogP, colour = log2FC > 0), alpha = 0.8) +
  scale_size_continuous(
    name   = "p-value",
    breaks = brks,
    labels = labs,
    range  = c(2, 10)
  ) +
  scale_size_continuous(
    name   = "p-value",
    breaks = pretty(plot_df$neglogP, n = 3),
    labels = function(b) format(signif(10^(-b), 2), sc = TRUE),
    range  = c(2, 10)
  ) +
  scale_colour_manual(
    name   = "Regulation",
    values = c("TRUE" = "#D62728", "FALSE"= "#1F77B4"),
    labels = c("TRUE"  = "Up", "FALSE" = "Down")
  ) +
  guides(
    colour = guide_legend(
      override.aes = list(size = 6)
    ),
    size   = guide_legend(
      override.aes = list(colour = "grey40"),
      keyheight    = unit(0.8, "cm")
    )
  ) +
  labs(x = "log\u2082 fold change", y = NULL,
       title = "Top differentially expressed genes\n(circle size = significance)") +
  theme_minimal(base_size = 12) +
  theme(panel.grid.major.y = element_blank(),
        axis.text.y = element_text(face = "italic"))

n_show  <- nrow(plot_df)
plot_h  <- 2 + 0.25 * n_show

ggsave("lollipop_top_genes.png", p,
  width  = 8,
  height = plot_h,
  units  = "in",
  bg     = "white")
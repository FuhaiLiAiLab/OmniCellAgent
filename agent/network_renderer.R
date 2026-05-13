#!/usr/bin/env Rscript
# ============================================================================
#  agent/network_renderer.R
#  -------------------------------------------------------------------------
#  Render a drug-target network spec (produced by the LangGraph reporter
#  LLM) into a PNG via ggraph + ggrepel + tidygraph. Designed to be invoked
#  as a subprocess from agent/network_renderer.py:
#
#      Rscript agent/network_renderer.R --spec <spec.json> --out <out.png>
#
#  The spec JSON schema is documented in agent/network_renderer.py and the
#  reporter prompt in agent/langgraph_agent.py; it is identical to the
#  Python renderer's input.
#
#  Design principles (from internal review of the previous matplotlib
#  renderer + Slack discussion with Yifei Lu & Kaiwen Fang):
#
#    1.  Never multiply layout coordinates by a fixed factor.  ggplot2
#        auto-scales axes, so coordinate multiplication is a no-op for
#        labels rendered at absolute physical sizes (pt/mm).
#    2.  Dynamic canvas size: figure dimensions grow with sqrt(vcount),
#        capped, so node markers and labels stay readable across graph
#        sizes from ~5 to ~150 nodes.
#    3.  `scale_*_continuous(expand = expansion(mult = …))` + `clip="off"`
#        gives ggrepel margin to push labels outwards.
#    4.  Layout selection by graph size: `fr` (Fruchterman-Reingold) for
#        tiny graphs, `stress` for medium (per Yifei), `graphopt` for
#        very large graphs (Kaiwen's recommendation — repulsion charge is
#        scaled by graph density).
#    5.  ggrepel: high `force`, generous `point.padding`/`box.padding`,
#        `bg.colour="white"` halo so labels never get visually clobbered.
#
#  We deliberately keep ALL nodes — no pruning of singletons or low-degree
#  vertices.  The LLM is the one deciding what to emit; the renderer's
#  job is to make whatever it produced legible.
# ============================================================================

suppressPackageStartupMessages({
  library(igraph); library(ggplot2); library(ggraph); library(ggrepel)
  library(tidygraph); library(jsonlite); library(optparse)
})

`%||%` <- function(a, b) if (is.null(a) || (length(a) == 1 && is.na(a))) b else a

# --------- Style constants (kept in sync with the Python renderer) -----------
STATUS_FILL  <- c(approved = "#a8ddb5", investigational = "#f6d58b",
                  preclinical = "#e8c4b1", unknown = "#d8dadd")
STATUS_EDGE  <- c(approved = "#2f855a", investigational = "#c88719",
                  preclinical = "#9a5b36", unknown = "#6b7280")
STATUS_DASH  <- c(approved = "solid",  investigational = "dashed",
                  preclinical = "dotted", unknown = "dashed")
ROLE_OUTLINE <- c(regular = "#34495e", hub = "#e67e22")
GENE_FILL    <- "#cfe2f3"
EDGE_GENE_GREY <- "#9ca3af"

# --------- Spec → tidygraph object ------------------------------------------
spec_to_graph <- function(spec) {
  rows_n <- lapply(spec$nodes, function(n) data.frame(
    name   = as.character(n$id),
    type   = tolower(as.character(n$type   %||% "gene")),
    weight = suppressWarnings(as.numeric(n$weight %||% 0.5)),
    role   = tolower(as.character(n$role   %||% "regular")),
    status = tolower(as.character(n$status %||% "unknown")),
    stringsAsFactors = FALSE
  ))
  nodes <- do.call(rbind, rows_n)
  nodes <- unique(nodes[!is.na(nodes$name) & nzchar(nodes$name), , drop = FALSE])
  nodes$role[!nodes$role %in% names(ROLE_OUTLINE)] <- "regular"
  nodes$status[!nodes$status %in% names(STATUS_FILL)] <- "unknown"
  nodes$weight[is.na(nodes$weight)] <- 0.5

  rows_e <- lapply(spec$edges, function(e) data.frame(
    from     = as.character(e$source),
    to       = as.character(e$target),
    type     = tolower(as.character(e$type %||% "gene-gene")),
    evidence = as.character(e$evidence %||% ""),
    stringsAsFactors = FALSE
  ))
  edges <- do.call(rbind, rows_e)
  if (is.null(edges) || nrow(edges) == 0) {
    edges <- data.frame(from = character(), to = character(),
                        type = character(), evidence = character(),
                        stringsAsFactors = FALSE)
  } else {
    # Drop edges referencing nodes the spec didn't declare.
    keep <- edges$from %in% nodes$name & edges$to %in% nodes$name
    edges <- edges[keep, , drop = FALSE]
  }

  graph_from_data_frame(edges, vertices = nodes, directed = FALSE)
}

# --------- Density-driven cosmetic parameters --------------------------------
density_params <- function(g) {
  v <- max(1L, vcount(g))
  e <- ecount(g)
  list(
    # Dynamic canvas: bigger-but-bounded.  Empirically tuned: 8" base,
    # +sqrt(v) inches per ~12 extra nodes, capped at 18" so the PDF
    # renderer keeps it inside a single page width.
    width  = min(18, 8 + sqrt(v) * 0.95),
    height = min(14, 6.5 + sqrt(v) * 0.78),
    # Gene marker size range scales softly with density.
    node_size_range = if (v <= 12) c(4, 9)
                      else if (v <= 40) c(3, 7)
                      else c(2, 5),
    label_size = if (v <= 12) 3.4
                 else max(2.4, 3.4 * (12 / v) ^ 0.30),
    show_edge_labels = e <= 30,
    edge_label_size = max(2.0, 2.6 * (12 / max(1, e)) ^ 0.25)
  )
}

# --------- Layout chooser (best-practice #3 + Yifei's recommendation) --------
choose_layout <- function(g) {
  v <- vcount(g)
  if (v <= 8) {
    return(create_layout(g, "fr"))
  }
  if (v <= 50) {
    return(create_layout(g, "stress"))
  }
  # Very dense — let graphopt do its thing with density-scaled repulsion.
  density <- edge_density(g)
  charge  <- 0.001 + density * 0.05
  create_layout(g, "igraph",
                algorithm = "graphopt",
                charge = charge,
                mass = 30, spring.length = 2, max.sa.movement = 0.1)
}

# --------- Render ------------------------------------------------------------
render <- function(spec, out_path) {
  g <- spec_to_graph(spec)
  if (vcount(g) == 0) {
    message("[network_renderer.R] empty spec — nothing to draw")
    return(invisible(NULL))
  }
  dp     <- density_params(g)
  layout <- choose_layout(g)

  # Per-edge target-drug status comes from the drug endpoint's `status`
  # vertex attribute.  We build it on the edges of `layout` (a data frame
  # of edge endpoints in the ggraph layout) so the colour scale receives
  # one value per row.
  status_lookup <- setNames(V(g)$status, V(g)$name)
  td_edge_status <- function(layer_data) {
    src <- layer_data$node1.name %||% layer_data$.from %||% layer_data$from
    tgt <- layer_data$node2.name %||% layer_data$.to   %||% layer_data$to
    drug_end <- ifelse(status_lookup[src] != "unknown" & src %in% V(g)$name &
                       V(g)$type[match(src, V(g)$name)] == "drug",
                       src, tgt)
    status_lookup[drug_end]
  }

  p <- ggraph(layout) +
    # gene-gene edges — thin curved grey lines
    geom_edge_arc(aes(filter = type == "gene-gene"),
                  colour = EDGE_GENE_GREY, alpha = 0.55,
                  width = 0.45, strength = 0.15) +
    # target-drug edges, coloured + dashed by drug status (via .N() to read
    # status from the destination vertex)
    geom_edge_arc(aes(filter = type == "target-drug",
                      colour   = .N()$status[to],
                      linetype = .N()$status[to]),
                  width = 0.75, alpha = 0.95, strength = 0.18) +
    scale_edge_colour_manual(values = STATUS_EDGE,
                             breaks = names(STATUS_EDGE), guide = "none") +
    scale_edge_linetype_manual(values = STATUS_DASH,
                               breaks = names(STATUS_DASH), guide = "none") +
    # gene nodes — circles, size by `weight`, outline by `role`
    geom_node_point(aes(filter = type == "gene",
                        size   = weight,
                        colour = role),
                    shape = 21, fill = GENE_FILL, stroke = 0.9) +
    scale_colour_manual(values = ROLE_OUTLINE, guide = "none") +
    scale_size(range = dp$node_size_range, guide = "none") +
    # drug nodes — squares, colour-coded fill = status
    geom_node_point(aes(filter = type == "drug", fill = status),
                    shape = 22,
                    size = if (vcount(g) <= 40) 5 else 4,
                    stroke = 0.9, colour = "black") +
    scale_fill_manual(name = "Drug status",
                      values = STATUS_FILL,
                      breaks = names(STATUS_FILL)) +
    # node labels via ggrepel with high force / padding (Kaiwen's tip)
    geom_node_text(aes(label = name),
                   size              = dp$label_size,
                   repel             = TRUE,
                   max.overlaps      = Inf,
                   point.padding     = 0.45,
                   box.padding       = 0.55,
                   force             = 2.5,
                   force_pull        = 0.1,
                   min.segment.length = 0.15,
                   segment.size      = 0.25,
                   segment.colour    = "#9ca3af",
                   bg.colour         = "white",
                   bg.r              = 0.12,
                   family            = "sans") +
    # Best-practice #2 — expand axes so ggrepel can push outwards
    scale_x_continuous(expand = expansion(mult = 0.22)) +
    scale_y_continuous(expand = expansion(mult = 0.22)) +
    coord_cartesian(clip = "off") +
    ggtitle(spec$title %||% "Drug-target network",
            subtitle = spec$subtitle %||% NULL) +
    theme_void(base_size = 11) +
    theme(plot.title       = element_text(face = "bold", hjust = 0.5),
          plot.subtitle    = element_text(hjust = 0.5, colour = "#374151"),
          legend.position  = "bottom",
          plot.margin      = margin(8, 18, 8, 18))

  # Optional edge labels (PMID/DOI) — only when sparse enough.
  # ggraph 2.2 doesn't ship a geom_edge_label helper, so we compute
  # midpoints from the layout data frame and add them as a regular
  # geom_text layer (with ggrepel-style halo).
  if (dp$show_edge_labels) {
    td_edges_idx <- which(igraph::edge_attr(g, "type") == "target-drug" &
                          nzchar(igraph::edge_attr(g, "evidence")))
    if (length(td_edges_idx) > 0) {
      ends_mat <- igraph::ends(g, td_edges_idx, names = TRUE)
      pos      <- as.data.frame(layout)[, c("name", "x", "y")]
      px       <- setNames(pos$x, pos$name)
      py       <- setNames(pos$y, pos$name)
      label_df <- data.frame(
        x     = (px[ends_mat[, 1]] + px[ends_mat[, 2]]) / 2,
        y     = (py[ends_mat[, 1]] + py[ends_mat[, 2]]) / 2,
        label = igraph::edge_attr(g, "evidence", index = td_edges_idx),
        stringsAsFactors = FALSE
      )
      p <- p + geom_label(
        data = label_df,
        aes(x = x, y = y, label = label),
        size = dp$edge_label_size,
        colour = "#1f2937",
        fill = scales::alpha("white", 0.85),
        label.size = NA,
        label.padding = unit(0.10, "lines"),
        inherit.aes = FALSE
      )
    }
  }

  dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
  ggsave(out_path, plot = p,
         width = dp$width, height = dp$height,
         dpi = 200, bg = "white", limitsize = FALSE)
  layout_name <- attr(layout, "ggraph.layout") %||% "auto"
  message("[network_renderer.R] wrote ", out_path,
          " (", vcount(g), " nodes, ", ecount(g), " edges, layout=",
          as.character(layout_name)[1], ")")
  invisible(out_path)
}

# --------- CLI ---------------------------------------------------------------
opt_list <- list(
  make_option("--spec",      type = "character", help = "Path to spec JSON"),
  make_option("--out",       type = "character", help = "Output PNG path"),
  make_option("--smoke-test", action = "store_true", default = FALSE,
              help = "Render a built-in synthetic spec (for testing)")
)
opt <- parse_args(OptionParser(option_list = opt_list))

if (isTRUE(opt$`smoke-test`)) {
  spec <- list(
    title = "Smoke test (R/ggraph)",
    subtitle = "Demo — not real data",
    filename_hint = "smoke_test_r",
    nodes = list(
      list(id = "TREM2", type = "gene", weight = 0.9, role = "hub"),
      list(id = "APP",   type = "gene", weight = 0.7, role = "hub"),
      list(id = "MAPT",  type = "gene", weight = 0.5),
      list(id = "APOE",  type = "gene", weight = 0.6),
      list(id = "Lecanemab", type = "drug", status = "approved"),
      list(id = "Donanemab", type = "drug", status = "approved"),
      list(id = "AL002",     type = "drug", status = "investigational"),
      list(id = "DemoX",     type = "drug", status = "preclinical")
    ),
    edges = list(
      list(source = "TREM2", target = "APP",  type = "gene-gene"),
      list(source = "APP",   target = "MAPT", type = "gene-gene"),
      list(source = "APOE",  target = "APP",  type = "gene-gene"),
      list(source = "APP",   target = "Lecanemab", type = "target-drug",
           evidence = "PMID:36625625"),
      list(source = "APP",   target = "Donanemab", type = "target-drug",
           evidence = "PMID:38157881"),
      list(source = "TREM2", target = "AL002", type = "target-drug",
           evidence = "NCT04592874"),
      list(source = "MAPT",  target = "DemoX", type = "target-drug",
           evidence = "PMID:00000000")
    )
  )
  out <- opt$out %||% "/tmp/r_smoke.png"
  render(spec, out)
} else {
  if (is.null(opt$spec) || is.null(opt$out)) {
    stop("Provide --spec <path> --out <path> (or --smoke-test).")
  }
  spec <- fromJSON(opt$spec, simplifyVector = FALSE)
  render(spec, opt$out)
}

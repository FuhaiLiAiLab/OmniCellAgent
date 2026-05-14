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
    # Dynamic canvas: prefer a near-square aspect so packed components
    # don't all line up in a single horizontal row.  Slightly more
    # generous than the previous tuning to give labels room to breathe.
    width  = min(16, 8.0 + sqrt(v) * 0.6),
    height = min(14, 7.0 + sqrt(v) * 0.6),
    node_size_range = if (v <= 12) c(4.5, 10)
                      else if (v <= 40) c(3.5, 8)
                      else c(2.5, 6),
    # Bigger labels across the board — previous floors (3.0 / 4.0)
    # rendered too small once the PDF fit the image to column width.
    label_size = if (v <= 12) 5.0
                 else max(3.8, 5.0 * (12 / v) ^ 0.28),
    show_edge_labels = e <= 30,
    edge_label_size = max(2.8, 3.2 * (12 / max(1, e)) ^ 0.22)
  )
}

# --------- Layout chooser (best-practice #3 + Yifei's recommendation) --------
#
# When the spec has several disconnected components (the common case when
# the LLM emits many drug→target dyads without gene-gene bridges), the
# default stress layout puts every component side-by-side and the figure
# reads as "several graphlets next to each other".  We use a custom
# grid packer that places component centroids on a near-square grid so
# the components fill the canvas evenly in both dimensions, instead of
# ggraph's default "concatenate side-by-side" or igraph's leftward
# rectangle packing.
choose_layout <- function(g) {
  v <- vcount(g)
  if (v <= 8) {
    return(create_layout(g, "fr"))
  }

  # Pick the per-component (a.k.a. sub-graph) layout function.
  sub_layout_fn <- if (v <= 50) {
    function(sg) graphlayouts::layout_with_stress(sg)
  } else {
    density <- edge_density(g)
    charge  <- 0.001 + density * 0.05
    function(sg) igraph::layout_with_graphopt(
      sg, charge = charge, mass = 30, spring.length = 2,
      max.sa.movement = 0.1
    )
  }

  comps  <- igraph::components(g)
  coords <- matrix(0, nrow = v, ncol = 2)

  if (comps$no <= 1) {
    coords <- sub_layout_fn(g)
  } else {
    # ── Custom packer ──────────────────────────────────────────────
    # Multi-node components are laid out on a near-square grid with
    # cell sizes that scale with sqrt(component size), so singletons
    # don't claim the same footprint as a 10-node cluster.  Any pure
    # singletons are then collapsed into a compact strip below the
    # grid, instead of getting one full grid cell each.
    comp_ids   <- comps$membership
    sizes_full <- as.integer(table(comp_ids))
    multi_ids  <- which(sizes_full > 1)
    single_ids <- which(sizes_full == 1)

    # If literally everything is a singleton, fall back to one tidy row.
    if (length(multi_ids) == 0) {
      n_s <- length(single_ids)
      cols <- max(1, ceiling(sqrt(n_s * 1.6)))
      for (k in seq_len(n_s)) {
        m <- which(comp_ids == single_ids[k])
        row_idx <- (k - 1) %/% cols
        col_idx <- (k - 1) %%  cols
        coords[m, 1] <- (col_idx - (cols - 1) / 2) * 1.6
        coords[m, 2] <- -row_idx * 1.6
      }
    } else {
      multi_sizes <- sizes_full[multi_ids]
      ord <- order(multi_sizes, decreasing = TRUE)
      multi_ids <- multi_ids[ord]
      multi_sizes <- multi_sizes[ord]

      # Grid dims for multi-node components (near-square, slight wide bias).
      n_m    <- length(multi_ids)
      n_cols <- max(1, ceiling(sqrt(n_m)))
      n_rows <- ceiling(n_m / n_cols)
      # Reference cell size — components get scale ∝ sqrt(size) but
      # capped so a single huge cluster doesn't push everything off-canvas.
      base_cell <- 2.6
      scale_of <- function(sz) min(1.4, max(0.5, sqrt(sz / 4)))

      for (k in seq_len(n_m)) {
        members <- which(comp_ids == multi_ids[k])
        sg      <- igraph::induced_subgraph(g, members)
        sub_xy  <- sub_layout_fn(sg)
        cx <- (max(sub_xy[, 1]) + min(sub_xy[, 1])) / 2
        cy <- (max(sub_xy[, 2]) + min(sub_xy[, 2])) / 2
        rx <- max(1e-6, (max(sub_xy[, 1]) - min(sub_xy[, 1])) / 2)
        ry <- max(1e-6, (max(sub_xy[, 2]) - min(sub_xy[, 2])) / 2)
        s  <- scale_of(multi_sizes[k])
        sub_xy[, 1] <- (sub_xy[, 1] - cx) / rx * s
        sub_xy[, 2] <- (sub_xy[, 2] - cy) / ry * s
        row_idx <- (k - 1) %/% n_cols
        col_idx <- (k - 1) %%  n_cols
        tx <- (col_idx - (n_cols - 1) / 2) * base_cell
        ty <- ((n_rows - 1) / 2 - row_idx) * base_cell
        coords[members, 1] <- sub_xy[, 1] + tx
        coords[members, 2] <- sub_xy[, 2] + ty
      }

      # Compact strip of singletons under the grid (snake-wrapped if many).
      if (length(single_ids) > 0) {
        n_s <- length(single_ids)
        strip_cols <- max(n_cols, ceiling(sqrt(n_s * 2.0)))
        # Place strip just below the grid's bottom row with a small gap.
        strip_y0 <- -((n_rows - 1) / 2 * base_cell) - base_cell * 0.9
        for (k in seq_len(n_s)) {
          m <- which(comp_ids == single_ids[k])
          row_idx <- (k - 1) %/% strip_cols
          col_idx <- (k - 1) %%  strip_cols
          coords[m, 1] <- (col_idx - (strip_cols - 1) / 2) * 1.1
          coords[m, 2] <- strip_y0 - row_idx * 0.9
        }
      }
    }
  }
  # Normalise to ~[-1, 1] so downstream cosmetic scales are predictable.
  rng <- apply(coords, 2, function(c) {
    r <- range(c); if (diff(r) == 0) 1 else diff(r) / 2
  })
  ctr <- apply(coords, 2, function(c) (max(c) + min(c)) / 2)
  coords[, 1] <- (coords[, 1] - ctr[1]) / rng[1]
  coords[, 2] <- (coords[, 2] - ctr[2]) / rng[2]
  create_layout(g, layout = data.frame(x = coords[, 1], y = coords[, 2]))
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
                   point.padding     = 0.3,    # was 0.45 — let labels sit closer to nodes
                   box.padding       = 0.32,   # was 0.55 — less aggressive box-around-text
                   force             = 1.4,    # was 2.5 — softer repulsion
                   force_pull        = 0.5,    # was 0.1 — stronger pull back toward node
                   min.segment.length = 0.15,
                   segment.size      = 0.25,
                   segment.colour    = "#9ca3af",
                   bg.colour         = "white",
                   bg.r              = 0.12,
                   family            = "sans") +
    # Axis expansion — narrower than v1; was 0.22, now 0.12 so the
    # network actually fills the canvas instead of looking lost in
    # whitespace.  ggrepel still has clip="off" to bleed into margin.
    scale_x_continuous(expand = expansion(mult = 0.12)) +
    scale_y_continuous(expand = expansion(mult = 0.12)) +
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
        # `label.size` was renamed to `linewidth` in ggplot2 3.5; use the
        # new name to silence the deprecation warning.
        linewidth = 0,
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

"""
agent/network_renderer.py
=========================

Render a *gene + drug* network spec produced by the LangGraph reporter LLM
into a static PNG using mature packages (NetworkX for the graph data
structure and layout, Matplotlib for drawing). No hand-rolled SVG, no
hand-tuned force-directed layout — both have been a maintenance burden in
the experimental/graph_recon prototypes.

Public entry point
------------------
    render_drug_target_network(spec: dict, out_dir: str | Path,
                               filename_hint: str | None = None) -> Path | None

`spec` is the JSON object the reporter LLM is asked to emit. Schema:

    {
        "title":         "Drug-target network for Alzheimer's disease",
        "subtitle":      "Top DEGs + therapeutic agents",  # optional
        "filename_hint": "ad_drug_target_network",         # optional
        "nodes": [
            {"id": "TREM2",   "type": "gene", "weight": 0.85, "role": "hub"},
            {"id": "APP",     "type": "gene", "weight": 0.42},
            {"id": "Lecanemab", "type": "drug", "status": "approved"},
            ...
        ],
        "edges": [
            {"source": "TREM2", "target": "APP",       "type": "gene-gene",  "weight": 0.35},
            {"source": "APP",   "target": "Lecanemab", "type": "target-drug","evidence": "PMID:36625625"},
            ...
        ]
    }

Node types
    - "gene": drawn as a circle. `weight` (0..1) scales node size; `role`
      = "hub" gets an orange outline.
    - "drug": drawn as a rounded rectangle. `status` ∈ {approved,
      investigational, preclinical, unknown} sets the fill colour.

Edge types
    - "gene-gene": thin grey curve (Bezier via matplotlib `connectionstyle`).
    - "target-drug": colour-coded by the drug's status, dashed for
      non-approved. Optional `evidence` string is rendered next to the edge.

Why these libraries
-------------------
- NetworkX is the de-facto Python graph library; `spring_layout`
  (Fruchterman-Reingold) and `kamada_kawai_layout` give acceptable layouts
  out of the box.
- Matplotlib has stable patches for circles, fancy rectangles and arrow
  paths; `connectionstyle='arc3,rad=…'` produces curved edges without
  custom Bezier math.

The renderer never raises on bad input — it logs and returns ``None`` so
the rest of the pipeline (PDF compile) keeps going.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; needed when running headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DRUG_STATUS_STYLE: dict[str, dict[str, Any]] = {
    "approved":         {"fill": "#a8ddb5", "edge": "#2f855a", "linestyle": "-"},
    "investigational":  {"fill": "#f6d58b", "edge": "#c88719", "linestyle": "--"},
    "preclinical":      {"fill": "#e8c4b1", "edge": "#9a5b36", "linestyle": ":"},
    "unknown":          {"fill": "#d8dadd", "edge": "#6b7280", "linestyle": "--"},
}

_GENE_FILL = "#cfe2f3"
_GENE_HUB_OUTLINE = "#e67e22"
_GENE_REGULAR_OUTLINE = "#34495e"

_GENE_EDGE_COLOR = "#9ca3af"
_GENE_EDGE_WIDTH = 0.9
_TARGET_DRUG_EDGE_WIDTH = 1.6

# Canvas defaults (small graphs). Scaled up by _density_params for dense
# networks so labels and nodes stay readable when the LLM emits ≫ 20 nodes.
_FIG_W = 11.0
_FIG_H = 8.5
_DPI = 180


def _density_params(n_nodes: int, n_edges: int) -> dict[str, float]:
    """
    Heuristic scaling: bigger figure + smaller fonts/node radii + lighter
    edges as the network grows. Tuned so a 10-node graph looks like the
    smoke test and a 100-node graph still fits.
    """
    n = max(1, n_nodes)
    e = max(0, n_edges)
    # Figure scales with sqrt(n) so a 4× node count gives a 2× canvas,
    # but capped so the embedded PDF figure fits on a single page.
    fw = _FIG_W * max(1.0, (n / 12.0) ** 0.5)
    fh = _FIG_H * max(1.0, (n / 12.0) ** 0.5)
    fw = min(fw, 16.0)
    fh = min(fh, 12.0)

    node_scale = 1.0 if n <= 12 else max(0.5, (12.0 / n) ** 0.5)
    # Keep label fonts large enough that they survive being scaled down
    # to PDF column width. The min floor (≥6.5 pt) was set after observing
    # 5.5 pt labels disappear when pandoc fits the image to ~6 in width.
    label_fs = 8.0 if n <= 12 else max(6.5, 8.0 * (12.0 / n) ** 0.3)
    drug_fs = 7.5 if n <= 12 else max(6.5, 7.5 * (12.0 / n) ** 0.3)
    # When edges are dense, hide PMID/DOI labels (just keep colours).
    show_citations = e <= 24
    return {
        "fw": fw, "fh": fh,
        "node_scale": node_scale,
        "label_fs": label_fs,
        "drug_fs": drug_fs,
        "show_citations": show_citations,
    }


# Block markers the reporter wraps the JSON in (post-processed below).
_GRAPH_SPEC_FENCE_RE = re.compile(
    r"```(?:graph-spec|graphspec|network-spec)\s*\n(?P<body>\{.*?\})\s*\n```",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_drug_target_network(
    spec: dict[str, Any],
    out_dir: str | Path,
    filename_hint: str | None = None,
) -> Path | None:
    """Render a single graph spec to PNG.

    Primary backend: ``Rscript agent/network_renderer.R`` (ggraph +
    ggrepel + tidygraph) — uses mature R graph-drawing packages that
    handle dense networks, label collisions, and curved edges without
    hand-rolled physics.

    Fallback: in-process matplotlib renderer (this module) — used when
    Rscript is missing, the R script errors out, or the R packages
    aren't installed.

    Side effect: alongside ``<base>.png`` we also write ``<base>.spec.json``
    (the exact spec dict consumed by the renderer) and drop a
    self-contained ``network_renderer.R`` + ``rerender.sh`` into the
    output folder so the figure can be re-rendered after manual edits
    to either the spec or the R script.

    Returns the PNG path on success, ``None`` on failure (so the
    pipeline keeps going).
    """
    out_dir = Path(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        if not spec or not spec.get("nodes"):
            print("[network_renderer] empty spec — nothing to draw")
            return None

        out_path = _resolve_output_path(spec, out_dir, filename_hint)

        # Always persist the spec next to the PNG so a human can edit
        # it and re-run the renderer without round-tripping through the
        # LLM. We write this BEFORE rendering so the JSON is available
        # even if rendering fails downstream.
        _write_spec_sidecar(spec, out_path)
        _ensure_local_render_assets(out_dir)

        # ── Try R first ─────────────────────────────────────────────
        r_path = _try_r_renderer(spec, out_path)
        if r_path is not None:
            return r_path

        # ── matplotlib fallback ─────────────────────────────────────
        print("[network_renderer] falling back to matplotlib backend")
        return _render_with_matplotlib(spec, out_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[network_renderer] render failed: {exc}")
        return None


def _write_spec_sidecar(spec: dict[str, Any], png_path: Path) -> Path:
    """Write the spec as JSON next to the PNG (same basename, ``.spec.json``)."""
    spec_path = png_path.with_suffix("").with_suffix(".spec.json")
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    return spec_path


def _ensure_local_render_assets(out_dir: Path) -> None:
    """Drop a self-contained renderer + helper script into ``out_dir``.

    The agent invokes the renderer from ``agent/network_renderer.R``; for
    user-facing re-renders we copy that script and a small ``rerender.sh``
    into the output folder so the figures can be regenerated after
    hand-edits to either the spec JSON or the R rendering code.

    Existing copies are left in place when their content matches, so we
    don't trample over a user's local edits to ``network_renderer.R``.
    """
    import shutil

    r_src = Path(__file__).resolve().parent / "network_renderer.R"
    r_dst = out_dir / "network_renderer.R"
    if r_src.exists():
        try:
            if not r_dst.exists() or r_dst.read_text() != r_src.read_text():
                shutil.copy2(r_src, r_dst)
        except OSError as exc:
            print(f"[network_renderer] could not copy R script to {r_dst}: {exc}")

    rerender_dst = out_dir / "rerender.sh"
    rerender_body = _RERENDER_SH_BODY
    try:
        if not rerender_dst.exists() or rerender_dst.read_text() != rerender_body:
            rerender_dst.write_text(rerender_body, encoding="utf-8")
            rerender_dst.chmod(0o755)
    except OSError as exc:
        print(f"[network_renderer] could not write {rerender_dst}: {exc}")

    readme_dst = out_dir / "README.md"
    if not readme_dst.exists():
        try:
            readme_dst.write_text(_RENDER_README_BODY, encoding="utf-8")
        except OSError as exc:
            print(f"[network_renderer] could not write {readme_dst}: {exc}")


_RERENDER_SH_BODY = """#!/usr/bin/env bash
# Re-render every *.spec.json in this directory back into a PNG using the
# local network_renderer.R. Edit the JSON (or the R script) and re-run.
#
#     ./rerender.sh                 # re-render all specs
#     ./rerender.sh foo.spec.json   # re-render a single spec
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v Rscript >/dev/null 2>&1; then
  echo "Rscript not found on PATH; install R or use the matplotlib fallback." >&2
  exit 1
fi

shopt -s nullglob
targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then
  targets=( *.spec.json )
fi
if [ ${#targets[@]} -eq 0 ]; then
  echo "no *.spec.json files found in $(pwd)" >&2
  exit 1
fi

for spec in "${targets[@]}"; do
  out="${spec%.spec.json}.png"
  echo "[rerender] $spec -> $out"
  Rscript --vanilla ./network_renderer.R --spec "$spec" --out "$out"
done
"""


_RENDER_README_BODY = """# network_plots/

Each figure here has three files with the same basename:

- ``<name>.png`` — the rendered figure embedded in the report
- ``<name>.spec.json`` — the graph spec (nodes / edges / styling) the
  reporter agent emitted; this is what the renderer consumed
- ``network_renderer.R`` — a copy of the R rendering code, so the figure
  can be regenerated locally without any project-side imports

## Adjusting a figure by hand

1. Open ``<name>.spec.json`` and edit nodes, edges, statuses, weights,
   ``title``/``subtitle``, etc. The schema is documented at the top of
   ``network_renderer.R``.
2. Optionally tweak ``network_renderer.R`` for styling changes (font
   size, layout choice, colour palette).
3. Re-render:

   ```bash
   ./rerender.sh                       # rebuild every PNG in this folder
   ./rerender.sh <name>.spec.json      # rebuild just one
   ```

   Requires ``Rscript`` plus the R packages listed at the top of
   ``network_renderer.R`` (``igraph``, ``ggraph``, ``ggrepel``,
   ``tidygraph``, ``jsonlite``, ``optparse``, ``graphlayouts``).
"""


def _try_r_renderer(spec: dict[str, Any], out_path: Path) -> Path | None:
    """Invoke the R renderer via subprocess. Returns out_path on success."""
    import shutil
    import subprocess
    import tempfile

    rscript = shutil.which("Rscript")
    if rscript is None:
        return None  # silently fall through; caller logs the fallback

    r_script = Path(__file__).resolve().parent / "network_renderer.R"
    if not r_script.exists():
        return None

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(spec, tmp, ensure_ascii=False)
        spec_path = Path(tmp.name)

    try:
        proc = subprocess.run(
            [rscript, "--vanilla", str(r_script),
             "--spec", str(spec_path),
             "--out", str(out_path)],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            # R-side failure — log a trimmed stderr, then fall back
            err = (proc.stderr or "").strip().splitlines()
            tail = "\n   ".join(err[-6:]) if err else "(no stderr)"
            print(f"[network_renderer] R renderer exit={proc.returncode}; "
                  f"tail of stderr:\n   {tail}")
            return None
        if not out_path.exists():
            print("[network_renderer] R renderer reported success but no PNG written")
            return None
        return out_path
    except subprocess.TimeoutExpired:
        print("[network_renderer] R renderer timed out after 120s")
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"[network_renderer] R renderer failed to launch: {exc}")
        return None
    finally:
        try:
            spec_path.unlink()
        except OSError:
            pass


def _render_with_matplotlib(spec: dict[str, Any], out_path: Path) -> Path | None:
    """Original in-process matplotlib renderer; used only as a fallback."""
    G, gene_nodes, drug_nodes, edges = _build_graph(spec)
    if G.number_of_nodes() == 0:
        print("[network_renderer] empty spec — nothing to draw")
        return None
    pos = _compute_layout(G, gene_nodes, drug_nodes)
    _draw_and_save(G, pos, spec, gene_nodes, drug_nodes, edges, out_path)
    return out_path


def render_specs_in_markdown(
    markdown_text: str,
    out_dir: str | Path,
    relative_image_dir: str = "network_plots",
) -> tuple[str, list[Path]]:
    """
    Find every ```graph-spec``` fence in the markdown, render each to PNG,
    and replace the fence with a markdown image reference. Returns the
    rewritten markdown and the list of generated PNG paths.

    `relative_image_dir` is the path used in the markdown ``![](...)``
    reference; the on-disk write target is ``out_dir/relative_image_dir``.
    """
    out_dir = Path(out_dir)
    image_dir = out_dir / relative_image_dir
    image_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    counter = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal counter
        body = match.group("body")
        try:
            spec = json.loads(body)
        except json.JSONDecodeError as exc:
            print(f"[network_renderer] malformed graph-spec JSON #{counter}: {exc}")
            return match.group(0)  # leave the original block in place

        counter += 1
        hint = spec.get("filename_hint") or f"network_{counter:02d}"
        png_path = render_drug_target_network(spec, image_dir, filename_hint=hint)
        if png_path is None:
            return match.group(0)
        generated.append(png_path)

        rel = f"{relative_image_dir}/{png_path.name}"
        title = spec.get("title") or "Drug-target network"
        # Replace the fence with an embedded image; keep title as alt-text + caption.
        return f"![{title}]({rel})\n\n*Figure — {title}.*\n"

    rewritten = _GRAPH_SPEC_FENCE_RE.sub(_replace, markdown_text)
    return rewritten, generated


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _build_graph(
    spec: dict[str, Any],
) -> tuple[nx.Graph, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Construct an undirected NetworkX graph and bucket nodes by type."""
    G: nx.Graph = nx.Graph()
    gene_nodes: list[dict[str, Any]] = []
    drug_nodes: list[dict[str, Any]] = []

    seen: set[str] = set()
    for node in spec.get("nodes", []) or []:
        nid = str(node.get("id", "")).strip()
        if not nid or nid in seen:
            continue
        seen.add(nid)
        ntype = (node.get("type") or "gene").strip().lower()
        attrs = {
            "type": ntype,
            "weight": float(node.get("weight", 0.5) or 0.5),
            "role": (node.get("role") or "regular").strip().lower(),
            "status": (node.get("status") or "unknown").strip().lower(),
        }
        G.add_node(nid, **attrs)
        clean = dict(node, **{"id": nid, "type": ntype})
        if ntype == "drug":
            drug_nodes.append(clean)
        else:
            gene_nodes.append(clean)

    edges: list[dict[str, Any]] = []
    for edge in spec.get("edges", []) or []:
        src = str(edge.get("source", "")).strip()
        tgt = str(edge.get("target", "")).strip()
        if not src or not tgt or src not in seen or tgt not in seen:
            continue
        etype = (edge.get("type") or "gene-gene").strip().lower()
        attrs = {
            "type": etype,
            "weight": float(edge.get("weight", 0.3) or 0.3),
            "evidence": (edge.get("evidence") or "").strip(),
        }
        # Skip duplicate undirected edges
        if G.has_edge(src, tgt):
            continue
        G.add_edge(src, tgt, **attrs)
        edges.append({**edge, "source": src, "target": tgt, "type": etype})

    return G, gene_nodes, drug_nodes, edges


def _compute_layout(
    G: nx.Graph,
    gene_nodes: Iterable[dict[str, Any]],
    drug_nodes: Iterable[dict[str, Any]],
) -> dict[str, tuple[float, float]]:
    """
    Layout strategy:

      1. Lay genes out on a single canvas. If the gene subgraph has
         multiple connected components (common when the LLM forgets to
         emit gene-gene edges), arrange the component centroids on a
         circle, then run spring inside each component. This stops the
         layout from collapsing all dyads on top of each other.
      2. Place every drug on a radial offset from its target centroid,
         far enough out that the drug rectangle does not overlap the
         gene circle. Multiple drugs sharing a target neighbourhood are
         fanned symmetrically.
      3. Run a short final spring relaxation on the whole graph with
         these positions as seed — just enough to clean up local
         overlaps without dragging components back into each other.
    """
    import math
    from collections import defaultdict

    gene_ids = [n["id"] for n in gene_nodes]
    drug_ids = [n["id"] for n in drug_nodes]
    if not gene_ids:
        return nx.spring_layout(G, seed=42, k=1.2 / max(1, G.number_of_nodes()) ** 0.5)

    sub_gene = G.subgraph(gene_ids).copy()

    # ── (1) Gene layout per component, then offset each component to a
    #        seat on a circle so disconnected dyads spread out.
    components = [list(c) for c in nx.connected_components(sub_gene)]
    components.sort(key=len, reverse=True)
    n_components = len(components)
    seed_pos: dict[str, tuple[float, float]] = {}

    # Radius of the ring on which component centroids sit. Tuned so that
    # even small dyads (1 gene + 1 drug) don't crash into other dyads.
    if n_components <= 1:
        ring_radius = 0.0
    else:
        ring_radius = 0.45 + 0.06 * n_components

    for idx, comp in enumerate(components):
        comp_sub = sub_gene.subgraph(comp).copy()
        local = nx.spring_layout(
            comp_sub,
            seed=42 + idx,
            k=1.4 / max(1, comp_sub.number_of_nodes()) ** 0.5,
            iterations=120,
        )
        # Centre this component on its assigned seat.
        if n_components <= 1:
            cx, cy = 0.0, 0.0
        else:
            angle = 2 * math.pi * idx / n_components
            cx, cy = ring_radius * math.cos(angle), ring_radius * math.sin(angle)
        # Re-centre local layout around (0,0) before offsetting.
        if local:
            lx = sum(p[0] for p in local.values()) / len(local)
            ly = sum(p[1] for p in local.values()) / len(local)
            for n, (x, y) in local.items():
                seed_pos[n] = (x - lx + cx, y - ly + cy)

    # ── (2) Drug placement. Group by sorted neighbour-tuple so multi-drug
    #        same-target buckets are fanned out together; otherwise each
    #        drug sits on a generous outward offset from its target(s).
    g_cx = sum(p[0] for p in seed_pos.values()) / max(1, len(seed_pos))
    g_cy = sum(p[1] for p in seed_pos.values()) / max(1, len(seed_pos))

    bucket_to_drugs: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for did in drug_ids:
        nbrs = tuple(sorted(n for n in G.neighbors(did) if n in seed_pos))
        bucket_to_drugs[nbrs].append(did)

    # Generous offset — gene radius ≈ 0.06, drug box ≈ 0.10 wide, so we
    # want centres at least 0.18 apart to avoid overlap at typical sizes.
    base_offset = 0.32
    spread = 0.22
    for nbrs, members in bucket_to_drugs.items():
        if nbrs:
            nx_avg = sum(seed_pos[n][0] for n in nbrs) / len(nbrs)
            ny_avg = sum(seed_pos[n][1] for n in nbrs) / len(nbrs)
        else:
            nx_avg, ny_avg = g_cx, g_cy
        dx, dy = nx_avg - g_cx, ny_avg - g_cy
        norm = math.hypot(dx, dy) or 1.0
        ux, uy = dx / norm, dy / norm
        px, py = -uy, ux
        for idx, did in enumerate(members):
            t = ((idx + 1) // 2) * (1 if idx % 2 == 0 else -1)
            seed_pos[did] = (
                nx_avg + base_offset * ux + t * spread * px,
                ny_avg + base_offset * uy + t * spread * py,
            )

    # ── (3) Light final relaxation: fix gene positions (don't let the
    #        spring drag dyads back together) and let drugs settle if
    #        the seed placement created residual conflicts.
    fixed = list(seed_pos.keys() & set(gene_ids))
    final_pos = nx.spring_layout(
        G,
        pos=seed_pos,
        fixed=fixed if fixed else None,
        seed=42,
        k=0.6 / max(1, G.number_of_nodes()) ** 0.5,
        iterations=20,
    )
    return final_pos


def _resolve_output_path(
    spec: dict[str, Any], out_dir: Path, filename_hint: str | None
) -> Path:
    base = filename_hint or spec.get("filename_hint") or "drug_target_network"
    # Strip extensions, sanitize for filesystem.
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", str(base).rstrip(".png").rstrip(".svg"))
    return out_dir / f"{base or 'drug_target_network'}.png"


def _node_size(node_attrs: dict[str, Any], scale: float = 1.0) -> float:
    """Map gene `weight` ∈ [0,1] → matplotlib scatter `s` (≈ area in pts²).
    `scale` shrinks gene circles in dense graphs (set by _density_params).
    """
    w = float(node_attrs.get("weight", 0.5) or 0.5)
    return (400.0 + 1400.0 * max(0.0, min(1.0, w))) * (scale ** 2)


def _draw_and_save(
    G: nx.Graph,
    pos: dict[str, tuple[float, float]],
    spec: dict[str, Any],
    gene_nodes: list[dict[str, Any]],
    drug_nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    out_path: Path,
) -> None:
    dp = _density_params(G.number_of_nodes(), G.number_of_edges())
    fig, ax = plt.subplots(figsize=(dp["fw"], dp["fh"]), dpi=_DPI)
    ax.set_axis_off()

    # ---- Edges: draw gene-gene first (background), then target-drug (foreground)
    gene_gene = [(e["source"], e["target"]) for e in edges if e["type"] == "gene-gene"]
    if gene_gene:
        nx.draw_networkx_edges(
            G, pos, edgelist=gene_gene, ax=ax,
            edge_color=_GENE_EDGE_COLOR, width=_GENE_EDGE_WIDTH, alpha=0.7,
            connectionstyle="arc3,rad=0.12",
        )

    # Group target-drug edges by status so we set linestyle and colour per group.
    by_status: dict[str, list[tuple[str, str]]] = {}
    evidence_labels: dict[tuple[str, str], str] = {}
    for e in edges:
        if e["type"] != "target-drug":
            continue
        # Find the drug end to read its status from node attrs.
        src_attrs = G.nodes[e["source"]]
        tgt_attrs = G.nodes[e["target"]]
        drug_attrs = src_attrs if src_attrs.get("type") == "drug" else tgt_attrs
        status = (drug_attrs.get("status") or "unknown").lower()
        if status not in _DRUG_STATUS_STYLE:
            status = "unknown"
        by_status.setdefault(status, []).append((e["source"], e["target"]))
        if e.get("evidence"):
            evidence_labels[(e["source"], e["target"])] = e["evidence"]

    for status, elist in by_status.items():
        style = _DRUG_STATUS_STYLE[status]
        nx.draw_networkx_edges(
            G, pos, edgelist=elist, ax=ax,
            edge_color=style["edge"], width=_TARGET_DRUG_EDGE_WIDTH, alpha=0.95,
            style=style["linestyle"], connectionstyle="arc3,rad=0.2",
        )

    # Evidence labels on target-drug edges. Shift each label off the line
    # by ~6% perpendicular so it doesn't collide with the gene/drug labels
    # at either endpoint. For dense graphs (>24 target-drug edges) we drop
    # the labels entirely — the colour-by-status edge encoding stays.
    import math as _m
    if not dp["show_citations"]:
        evidence_labels = {}
    for (u, v), text in evidence_labels.items():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        midx, midy = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = (x1 - x0), (y1 - y0)
        L = _m.hypot(dx, dy) or 1.0
        # Perpendicular offset away from the gene's centre (push outward).
        nx_perp, ny_perp = -dy / L, dx / L
        offset_mag = 0.045
        # Choose perp direction that moves the label AWAY from graph centre.
        if (midx + nx_perp * 0.01) ** 2 + (midy + ny_perp * 0.01) ** 2 < midx ** 2 + midy ** 2:
            nx_perp, ny_perp = -nx_perp, -ny_perp
        lx, ly = midx + nx_perp * offset_mag, midy + ny_perp * offset_mag
        ax.text(lx, ly, text, fontsize=6.0, color="#1f2937",
                ha="center", va="center",
                bbox={"facecolor": "white", "edgecolor": "#cbd5e1",
                      "alpha": 0.85, "pad": 1.6, "boxstyle": "round,pad=0.18"})

    # ---- Gene nodes (circles)
    if gene_nodes:
        gene_ids = [n["id"] for n in gene_nodes]
        sizes = [_node_size(G.nodes[i], scale=dp["node_scale"]) for i in gene_ids]
        hub = ["hub" == (G.nodes[i].get("role") or "regular") for i in gene_ids]
        outline = [_GENE_HUB_OUTLINE if h else _GENE_REGULAR_OUTLINE for h in hub]
        nx.draw_networkx_nodes(
            G, pos, nodelist=gene_ids, ax=ax,
            node_size=sizes, node_color=_GENE_FILL,
            edgecolors=outline, linewidths=[2.2 if h else 1.0 for h in hub],
        )
        nx.draw_networkx_labels(
            G, pos, labels={i: i for i in gene_ids}, ax=ax,
            font_size=dp["label_fs"], font_weight="bold", font_color="#1f2937",
        )

    # ---- Drug nodes (rounded rectangles) — drawn manually with FancyBboxPatch
    # Width/height also scale down for dense networks.
    drug_w_scale = max(0.5, dp["node_scale"])
    for d in drug_nodes:
        did = d["id"]
        if did not in pos:
            continue
        status = (d.get("status") or "unknown").lower()
        style = _DRUG_STATUS_STYLE.get(status, _DRUG_STATUS_STYLE["unknown"])
        x, y = pos[did]
        text_len = max(6, len(did))
        w = (0.025 + 0.012 * text_len) * drug_w_scale
        h = 0.055 * drug_w_scale
        patch = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.005,rounding_size=0.012",
            linewidth=1.6, edgecolor=style["edge"], facecolor=style["fill"],
            transform=ax.transData, zorder=3,
        )
        ax.add_patch(patch)
        ax.text(x, y, did, fontsize=dp["drug_fs"], fontweight="bold",
                color="#1f2937", ha="center", va="center", zorder=4)

    # ---- Title + legend
    title = spec.get("title") or "Drug-target network"
    subtitle = spec.get("subtitle")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    if subtitle:
        ax.text(0.5, 1.005, subtitle, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=10, color="#374151")

    legend_handles = [
        mpatches.Patch(facecolor=_GENE_FILL, edgecolor=_GENE_REGULAR_OUTLINE,
                       label="Gene"),
        mpatches.Patch(facecolor=_GENE_FILL, edgecolor=_GENE_HUB_OUTLINE,
                       linewidth=2.2, label="Gene (hub)"),
    ]
    for status, style in _DRUG_STATUS_STYLE.items():
        if status in {n.get("status") for n in drug_nodes}:
            legend_handles.append(
                mpatches.Patch(facecolor=style["fill"], edgecolor=style["edge"],
                               label=f"Drug — {status}")
            )
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower left", fontsize=8,
                  frameon=True, framealpha=0.9, edgecolor="#cbd5e1")

    # Auto-fit with a small margin
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    if xs and ys:
        x_pad = 0.18 * (max(xs) - min(xs) or 1.0)
        y_pad = 0.18 * (max(ys) - min(ys) or 1.0)
        ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
        ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)

    fig.tight_layout()
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI for smoke-testing
# ---------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, help="Path to a JSON spec file")
    parser.add_argument("--out-dir", type=Path, default=Path("./network_plots"))
    parser.add_argument("--filename-hint", type=str, default=None)
    args = parser.parse_args()

    if args.spec:
        spec = json.loads(args.spec.read_text())
    else:
        # Built-in smoke test with a tiny synthetic AD-style graph.
        spec = {
            "title": "Smoke-test drug-target network (AD-like)",
            "subtitle": "Demo — not real data",
            "filename_hint": "smoke_test",
            "nodes": [
                {"id": "TREM2", "type": "gene", "weight": 0.9, "role": "hub"},
                {"id": "APP",   "type": "gene", "weight": 0.7, "role": "hub"},
                {"id": "MAPT",  "type": "gene", "weight": 0.5},
                {"id": "APOE",  "type": "gene", "weight": 0.6},
                {"id": "Lecanemab",   "type": "drug", "status": "approved"},
                {"id": "Donanemab",   "type": "drug", "status": "approved"},
                {"id": "AL002",       "type": "drug", "status": "investigational"},
                {"id": "DemoX",       "type": "drug", "status": "preclinical"},
            ],
            "edges": [
                {"source": "TREM2", "target": "APP",   "type": "gene-gene", "weight": 0.4},
                {"source": "APP",   "target": "MAPT",  "type": "gene-gene", "weight": 0.5},
                {"source": "APOE",  "target": "APP",   "type": "gene-gene", "weight": 0.6},
                {"source": "APP",   "target": "Lecanemab", "type": "target-drug", "evidence": "PMID:36625625"},
                {"source": "APP",   "target": "Donanemab", "type": "target-drug", "evidence": "PMID:38157881"},
                {"source": "TREM2", "target": "AL002",     "type": "target-drug", "evidence": "NCT04592874"},
                {"source": "MAPT",  "target": "DemoX",     "type": "target-drug", "evidence": "PMID:00000000"},
            ],
        }

    out = render_drug_target_network(spec, args.out_dir,
                                     filename_hint=args.filename_hint)
    if out is None:
        print("render failed", file=sys.stderr)
        sys.exit(1)
    print(f"wrote {out}")

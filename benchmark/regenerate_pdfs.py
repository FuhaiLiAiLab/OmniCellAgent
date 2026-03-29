#!/usr/bin/env python3
"""
Regenerate PDF reports from existing Markdown files with improved formatting.

Fixes:
  - Unicode/Greek characters (α, β, μ, etc.) via DejaVu fonts + fontspec
  - Wide tables overflowing page margins (longtable + small font + word-wrap)
  - Long code blocks in appendix (breaklines, small font)
  - Proper page breaks before major sections
  - Table of contents with correct depth
  - Image sizing and centering

Usage:
    python benchmark/regenerate_pdfs.py                  # regenerate all 3
    python benchmark/regenerate_pdfs.py --session AD-test # regenerate one
"""
import argparse
import glob
import os
import subprocess
import shutil
import sys
import tempfile
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSIONS_DIR = os.path.join(ROOT, "webapp", "sessions")

# ---------- LaTeX header injected via pandoc -H ----------
LATEX_HEADER = r"""
% ── Fonts (Unicode-capable) ──────────────────────────────────────────
\usepackage{fontspec}
\setmainfont{DejaVu Serif}
\setsansfont{DejaVu Sans}
\setmonofont[Scale=0.85]{DejaVu Sans Mono}

% ── Page layout ──────────────────────────────────────────────────────
% geometry is loaded by pandoc via -V geometry:margin=…  ; do NOT reload
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\textit{OmniCellAgent Analysis Report}}
\fancyhead[R]{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\fancyfoot{}

% ── Tables ───────────────────────────────────────────────────────────
% longtable, booktabs loaded by pandoc; only add extras
\usepackage{makecell}
\usepackage{etoolbox}
\renewcommand{\arraystretch}{1.35}
\AtBeginEnvironment{longtable}{\small}

% ── Images ───────────────────────────────────────────────────────────
% graphicx loaded by pandoc; just tweak max-width
\makeatletter
\def\maxwidth{\ifdim\Gin@nat@width>0.92\linewidth 0.92\linewidth\else\Gin@nat@width\fi}
\makeatother
\setkeys{Gin}{width=\maxwidth,keepaspectratio}

% ── Code blocks ──────────────────────────────────────────────────────
\usepackage{fvextra}
\fvset{breaklines,breakanywhere,fontsize=\scriptsize}

% ── Section spacing & page-breaks ────────────────────────────────────
\usepackage{titlesec}
\titleformat{\section}{\Large\bfseries}{}{0em}{}[\vspace{4pt}\hrule\vspace{6pt}]
\titleformat{\subsection}{\large\bfseries}{}{0em}{}
\titleformat{\subsubsection}{\normalsize\bfseries}{}{0em}{}
% page break before each \section
\let\oldsection\section
\renewcommand{\section}{\clearpage\oldsection}

% ── Hyperlinks ───────────────────────────────────────────────────────
% hyperref loaded by pandoc; just configure colors via passOptionsToPackage
\PassOptionsToPackage{colorlinks=true,linkcolor=blue!60!black,urlcolor=blue!70!black}{hyperref}

% ── Misc ─────────────────────────────────────────────────────────────
\usepackage{microtype}
"""

# ---------- helpers -----------------------------------------------------------

def find_reports(session_filter: str | None = None) -> list[dict]:
    """Return list of {session, md_path, pdf_path} for all reports."""
    reports = []
    if session_filter:
        dirs = [os.path.join(SESSIONS_DIR, session_filter)]
    else:
        dirs = sorted(glob.glob(os.path.join(SESSIONS_DIR, "*-test")))

    for d in dirs:
        if not os.path.isdir(d):
            print(f"⚠️  Session dir not found: {d}")
            continue
        mds = sorted(glob.glob(os.path.join(d, "report_*.md")))
        for md in mds:
            reports.append({
                "session": os.path.basename(d),
                "md_path": md,
                "pdf_path": md.replace(".md", ".pdf"),
            })
    return reports


def preprocess_markdown(md_path: str) -> str:
    """Read the markdown and fix formatting issues that break pandoc 2.x.

    Key fix: pandoc 2.x requires a **blank line** before and after a pipe
    table, otherwise the whole table is emitted as literal \textbar{} text.
    The LLM-generated reports often omit that blank line.
    """
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    out: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.rstrip()
        is_table_row = stripped.startswith("|") and stripped.endswith("|")

        if is_table_row:
            # Ensure a blank line exists before the first row of a table
            if out and out[-1].strip() != "" and not (
                out[-1].rstrip().startswith("|") and out[-1].rstrip().endswith("|")
            ):
                out.append("\n")
        else:
            # Ensure a blank line exists after the last row of a table
            if (
                out
                and out[-1].rstrip().startswith("|")
                and out[-1].rstrip().endswith("|")
                and stripped != ""
            ):
                out.append("\n")

        out.append(line)

    return "".join(out)


def compile_pdf(md_path: str, pdf_path: str, header_file: str) -> bool:
    """Run pandoc + xelatex to produce PDF with the custom header."""
    session_dir = os.path.dirname(md_path)

    # ── Preprocess: fix blank-line issues so pandoc parses tables ──
    fixed_md = preprocess_markdown(md_path)
    tmp_md_fd, tmp_md_path = tempfile.mkstemp(
        suffix=".md", prefix="oca_fixed_", dir=session_dir
    )
    with os.fdopen(tmp_md_fd, "w", encoding="utf-8") as f:
        f.write(fixed_md)

    cmd = [
        "pandoc",
        tmp_md_path,
        "-o", pdf_path,
        "--pdf-engine=xelatex",
        "-H", header_file,              # inject our LaTeX preamble
        "--toc", "--toc-depth=2",
        "--highlight-style=tango",
        "--resource-path", session_dir,
        # pandoc variables
        "-V", "geometry:margin=0.9in",
        "-V", "fontsize=11pt",
        "-V", "documentclass=article",
        "-V", "papersize=a4",
        "-V", "lang=en",
        # table handling – force longtable for wide tables
        "--columns=72",
    ]

    print(f"  ⏳  Running pandoc → xelatex …")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=session_dir,
        timeout=300,
    )

    _cleanup(tmp_md_path)  # remove preprocessed temp file

    if result.returncode == 0:
        size_kb = os.path.getsize(pdf_path) / 1024
        print(f"  ✅  PDF generated ({size_kb:.0f} KB): {pdf_path}")
        return True

    # ── If xelatex failed, fall back to wkhtmltopdf ──────────────────
    print(f"  ⚠️  xelatex failed, trying wkhtmltopdf fallback …")
    if result.stderr:
        # Print only the last few useful lines
        err_lines = result.stderr.strip().split("\n")
        for ln in err_lines[-8:]:
            print(f"      {ln}")

    return _fallback_wkhtmltopdf(md_path, pdf_path, session_dir, fixed_md)


def _fallback_wkhtmltopdf(md_path: str, pdf_path: str, session_dir: str,
                           preprocessed_text: str | None = None) -> bool:
    """Strategy 2: pandoc → HTML → wkhtmltopdf with enhanced CSS."""
    if not shutil.which("wkhtmltopdf"):
        print("  ❌  wkhtmltopdf not available either")
        return False

    html_path = md_path.replace(".md", ".html")

    # Write preprocessed markdown to a temp file for this fallback too
    if preprocessed_text is None:
        preprocessed_text = preprocess_markdown(md_path)
    tmp_md_fd, tmp_md_path = tempfile.mkstemp(
        suffix=".md", prefix="oca_html_", dir=session_dir
    )

    css_content = textwrap.dedent("""\
    body {
        font-family: "DejaVu Sans", "Noto Sans", "Segoe UI", Roboto, Arial, sans-serif;
        font-size: 11pt;
        line-height: 1.55;
        color: #1a1a1a;
        max-width: 100%;
    }
    h1 { font-size: 1.6em; border-bottom: 2px solid #2c3e50; padding-bottom: 6px; margin-top: 30px; }
    h2 { font-size: 1.3em; border-bottom: 1px solid #bdc3c7; padding-bottom: 4px; margin-top: 24px; }
    h3 { font-size: 1.1em; margin-top: 18px; }
    h1, h2, h3 { page-break-after: avoid; }

    img {
        max-width: 100% !important;
        max-height: 600px !important;
        height: auto !important;
        width: auto !important;
        display: block;
        margin: 12px auto;
        page-break-inside: avoid;
    }

    table {
        width: 100%;
        border-collapse: collapse;
        margin: 14px 0;
        font-size: 9pt;
        page-break-inside: avoid;
        table-layout: fixed;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    th, td {
        border: 1px solid #ccc;
        padding: 5px 7px;
        text-align: left;
        vertical-align: top;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    th { background-color: #ecf0f1; font-weight: 600; }
    tr:nth-child(even) { background-color: #f9f9f9; }

    pre, code {
        font-family: "DejaVu Sans Mono", "Consolas", monospace;
        font-size: 8pt;
        background-color: #f5f5f5;
        border-radius: 3px;
    }
    pre {
        padding: 8px 10px;
        overflow-x: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
        border: 1px solid #e0e0e0;
        page-break-inside: auto;
    }

    blockquote { border-left: 3px solid #3498db; padding-left: 12px; color: #555; }
    hr { border: none; border-top: 1px solid #bbb; margin: 20px 0; }
    """)

    css_path = os.path.join(session_dir, "_report_style.css")
    with open(css_path, "w") as f:
        f.write(css_content)

    with os.fdopen(tmp_md_fd, "w", encoding="utf-8") as f:
        f.write(preprocessed_text)

    html_cmd = [
        "pandoc", tmp_md_path, "-o", html_path,
        "--standalone",
        "--self-contained",
        "--toc", "--toc-depth=2",
        "--highlight-style=tango",
        "--resource-path", session_dir,
        "-c", css_path,
        "--columns=72",
    ]
    r1 = subprocess.run(html_cmd, capture_output=True, text=True, cwd=session_dir, timeout=120)
    _cleanup(tmp_md_path)  # remove preprocessed temp
    if r1.returncode != 0:
        print(f"  ❌  Pandoc HTML conversion failed: {(r1.stderr or '')[:200]}")
        _cleanup(css_path)
        return False

    pdf_cmd = [
        "wkhtmltopdf",
        "--enable-local-file-access",
        "--margin-top", "18mm",
        "--margin-bottom", "18mm",
        "--margin-left", "14mm",
        "--margin-right", "14mm",
        "--footer-center", "[page]",
        "--footer-font-size", "9",
        html_path, pdf_path,
    ]
    r2 = subprocess.run(pdf_cmd, capture_output=True, text=True, cwd=session_dir, timeout=180)
    _cleanup(css_path, html_path)

    if r2.returncode == 0:
        size_kb = os.path.getsize(pdf_path) / 1024
        print(f"  ✅  PDF generated via wkhtmltopdf ({size_kb:.0f} KB): {pdf_path}")
        return True

    print(f"  ❌  wkhtmltopdf failed: {(r2.stderr or '')[:200]}")
    return False


def _cleanup(*paths):
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


# ---------- main --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Regenerate PDF reports from markdown")
    parser.add_argument("--session", type=str, default=None,
                        help="Regenerate only this session (e.g. AD-test)")
    args = parser.parse_args()

    reports = find_reports(args.session)
    if not reports:
        print("No reports found. Check webapp/sessions/*-test/report_*.md")
        sys.exit(1)

    print(f"📄 Found {len(reports)} report(s) to regenerate\n")

    # Write the LaTeX header to a temp file (shared across all runs)
    header_fd, header_path = tempfile.mkstemp(suffix=".tex", prefix="oca_header_")
    with os.fdopen(header_fd, "w") as f:
        f.write(LATEX_HEADER)

    success = 0
    for rpt in reports:
        print(f"── {rpt['session']} ──")
        print(f"  📝  Source: {rpt['md_path']}")
        if compile_pdf(rpt["md_path"], rpt["pdf_path"], header_path):
            success += 1
        print()

    _cleanup(header_path)

    print(f"{'='*50}")
    print(f"Done: {success}/{len(reports)} PDFs regenerated successfully")
    if success < len(reports):
        sys.exit(1)


if __name__ == "__main__":
    main()

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
SESSIONS_DIR = os.path.join(ROOT, "webapp", "assets", "sessions")

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
\usepackage{array}
\usepackage{tabularx}
\renewcommand{\arraystretch}{1.5}
% Use smaller font and more padding for tables
\AtBeginEnvironment{longtable}{\footnotesize\setlength{\tabcolsep}{10pt}}
% Allow line breaks in table cells and add padding
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcolumntype{C}[1]{>{\centering\arraybackslash}p{#1}}
% Add visible column rules
\setlength{\arrayrulewidth}{0.4pt}
% Force vertical lines between all columns in longtable
\AtBeginEnvironment{longtable}{%
  \setlength{\tabcolsep}{8pt}%
  \renewcommand{\arraystretch}{1.4}%
}
% Add a thin border between header cells
\newcommand{\thmark}{\rule[-0.5ex]{0.4pt}{2.5ex}}

% ── Images ───────────────────────────────────────────────────────────
% graphicx loaded by pandoc; just tweak max-width
\makeatletter
\def\maxwidth{\ifdim\Gin@nat@width>0.85\linewidth 0.85\linewidth\else\Gin@nat@width\fi}
\makeatother
\setkeys{Gin}{width=\maxwidth,keepaspectratio}

% Force figures to be centered with vertical space
\usepackage{float}
\let\origfigure\figure
\let\endorigfigure\endfigure
\renewenvironment{figure}[1][htbp]{%
  \origfigure[H]%
  \centering
}{%
  \endorigfigure
}

% Add space around standalone images (not in figures)
\let\oldincludegraphics\includegraphics
\renewcommand{\includegraphics}[2][]{%
  \par\vspace{12pt}%
  \begin{center}%
    \oldincludegraphics[#1]{#2}%
  \end{center}%
  \vspace{12pt}\par%
}

% ── Code blocks ──────────────────────────────────────────────────────
\usepackage{fvextra}
\fvset{breaklines,breakanywhere,fontsize=\scriptsize,breaksymbol=,breakanywheresymbolpre=,breakbeforesymbolpre=,breakaftersymbolpre=}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,breakanywhere,commandchars=\\\{\},fontsize=\scriptsize}

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
\usepackage{seqsplit}
"""

# ---------- helpers -----------------------------------------------------------

def find_reports(session_filter: str | None = None, include_revised: bool = True) -> list[dict]:
    """Return list of {session, md_path, pdf_path, is_revised} for all reports."""
    reports = []
    if session_filter:
        dirs = [os.path.join(SESSIONS_DIR, session_filter)]
    else:
        dirs = sorted(glob.glob(os.path.join(SESSIONS_DIR, "*-test")))

    for d in dirs:
        if not os.path.isdir(d):
            print(f"⚠️  Session dir not found: {d}")
            continue
        # Find original reports
        mds = sorted(glob.glob(os.path.join(d, "report_*.md")))
        for md in mds:
            reports.append({
                "session": os.path.basename(d),
                "md_path": md,
                "pdf_path": md.replace(".md", ".pdf"),
                "is_revised": False,
            })
        # Find revised reports
        if include_revised:
            revised_mds = sorted(glob.glob(os.path.join(d, "report-revised_*.md")))
            for md in revised_mds:
                reports.append({
                    "session": os.path.basename(d),
                    "md_path": md,
                    "pdf_path": md.replace(".md", ".pdf"),
                    "is_revised": True,
                })
    return reports


def preprocess_markdown(md_path: str) -> str:
    """Read the markdown and fix formatting issues that break pandoc 2.x.

    Fixes:
    1. pandoc 2.x requires a **blank line** before and after a pipe table
    2. Images need blank lines before/after for proper figure handling
    3. Long lines in code blocks need to be broken for proper wrapping
    """
    import re
    
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    out: list[str] = []
    in_code_block = False
    
    for i, line in enumerate(lines):
        stripped = line.rstrip()
        
        # Track code block state
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            out.append(line)
            continue
        
        # Inside code blocks: break very long lines
        if in_code_block:
            if len(stripped) > 100:
                # Break at JSON-like boundaries
                broken = stripped
                # Insert newlines after common JSON separators
                broken = re.sub(r"('\s*,\s*')", r"',\n'", broken)
                broken = re.sub(r'("\s*,\s*")', r'",\n"', broken)
                broken = re.sub(r"(},\s*{)", r"},\n{", broken)
                broken = re.sub(r"(\],\s*\[)", r"],\n[", broken)
                broken = re.sub(r"(:\s*True,)", r": True,\n", broken)
                broken = re.sub(r"(:\s*False,)", r": False,\n", broken)
                # Break at sentence boundaries in result_summary
                broken = re.sub(r"(\.\s+)([A-Z])", r".\n\2", broken)
                # Break very long unbroken sequences
                if any(len(segment) > 90 for segment in broken.split('\n')):
                    # Force break every 80 chars at word boundaries
                    new_broken = []
                    for segment in broken.split('\n'):
                        if len(segment) > 90:
                            words = segment.split(' ')
                            current_line = []
                            current_len = 0
                            for word in words:
                                if current_len + len(word) + 1 > 80 and current_line:
                                    new_broken.append(' '.join(current_line))
                                    current_line = [word]
                                    current_len = len(word)
                                else:
                                    current_line.append(word)
                                    current_len += len(word) + 1
                            if current_line:
                                new_broken.append(' '.join(current_line))
                        else:
                            new_broken.append(segment)
                    broken = '\n'.join(new_broken)
                out.append(broken + '\n')
            else:
                out.append(line)
            continue
        
        is_table_row = stripped.startswith("|") and stripped.endswith("|")
        is_image = stripped.startswith("![") and "](" in stripped

        # Handle images - ensure blank lines before and after
        if is_image:
            if out and out[-1].strip() != "":
                out.append("\n")
            out.append(line)
            # Check if next line exists and is not blank
            if i + 1 < len(lines) and lines[i + 1].strip() != "":
                out.append("\n")
            continue

        if is_table_row:
            # Add explicit padding to table cells for better column separation
            # Replace | with |  (double space) to ensure visible column gaps
            padded_line = re.sub(r'\|([^|])', r'|  \1', stripped)
            padded_line = re.sub(r'([^|])\|', r'\1  |', padded_line)
            line = padded_line + '\n'
            stripped = padded_line
            
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

    /* Center images */
    p > img, figure, .figure {
        display: block !important;
        margin-left: auto !important;
        margin-right: auto !important;
        text-align: center !important;
        max-width: 85% !important;
        max-height: 550px !important;
        height: auto !important;
        width: auto !important;
        page-break-inside: avoid;
    }
    figure { text-align: center; }
    figcaption { text-align: center; font-style: italic; margin-top: 8px; }

    /* Tables with proper spacing */
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 14px 0;
        font-size: 9pt;
        page-break-inside: avoid;
        table-layout: auto;
    }
    th, td {
        border: 1px solid #ccc;
        padding: 6px 10px;
        text-align: left;
        vertical-align: top;
        word-wrap: break-word;
        overflow-wrap: break-word;
        max-width: 200px;
    }
    th { background-color: #ecf0f1; font-weight: 600; white-space: nowrap; }
    tr:nth-child(even) { background-color: #f9f9f9; }

    /* Code blocks with proper wrapping */
    pre, code {
        font-family: "DejaVu Sans Mono", "Consolas", monospace;
        font-size: 7.5pt;
        background-color: #f5f5f5;
        border-radius: 3px;
    }
    pre {
        padding: 10px 12px;
        overflow-x: hidden;
        white-space: pre-wrap;
        word-wrap: break-word;
        word-break: break-all;
        border: 1px solid #e0e0e0;
        max-width: 100%;
        line-height: 1.4;
    }
    code {
        word-break: break-all;
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
    parser.add_argument("--revised-only", action="store_true",
                        help="Only regenerate revised reports")
    parser.add_argument("--original-only", action="store_true",
                        help="Only regenerate original (non-revised) reports")
    args = parser.parse_args()

    reports = find_reports(args.session, include_revised=not args.original_only)
    
    # Filter if needed
    if args.revised_only:
        reports = [r for r in reports if r.get("is_revised", False)]
    elif args.original_only:
        reports = [r for r in reports if not r.get("is_revised", False)]
    
    if not reports:
        print("No reports found. Check webapp/assets/sessions/*-test/report*.md")
        sys.exit(1)

    print(f"📄 Found {len(reports)} report(s) to regenerate\n")

    # Write the LaTeX header to a temp file (shared across all runs)
    header_fd, header_path = tempfile.mkstemp(suffix=".tex", prefix="oca_header_")
    with os.fdopen(header_fd, "w") as f:
        f.write(LATEX_HEADER)

    success = 0
    for rpt in reports:
        version_tag = " (REVISED)" if rpt.get("is_revised") else " (ORIGINAL)"
        print(f"── {rpt['session']}{version_tag} ──")
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

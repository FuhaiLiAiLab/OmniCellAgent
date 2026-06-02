#!/usr/bin/env python3
"""
Combine first-run and revised PDF reports into a single supplementary document.

This script:
1. Finds all report markdown files from benchmark sessions (AD, PDAC, LungCancer)
2. Regenerates PDFs with the same pandoc/xelatex conversion used by benchmark/regenerate_pdfs.py
3. Combines everything into a single supplementary PDF

Outputs to: logs/appendix/supplementary_reports.pdf

Usage:
    python scripts/combine_supplementary_pdfs.py
    python scripts/combine_supplementary_pdfs.py --skip-regenerate  # Use existing PDFs

Requirements:
    pip install PyPDF2
    pandoc + xelatex available on PATH for the preferred PDF conversion path
"""
import argparse
import os
import sys
import tempfile
import subprocess
import shutil
import textwrap
from pathlib import Path

try:
    from PyPDF2 import PdfMerger, PdfReader
except ImportError:
    print("❌ PyPDF2 not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2"])
    from PyPDF2 import PdfMerger, PdfReader


# ---------- LaTeX header for PDF generation (same as regenerate_pdfs.py) ----------
LATEX_HEADER = r"""
% ── Fonts (Unicode-capable) ──────────────────────────────────────────
\usepackage{fontspec}
\setmainfont{DejaVu Serif}
\setsansfont{DejaVu Sans}
\setmonofont[Scale=0.85]{DejaVu Sans Mono}

% ── Page layout ──────────────────────────────────────────────────────
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\textit{OmniCellAgent Analysis Report}}
\fancyhead[R]{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\fancyfoot{}

% ── Tables ───────────────────────────────────────────────────────────
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
\let\oldsection\section
\renewcommand{\section}{\clearpage\oldsection}

% ── Hyperlinks ───────────────────────────────────────────────────────
\PassOptionsToPackage{colorlinks=true,linkcolor=blue!60!black,urlcolor=blue!70!black}{hyperref}

% ── Misc ─────────────────────────────────────────────────────────────
\usepackage{microtype}
\usepackage{seqsplit}
"""


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

    out = []
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
            for raw, display in {
                "OmicMiningAgent": "Omic-Mining Agent",
                "BioMarkerKGAgent": "BioMarker-KG Agent",
                "PubMedResearcher": "PubMed Researcher",
                "GoogleSearcher": "Google Searcher",
                "ScientistsAgent": "Scientists Agent",
            }.items():
                stripped = stripped.replace(raw, display)
            # Add explicit padding to table cells for better column separation
            padded_line = re.sub(r'\|([^|])', r'|  \1', stripped)
            padded_line = re.sub(r'([^|])\|', r'\1  |', padded_line)
            line = padded_line + '\n'
            stripped = padded_line
            
            if out and out[-1].strip() != "" and not (
                out[-1].rstrip().startswith("|") and out[-1].rstrip().endswith("|")
            ):
                out.append("\n")
        else:
            if (
                out
                and out[-1].rstrip().startswith("|")
                and out[-1].rstrip().endswith("|")
                and stripped != ""
            ):
                out.append("\n")

        out.append(line)

    return "".join(out)


def compile_pdf_from_md(md_path: str, pdf_path: str, header_file: str) -> bool:
    """Run pandoc + xelatex to produce PDF with the custom header."""
    session_dir = os.path.dirname(md_path)

    # Preprocess markdown so pandoc parses tables and long code blocks cleanly.
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
        "-H", header_file,
        "--toc", "--toc-depth=2",
        "--highlight-style=tango",
        "--resource-path", session_dir,
        "-V", "geometry:margin=0.9in",
        "-V", "fontsize=11pt",
        "-V", "documentclass=article",
        "-V", "papersize=letter",
        "-V", "lang=en",
        "--columns=72",
    ]

    print(f"    ⏳ Running pandoc → xelatex …")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=session_dir,
        timeout=300,
    )

    _cleanup(tmp_md_path)

    if result.returncode == 0:
        size_kb = os.path.getsize(pdf_path) / 1024
        print(f"    ✅ PDF generated ({size_kb:.0f} KB): {pdf_path}")
        return True

    print(f"    ⚠️ xelatex failed, trying wkhtmltopdf fallback …")
    if result.stderr:
        for line in result.stderr.strip().split("\n")[-8:]:
            print(f"      {line}")

    return _fallback_wkhtmltopdf(md_path, pdf_path, session_dir, fixed_md)


def _fallback_wkhtmltopdf(md_path: str, pdf_path: str, session_dir: str,
                           preprocessed_text: str | None = None) -> bool:
    """Strategy 2: pandoc → HTML → wkhtmltopdf with enhanced CSS."""
    if not shutil.which("wkhtmltopdf"):
        print("    ❌ wkhtmltopdf not available either")
        return False

    html_path = md_path.replace(".md", ".html")

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

    _cleanup(tmp_md_path)

    if r1.returncode != 0:
        print(f"    ❌ Pandoc HTML conversion failed: {(r1.stderr or '')[:200]}")
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
        print(f"    ✅ PDF generated via wkhtmltopdf ({size_kb:.0f} KB): {pdf_path}")
        return True

    print(f"    ❌ wkhtmltopdf failed: {(r2.stderr or '')[:200]}")
    return False


def _cleanup(*paths):
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass


def find_report_mds(sessions_dir: Path, session_suffix: str = "-test") -> dict:
    """Find all report markdown files organized by case study.

    session_suffix is appended to the case key to form the session dir name
    (e.g. 'AD' + '-test' → 'AD-test'; '-test-2' → 'AD-test-2').
    """
    case_studies = ['AD', 'PDAC', 'LungCancer']
    results = {}

    for case in case_studies:
        session_dir = sessions_dir / f"{case}{session_suffix}"
        if not session_dir.exists():
            print(f"⚠️  Session directory not found: {session_dir}")
            continue
        
        results[case] = {'first_run': None, 'revised': None}
        
        # Find original reports (report_*.md but not static versions)
        for md_file in session_dir.glob("report_*.md"):
            if '_static' in md_file.name:
                continue
            if results[case]['first_run'] is None or md_file.stat().st_mtime > results[case]['first_run'].stat().st_mtime:
                results[case]['first_run'] = md_file
        
        # Find revised reports
        for md_file in session_dir.glob("report-revised_*.md"):
            if results[case]['revised'] is None or md_file.stat().st_mtime > results[case]['revised'].stat().st_mtime:
                results[case]['revised'] = md_file
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Combine report PDFs into supplementary document")
    parser.add_argument("--skip-regenerate", action="store_true",
                        help="Skip PDF regeneration, use existing PDFs")
    parser.add_argument(
        "--session-suffix", default="-test",
        help="Suffix appended to case keys to form session dir name (default '-test'). "
             "Use '-test-2' to compile from re-runs.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output PDF filename (under logs/appendix/). Default: "
             "'supplementary_reports.pdf' for -test, 'supplementary_reports<suffix>.pdf' otherwise.",
    )
    parser.add_argument(
        "--sessions-dir", default=None,
        help="Override sessions directory (default webapp/sessions). "
             "The original script used webapp/assets/sessions; pass the right one for your layout.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.absolute()
    # Default sessions dir: try webapp/sessions first (current layout), fall back to assets/sessions
    if args.sessions_dir:
        sessions_dir = Path(args.sessions_dir)
        if not sessions_dir.is_absolute():
            sessions_dir = project_root / args.sessions_dir
    else:
        sessions_dir = project_root / "webapp" / "sessions"
        if not sessions_dir.exists():
            sessions_dir = project_root / "webapp" / "assets" / "sessions"
    export_dir = project_root / "logs" / "appendix"
    export_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_path = export_dir / args.output
    elif args.session_suffix == "-test":
        output_path = export_dir / "supplementary_reports.pdf"
    else:
        # Sanitize suffix for filename use (-test-2 → _test_2)
        sanitized = args.session_suffix.replace("/", "_").lstrip("-")
        output_path = export_dir / f"supplementary_reports_{sanitized}.pdf"

    print(f"🔍 Searching for report markdown files (session_suffix='{args.session_suffix}', sessions_dir={sessions_dir})...")
    reports = find_report_mds(sessions_dir, session_suffix=args.session_suffix)
    
    if not reports:
        print("❌ No reports found!")
        return 1
    
    # Print what was found
    print("\n📄 Found reports:")
    for case, paths in reports.items():
        print(f"\n  {case}:")
        print(f"    First run: {paths['first_run'].name if paths['first_run'] else '❌ Not found'}")
        print(f"    Revised:   {paths['revised'].name if paths['revised'] else '❌ Not found'}")
    
    # Create LaTeX header file
    header_fd, header_path = tempfile.mkstemp(suffix=".tex", prefix="oca_header_")
    with os.fdopen(header_fd, "w") as f:
        f.write(LATEX_HEADER)
    
    # Process and combine PDFs
    print(f"\n📚 Processing PDFs...")
    print("  Using original style: regenerate from Markdown, then merge report PDFs directly.")
    merger = PdfMerger()
    
    case_order = ['AD', 'PDAC', 'LungCancer']
    
    total_pages = 0
    
    # First pass: regenerate PDFs and collect page counts.
    pdf_entries = []
    for case in case_order:
        if case not in reports:
            continue
        paths = reports[case]
        
        for version, md_path in [('First Run', paths['first_run']), ('Revised', paths['revised'])]:
            if md_path is None:
                continue
            
            pdf_path = md_path.with_suffix('.pdf')
            
            # Regenerate PDF if needed
            if not args.skip_regenerate or not pdf_path.exists():
                print(f"  ⏳ Regenerating: {case} ({version})...")
                if compile_pdf_from_md(str(md_path), str(pdf_path), header_path):
                    print(f"    ✅ Generated: {pdf_path.name}")
                else:
                    print(f"    ❌ Failed to generate PDF for {md_path.name}")
                    continue
            
            if pdf_path.exists():
                try:
                    reader = PdfReader(str(pdf_path))
                    num_pages = len(reader.pages)
                    pdf_entries.append({
                        'case': case,
                        'version': version,
                        'pdf_path': pdf_path,
                        'num_pages': num_pages
                    })
                except Exception as e:
                    print(f"  ⚠️  Could not read {pdf_path.name}: {e}")

    # Second pass: merge report PDFs directly, matching supplementary_reports.pdf.
    for entry in pdf_entries:
        pdf_path = entry['pdf_path']
        num_pages = entry['num_pages']

        merger.append(str(pdf_path))
        print(f"  ✅ Added {pdf_path.name} ({num_pages} pages)")
        total_pages += num_pages
    
    # Write output
    merger.write(str(output_path))
    merger.close()
    
    # Clean up
    try:
        os.remove(header_path)
    except OSError:
        pass
    
    print(f"\n✅ Supplementary document created: {output_path}")
    print(f"   Total pages: {total_pages}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

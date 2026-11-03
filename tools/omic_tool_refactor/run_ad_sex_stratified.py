"""AD sex-stratified figure: the female-vs-male four-sector scatter of the AD case study.

1. Cohorts: for each AD cell type the agent runs (CASE_STUDY_COHORTS: microglial cell, astrocyte,
   glutamatergic neuron), the omic workflow with the call run_ad_microglia.py makes, into
   <output_root>/omic_<slug>/ (the agent tool's folder names). A folder that already holds the
   four files the sex model reads is reused, so an output root of run_ad_microglia.py can be
   extended. Each workflow result goes to <output_root>/omic_result_<slug>.json.
2. Sex models: sex_stratified_de.R on the three cohort folders in parallel, into
   <output_root>/sex_stratified/<slug>/.
3. Figure: plot_sex_stratified_scatter.py, one panel per cohort, into
   <output_root>/sex_stratified/foursector_scatter_AD.{png,pdf}.

Run with R set up (scripts/r_env.sh), from any directory:
  scripts/r_env.sh <omnicellagent python> tools/omic_tool_refactor/run_ad_sex_stratified.py <output_root>
"""
from concurrent.futures import ThreadPoolExecutor
import glob
import json
import os
import re
import subprocess
import sys

from omic_fetch_analysis_workflow import omic_fetch_analysis_workflow
from subprocess_r import run_r_script

DISEASE = "Alzheimer's disease"
ORGAN = "brain"
# (cell type, panel title) in panel order: the agent's AD case-study cell types
COHORTS = (
    ("microglial cell", "Microglia"),
    ("astrocyte", "Astrocyte"),
    ("glutamatergic neuron", "Glutamatergic Neuron"),
)
TITLE = "Sex-Specific DEG Landscape in Alzheimer's Disease (AD vs Control, joint model)"
OUT_NAME = "foursector_scatter_AD"
# The full-gene models took 17, 27 and 39 minutes on 2026-11-02
SEX_MODEL_TIMEOUT = 2 * 3600
HERE = os.path.dirname(os.path.abspath(__file__))
# The files sex_stratified_de.R reads from a cohort folder, one of each
SEX_MODEL_INPUTS = ("foranalysis_combined_normal_df_*.csv", "foranalysis_combined_disease_df_*.csv",
                    "de_covariates_*.csv", "labels.csv")


def slug(cell_type):
    """omic_analysis_tool's cohort folder name is omic_<slug>."""
    return re.sub(r"[^a-z0-9]+", "_", cell_type.split(";")[0].lower()).strip("_")


def has_sex_model_inputs(cohort_dir):
    return all(len(glob.glob(os.path.join(glob.escape(cohort_dir), pattern))) == 1
               for pattern in SEX_MODEL_INPUTS)


def run_cohort(output_root, cell_type):
    """Return the cohort folder, running the omic workflow unless the folder already has the inputs."""
    cohort_dir = os.path.join(output_root, f"omic_{slug(cell_type)}")
    if has_sex_model_inputs(cohort_dir):
        print(f"[Cohort] {cell_type}: reusing {cohort_dir}")
        return cohort_dir
    result = omic_fetch_analysis_workflow(disease=DISEASE, cell_type=cell_type, organ=ORGAN,
                                          session_dir=cohort_dir)
    with open(os.path.join(output_root, f"omic_result_{slug(cell_type)}.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1, default=str)
    print(f"[Cohort] {cell_type}: success={result.get('success')} num_samples={result.get('num_samples')}")
    if not has_sex_model_inputs(cohort_dir):
        raise RuntimeError(f"{cell_type}: the workflow wrote no DE inputs ({result.get('message')})")
    return cohort_dir


def run_sex_models(cohort_dirs, out_dir):
    """sex_stratified_de.R on every cohort folder at once; an R failure raises."""
    script = os.path.join(HERE, "sex_stratified_de.R")
    with ThreadPoolExecutor(max_workers=len(cohort_dirs)) as pool:
        futures = [pool.submit(run_r_script, script, [cohort_dir, os.path.join(out_dir, name)],
                               timeout=SEX_MODEL_TIMEOUT)
                   for name, cohort_dir in cohort_dirs.items()]
        for future in futures:
            future.result()


def draw_figure(out_dir):
    command = [sys.executable, os.path.join(HERE, "plot_sex_stratified_scatter.py")]
    for cell_type, label in COHORTS:
        command += ["--panel", f"{label}={os.path.join(out_dir, slug(cell_type))}"]
    command += ["--out-dir", out_dir, "--out-name", OUT_NAME, "--title", TITLE, "--disease-label", "AD"]
    subprocess.run(command, check=True)
    return [os.path.join(out_dir, f"{OUT_NAME}.{extension}") for extension in ("png", "pdf")]


def main(output_root):
    output_root = os.path.abspath(output_root)
    os.makedirs(output_root, exist_ok=True)
    cohort_dirs = {slug(cell_type): run_cohort(output_root, cell_type) for cell_type, _ in COHORTS}
    out_dir = os.path.join(output_root, "sex_stratified")
    run_sex_models(cohort_dirs, out_dir)
    return draw_figure(out_dir)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: run_ad_sex_stratified.py <output_root>")
    for path in main(sys.argv[1]):
        print(f"Figure: {path}")

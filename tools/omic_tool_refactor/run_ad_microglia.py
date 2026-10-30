"""AD microglia case study without the agent: the call omic_analysis_tool makes for this cohort.

disease "Alzheimer's disease", cell type "microglial cell", organ "brain", label "disease".
Writes <output_root>/omic_microglial_cell/ (the tool's per-cell-type session subfolder) and,
from the command line, the tool's return value to <output_root>/omic_result.json.

Run with R set up (scripts/r_env.sh), from any directory:
  scripts/r_env.sh <omnicellagent python> tools/omic_tool_refactor/run_ad_microglia.py <output_root>
"""
import json
import os
import sys

from omic_fetch_analysis_workflow import omic_fetch_analysis_workflow

DISEASE = "Alzheimer's disease"
CELL_TYPE = "microglial cell"
ORGAN = "brain"


def main(output_root):
    """Run the workflow as omic_analysis_tool does and return the tool's result dict."""
    try:
        # The tool's subfolder name: omic_ + the cell type in lower case, non-alphanumerics as "_"
        session_dir = os.path.join(output_root, "omic_microglial_cell")
        result = omic_fetch_analysis_workflow(disease=DISEASE, cell_type=CELL_TYPE, organ=ORGAN,
                                              session_dir=session_dir)
        # Cohort diagnostics stay in the session files (cohort_diagnostics.json,
        # COHORT_DIAGNOSTICS.txt); no agent, prompt or report receives them.
        if isinstance(result, dict):
            result.pop("cohort_diagnostics", None)
            result.pop("cohort_verdict", None)
            # The most abundant genes only confirm the cell type; pass them under a name that says so
            names = result.pop("top_gene_names", None) or []
            values = result.pop("top_gene_values", None) or []
            result.pop("top_gene_indices", None)
            if names:
                result["qc_highest_mean_expression_genes"] = {
                    "note": ("Highest mean expression (CP10K) over both groups: a check that the intended "
                             "cell type was retrieved, not a disease signal or differential expression."),
                    "genes": [{"gene": g, "mean_cp10k": round(float(v), 1)} for g, v in zip(names, values)],
                }
        return result

    except Exception as e:
        import traceback
        return {
            "success": False,
            "message": f"Error in omic analysis: {str(e)}",
            "traceback": traceback.format_exc()
        }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: run_ad_microglia.py <output_root>")
    output_root = os.path.abspath(sys.argv[1])
    result = main(output_root)
    with open(os.path.join(output_root, "omic_result.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1, default=str)
    print(f"success={result.get('success')} num_samples={result.get('num_samples')}")
    sys.exit(0 if result.get("success") else 1)

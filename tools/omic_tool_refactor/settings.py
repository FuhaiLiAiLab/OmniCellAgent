"""Paths the mini omic pipeline reads; in the agent these come from configs/paths.yaml."""
from pathlib import Path

# OmniCellTOSG code checkout (provides CellTOSG_Loader); external.omnicell_root in paths.yaml
OMNICELL_ROOT = "/storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellTOSG"
# OmniCellTOSG v3 dataset; external.omnicell_data_root in paths.yaml. configs/paths.yaml is
# gitignored (not part of bb92fab): the reference run set this path in its own paths.yaml, and
# the main checkout's configs/paths.yaml still names the older OmniCellTOSG_dataset
OMNICELL_DATA_ROOT = "/storage3/fs1/fuhai.li/Active/Shared/dataset/OmniCellTOSG_dataset_v3"
# KEGG / pathway plot script; enrichment.kegg_script in paths.yaml
KEGG_SCRIPT = str(Path(__file__).resolve().with_name("kegg_simple.R"))
# HGNC protein-coding snapshot (1.4 MB data file, tracked at tools/omic_tools/reference/ and
# identical in bb92fab); read in place rather than copied
HGNC_REFERENCE = str(Path(__file__).resolve().parent.parent / "omic_tools" / "reference"
                     / "hgnc_protein_coding.tsv")

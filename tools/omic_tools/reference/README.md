# HGNC protein-coding reference

`hgnc_protein_coding.tsv` is a compact, reproducible subset of the user-supplied
`hgnc_complete_set.txt`, added on 2026-09-10. It retains `hgnc_id`, `symbol`,
`locus_group`, `locus_type`, and `status` for rows where:

```python
row["status"] == "Approved" and row["locus_group"] == "protein-coding gene"
```

The attachment has 45,045 rows; 19,297 satisfy this rule. No symbols are renamed
or resolved through aliases or previous symbols. Immunoglobulin and T-cell
receptor genes classified as `other` in this snapshot are excluded by the rule.

- Original attachment SHA-256: `854162118530e929f06249f3349465dd5fe0515fcccf0347f463e833609c1270`
- Derived TSV SHA-256: `37446ba01ef9ac00618e2e451972385c46aaead626ea73e286a0eeaa2a91fe3a`

`celltosg_runtime_adapter.py` intersects this reference with `gene_name` in the
dataset's `bmg_gene_index.csv`. It writes a filtered copy in each temporary
loader root, retaining BMG row order and candidate indices. Expression and graph
inputs are linked from the shared dataset; the shared gene mapping is not edited.
Disease queries also supply preselected known-donor metadata in the temporary
root, so gene extraction operates on precisely that selected cohort.

For the currently configured BMG mapping, this keeps 19,109 of 41,149 symbols
and removes 22,040. Another 188 protein-coding symbols in the reference are not
present in BMG. These counts describe these snapshots; they are not fixed matrix
dimensions enforced by the code.

Filtering occurs before CellTOSG selects representative expression columns and
writes `expression_gene.csv` and `bmg_to_gene_choice.csv`. Thus CP10K denominators,
abundance rankings, DE and enrichment inputs use the retained protein-coding
features. Source expression files still have the full 412,039-column storage
axis. Existing session outputs are not regenerated. Direct calls to the generic
`omic_analysis()` component still analyze the matrices supplied by their caller.

To regenerate the compact reference from a replacement complete-set file:

```python
import csv

columns = ["hgnc_id", "symbol", "locus_group", "locus_type", "status"]
with open("hgnc_complete_set.txt", encoding="utf-8", newline="") as source:
    rows = [
        row for row in csv.DictReader(source, delimiter="\t")
        if row["status"] == "Approved" and row["locus_group"] == "protein-coding gene"
    ]
with open("hgnc_protein_coding.tsv", "w", encoding="utf-8", newline="") as target:
    writer = csv.DictWriter(
        target, fieldnames=columns, delimiter="\t", extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
```

Update the snapshot counts and checksums here and the reference regression test
when replacing this reference.

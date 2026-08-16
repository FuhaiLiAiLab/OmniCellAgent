
import pandas as pd
import numpy as np
import plotnine as p9
import warnings

# Suppress plotnine warnings about sample sizes for guides
warnings.filterwarnings('ignore', category=UserWarning, module='plotnine')

# Set seed for reproducibility (though not used for random number generation in this script)
np.random.seed(1234)

# --- Global Settings ---
# Maximum number of gene names to display per pathway
MAX_GENE_SHOW = 9

# Color palette for different ontologies/databases
# Corresponds to the R palette: c("BP"="#4DBBD5", "CC"="#00A087", "MF"="#E64B35", "KEGG"="#7E6148", "DisGeNET"="#3C5488")
PALETTE = {
    "BP": "#4DBBD5",
    "CC": "#00A087",
    "MF": "#E64B35",
    "KEGG": "#7E6148",
    "DisGeNET": "#3C5488"
}

# --- Data Loading and Conversion Functions ---

def convert_enrichment_csv(file_path: str, ontology: str = None) -> pd.DataFrame:
    """
    Reads and processes an enrichment results CSV file from Enrichr.
    This function combines the logic of R's `convert_GO_file` and `convert_kegg_csv`.

    Args:
        file_path (str): The path to the input CSV file.
        ontology (str, optional): The ontology name (e.g., 'BP', 'CC', 'MF').
                                  If None, it's treated as a KEGG file.

    Returns:
        pd.DataFrame: A cleaned and formatted DataFrame.
    """
    df = pd.read_csv(file_path)

    # Standardize column names for easier access
    df.columns = [col.replace(' ', '_') for col in df.columns]

    # Extract ID and Description from the 'Term' column
    if ontology is None or ontology == "KEGG": # KEGG file
        # For KEGG, R code extracts 'K\d+' pattern
        df['ID'] = df['Term'].str.extract(r'(K\d+)')[0]
        df['Description'] = df['Term'].str.replace(r'\s*K\d+', '', regex=True).str.strip()
    else: # GO or DisGeNET file
        df['ID'] = df['Term'].str.extract(r'(GO:\d+)')[0]
        df['Description'] = df['Term'].str.replace(r'\s*\(GO:\d+\)', '', regex=True).str.strip()

    # Process gene list
    df['geneID'] = df['Genes'].str.replace(';', '/')
    df['Count'] = df['geneID'].str.split('/').str.len()

    # Create placeholder/calculated columns similar to the R script
    df['GeneRatio'] = df['Count'].astype(str) + '/' + (df['Count'] + 100).astype(str)
    df['BgRatio'] = "1000/20000"
    df = df.rename(columns={'P-value': 'pvalue', 'Adjusted_P-value': 'p.adjust'})
    df['qvalue'] = df['p.adjust']
    
    # Add ontology column if provided
    if ontology:
        df['ONTOLOGY'] = ontology

    # Select and reorder columns
    if ontology:
        return df[['ONTOLOGY', 'ID', 'Description', 'GeneRatio', 'BgRatio',
                   'pvalue', 'p.adjust', 'qvalue', 'geneID', 'Count']]
    else:
        return df[['ID', 'Description', 'GeneRatio', 'BgRatio',
                   'pvalue', 'p.adjust', 'qvalue', 'geneID', 'Count']]

# --- Main Analysis ---

# NOTE: Replace these placeholder file paths with the actual paths to your data.


root_path = "/storage1/fs1/fuhai.li/Active/di.huang/Research/LLM/RAG-MLLM/bioRAGUI/bioRAG/tools/dataset_outputs/enrichment_results/Alzheimer's_Disease_all_regulated/"
file_paths = {
    'bp': root_path + "GO_Biological_Process_2021_results.csv",
    'cc': root_path + "GO_Cellular_Component_2021_results.csv",
    'mf': root_path + "GO_Molecular_Function_2021_results.csv",
    'disgenet': root_path + "DisGeNET_results.csv",
    'kegg': root_path + "KEGG_2021_Human_results.csv"
}

# Load and process all enrichment files
go_bp = convert_enrichment_csv(file_paths['bp'], "BP")
go_cc = convert_enrichment_csv(file_paths['cc'], "CC")
go_mf = convert_enrichment_csv(file_paths['mf'], "MF")
disgenet_df = convert_enrichment_csv(file_paths['disgenet'], "DisGeNET")
kegg_df = convert_enrichment_csv(file_paths['kegg'], "KEGG")


#----------------------------------------------------
## KEGG Dotplot
#----------------------------------------------------

# Prepare data for the KEGG dotplot
top_n = 20
plot_df_kegg = (
    kegg_df
    .sort_values('p.adjust')
    .head(top_n)
    .assign(log10_padj=lambda x: -np.log10(x['p.adjust']))
)

# Reorder 'Description' based on 'Count' for plotting (like fct_reorder)
plot_df_kegg['Description'] = pd.Categorical(
    plot_df_kegg['Description'],
    categories=plot_df_kegg.sort_values('Count', ascending=True)['Description'].unique()
)

# Create the KEGG dotplot
kegg_dotplot = (
    p9.ggplot(plot_df_kegg, p9.aes(x='Count', y='Description'))
    + p9.geom_point(p9.aes(size='Count', color='log10_padj'))
    + p9.scale_color_gradient(low="blue", high="red", name="-log10(p.adjust)")
    + p9.scale_size(range=(3, 8))
    + p9.labs(
        title="KEGG Enrichment Dotplot",
        x="Gene Count",
        y="KEGG Pathway"
    )
    + p9.theme_minimal()
    + p9.theme(
        axis_text_y=p9.element_text(size=10),
        plot_title=p9.element_text(
            # hjust=0.5,
                                    weight='bold')
    )
)

# Save the plot
kegg_dotplot.save("kegg_dotplot_python.png", width=15, height=10, dpi=300)

print("KEGG dotplot saved to kegg_dotplot_python.png")

#----------------------------------------------------
## Combined Pathway Plot
#----------------------------------------------------

# Combine all data into a single DataFrame
all_pathways = pd.concat([go_bp, go_cc, go_mf, disgenet_df, kegg_df])

# Prepare data for the combined plot: select top 5 from each category
use_pathway = (
    all_pathways
    .sort_values(['p.adjust', 'Count'], ascending=[True, False])
    .groupby('ONTOLOGY')
    .head(5)
    .reset_index(drop=True)
)

# Set the order for ONTOLOGY and Description for plotting
ontology_order = ["DisGeNET", "KEGG", "MF", "CC", "BP"]
use_pathway['ONTOLOGY'] = pd.Categorical(use_pathway['ONTOLOGY'], categories=ontology_order)
use_pathway = use_pathway.sort_values(['ONTOLOGY', 'p.adjust'], ascending=[True, False])
use_pathway['Description'] = pd.Categorical(use_pathway['Description'], categories=use_pathway['Description'].unique())

# Add index column like in R
use_pathway = use_pathway.reset_index(drop=True)
use_pathway['index'] = use_pathway.index

# --- Prepare data for rectangular category labels ---
rect_data = (
    use_pathway['ONTOLOGY']
    .value_counts()
    .rename_axis('ONTOLOGY')
    .reset_index(name='n')
)
rect_data['ONTOLOGY'] = pd.Categorical(rect_data['ONTOLOGY'], categories=ontology_order)
rect_data = rect_data.sort_values('ONTOLOGY', ascending=False).reset_index(drop=True)

rect_xmin, rect_xmax = -0.8, -0.4
rect_data['ymax'] = rect_data['n'].cumsum()
rect_data['ymin'] = rect_data['ymax'] - rect_data['n']

# Adjust ymin/ymax for padding (index in plotnine is 0-based)
rect_data['ymin'] = rect_data['ymin'] - 0.4
rect_data['ymax'] = rect_data['ymax'] - 0.6


# --- Helper function to truncate gene lists ---
def truncate_genes(gene_string: str, max_genes: int = MAX_GENE_SHOW) -> str:
    """Truncates a '/' separated string of genes."""
    genes = gene_string.split('/')
    if len(genes) <= max_genes:
        return gene_string
    return '/'.join(genes[:max_genes]) + '/...'

# Apply the function to the geneID column
use_pathway['geneID_limited'] = use_pathway['geneID'].apply(truncate_genes)
use_pathway['-log10(p.adjust)'] = -np.log10(use_pathway['p.adjust'])

# --- Create the combined plot ---
xaxis_max = use_pathway['-log10(p.adjust)'].max() + 1.5
count_x, desc_x, gene_x = -0.2, 0.05, 0.05

combined_plot = (
    p9.ggplot(use_pathway, p9.aes(x='-log10(p.adjust)', y='index', fill='ONTOLOGY'))
    # Bar plot for p-values
    + p9.geom_col(width=0.6, alpha=0.7)
    # Pathway description labels
    + p9.geom_text(p9.aes(x=desc_x, y='Description', label='Description'), ha='left', va='center',
                    size=12)
    # Gene list labels - positioned below each bar (like R's vjust=2.6)
    + p9.geom_text(
        p9.aes(x=gene_x, y='Description', label='geneID_limited', color='ONTOLOGY'),
        ha='left', va='top', nudge_y=-0.35, size=8,
        show_legend=False
    )
    # Points for gene counts
    + p9.geom_point(
        p9.aes(x=count_x, y='Description', size='Count'),
        shape='o', fill='white', show_legend=True
    )
    # Labels for gene counts
    + p9.geom_text(
        p9.aes(x=count_x, y='Description', label='Count'),
        ha='center', va='center', size=8, color='black'
    )
    + p9.scale_size_continuous(name='Count', range=(5, 12))
    # Rectangular category labels on the left
    + p9.geom_rect(
        p9.aes(xmin=rect_xmin, xmax=rect_xmax, ymin='ymin', ymax='ymax', fill='ONTOLOGY'),
        data=rect_data,
        inherit_aes=False
    )
    # Text inside the rectangular labels
    + p9.geom_text(
        p9.aes(x=(rect_xmin + rect_xmax) / 2, y='(ymin + ymax) / 2 + 0.5', label='ONTOLOGY'),
        data=rect_data,
        inherit_aes=False,
        ha='center', va='center', size=10, color='white'
    )
    # Manually drawn x-axis line
    + p9.geom_segment(
        p9.aes(x=0, y=-0.5, xend=xaxis_max, yend=-0.5),
        inherit_aes=False, size=1.5
    )
    + p9.labs(y=None, x='-log10(p.adjust)')
    # Manual color scales
    + p9.scale_fill_manual(values=PALETTE, name='Category')
    + p9.scale_color_manual(values=PALETTE)
    # X-axis configuration
    + p9.scale_x_continuous(
        breaks=np.arange(0, np.ceil(xaxis_max), 2),
        expand=(0, 0.05) # Add padding to the right
    )
    + p9.coord_cartesian(xlim=(rect_xmin, xaxis_max + 1))
    # Theme configuration (approximating theme_prism)
    + p9.theme_classic()
    + p9.theme(
        axis_text_y=p9.element_blank(),
        axis_line=p9.element_blank(),
        axis_ticks_major_y=p9.element_blank(),
        axis_title=p9.element_text(size=12),
        axis_text_x=p9.element_text(size=10),
        legend_title=p9.element_text(size=11),
        legend_text=p9.element_text(size=10),
        panel_grid_major=p9.element_blank(),
        panel_grid_minor=p9.element_blank()
    )
)

# Save the final combined plot
combined_plot.save("pathway_combined_plot_python.png", width=12, height=10, dpi=300)

print("Combined pathway plot saved to pathway_combined_plot_python.png")

#!/bin/bash
Rscript -e 'install.packages("ggprism", repos = "https://cloud.r-project.org")'
Rscript -e 'install.packages("tidyverse", repos = "https://cloud.r-project.org")'
Rscript -e 'install.packages("dplyr", repos = "https://cloud.r-project.org")' # optional
Rscript -e 'install.packages("Cairo", repos="https://cloud.r-project.org")'
Rscript -e 'if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager", repos = "https://cloud.r-project.org")'
Rscript -e 'if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager", repos="https://cloud.r-project.org"); BiocManager::install(c("ggtangle", "enrichplot"))'
Rscript -e 'BiocManager::install("org.Hs.eg.db")'
Rscript -e 'BiocManager::install("clusterProfiler")'
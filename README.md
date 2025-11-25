# Influential CpG Identification Pipeline

This project implements a **Logistic Group Lasso** pipeline to identify influential CpGs (methylation sites) that affect Gene Expression (mRNA).

## Overview

The pipeline connects Methylation data with Gene Expression data using Ensembl IDs and uses a custom Logistic Group Lasso model to select the most influential CpGs for each gene.

### Key Features
-   **Data Standardization**: Standardizes Ensembl IDs across CNV, mRNA, and Methylation datasets.
-   **Logistic Group Lasso**: Implements a custom Logistic Group Lasso algorithm (using Proximal Gradient Descent) to handle sparse feature selection.
-   **Full Genome Analysis**: Capable of analyzing all genes in the dataset efficiently.

## Project Structure

-   `main.py`: The main entry point for the analysis.
-   `models/logistic_group_lasso.py`: Custom implementation of the Logistic Group Lasso model.
-   `data_loader.py`: Utilities for loading and processing large biological datasets.
-   `prepare_data.py`: Script to prepare and standardize raw data.
-   `init.py`: Helper script for initial CpG-to-Gene mapping.

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Prepare Data
Ensure your raw data files (`dnameth.tsv`, `mrna.tsv`, `cnv.tsv`, etc.) are in the `data/` directory.
Run the preparation script to standardize IDs and identify common samples:
```bash
python3 prepare_data.py
```

### 2. Run Analysis
Run the main analysis script to identify influential CpGs for all genes:
```bash
python3 main.py
```
The results will be saved to `all_influential_cpgs.csv`.

## Methodology
1.  **Target Binarization**: mRNA expression is binarized into High/Low based on the median.
2.  **Model Training**: A Logistic Group Lasso model is trained for each gene to predict High/Low expression from CpG methylation levels.
3.  **Selection**: CpGs with non-zero weights are selected as influential.

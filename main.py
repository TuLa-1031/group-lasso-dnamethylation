import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tqdm import tqdm
import os

# Import loading function from data_loader.py
from data_loader import load_real_data
from models.logistic_group_lasso import LogisticGroupLasso

def analyze_all_genes(dnameth_path, mrna_path, mapping_path, common_samples_path, common_genes_path, output_file="all_influential_cpgs.csv"):
    print("--- Starting Analysis for ALL Genes ---")
    
    # 1. Load Data (Into Memory)
    print("Loading mRNA data...")
    mrna_df = pd.read_csv(mrna_path, sep='\t', index_col=0)
    
    print("Loading Mapping...")
    mapping_df = pd.read_csv(mapping_path)
    mapping_df = mapping_df.dropna(subset=['Ensembl_Gene_ID'])
    
    print("Loading Methylation Data (this may take a moment)...")
    # Load all common samples
    with open(common_samples_path, 'r') as f:
        common_samples = [line.strip() for line in f]
        
    # Read header to filter samples
    header = pd.read_csv(dnameth_path, sep='\t', nrows=0).columns.tolist()
    valid_samples = [s for s in common_samples if s in header]
    
    # Load full dnameth for valid samples
    # Assuming it fits in memory (~1GB for 450k CpGs x 450 Samples)
    usecols = [header[0]] + valid_samples
    dnam_df = pd.read_csv(dnameth_path, sep='\t', usecols=usecols, index_col=0)
    print(f"Methylation Data Loaded: {dnam_df.shape}")
    
    # Transpose X to (Samples x CpGs)
    X_full = dnam_df.T
    
    # Align Samples globally
    # mrna_df is typically Genes x Samples. Transpose to Samples x Genes for easier alignment.
    if len(X_full.index.intersection(mrna_df.columns)) > 0: # Check if mRNA columns are samples
        mrna_df = mrna_df.T # Now Samples x Genes
    
    common_samples_final = X_full.index.intersection(mrna_df.index)
    print(f"Common Samples for Analysis: {len(common_samples_final)}")
    
    X_full = X_full.loc[common_samples_final]
    mrna_df = mrna_df.loc[common_samples_final]
    
    # Load Gene List
    with open(common_genes_path, 'r') as f:
        genes_to_analyze = [line.strip() for line in f]
        
    print(f"Analyzing {len(genes_to_analyze)} genes...")
    
    all_results = []
    
    # Iterate with progress bar
    for gene_id in tqdm(genes_to_analyze):
        try:
            # 1. Get Target
            if gene_id not in mrna_df.columns:
                continue
            y = mrna_df[gene_id]
            
            # 2. Get CpGs
            gene_cpgs = mapping_df[mapping_df['Ensembl_Gene_ID'] == gene_id]['IlmnID'].unique()
            if len(gene_cpgs) == 0:
                continue
                
            # 3. Get Features
            # Filter X for these CpGs
            # Only keep CpGs that exist in X_full
            valid_cpgs = [c for c in gene_cpgs if c in X_full.columns]
            if not valid_cpgs:
                continue
                
            X_gene = X_full[valid_cpgs]
            
            # 4. Preprocess
            # Drop columns that are all NaN
            X_gene = X_gene.dropna(axis=1, how='all')
            
            # Update valid_cpgs to match remaining columns
            valid_cpgs = X_gene.columns.tolist()
            
            if not valid_cpgs:
                continue
                
            # Impute remaining NaNs (fast mean)
            X_gene_vals = X_gene.values
            if np.isnan(X_gene_vals).any():
                imputer = SimpleImputer(strategy='mean')
                X_gene_vals = imputer.fit_transform(X_gene_vals)
            
            # Scale
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_gene_vals)
            
            # Binarize Target
            median_val = y.median()
            y_binary = (y > median_val).astype(int)
            
            # Skip if only one class
            if len(np.unique(y_binary)) < 2:
                continue
            
            # 5. Train Model
            groups = np.arange(X_scaled.shape[1])
            # Use smaller max_iter or larger tol for speed if needed
            model = LogisticGroupLasso(groups=groups, alpha=0.05, max_iter=500, learning_rate=0.1, tol=1e-3)
            model.fit(X_scaled, y_binary)
            
            # 6. Extract Results
            weights = model.coef_
            # Vectorized creation of results
            nonzero_mask = np.abs(weights) > 0
            if np.any(nonzero_mask):
                influential_cpgs = np.array(valid_cpgs)[nonzero_mask]
                influential_weights = weights[nonzero_mask]
                
                for cpg, w in zip(influential_cpgs, influential_weights):
                    all_results.append({
                        'Gene_ID': gene_id,
                        'IlmnID': cpg,
                        'Weight': w,
                        'AbsWeight': abs(w)
                    })
                    
        except Exception as e:
            # print(f"Error analyzing {gene_id}: {e}") # Uncomment for debugging specific gene errors
            continue
            
    # Save Results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values(by=['Gene_ID', 'AbsWeight'], ascending=[True, False])
        results_df.to_csv(output_file, index=False)
        print(f"\n✅ Analysis Complete. Results saved to {output_file}")
        print(f"Total influential associations found: {len(results_df)}")
    else:
        print("\n⚠️ No influential CpGs found.")

def main():
    # Configuration
    dnameth_path = "data/dnameth.tsv"
    mrna_path = "data/mrna_standardized.tsv"
    mapping_path = "data/cg_to_ensembl.csv"
    common_samples_path = "data/common_samples.txt"
    common_genes_path = "data/common_genes.txt"
    
    analyze_all_genes(dnameth_path, mrna_path, mapping_path, common_samples_path, common_genes_path)

if __name__ == "__main__":
    main()

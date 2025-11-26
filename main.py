import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tqdm import tqdm
import os

from data_loader import load_real_data
from models.logistic_group_lasso import LogisticGroupLasso

def analyze_all_genes(dnameth_path, mrna_path, mapping_path, common_samples_path, common_genes_path, output_file="all_influential_cpgs.csv"):
    print("--- Starting Analysis for ALL Genes ---")
    
    print("Loading mRNA data:")
    mrna_df = pd.read_csv(mrna_path, sep='\t', index_col=0)
    
    print("Loading Mapping:")
    mapping_df = pd.read_csv(mapping_path)
    mapping_df = mapping_df.dropna(subset=['Ensembl_Gene_ID'])
    
    print("Loading Methylation Data:")
    with open(common_samples_path, 'r') as f:
        common_samples = [line.strip() for line in f]
        
    header = pd.read_csv(dnameth_path, sep='\t', nrows=0).columns.tolist()
    valid_samples = [s for s in common_samples if s in header]
    
    usecols = [header[0]] + valid_samples
    dnam_df = pd.read_csv(dnameth_path, sep='\t', usecols=usecols, index_col=0)
    print(f"Methylation Data Loaded: {dnam_df.shape}")
    
    X_full = dnam_df.T

    if len(X_full.index.intersection(mrna_df.columns)) > 0:
        mrna_df = mrna_df.T
    
    common_samples_final = X_full.index.intersection(mrna_df.index)
    print(f"Common Samples: {len(common_samples_final)}")
    
    X_full = X_full.loc[common_samples_final]
    mrna_df = mrna_df.loc[common_samples_final]
    
    # Load Gene List
    with open(common_genes_path, 'r') as f:
        genes_to_analyze = [line.strip() for line in f]
        
    print(f"Analyzing {len(genes_to_analyze)} genes:")
    
    all_results = []
    
    for gene_id in tqdm(genes_to_analyze):
        try:
            if gene_id not in mrna_df.columns:
                continue
            y = mrna_df[gene_id]
            
            gene_cpgs = mapping_df[mapping_df['Ensembl_Gene_ID'] == gene_id]['IlmnID'].unique()
            if len(gene_cpgs) == 0:
                continue

            valid_cpgs = [c for c in gene_cpgs if c in X_full.columns]
            if not valid_cpgs:
                continue
                
            X_gene = X_full[valid_cpgs]
            
            X_gene = X_gene.dropna(axis=1, how='all')
            
            valid_cpgs = X_gene.columns.tolist()
            
            if not valid_cpgs:
                continue
                
            X_gene_vals = X_gene.values
            if np.isnan(X_gene_vals).any():
                imputer = SimpleImputer(strategy='mean')
                X_gene_vals = imputer.fit_transform(X_gene_vals)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_gene_vals)
            
            median_val = y.median()
            y_binary = (y > median_val).astype(int)
            
            if len(np.unique(y_binary)) < 2:
                continue
            
            groups = np.arange(X_scaled.shape[1])
            model = LogisticGroupLasso(groups=groups, alpha=0.05, max_iter=500, learning_rate=0.1, tol=1e-3)
            model.fit(X_scaled, y_binary)
            
            weights = model.coef_
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
            continue
            
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values(by=['Gene_ID', 'AbsWeight'], ascending=[True, False])
        results_df.to_csv(output_file, index=False)
        print(f"\nComplete. Results saved to {output_file}")
        print(f"Total influential associations found: {len(results_df)}")
    else:
        print("\not found")

def main():
    dnameth_path = "data/dnameth.tsv"
    mrna_path = "data/mrna_standardized.tsv"
    mapping_path = "data/cg_to_ensembl.csv"
    common_samples_path = "data/common_samples.txt"
    common_genes_path = "data/common_genes.txt"
    
    analyze_all_genes(dnameth_path, mrna_path, mapping_path, common_samples_path, common_genes_path)

if __name__ == "__main__":
    main()

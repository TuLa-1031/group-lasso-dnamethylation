
import pandas as pd
import numpy as np

def load_real_data(dnameth_path, mrna_path, mapping_path, common_samples_path):
    """
    Loads and aligns Methylation and mRNA data.
    
    Args:
        dnameth_path (str): Path to dnameth.tsv
        mrna_path (str): Path to mrna_standardized.tsv
        mapping_path (str): Path to cg_to_ensembl.csv
        common_samples_path (str): Path to file containing list of allowed samples.
    
    Returns:
        X_df (pd.DataFrame): Methylation data (Samples x CpGs)
        mrna_df (pd.DataFrame): mRNA data (Samples x Genes)
        mapping_df (pd.DataFrame): Mapping table
    """
    print("--- Loading Data ---")
    
    print(f"Loading mapping from {mapping_path}:")
    mapping_df = pd.read_csv(mapping_path)
    mapping_df = mapping_df.dropna(subset=['Ensembl_Gene_ID'])
    
    print(f"Loading mRNA from {mrna_path}:")
    mrna_df = pd.read_csv(mrna_path, sep='\t', index_col=0)
    
    print(f"Loading Methylation from {dnameth_path}:")
    with open(common_samples_path, 'r') as f:
        common_samples = [line.strip() for line in f]
        
    header = pd.read_csv(dnameth_path, sep='\t', nrows=0).columns.tolist()
    valid_samples = [s for s in common_samples if s in header]
    
    usecols = [header[0]] + valid_samples
    dnam_df = pd.read_csv(dnameth_path, sep='\t', usecols=usecols, index_col=0)
    
    X_df = dnam_df.T
    
    if len(X_df.index.intersection(mrna_df.columns)) > 0:
        mrna_df = mrna_df.T
        
    common_samples_final = X_df.index.intersection(mrna_df.index)
    print(f"Common aligned samples: {len(common_samples_final)}")
    
    X_df = X_df.loc[common_samples_final]
    mrna_df = mrna_df.loc[common_samples_final]
    
    print(f"Final X shape: {X_df.shape}")
    print(f"Final mRNA shape: {mrna_df.shape}")
    
    return X_df, mrna_df, mapping_df


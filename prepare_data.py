import pandas as pd
import numpy as np

def standardize_ensembl_id(id_str):
    if isinstance(id_str, str):
        return id_str.split('.')[0]
    return id_str

def prepare_data():
    print("Loading CNV data:")
    cnv_df = pd.read_csv("data/cnv.tsv", sep='\t', index_col=0)
    print(f"CNV shape: {cnv_df.shape}")
    
    print("Loading mRNA data:")
    mrna_df = pd.read_csv("data/mrna.tsv", sep='\t', index_col=0)
    print(f"mRNA shape: {mrna_df.shape}")
    
    print("Standardizing CNV IDs:")
    cnv_df.index = cnv_df.index.map(standardize_ensembl_id)
    cnv_df = cnv_df.groupby(cnv_df.index).mean() 
    
    print("Standardizing mRNA IDs:")
    mrna_df.index = mrna_df.index.map(standardize_ensembl_id)
    mrna_df = mrna_df.groupby(mrna_df.index).mean()
    
    print("Loading CpG to Ensembl mapping:")
    mapping_df = pd.read_csv("data/cg_to_ensembl.csv")
    mapping_df = mapping_df.dropna(subset=['Ensembl_Gene_ID'])
    
    mapped_genes = set(mapping_df['Ensembl_Gene_ID'].unique())
    print(f"Found {len(mapped_genes)} genes in CpG mapping.")
    
    common_genes = set(cnv_df.index).intersection(mrna_df.index).intersection(mapped_genes)
    print(f"Found {len(common_genes)} common genes across CNV, mRNA, and CpG mapping.")
    
    print("Saving standardized CNV data (filtered by common genes):")
    cnv_df.loc[list(common_genes)].to_csv("data/cnv_standardized.tsv", sep='\t')
    
    print("Saving standardized mRNA data (filtered by common genes):")
    mrna_df.loc[list(common_genes)].to_csv("data/mrna_standardized.tsv", sep='\t')
    
    
    print("Checking dnameth samples:")
    dnameth_samples = pd.read_csv("data/dnameth.tsv", sep='\t', nrows=0).columns.tolist()
    
    common_samples = set(dnameth_samples).intersection(cnv_df.columns).intersection(mrna_df.columns)
    print(f"Found {len(common_samples)} common samples")
    
    with open("data/common_samples.txt", "w") as f:
        for sample in common_samples:
            f.write(sample + "\n")
            
    with open("data/common_genes.txt", "w") as f:
        for gene in common_genes:
            f.write(gene + "\n")
            
    print("Done")

if __name__ == "__main__":
    prepare_data()

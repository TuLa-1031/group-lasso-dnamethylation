import pandas as pd
import numpy as np

def load_real_data(dnameth_path, mapping_path, num_samples=None, common_genes_path=None, common_samples_path=None):
    """
    Loads real methylation data and mapping.
    
    Args:
        dnameth_path (str): Path to dnameth.tsv
        mapping_path (str): Path to cg_to_ensembl.csv
        num_samples (int, optional): Number of samples to load (columns from dnameth.tsv).
        common_genes_path (str, optional): Path to file containing list of allowed genes.
        common_samples_path (str, optional): Path to file containing list of allowed samples.
    
    Returns:
        X (np.array): Methylation matrix (samples x cpgs)
        y (np.array): Dummy labels (since we don't have labels file yet)
        cpg_mapping (pd.DataFrame): Mapping aligned with X columns.
        gene_names (list): List of gene names.
        capsule_sizes (list): List of capsule sizes.
    """
    print(f"Loading mapping from {mapping_path}...")
    mapping_df = pd.read_csv(mapping_path)
    
    # Use Ensembl_Gene_ID as the gene identifier. Filter out NaNs.
    mapping_df = mapping_df.dropna(subset=['Ensembl_Gene_ID'])
    mapping_df['gene'] = mapping_df['Ensembl_Gene_ID']
    
    if common_genes_path:
        print(f"Filtering by genes in {common_genes_path}...")
        with open(common_genes_path, 'r') as f:
            allowed_genes = set(line.strip() for line in f)
        mapping_df = mapping_df[mapping_df['gene'].isin(allowed_genes)]
        print(f"Mapping filtered to {len(mapping_df)} rows.")
    
    print(f"Loading methylation data from {dnameth_path}...")
    # Read dnameth.tsv. Rows=CpGs, Cols=Samples.
    
    if common_samples_path:
        print(f"Filtering by samples in {common_samples_path}...")
        with open(common_samples_path, 'r') as f:
            allowed_samples = [line.strip() for line in f]
        
        # We need to read the header first to find indices of these samples
        header = pd.read_csv(dnameth_path, sep='\t', nrows=0).columns.tolist()
        
        # Filter allowed_samples to those actually in header
        valid_samples = [s for s in allowed_samples if s in header]
        
        if num_samples:
            valid_samples = valid_samples[:num_samples]
            
        usecols = [header[0]] + valid_samples
        dnam_df = pd.read_csv(dnameth_path, sep='\t', usecols=usecols, index_col=0)
        
    elif num_samples:
        # Read header to get column names
        header = pd.read_csv(dnameth_path, sep='\t', nrows=0).columns.tolist()
        usecols = header[:num_samples+1] # +1 for index column
        dnam_df = pd.read_csv(dnameth_path, sep='\t', usecols=usecols, index_col=0)
    else:
        dnam_df = pd.read_csv(dnameth_path, sep='\t', index_col=0)
        
    print(f"Data shape: {dnam_df.shape}")
    
    # Intersect CpGs
    common_cpgs = dnam_df.index.intersection(mapping_df['IlmnID'])
    print(f"Found {len(common_cpgs)} common CpGs between data and mapping.")
    
    dnam_df = dnam_df.loc[common_cpgs]
    mapping_df = mapping_df[mapping_df['IlmnID'].isin(common_cpgs)]
    
    # Create a map from CpG_ID to Index in X.
    cpg_to_idx = {cpg: i for i, cpg in enumerate(dnam_df.index)}
    
    # Map IlmnID to index using the dictionary
    mapping_df['index'] = mapping_df['IlmnID'].map(cpg_to_idx)
    
    # Filter out CpGs that are not in dnam_df
    mapping_df = mapping_df.dropna(subset=['index'])
    
    # Convert index to int
    mapping_df['index'] = mapping_df['index'].astype(int)
    
    # Create the final mapping dataframe
    cpg_mapping_final = mapping_df[['index', 'gene', 'IlmnID']].set_index('index')
    
    # X needs to be transposed: (Samples x CpGs)
    X = dnam_df.T.values.astype(np.float32)
    
    # Dummy y
    y = np.zeros(X.shape[0], dtype=int)
    
    gene_names = sorted(cpg_mapping_final['gene'].unique().tolist())
    
    # Calculate capsule sizes
    capsule_sizes_series = cpg_mapping_final.groupby('gene').size()
    capsule_sizes = capsule_sizes_series.reindex(gene_names, fill_value=0).tolist()
        
    print(f"Prepared data: X shape {X.shape}, {len(gene_names)} genes.")
    
    return X, y, cpg_mapping_final, gene_names, capsule_sizes

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

def harmonize_namespaces(df, mapping_path):
    """
    Maps Entrez gene IDs in column names to HUGO gene symbols.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with genes as columns.
    mapping_path : str or Path
        Path to the tcga_entrez_to_hugo.pkl mapping file.
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with mapped columns, dropping any unmapped columns.
    """
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Mapping file not found at: {mapping_path}")
    
    entrez_to_hugo = joblib.load(mapping_path)
    # Ensure keys and values are string
    mapping_dict = {str(k).strip(): str(v).strip() for k, v in entrez_to_hugo.items()}
    
    # Clean whitespace in column names
    clean_cols = [str(col).strip() for col in df.columns]
    df_clean = df.copy()
    df_clean.columns = clean_cols
    
    mapped_cols = df_clean.columns.map(mapping_dict)
    df_clean.columns = mapped_cols
    
    # Drop unmapped columns (NaNs)
    df_clean = df_clean.loc[:, df_clean.columns.notna()]
    
    # Group by mapped name and take mean if there are duplicates
    if not df_clean.columns.is_unique:
        df_clean = df_clean.groupby(df_clean.columns, axis=1).mean()
        
    return df_clean

def scale_cohort(df):
    """
    Performs independent Z-score normalization (StandardScaler) across features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Expression dataframe (samples as rows, genes as columns).
        
    Returns:
    --------
    pd.DataFrame
        Z-score normalized expression dataframe.
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, index=df.index, columns=df.columns)

def align_features(df, required_features, fill_value=0.0):
    """
    Aligns dataframe columns to match a set of required features,
    sorting alphabetically, and filling missing features with a default value.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input expression dataframe.
    required_features : list
        List of required gene names.
    fill_value : float, default 0.0
        Value to fill for missing genes.
        
    Returns:
    --------
    pd.DataFrame
        Aligned dataframe with sorted required features.
    """
    df_aligned = df.copy()
    
    # Find missing genes
    missing_genes = list(set(required_features) - set(df_aligned.columns))
    
    # Fill missing genes with the default value
    for gene in missing_genes:
        df_aligned[gene] = fill_value
        
    # Reindex to keep only required features and sort alphabetically
    sorted_features = sorted(required_features)
    df_aligned = df_aligned.reindex(columns=sorted_features)
    
    return df_aligned

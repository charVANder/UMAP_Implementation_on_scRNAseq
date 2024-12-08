import pandas as pd
from scipy.io import mmread

def load_data(counts_matrix_path, features_path, barcodes_path):
    '''Loads the gene expression counts matrix and associated feature and barcode files. Uses dense matrices for easier processing.

    Parameters:
        counts_matrix_path (str): path to the gene expression counts matrix file in MTX format.
        features_path (str): path to the features file that has the gene identifiers/names.
        barcodes_path (str): path to the barcodes file that has the cell identifiers.

    Returns: A tuple with the following three things:
        matrix (pd.DataFrame): Dense matrix of gene expression counts with shape (genes, cells).
        features (pd.DataFrame): DataFrame with the gene IDs and names.
        barcodes (pd.DataFrame): DataFrame with the cell barcodes.
    '''
    matrix = mmread(counts_matrix_path).todense()  # converting to dense for easier processing
    features = pd.read_csv(features_path, sep='\t', header=None)
    barcodes = pd.read_csv(barcodes_path, sep='\t', header=None)
        
    return matrix, features, barcodes

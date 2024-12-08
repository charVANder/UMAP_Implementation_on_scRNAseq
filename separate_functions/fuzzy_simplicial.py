import numpy as np
from scipy.sparse import csr_matrix

def get_fuzzy_simplicial(similarity_matrix):
    ''' Creates the fuzzy-simplicial set (pairwise probability distributions) using the similarity matrix.
    
    Parameters:
        similarity_matrix: the high dimension pairwise similarity matrix
    
    Returns:
        fuzzy_simplicial_set: matrix of pairwise probabilities.
    '''
    # Normalizing the similarity matrix
    row_sums = similarity_matrix.sum(axis=1)
    probability_matrix = similarity_matrix / row_sums[:, np.newaxis]

    # The graph needs to be undirected. Need to symmetrize matrix -> P(i,j) = P(j,i)
    probability_matrix = (probability_matrix + probability_matrix.T) / 2
    
    # Use sparse matrix for memory efficiency
    fuzzy_simplicial_set = csr_matrix(probability_matrix)
    
    return fuzzy_simplicial_set

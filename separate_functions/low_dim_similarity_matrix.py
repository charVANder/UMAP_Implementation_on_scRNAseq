import numpy as np

def get_low_dim_similarity_matrix(embedding, alpha=1.0):
    ''' Calculates pairwise similarities b/w points in the low-dimensional space using t-distribution.
    
    Parameters:
        embedding (numpy array): low-dimensional embedding of shape (n_points, n_components).
        alpha (float): degrees of freedom for the t-distribution (default for umap-learn is alpha=1.0--Cauchy distribution).
        
    Returns:
        np.ndarray: pairwise similarity matrix in the low-dimensional space.
    '''    
    # Calculating Euclidean pairwise distances in low-dimension space
    distances = np.linalg.norm(embedding[:, np.newaxis] - embedding, axis=2)
    
    # Applying t-distribution kernel to compute pairwise similarities
    low_similarity_matrix = (1 + (distances ** 2) / alpha) ** -alpha

    # Normalize to convert into probabilities
    row_sums = low_similarity_matrix.sum(axis=1)
    low_similarity_matrix = low_similarity_matrix / row_sums[:, np.newaxis]

    # Verification - make sure row sums are 1
    if not np.allclose(np.sum(low_similarity_matrix, axis=1), 1, atol=1e-6):
        print("Warning: The row sums do not equal 1")
    
    return low_similarity_matrix

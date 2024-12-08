import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import csr_matrix

def estimate_sigma(data, n_neighbors=15): # default is 15 in umap-learn
    # Get distances to n_neighbors nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    neighbors.fit(data)
    distances, blah = neighbors.kneighbors(data)
    
    # Get the median distance of the knn
    median_distance = np.median(distances[:, -1])  # using distance to the k-th nearest neighbor
    
    return median_distance

def get_high_dim_similarity_matrix(data, n_neighbors=15, sigma=None):
    ''' Computes the pairwise Gaussian similarity matrix for the given data using UMAP-like sigma calculation.
    
    Parameters:
        data: the DataFrame of gene expression data
        n_neighbors: number of nearest neighbors (default 15)
        sigma: scaling parameter of Gaussian kernel. If None, it will be estimated automatically similar to umap-learn
    
    Returns:
        similarity_matrix: matrix where the value at [a, b] is the similarity b/w data points a and b.
    '''
    if sigma is None:
        sigma = estimate_sigma(data, n_neighbors) # getting sigma like umap-learn
    
    # Calculating pairwise Euclidean distances for all points
    distances = euclidean_distances(data, data)

    # Convert distances to similarities with gaussian
    similarity_matrix = np.exp(-0.5 * (distances / sigma) ** 2)

    # Sparse matrices allow faster computation of operations by focusing only on the non-zero elements.
    similarity_matrix_sparse = csr_matrix(similarity_matrix)

    return similarity_matrix_sparse

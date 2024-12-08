import numpy as np

def initialize_embedding(n_points, n_components=2, random_seed=42):
    ''' Initializes the embedding randomly in lower-dimensional space.
    
    Parameters:
        n_points (int): number of data points (number rows/columns in the similarity matrix).
        n_components (int): dimension of the embedding (default is 2 for 2D which is standard).
        random_seed: random seed for reproducibility.
        
    Returns:
        embedding (numpy array): Randomly initialized embedding of shape (n_points, n_components).
    '''
    np.random.seed(random_seed)  # for reproducibility
    embedding = np.random.randn(n_points, n_components)
    return embedding

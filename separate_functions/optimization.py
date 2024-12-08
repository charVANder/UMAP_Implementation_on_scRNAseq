import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform


def cross_entropy_loss(P, Q, epsilon=1e-10):
    '''Computes the cross-entropy loss between the high-dimensional fuzzy set (P) and low-dimensional (Q) probability matrices.
    
    Parameters:
        P: high-dimensional (fuzzy-simplicial set) probability matrix
        Q: low-dimensional probability matrix
        epsilon: small value for stability

    Returns:
        loss = the cross-entropy loss
    '''
    # Make sure P and Q are dense numpy arrays. Convert as needed like if they are sparse
    P = P.toarray() if isinstance(P, csr_matrix) else P
    Q = Q.toarray() if isinstance(Q, csr_matrix) else Q
    
    # To avoid log(0)
    P = np.clip(P, epsilon, 1 - epsilon)
    Q = np.clip(Q, epsilon, 1 - epsilon)

    # Calculating the cross-entropy loss
    loss = -np.sum(P * np.log(Q) + (1 - P) * np.log(1 - Q))
    return loss


def calculate_gradient(P, Q, embedding, epsilon=1e-10):    
    '''Calculates the gradient of the loss function with respect to the embedding.

    Parameters:
        P: high-dimensional (fuzzy-simplicial set) probability matrix
        Q: low-dimensional probability matrix
        embedding (numpy array): low-dimensional embedding of shape (n_points, n_components).
        epsilon: small value for stability

    Returns:
        gradient: 
    '''
    # Convert embedding to dense array if needed
    embedding = np.array(embedding)
    
    # Calculating pairwise distances
    distances = squareform(pdist(embedding))
    gradient = np.zeros_like(embedding)
    
    for index1 in range(embedding.shape[0]):
        for index2 in range(index1 + 1, embedding.shape[0]):
            diff = embedding[index1] - embedding[index2]
            dist = distances[index1, index2] + epsilon  # preventing division by zero
            
            # Gradient for the t-distribution (chain rule)
            diff_term = (P[index1, index2] - Q[index1, index2]) * diff / (dist ** 2)
            gradient[index1] += diff_term
            gradient[index2] -= diff_term
    return gradient
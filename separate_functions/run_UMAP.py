import numpy as np
from low_dim_similarity_matrix import get_low_dim_similarity_matrix
from optimization import cross_entropy_loss, calculate_gradient

def run_UMAP(P, embedding, learning_rate=1.0, n_iterations=500, alpha=1.0, tol=1e-4):
    '''Runs the UMAP algorithm all the way to the optimization of the low-dimensional embedding via gradient descent.
    
    Parameters:
        P: high-dimensional (fuzzy-simplicial set) probability matrix
        embedding: initial low-dimensional embedding.
        learning_rate: step size for gradient descent.
        n_iterations: number of iterations for the optimization.
        alpha: t-distribution parameter for low-dimensional similarity calculation. Default is usually 1.0
        tol: convergence threshold for the loss difference.
        
    Returns:
        embedding: The optimized low-dimensional embedding.
    '''
    prev_loss = float('inf')  # initializing previous loss as infinity

    for iteration in range(n_iterations):  # adjust as needed
        # Recalculating the low-dimensional similarities based on the current embedding
        Q = get_low_dim_similarity_matrix(embedding, alpha=alpha)
        
        # Updating w/ gradient
        gradient = calculate_gradient(P, Q, embedding)  # getting loss gradient
        embedding -= learning_rate * gradient

        # Loss difference for convergence check
        current_loss = cross_entropy_loss(P, Q, epsilon=1e-10)
        loss_diff = np.abs(prev_loss - current_loss)

        # If the change in loss is smaller than the tolerance, break early
        if loss_diff < tol:
            print(f"Converged at iteration {iteration} with loss difference {loss_diff:.5f}")
            break

        # Printing every 10 iterations to track progress
        if iteration % 10 == 0:
            print(f"Iteration {iteration} of {n_iterations}, Loss: {current_loss:.6f}")

    return embedding

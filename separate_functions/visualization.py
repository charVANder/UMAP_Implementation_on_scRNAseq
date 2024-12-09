import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def plot_umap(embedding, labels=None, title="UMAP Projection", cmap='viridis'):
    ''' Creates visualization of 2D UMAP embedding with a scatter plot.

    Parameters:
        embedding (2D array): final low-dimensional embedding of the data
        labels: cluster labels for each point, used to add color. Default is None
        title (str): title of the plot. The default is "UMAP Projection"
        cmap: matplotlib colormap option. Default is viridis

    Returns:
        None: displays the UMAP projection as a scatter plot.
    '''
    plt.figure(figsize=(8, 6))
    
    # Option to show labels if provided.
    # For example, running KMeans or DBSCAN on the UMAP embeddings to pass cluster labels as labels.
    # UMAP for dimension reduction and then clustering is often done in industry.
    if labels is not None:
        plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap=cmap, s=5)
        plt.colorbar()
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], color='tab:blue', s=5)
    plt.title(title)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    #plt.axis('equal')
    #plt.tight_layout()
    plt.show()


def plot_optimization(iterations, loss_values, title):
    ''' Plots the UMAP optimization progress by visualizing the cross-entropy loss over iterations.

    Parameters:
        iterations (list): a list of integers containing the iteration numbers
        loss_values (list): a list containing the cross-entropy loss values at each iteration.
        title (str): title of the plot.

    Returns:
        None: displays the plot with loss values on the y-axis and iteration numbers on the x-axis.
    '''
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, loss_values, marker='o', linestyle='-', color='tab:blue')
    plt.title(title, fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.show()
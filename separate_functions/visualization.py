import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

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

def main():
    # Loading the saved embeddings and etc.
    final_fem_embedding = np.load('../final_embeddings/fem_embedding.npy')
    final_male_embedding = np.load('../final_embeddings/male_embedding.npy')
    fem_dbscan = DBSCAN(eps=0.5, min_samples=12)
    male_dbscan = DBSCAN(eps=0.5, min_samples=12)
    fem_dbscan_labels = fem_dbscan.fit_predict(final_fem_embedding)
    male_dbscan_labels = male_dbscan.fit_predict(final_male_embedding)


    iterations = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    fem_loss_values = [
        94717.808394, 92707.100954, 92042.217898, 91648.584677,
        91288.015226, 90976.559372, 90714.265300, 90479.552085,
        90246.052480, 90054.146767
    ]
    male_loss_values = [
        95247.281294, 93035.649956, 92187.536731, 91675.211230,
        91315.740198, 91017.201041, 90754.578002, 90514.253475,
        90289.576375, 90096.138191
    ]
    
    # Female UMAP Visuals (projection, DBSCAN projection, and optimization)
    plot_umap(final_fem_embedding, title="Female Patient Single-Cell UMAP Projection: Aortic Valve Leaflet Cells in AVS")
    plot_umap(final_fem_embedding, labels=fem_dbscan_labels, title="Female Patient UMAP Projection with DBSCAN Clustering", cmap='Set1')
    plot_optimization(iterations, fem_loss_values, "Loss over Iterations during UMAP Optimization (female dataset)")

    # Male UMAP Visuals (projection, DBSCAN projection, and optimization)
    plot_umap(final_male_embedding, title=" Male Patient Single-Cell UMAP Projection: Aortic Valve Leaflet Cells in AVS")
    plot_umap(final_male_embedding, labels=male_dbscan_labels, title="Male Patient UMAP Projection with DBSCAN Clustering", cmap='Set1')
    plot_optimization(iterations, male_loss_values, "Loss over Iterations during UMAP Optimization (male dataset)")
if __name__ == "__main__":
    main()
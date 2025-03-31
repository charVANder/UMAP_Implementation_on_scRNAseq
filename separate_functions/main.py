import pandas as pd
import numpy as np
import tarfile
import os
from load_data import load_data
from run_UMAP import run_UMAP
from high_dim_similarity_matrix import estimate_sigma, get_high_dim_similarity_matrix
from fuzzy_simplicial import get_fuzzy_simplicial
from initialize_embedding import initialize_embedding

def extract_tar(tar_path, extract_to):
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=extract_to)

def main():
    # Extracting from tar file
    tar_file_path = '../data/GSE273980_RAW.tar'
    extract_dir = '../data/GSE273980_RAW'  # dir to extract files to
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)  # making sure directory exists
        extract_tar(tar_file_path, extract_dir)
    
    # Reading in data
    fem_counts_matrix_path = os.path.join(extract_dir, 'GSM8441017_female_AVS_HH6_matrix.mtx.gz')
    fem_features_path = os.path.join(extract_dir, 'GSM8441017_female_AVS_HH6_features.tsv.gz')
    fem_barcodes_path = os.path.join(extract_dir, 'GSM8441017_female_AVS_HH6_barcodes.tsv.gz')
    male_counts_matrix_path = os.path.join(extract_dir, 'GSM8441018_male_AVS_HH4_matrix.mtx.gz')
    male_features_path = os.path.join(extract_dir, 'GSM8441018_male_AVS_HH4_features.tsv.gz')
    male_barcodes_path = os.path.join(extract_dir, 'GSM8441018_male_AVS_HH4_barcodes.tsv.gz')

    # Loading data
    fem_counts_matrix, fem_features, fem_barcodes = load_data(fem_counts_matrix_path, fem_features_path, fem_barcodes_path)
    male_counts_matrix, male_features, male_barcodes = load_data(male_counts_matrix_path, male_features_path, male_barcodes_path)

    # Creating DataFrames
    fem_df = pd.DataFrame(fem_counts_matrix.T, index=fem_barcodes[0], columns=fem_features[1]) 
    male_df = pd.DataFrame(male_counts_matrix.T, index=male_barcodes[0], columns=male_features[1])

    # Getting sigmas for both datasets
    fem_sigma = estimate_sigma(fem_df.values, n_neighbors=15)
    male_sigma = estimate_sigma(male_df.values, n_neighbors=15)

    # Creating high-dimension similarity matrices and then fuzzy-simplicial sets
    fem_similarity_matrix = get_high_dim_similarity_matrix(fem_df.values, n_neighbors=15, sigma=fem_sigma)
    male_similarity_matrix = get_high_dim_similarity_matrix(fem_df.values, n_neighbors=15, sigma=male_sigma)
    fem_fuzzy_simplicial = get_fuzzy_simplicial(fem_similarity_matrix)
    male_fuzzy_simplicial = get_fuzzy_simplicial(male_similarity_matrix)

    # Getting the initial low-dimensional embeddings
    fem_embedding = initialize_embedding(fem_similarity_matrix.shape[0], n_components=2)
    male_embedding = initialize_embedding(male_similarity_matrix.shape[0], n_components=2)

    # Running gradient descent optimization to refine the embeddings
    # UMAP is usually 500 iterations by default, but this takes a LONG TIME to run, so iterations have been lowered
    # This was more of a proof of concept. In the future, it would be better to try smaller datasets as well
    final_fem_embedding = run_UMAP(fem_fuzzy_simplicial, fem_embedding, learning_rate=1.0, n_iterations=100, alpha=1.0, tol=1e-4)
    final_male_embedding = run_UMAP(male_fuzzy_simplicial, male_embedding, learning_rate=1.0, n_iterations=100, alpha=1.0, tol=1e-4)

    # # OPTIONAL - saving final embeddings as numpy files because running this takes forever and a day
    # np.save('embeddings/fem_embedding.npy', final_fem_embedding)
    # np.save('embeddings/male_embedding.npy', final_male_embedding)

    # # VISUALIZATION
    # Because running the algorithm took too long, the embeddings have been saved into a file.
    # You can run the visualization.py file to view the UMAP projections along with the optimization plots (iteration vs. cross-entropy loss score).


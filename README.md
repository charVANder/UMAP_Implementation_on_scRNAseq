# UMAP_Implementation
Simplified UMAP implementation done from scratch. Mini project for BINF6250 (Algorithmic Foundations in Bioinformatics) course at the Roux Institute at Northeastern University.

## <u>Reflection Notes</u>
I would first like to emphasize that this project was not done to create an efficient or modularized algorithm ready for widespread use. Rather, it was more of a proof of concept, and a way to better learn about the intricacies of UMAP by creating a python algorithm from scratch without sklearn or scipy. I personally find single-cell RNA sequencing data very interesting, and UMAP is one of the best ways to capture local and global structures in those datasets. It is also fairly new and used widely in industry today.

The biggest issue with this implementation was the runtime and computational efficiency. Normally, I would have the UMAP algorithm run about 500 iterations in the optimization step by default (I was trying to emulate what I knew about `umap-learn` defaults). However, this implementation was taking much too long to run on the datasets (which were scRNAseq datasets from GEO). This was even the case when I lowered `n_iterations` to only 100 (ideally, I would have liked to set it closer to 500). Alternative options to fix the long runtimes might be to simplify the datasets (gene expression files are always large and complex), make the functions more efficient, run batches, find a new/simpler dataset to run, or just run UMAP the normal way with a standard library and tools such as umap-learn/scanpy. Another thing I would want to do if I had more time would be to try PCA on the dataset beforehand for noise reduction and computational efficiency. I would also try running it through a scaler (such as sklearn's StandardScaler) to help improve the convergence with gradient descent. The scaler would ensure that distance-based methods worked more effectively.

It should be noted that for ease of use, I chose to let the implementation run separately in the background, and then saved the final embeddings as separate files so that I could use them for visualization without having to rerun everything. Despite long runtimes and inefficiency, this entire process really taught me a lot about the in-depth workings and mathematics behind UMAP. The next time I use this algorithm I feel that I'll be more confident about tuning hyperparameters to best suit the needs of my dataset!

## <u>The Data</u>
The data that supports any possible findings in this project have been deposited in GEO with the accession code [GSE273980](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE273980) (recently made public December 1st, 2024). These are single cell gene expression profiles of cells isolated from human aortic valve leaflets. According to the studies done in the Aguado lab at University of California San Diego, aortic valve cell heterogeneity increases during AVS (aortic valve stenosis) progression. The researches are using single cell RNA sequencing to characterize valve cell phenotypes in AVS patients. Whole cells were isolated from aortic valve leaflets of age and disease-matched AVS patients--one male and one female.
* GSM8441017_female_AVS_HH6_barcodes.tsv.gz
* GSM8441017_female_AVS_HH6_features.tsv.gz
* GSM8441017_female_AVS_HH6_matrix.mtx.gz
* GSM8441018_male_AVS_HH4_barcodes.tsv.gz
* GSM8441018_male_AVS_HH4_features.tsv.gz
* GSM8441018_male_AVS_HH4_matrix.mtx.gz

In this project, the data files shown above were refactored to create two gene expression DataFrames, one for the female patient, and one for the male patient. The intention was to implement my scratch UMAP algorithm on both sets and compare the final embeddings.

## <u>The Algorithm</u>
Before starting on the implementation, I tried to organize what I knew about UMAP into separate steps. I would then focus on creating functions for each part and then combine them in the end.

#### 1. Creating a High-Dimensional (original data) Similarity Matrix
* This would be done by converting high-dimensional Euclidean distances between data points (the cells) into similarities using a Gaussian kernel.
* In the `umap-learn` library, the default value for `sigma` (basically the "spread" of the Gaussian kernel) is tuned based on the data. It uses the median distance to the k-nearest neighbors (default `n_neighbors=15`) as the estimate for `sigma`. My original code was setting `sigma` to the median of the pairwise distances, meaning that I would be looking through every single point rather than just the nearest neighbors. Unfortunately, this could result in outliers influencing the value of sigma, potentially making the kernel scale too large or too small for certain points in denser areas. Because of this, I made a last minute switch to mimic `umap-learn` with sklearn's `NearestNeighbors`.
* See `high_dim_similarity_matrix.py` in the `separate_functions` directory.

<p align="center">
<img width="325" alt="high_dim_similarity" src="figs/high_dim_sim.png">
</p> 

#### 2. Creating the Fuzzy-Simplicial Set
* Converting the high-dimensional similarity matrix into a probability distribution. A weighted graph where the weights represent the probability of edges between the points.
* See `fuzzy_simplicial.py` in the `separate_functions` directory.

#### 3. Randomly initialize the embedding in a lower-dimensional space (usually 2D).
* This function would serve as the starting point for the optimization.
* See `initialize_embedding` in the `separate_functions` directory.

#### 4. Create the Low-Dimensional Similarity Matrix/Probabilities
* In UMAP, low-dimensional similarities are calculated using a t-distribution, which helps to maintain local structure.
* See `low_dim_similarity_matrix.py` in the `separate_functions` directory.

<p align="center">
<img width="200" alt="fuzzy_simplicial" src="figs/fuzzy_simplicial.png">
</p> 

#### 5. Optimization and Refinement
* In UMAP, the final embedding is optimized by minimizing the cross-entropy loss between the high-dimensional and low-dimensional similarities. It is minimizing the cross-entropy loss between high-dimensional fuzzy simplicial set and low-dimensional set, usually with stochastic gradient descent or other gradient based methods. UMAP keeps iterating these steps until convergence is reached.
* Gradient descent optimizes the low-dimensional embedding by adjusting the points to minimize the difference between high-dimensional and low-dimensional similarities. The chain rule is used to compute the gradient of the loss function with respect to the embedding, capturing how small changes in the embedding affect the pairwise similarities, and guiding the updates as such.
* See `optimization.py, run_UMAP.py, and main.py` in the `separate_functions` directory.
* To avoid rerunning the implmentation (which takes an unreasonably long time on the GEO datasets), the final embedding files were saved as `.npy`.

<p align="center">
<img width="475" alt="cross_entropy" src="figs/cross_entropy.png">
</p> 
<p align="center">
<img width="375" alt="chain_rule" src="figs/chain_rule.png">
</p> 

6. Visualization
    * Additional functions to visualize results were created and can be viewed in `separate_functions/visualization.py`.

## <u>Results</u>
### UMAP Projections for Female/Male Datasets:
<p float="left", align="center">
  <img src="figs/fem_umap_projection.png" width="400"/>
  <img src="figs/male_umap_projection.png" width="396"/> 
</p>

### Cross-Entropy Optimization with Gradient Descent (100 iterations):
By tracking the cross-entropy loss for every 10 iteration, we can see that the algorithm was working and loss was being minimized for the 100 iterations. That said, if runtime were not an issue, then running 500+ iterations would have been better, allowing the loss to taper off at the end and break the loop once convergence was actually reached.
<p float="left", align="center">
  <img src="figs/fem_optimization.png" width="400"/>
  <img src="figs/male_optimization.png" width="400"/> 
</p>


## <u>References</u>
* GEO data acquisition [here](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE273980)
* Numpy, Scikit-learn, and Scipy documentation was used throughout
* Umap-learn documentation [here](https://umap-learn.readthedocs.io/en/latest/)
* UMAP example with math formulas [here](https://towardsdatascience.com/how-to-program-umap-from-scratch-e6eff67f55fe)
* More on UMAP [here](https://pair-code.github.io/understanding-umap/)
* StackOverflow - avoiding log(0) in cross-entropy loss function [here]( https://stackoverflow.com/questions/47377222/what-is-the-problem-with-my-implementation-of-the-cross-entropy-function)

### AI Appendix:
* Prompt: *How do I apply gaussian kernel to convert distances to similarities in umap by scratch?*
    * This helped me by providing the general math equation used to make the calculations.
* Prompt: *Can you give me a mathematical equation that represents cross-entropy loss (also used gradient descent, t-dist kernel, etc)?*
    * This helped provide the math equation images in this README.
* Prompt: *Can you help me find the error in my gradient descent function?*
    * I was using the chain rule incorrectly and was able to fix the issue.
* Prompt: *Why do I keep getting this same error "ValueError the truth value of an array with more than one element is ambiguous. Use a.any() or a.all()"*
    * This helped fix an error in my `cross_entropy_loss` function where I was unknowingly using a CSR sparse matrix in a boolean context. I was using `np.clip()` before converting to the dense array.
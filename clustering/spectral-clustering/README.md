# Spectral Clustering Lab

This folder contains a from-scratch implementation of **Spectral Clustering**, a technique that uses the eigenvalues (spectrum) of the similarity matrix of the data to perform dimensionality reduction before clustering in fewer dimensions.

## Files Overview

- `spectral_clustering.py`: The main implementation of the spectral clustering algorithm.
- `datasets.py`: Utility functions to generate synthetic datasets, including interleaved "moons" and multi-dimensional Gaussians.

## How the Algorithm Works

The implementation in `spectral_clustering.py` follows these mathematical steps:

1.  **Affinity Matrix ($A$):** Computes the similarity between all pairs of points using a Radial Basis Function (RBF) kernel:  
    $A_{ij} = \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right)$
2.  **Degree Matrix ($D$):** A diagonal matrix where each entry is the sum of the corresponding row in the affinity matrix.
3.  **Laplacian Matrix ($L$):** Computes the unnormalized Laplacian as $L = D - A$.
4.  **Eigen-decomposition:** Calculates the eigenvalues and eigenvectors of $L$.
5.  **Clustering:**
    -   **Fiedler Solution:** For binary clustering, it uses the second smallest eigenvector (the Fiedler vector) and thresholds it at zero.
    -   **K-Means Solution:** Projects the data into the subspace formed by the first $k$ eigenvectors and applies the K-Means algorithm.

## Requirements

To run this code, you need the following Python libraries:

```bash
pip install numpy matplotlib scikit-learn
```

## Usage

You can run the script directly to see the clustering results on a Gaussian dataset:

```bash
python spectral_clustering.py
```

Inside `spectral_clustering.py`, you can toggle between `fiedler_solution=True` for binary partitioning or `fiedler_solution=False` to use the K-Means approach for $n$ clusters. You can also tune the `sigma` parameter to adjust the sensitivity of the RBF kernel.
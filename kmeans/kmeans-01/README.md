# K-Means Clustering Project

This folder contains a custom implementation of the K-Means clustering algorithm and several scripts to demonstrate its effectiveness on both synthetic datasets and real-world image segmentation tasks.

## Files Overview

- **cluster.py**: This is the core module containing the `KMeans` class. It implements the algorithm from scratch, including logic for center initialization, cluster assignment, and centroid updates. It also supports multiple restarts (`n_init`) to avoid local minima by selecting the result with the lowest cost.

- **datasets.py**: Contains utility functions to generate synthetic data for testing.
  - `gaussians_dataset`: Generates points from multiple multivariate Gaussian distributions.
  - `two_moon_dataset`: Generates the classic "two moons" interleaving circles, useful for showing the limitations of K-Means on non-spherical clusters.

- **avengers.py**: Implements an image segmentation application. It uses K-Means to cluster pixels based on color information to isolate subjects (like Tony Stark) from their background. It also includes a `blend` function to composite the segmented subject onto a new background image.

- **main.py**: The primary entry point for running experiments. It uses `argparse` to allow switching between different scenarios via the command line.

## How to Run

You can execute the experiments by running `main.py` with the `--dataset` flag:

### 1. Synthetic Gaussian Clusters
Visualizes how K-Means partitions clusters generated from different Gaussian distributions.
```bash
python main.py --dataset gaussians
```

### 2. Two Moons Dataset
Demonstrates K-Means' behavior on non-linearly separable data.
```bash
python main.py --dataset twomoon
```

### 3. Image Segmentation (Avengers)
Performs color-based segmentation on an image and blends the result with a background.
```bash
python main.py --dataset avengers
```

## Requirements

- `numpy`, `matplotlib`, `opencv-python` (cv2), and `scikit-learn`.
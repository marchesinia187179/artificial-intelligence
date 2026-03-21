# LDA Implementation and Iris Demo

This folder contains a minimal Linear Discriminant Analysis (LDA) implementation (`lda.py`) and a simple test/demo script (`lda_tests.py`) that applies LDA to the Iris dataset.

## Files

- `lda.py`: LDA class implementing fit and transform methods.
- `lda_tests.py`: demo script loading `sklearn.datasets.load_iris`, fitting LDA, projecting data into 2 components, and plotting the result.

## Overview

`lda.py` follows the standard Fisher LDA pattern:
- compute within-class scatter matrix `S_W`
- compute between-class scatter matrix `S_B`
- solve generalized eigenproblem for `S_W^{-1} S_B`
- sort eigenvectors by eigenvalue magnitude
- select top `n_components` discriminant directions

`transform(X)` projects input data onto the learned discriminants.

## Usage

Install dependencies:
```bash
pip install numpy scikit-learn matplotlib
```

Run the demo:
```bash
cd linear-discriminant-analysis/lda00
python lda_tests.py
```

You should see the Iris dataset projected onto the first two linear discriminants with points colored by species.

## `lda.py` details

Constructor:
- `LDA(n_components)`: number of projection components.

Methods:
- `fit(X, y)`: computes means, `S_W`, `S_B`, eigenvalues/vectors, and stores top `n_components` linear discriminants.
- `transform(X)`: returns `X` projected into discriminant subspace.

## `lda_tests.py` details

- Loads Iris dataset (`X`, `y`).
- Instantiates `LDA(2)`.
- Fits and projects `X` to 2D.
- Prints shapes and plots projected points with color mapping.

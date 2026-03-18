# Cat-or-Dog Classifier

This folder contains a simple Gaussian Naive Bayes classifier (`cat-or-dog.py`) that predicts whether a sample is a **cat** or a **dog** based on two numerical features (e.g., weight and height).

## What `cat-or-dog.py` does

- Reads a CSV dataset from `cat-or-dog-classifier/data.csv`
- Plots the data using two selected numeric features (`Weight_kg` vs `Height_cm`)
- Trains a **Gaussian Naive Bayes** model from scratch
- Evaluates the model on a hold-out test set and prints the accuracy

The implementation uses a custom `NaiveBayes` class (no scikit-learn model is used for the classifier itself).

## Requirements

- Python 3.x
- numpy
- pandas
- matplotlib
- scikit-learn

Install dependencies with:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## How to run

From the repository root (or from within `cat-or-dog-classifier`), run:

```bash
python cat-or-dog-classifier/cat-or-dog.py
```

The script will show a scatter plot and print the classification accuracy on the test split.

## File overview

- `cat-or-dog.py`: main script that loads data, plots it, trains the classifier, and prints accuracy.
- `data.csv`: dataset with numeric features and a `Class` column (`cat`/`dog`).# Cat-or-Dog Classifier

This folder contains a simple Gaussian Naive Bayes classifier (`cat-or-dog.py`) that predicts whether a sample is a **cat** or a **dog** based on two numerical features (e.g., weight and height).

## What `cat-or-dog.py` does

- Reads a CSV dataset from `cat-or-dog-classifier/data.csv`
- Plots the data using two selected numeric features (`Weight_kg` vs `Height_cm`)
- Trains a **Gaussian Naive Bayes** model from scratch
- Evaluates the model on a hold-out test set and prints the accuracy

The implementation uses a custom `NaiveBayes` class (no scikit-learn model is used for the classifier itself).

## Requirements

- Python 3.x
- numpy
- pandas
- matplotlib
- scikit-learn

Install dependencies with:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## How to run

From the repository root (or from within `cat-or-dog-classifier`), run:

```bash
python cat-or-dog-classifier/cat-or-dog.py
```

The script will show a scatter plot and print the classification accuracy on the test split.

## File overview

- `cat-or-dog.py`: main script that loads data, plots it, trains the classifier, and prints accuracy.
- `data.csv`: dataset with numeric features and a `Class` column (`cat`/`dog`).
# Orange or Grapefruit Classifier

This project implements a Naive Bayes classifier from scratch to distinguish between oranges and grapefruits based on physical characteristics such as diameter, weight, and color components (red, green, blue).

## Overview

The `main.py` file contains a complete implementation of a Gaussian Naive Bayes classifier. It loads data from `data.csv`, trains the model, makes predictions, and evaluates accuracy. The classifier assumes that the features follow a Gaussian distribution for each class.

## Data Description

The dataset in `data.csv` contains measurements for oranges and grapefruits. Each row represents a fruit with the following columns:

- `name`: The class label ("orange" or "grapefruit")
- `diameter`: The diameter of the fruit (in cm)
- `weight`: The weight of the fruit (in grams)
- `red`: Red color component (0-255)
- `green`: Green color component (0-255)
- `blue`: Blue color component (0-255)

### Sample Data

| name      | diameter | weight | red | green | blue |
|-----------|----------|--------|-----|-------|------|
| orange    | 2.96     | 86.76  | 172 | 85    | 2    |
| orange    | 3.91     | 88.05  | 166 | 78    | 3    |
| grapefruit| 4.42     | 95.17  | 156 | 81    | 2    |
| grapefruit| 4.47     | 95.6   | 163 | 81    | 4    |
| ...       | ...      | ...    | ... | ...   | ...  |

## Code Explanation

### Imports
- `numpy`: For numerical operations and array handling.
- `pandas`: For data manipulation and CSV reading.
- `sklearn.model_selection.train_test_split`: To split the dataset into training and testing sets.
- `matplotlib.pyplot`: For plotting the data.

### Naive_Bayes Class

#### `__init__`
Initializes the classifier with placeholders for classes, number of classes, means, variances, and priors.

#### `fit(X_train, y_train)`
Trains the model by:
- Converting data to numpy arrays.
- Identifying unique classes.
- Calculating mean, variance for each feature per class, and prior probabilities.
- Printing the computed statistics for each class.

#### `predict(X_test)`
Makes predictions for test data by:
- For each test sample, computing the posterior probability for each class using the log of priors and likelihoods.
- Selecting the class with the highest posterior.

#### `_likelihood(x_test, idx)`
Computes the Gaussian likelihood for a feature given the class mean and variance.

### Helper Functions

#### `plot_data(X, Y, x_col, y_col)`
Plots a scatter plot of two features, colored by class.

### Main Function

1. Loads the CSV data.
2. Separates features (X) and labels (Y).
3. Plots diameter vs. weight.
4. Splits data into train/test sets (80/20).
5. Prints dataset statistics.
6. Trains the Naive Bayes model.
7. Makes predictions on test set.
8. Calculates and prints accuracy.

## How to Run

Ensure you have Python installed with the required libraries:
```bash
pip install numpy pandas scikit-learn matplotlib
```

Run the script:
```bash
python main.py
```

## Output

The script will:
- Display a scatter plot of diameter vs. weight.
- Print dataset shapes and statistics.
- Show mean, variance, and prior for each class during training.
- Output the test accuracy percentage.

Example output:
```
Orange or Grapefruit Classifier...
------------------------------
data: (10000, 6)
n_classes: 2, [grapefruit, orange]
n_features: 5, [diameter, weight, red, green, blue]
------------------------------
X: (10000, 5)
Y: (10000,)
X_train: (8000, 5)
X_test: (2000, 5)
y_train: (8000,)
y_test: (2000,)
------------------------------
Class: grapefruit
    mean: [ 11.47914257 197.32543121 150.8668993   70.00498504  15.52467597]
    var: [  1.51600186 374.88632127 101.59494622 100.10067306  85.44330934]
    prior: 0.5015
Class: orange
    mean: [  8.47952106 152.87645687 156.78435306  81.94658977   7.11985958]
    var: [  1.5576663  342.77234575  97.38378727 101.76470001  42.01472094]
    prior: 0.4985
------------------------------
accuracy: 92.00%
```
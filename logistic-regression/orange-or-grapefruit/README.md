# Orange or Grapefruit Classifier using Logistic Regression

This project implements a Logistic Regression classifier from scratch to distinguish between oranges and grapefruits based on physical characteristics such as diameter, weight, and color components (red, green, blue).

## Overview

The `main.py` file contains a complete implementation of a Logistic Regression classifier. It loads data from `data.csv`, trains the model using gradient descent optimization, makes predictions, and evaluates accuracy. The classifier uses a sigmoid activation function to model the binary classification problem.

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

### Logistic_Regression Class

#### `__init__`
Initializes the classifier with a placeholder for weights.

#### `fit(X_train, y_train, training_steps, learning_rate)`
Trains the model by:
- Initializing weights with small random values.
- Iteratively updating weights using gradient descent.
- Computing predictions using the sigmoid function.
- Computing the gradient of the loss function.
- Updating weights in the direction of negative gradient.

#### `predict(X_test)`
Makes predictions for test data by:
- Computing the linear combination of features and weights.
- Applying the sigmoid function to get probabilities.
- Classifying as 1 if probability >= 0.5, else 0.

### Helper Functions

#### `sigmoid(x)`
Computes the sigmoid activation function: $\sigma(x) = \frac{1}{1 + e^{-x}}$

#### `dloss(X, Y, F)`
Computes the gradient of the loss function with respect to weights:
$$\frac{\partial L}{\partial w} = -\frac{1}{N} X^T (Y - F)$$

where $Y$ is the true labels and $F$ is the predicted probabilities.

### Main Function

1. Loads the CSV data.
2. Separates features (X) and labels (Y).
3. Adds a bias term (column of 1s) to the feature matrix.
4. Splits data into train/test sets (80/20).
5. Converts class labels to binary (1 for the first class, 0 for the second).
6. Trains the logistic regression model with 1000 training steps and learning rate 0.01.
7. Makes predictions on the test set.
8. Calculates and prints accuracy.

## How to Run

Ensure you have Python installed with the required libraries:
```bash
pip install numpy pandas scikit-learn
```

Run the script from the project root directory:
```bash
python logistic-regression/orange-or-grapefruit/main.py
```

## Output

The script will output the test accuracy percentage.

Example output:
```
accuracy: 92.7%
```


# Gaussian Naive Bayes from Scratch

This repository contains a Python implementation of the **Gaussian Naive Bayes** algorithm built from scratch using `NumPy` and `Pandas`. The model is applied to a clinical dataset to classify tumors as either Malignant or Benign.

## Project Overview
The goal of this project is to demonstrate the underlying mechanics of the Naive Bayes classifier without relying on high-level machine learning libraries like Scikit-Learn for the model logic. It implements the Gaussian probability density function to handle continuous numerical features.

## Dataset
The project uses the **Breast Cancer Wisconsin Diagnostic** dataset (`data.csv`).
- **Features**: 30 real-valued columns describing characteristics of cell nuclei (e.g., radius, texture, perimeter, area).
- **Target**: The `diagnosis` column, where **M** stands for Malignant and **B** stands for Benign.
- **Preprocessing**: Identifiers like `id` are removed, and the data is split into training (70%) and testing (30%) sets.

## Code Structure

### 1. Data Preparation
The data is loaded using `pandas`. We separate the features ($X$) from the labels ($y$) and perform a stratified split to ensure the model is evaluated on unseen data.
```python
X = data.drop(columns=['id', 'diagnosis'])
y = data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
## 2. The NaiveBayes Class
The core logic is encapsulated in a class that handles the statistical profiles of the training data.

### Training (`fit` method)
During the training phase, the model calculates three key components for each class:

* **Priors**: The baseline probability of each class.
* **Means**: The average value of each feature per class.
* **Stds**: The standard deviation of each feature per class.

### Likelihood Calculation
The model assumes that features follow a Normal (Gaussian) distribution. The likelihood of a feature value is calculated using:

$$P(x_i | y) = \frac{1}{\sigma_y \cdot \sqrt{2\pi}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma^2_y}\right)$$

### Prediction (`predict` method)
To classify a new sample, the model applies the **Bayes Theorem**. It calculates the "Posterior" probability for each class by multiplying the Prior by the Likelihood of all features. The class with the highest posterior probability is selected as the final prediction.

## 3. Evaluation
The model's performance is measured by comparing the predicted labels against the actual labels in the test set.

* **Current Accuracy**: ~63.16%

## Requirements
* Python 3.x
* NumPy
* Pandas
* Scikit-Learn (only for `train_test_split`)

## How to Use
1. Clone the repository.
2. Ensure `data.csv` is in the same directory.
3. Run the Jupyter Notebook `naive-bayes.ipynb`.

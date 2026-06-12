# Logistic Regression from Scratch

This project provides a clean, modular implementation of a **Logistic Regression** classifier built from scratch using Python and NumPy. It includes synthetic data generation via Gaussian distributions and uses Gradient Descent for model optimization.

## Project Structure

- **`log_reg_students.py`**: Contains the core logic for the Logistic Regression model, including the sigmoid activation function, Binary Cross-Entropy loss calculation, and the Gradient Descent training loop.
- **`datasets.py`**: A utility script to generate synthetic datasets using multivariate Gaussian distributions. It handles data shuffling and Min-Max normalization.
- **`main_students.py`**: The entry point of the application. It orchestrates the data generation, model training, evaluation, and saves the learned weights.

## Features

- **Manual Implementation**: No high-level ML libraries like Scikit-Learn are used for the model logic, making it ideal for educational purposes.
- **Gradient Descent**: Implements the standard optimization algorithm to minimize the loss function.
- **Data Preprocessing**: Includes automated normalization to ensure feature scaling, which helps the model converge faster.
- **Evaluation**: Calculates accuracy on a test set to verify model performance.

## Mathematical Overview

The model utilizes the following standard components:
1.  **Sigmoid Function**: $\sigma(z) = \frac{1}{1 + e^{-z}}$
2.  **Loss Function**: Binary Cross-Entropy (Log Loss).
3.  **Optimization**: Gradient Descent update rule: $w = w - \eta \cdot \frac{\partial L}{\partial w}$

## Getting Started

### Prerequisites

You only need Python 3 and NumPy installed:

```bash
pip install numpy
```

### Running the Project

To train the model and see the results, simply run the main script:

```bash
python main_students.py
```

The script will:
1.  Generate a synthetic dataset.
2.  Print the training loss every 500 epochs.
3.  Display the final accuracy on the test set.
4.  Save the trained weights into a file named `weights.npy`.

## License
This project is open-source and available for educational use.
import numpy as np

# Small constant to avoid numerical instability (log(0))
eps = np.finfo(float).eps


def sigmoid(x):
    """
    Compute the sigmoid activation function.
    
    Args:
        x (ndarray): Input data (z = wx + b).
    Returns:
        ndarray: Probabilities between 0 and 1.
    """
    return 1 / (1 + np.exp(-x))


def loss(y_true, y_pred):
    """
    Compute the Binary Cross-Entropy loss.
    
    Args:
        y_true (ndarray): Ground truth labels.
        y_pred (ndarray): Predicted probabilities.
    Returns:
        float: Mean cross-entropy loss.
    """
    return - np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps), axis=0)


def dloss_dw(y_true, y_pred, X):
    """
    Compute the gradient of the loss function with respect to the weights.
    
    Args:
        y_true (ndarray): Ground truth labels.
        y_pred (ndarray): Predicted probabilities.
        X (ndarray): Feature matrix.
    Returns:
        ndarray: Gradient vector.
    """
    N = X.shape[0]
    return - X.T @ (y_true - y_pred) / N


class LogisticRegression:
    """ 
    Simple Logistic Regression classifier implemented using Gradient Descent.
    """

    def __init__(self):
        self._w = None  # Weights vector

    def fit_gd(self, X, Y, n_epochs, learning_rate, verbose=False):
        """
        Train the model using Gradient Descent.
        
        Args:
            X (ndarray): Training features.
            Y (ndarray): Training labels.
            n_epochs (int): Number of iterations.
            learning_rate (float): Step size for weight updates.
            verbose (bool): If True, prints loss every 500 epochs.
        """
        n_samples, n_features = X.shape
        # Initialize weights with small random values
        self._w = np.random.randn(n_features) * 0.001

        for e in range(n_epochs):
            # Forward pass
            y_pred = sigmoid(X @ self._w)
            
            # Backward pass (gradient update)
            self._w -= learning_rate * dloss_dw(Y, y_pred, X)

            if verbose and e % 500 == 0:
                current_loss = loss(Y, y_pred)
                print(f'Epoch {e:4d}: loss={current_loss:.6f}')

    def predict(self, X):
        """
        Predict binary class labels for samples in X.
        
        Args:
            X (ndarray): Input features.
        Returns:
            ndarray: Binary predictions (0 or 1).
        """
        probabilities = sigmoid(X @ self._w)
        return (probabilities >= 0.5).astype(int)

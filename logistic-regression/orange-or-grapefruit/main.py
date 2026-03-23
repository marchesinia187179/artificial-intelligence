import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
    
def dloss(X, Y, F):
    N = X.shape[0]
    return - np.dot(X.T, (Y - F)) / N


class Logistic_Regression:

    def __init__(self):
        self._weights = None
        pass

    def fit(self, X_train, y_train, training_steps, learning_rate):
        n_samples, n_features = X_train.shape
        self._weights = np.random.randn(n_features) * 0.001

        for step in range(training_steps):
            x = np.dot(X_train, self._weights.T)
            F = sigmoid(x)
            dL = dloss(X_train, y_train, F)
            self._weights -= learning_rate * dL

    def predict(self, X_test):
        x = np.dot(X_test, self._weights.T)
        F = sigmoid(x)
        return (F >= 0.5).astype(int)


def main():
    data = pd.read_csv("logistic-regression/orange-or-grapefruit/data.csv")
    X = data.drop(columns="name")
    y = data["name"]

    # bias
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=42)

    classes = np.unique(y_train)
    y_train = np.where(y_train == classes[0], 1, 0)
    y_test = np.where(y_test == classes[0], 1, 0)

    LR = Logistic_Regression()
    LR.fit(X_train, y_train, 1000, 0.01)

    predictions = LR.predict(X_test)

    accuracy = np.mean(predictions == y_test) * 100
    print(f"accuracy: {accuracy}%")


if __name__ == '__main__':
    main()
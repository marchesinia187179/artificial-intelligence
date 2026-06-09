import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class LogReg:
    def __init__(self):
        self._weights = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _dloss(self, X, y, F):
        return - (X.T @ (y - F)) / X.shape[0]
    
    def fit(self, X, y, training_steps, learning_rate):
        n_samples, n_features = X.shape
        self._weights = np.zeros(n_features)
        
        for _ in range(training_steps):
            z = X @ self._weights
            F = self._sigmoid(z)
            grad = self._dloss(X, y, F)
            self._weights -= learning_rate * grad
    
    def predict(self, X):
        x = X @ self._weights
        F = self._sigmoid(x)
        return (F >= 0.5).astype(int)
            

def main():
    dataset = datasets.load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=42)
    model = LogReg()
    model.fit(X_train, y_train, 500, 0.1)
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")


if __name__ == "__main__":
    main()
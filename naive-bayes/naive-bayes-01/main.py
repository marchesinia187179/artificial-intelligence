from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np


class NaiveBayes():
    def __init__(self):
        self._classes = None
        self._prior = None
        self._mean = None
        self._var = None


    def fit(self, X, y):
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        n_features = X.shape[1]

        self._prior = np.zeros(n_classes)
        self._mean = np.zeros((n_classes, n_features))
        self._var = np.zeros((n_classes, n_features))

        for i, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[i, :] = X_c.mean(axis=0)
            self._var[i, :] = X_c.var(axis=0)
            self._prior[i] = X_c.shape[0] / X.shape[0]


    def predict(self, X):
        logprior = np.log(self._prior)
        log_likelihood = self._log_likelihood(X)
        posterior = logprior + log_likelihood
        return self._classes[np.argmax(posterior, axis=1)]


    def _log_likelihood(self, X):
        # Aggiungiamo un piccolo epsilon alla varianza per stabilità numerica
        var = self._var + 1e-9
        return - 0.5 * np.sum(np.log(2 * np.pi * var) + (X[:, np.newaxis, :] - self._mean) ** 2 / var, axis=2)


def main():
    dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=42)
    model = NaiveBayes()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuracy: {np.mean(y_pred == y_test):.2%}")


if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class NaiveBayes:
    """Gaussian Naive Bayes classifier (continuous features)."""

    def __init__(self):
        self._classes = None
        self._priors = None
        self._mean = None
        self._var = None

    def fit(self, X_train, y_train):
        """
        Fit the classifier.

        Args:
            X_train: array-like with shape (n_samples, n_features)
            y_train: array-like with shape (n_samples,)

        Returns:
            self
        """
        X = np.asarray(X_train, dtype=float)
        self._classes = np.unique(y_train)

        n_samples, n_features = X.shape
        self._priors = np.zeros(len(self._classes), dtype=float)
        self._mean = np.zeros((len(self._classes), n_features), dtype=float)
        self._var = np.zeros((len(self._classes), n_features), dtype=float)

        for idx, c in enumerate(self._classes):
            X_c = X[np.asarray(y_train) == c]
            self._mean[idx] = X_c.mean(axis=0)
            self._var[idx] = X_c.var(axis=0) + 1e-9
            self._priors[idx] = X_c.shape[0] / n_samples

        return self

    def predict(self, X_test):
        """
        Predict labels for X_test.

        Args:
            X_test: array-like with shape (n_samples, n_features)

        Returns:
            list: predicted class labels
        """
        X = np.asarray(X_test, dtype=float)
        predictions = []

        for x in X:
            posteriors = []
            for idx, _ in enumerate(self._classes):
                log_likelihood = np.sum(self._log_likelihood(x, idx))
                log_prior = np.log(self._priors[idx])
                posteriors.append(log_likelihood + log_prior)

            predictions.append(self._classes[np.argmax(posteriors)])

        return predictions

    def _log_likelihood(self, x, class_idx):
        """
        Compute log-likelihood of x under Gaussian for class_idx.

        Args:
            x: 1D array of shape (n_features,)
            class_idx: int

        Returns:
            1D array of log-probabilities per feature
        """
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        return -0.5 * (np.log(2 * np.pi * var) + (x - mean)**2 / var)


def plot_data(X, y, x_col, y_col):
    """
    Plot two features colored by class.

    Args:
        X: DataFrame with feature columns
        y: Series or array-like with class labels
        x_col: str name of feature for x axis
        y_col: str name of feature for y axis
    """
    df = X.copy()
    df["Class"] = y
    for label, group in df.groupby("Class"):
        plt.scatter(group[x_col], group[y_col], label=label, alpha=0.7)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.title(f"{x_col} vs {y_col}")
    plt.grid(True)
    plt.show()


def main():
    data = pd.read_csv("cat-or-dog-classifier/data.csv")
    X = data.drop(columns=["Class"])
    y = data["Class"]

    plot_data(X, y, "Weight_kg", "Height_cm")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    nb = NaiveBayes().fit(X_train, y_train)
    preds = nb.predict(X_test)
    accuracy = np.mean(np.asarray(preds) == np.asarray(y_test))

    print(accuracy)


if __name__ == "__main__":
    main()
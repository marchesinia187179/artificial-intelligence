import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

class Linear_Discriminant_Analysis:

    def __init__(self):
        self._classes = None
        self._n_classes = 0
        self._priors = None
        self._means = None
        self._cov_matrix = None

    def fit(self, X_train, y_train):
        self._classes = np.unique(y_train)
        self._n_classes = len(self._classes)
        n_samples, n_features = X_train.shape

        self._priors = np.zeros(self._n_classes, dtype=float)
        self._means = np.zeros((self._n_classes, n_features), dtype=float)
        self._cov_matrix = np.zeros((n_features, n_features), dtype=float)

        for idx, c in enumerate(self._classes):
            X_c = X_train[y_train == c]
            self._means[idx] = X_c.mean(axis=0)
            self._priors[idx] = X_c.shape[0] / n_samples
            
            z = X_c - self._means[idx]
            self._cov_matrix += z.T @ z

            print(f"Class: {c}")
            print(f"mean: {self._means[idx]}")
            print(f"prior: {self._priors[idx]}")
            print("-"*10)
        
        self._cov_matrix /= (n_samples - self._n_classes)

        print(f"Covariance matrix: {self._cov_matrix.shape}")
        print("-"*30)

    def predict(self, X_test):
        predictions = []
        for x_test in X_test:
            posteriors = []
            for idx, c in enumerate(self._classes):
                log_prior_c = np.log(self._priors[idx])
                log_likelihood = self._log_likelihood(x_test, idx)
                posteriors.append(log_likelihood + log_prior_c)
            
            predictions.append(self._classes[np.argmax(posteriors)])

        return predictions

    def _log_likelihood(self, x_test, idx):
        z = x_test - self._means[idx]
        inv_cov = np.linalg.inv(self._cov_matrix + 1e-9 * np.eye(self._cov_matrix.shape[0]))
        mahalanobis_dist = z.T @ inv_cov @ z

        log_det = np.log(np.linalg.det(2 * np.pi * self._cov_matrix))

        return -0.5 * (log_det + mahalanobis_dist)


def plot_data(X, Y, x_col, y_col):
    df = X.copy()
    df["name"] = Y
    for label, group in df.groupby("name"):
        plt.scatter(group[x_col], group[y_col], label=label, alpha=0.7)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.title(f"{x_col} vs {y_col}")
    plt.grid(True)
    plt.show()


def main():
    data = pd.read_csv("linear-discriminant-analysis/orange-or-grapefruit/data.csv")
    X = data.drop(columns=["name"])
    Y = data["name"]

    plot_data(X, Y, "diameter", "weight")

    X = np.asarray(X, dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    print("\nOrange or Grapefruit Classifier...")
    print("-"*30)
    print(f"data: {data.shape}")
    print("n_classes: 2, [grapefruit, orange]")
    print("n_features: 5, [diameter, weight, red, green, blue]")
    print("-"*30)
    print(f"X: {X.shape}")
    print(f"Y: {Y.shape}")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    print("-"*30)
    
    LDA = Linear_Discriminant_Analysis()
    LDA.fit(X_train, y_train)
    predictions = LDA.predict(X_test)
    accuracy = np.mean(y_test == predictions) * 100

    print(f"accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    main()
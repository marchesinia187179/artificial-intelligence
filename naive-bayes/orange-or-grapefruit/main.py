import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

class Naive_Bayes:

    def __init__(self):
        self._classes = None
        self._n_classes = 0
        self._means = None
        self._vars = None
        self._priors = None

    def fit(self, X_train, y_train):
        X = np.array(X_train, dtype='float')

        n_sample, n_feature = X_train.shape
        self._classes = np.unique(y_train)
        self._n_classes = len(self._classes)

        self._means = np.zeros((self._n_classes, n_feature), dtype='float')
        self._vars = np.zeros((self._n_classes, n_feature), dtype='float')
        self._priors = np.zeros(self._n_classes, dtype='float')

        for idx, c in enumerate(self._classes):
            print(f"Class: {c}")

            self._means[idx] = X_train[y_train == c].mean(axis=0)
            self._vars[idx] = X_train[y_train == c].var(axis=0)
            self._priors[idx] = X_train[y_train == c].shape[0] / n_sample

            print(f"    mean: {self._means[idx]}")
            print(f"    var: {self._vars[idx]}")
            print(f"    prior: {self._priors[idx]}")

    def predict(self, X_test):
        predictions = []
        for x_test in X_test:
            posteriors = []
            for idx, c in enumerate(self._classes):
                prior = np.log(self._priors[idx])
                likelihood = np.sum(np.log(self._likelihood(x_test, idx)))
                posteriors.append(prior + likelihood)
            
            predictions.append(self._classes[np.argmax(posteriors)])

        return predictions

    def _likelihood(self, x_test, idx):
        mean = self._means[idx]
        var = self._vars[idx] + 1e-9
        numerator = np.exp((-(x_test - mean)**2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator
        

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
    data = pd.read_csv("naive-bayes/orange-or-grapefruit/data.csv")
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
    
    NB = Naive_Bayes()
    NB.fit(X_train, y_train)
    predictions = NB.predict(X_test)
    accuracy = np.mean(y_test == predictions) * 100

    print("-"*30)
    print(f"accuracy: {accuracy}%")


if __name__ == '__main__':
    main()
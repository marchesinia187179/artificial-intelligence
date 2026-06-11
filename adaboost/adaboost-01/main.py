import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Weakclassifier:
    
    def __init__(self):
        self._dim = None
        self._threshold = None
        self._label_above_split = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        n, d = X.shape
        possible_labels = np.unique(Y)

        self._dim = np.random.choice(d)
        self._threshold = np.random.uniform(np.min(X[:, self._dim]), np.max(X[:, self._dim]))
        self._label_above_split = np.random.choice(possible_labels) 

    def predict(self, X: np.ndarray):
        y_pred = np.where(X[:, self._dim] >= self._threshold, self._label_above_split, -self._label_above_split)
        return y_pred
    

class Adaboost:
    
    def __init__(self, n_learners: int, n_max_trials: int = 200):
        self.n_learners = n_learners
        self.learners = []
        self.alphas = np.zeros(shape=n_learners)
        self.n_max_trials = n_max_trials

    def fit(self, X: np.ndarray, Y: np.ndarray):
        n, d = X.shape
        possible_labels = np.unique(Y)

        sample_weights = np.ones(shape=n) / n

        for l in range(self.n_learners):
            cur_idx = np.random.choice(n, size=(int(0.5 * n)), replace=True, p=sample_weights)

            cur_X = X[cur_idx]
            cur_Y = Y[cur_idx]

            error = 1
            n_trials = 0
            cur_wclass = None
            y_pred = None

            while error > 0.5:
                cur_wclass = Weakclassifier()
                cur_wclass.fit(cur_X, cur_Y)
                y_pred = cur_wclass.predict(cur_X)
                error = np.sum(sample_weights[cur_idx] * (y_pred != cur_Y)) / np.sum(sample_weights[cur_idx])

                n_trials += 1
                if n_trials > self.n_max_trials:
                    sample_weights = np.ones(shape=n) / n

            alpha = np.log((1 - error + 1e-10) / (error + 1e-10)) / 2

            self.alphas[l] = alpha
            self.learners.append(cur_wclass)

            full_y_pred = cur_wclass.predict(X)
            sample_weights = sample_weights * np.exp(-alpha * Y * full_y_pred)
            sample_weights = sample_weights / np.sum(sample_weights)

    def predict(self, X: np.ndarray):
        all_preds = np.array([learner.predict(X) for learner in self.learners])
        y_pred = np.dot(self.alphas, all_preds)
        return np.where(y_pred > 0, 1, -1)


def main():
    dataset = load_iris()
    X = dataset.data
    y = dataset.target
    y[y == 0] = -1
    y[y == 1] = 1
    y[y == 2] = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = Adaboost(n_learners=500)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions)}")


if __name__ == "__main__":
    main()
import numpy as np

from sklearn.datasets import make_moons, make_circles
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import Optional


class KMeans:
    def __init__(self, k, max_iters=100, initial_centers: Optional[np.ndarray] = None):
        self.k = k
        self.max_iters = max_iters
        self.initial_centers = initial_centers

    def _init_centers(self, X, use_samples=False):
        n_samples, dim = X.shape

        if use_samples:
            return X[np.random.choice(n_samples, size=self.k)]
        
        centers = np.zeros((self.k, dim))
        for i in range(dim):
            min_f, max_f = np.min(X[:, i]), np.max(X[:, i])
            centers[:, i] = np.random.uniform(low=min_f, high=max_f, size=self.k)
        return centers
    
    def compute_cost_function(self, X, centers, assignments):
        return np.sum(np.sum((X - centers[assignments]) ** 2, axis=1))
    
    def single_fit_predict(self, X):
        n_samples, dim = X.shape
        
        centers = np.array(self._init_centers(X)) \
            if self.initial_centers is None \
            else np.array(self.initial_centers)
        
        old_assignments = np.zeros(shape=n_samples)

        while True:
            new_assignments = np.argmin(np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1)
            
            # Calcola i nuovi centri, mantenendo quelli vecchi se un cluster rimane vuoto
            centers = np.array([
                X[new_assignments == i].mean(axis=0) if np.any(new_assignments == i) else centers[i]
                for i in range(self.k)
            ])
            
            if np.array_equal(old_assignments, new_assignments):
                break
            
            old_assignments = new_assignments

        return centers, new_assignments
    
    def fit_predict(self, X):
        for i in range(self.max_iters):
            centers, assignments = self.single_fit_predict(X)
            cost = self.compute_cost_function(X, centers, assignments)
            if i == 0 or cost < best_cost:
                best_cost = cost
                best_assignments = assignments
        
        return best_assignments


def main_1():
    dataset = load_iris()
    X = dataset.data
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    model = KMeans(k=2, max_iters=100)
    y_pred = model.fit_predict(X_train)
    print(accuracy_score(y_train, y_pred))


def main_2():
    X, y = make_moons(n_samples=1000, noise=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    model = KMeans(k=2, max_iters=100)
    y_pred = model.fit_predict(X_train)
    print(accuracy_score(y_train, y_pred))


def main_3():
    X, y = make_circles(n_samples=1000, noise=0.05, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    model = KMeans(k=2, max_iters=100)
    y_pred = model.fit_predict(X_train)
    print(accuracy_score(y_train, y_pred))


if __name__ == '__main__':
    main_2()
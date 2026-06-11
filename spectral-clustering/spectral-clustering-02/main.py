import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def spectral_clustering(data, n_cl, sigma=1.):
    n_samples = data.shape[0]

    # compute affinity matrix
    distances = np.linalg.norm(data[:, None] - data[None, :], axis=2)
    affinity_matrix = np.exp(- (distances ** 2) / (2 * (sigma ** 2)))
    # compute degree matrix
    degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))

    # compute symmetric normalized laplacian
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(affinity_matrix, axis=1)))
    laplacian_matrix = np.eye(n_samples) - d_inv_sqrt @ affinity_matrix @ d_inv_sqrt

    # compute eigenvalues and vectors
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)

    # ensure we are not using complex numbers
    if eigenvalues.dtype == 'complex128':
        eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real

    # sort eigenvalues and eigenvectors
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Fiedler-vector solution
    fiedler_vector = eigenvectors[:, 1]
    labels = (fiedler_vector > 0).astype(int)

    return labels

def main():
    X, y = make_moons(n_samples=300, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    labels = spectral_clustering(X_train, n_cl=2, sigma=0.15)
    y_pred = labels[np.argmin(np.linalg.norm(X_test[:, None] - X_train[None, :], axis=2), axis=1)]

    accuracy = accuracy_score(y_test, y_pred)
    accuracy = max(accuracy, 1 - accuracy)
    print(f"Accuracy: {accuracy}")

if __name__ == '__main__':
    main()
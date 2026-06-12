import csv
import numpy as np

def gaussians_dataset(n_gaussian, n_points, mus, stds):
    """
    Generates a synthetic dataset consisting of multiple Gaussian distributions.

    Args:
        n_gaussian (int): Number of Gaussian components.
        n_points (list): Number of points for each Gaussian component.
        mus (list): List of mean vectors for each Gaussian.
        stds (list): List of standard deviation vectors (diagonal of covariance).

    Returns:
        tuple: (X_train, Y_train, X_test, Y_test) split of the generated data.
    """

    assert n_gaussian == len(mus) == len(stds) == len(n_points)

    X = []
    Y = []
    
    # Generate data for each Gaussian component
    for i in range(0, n_gaussian):
        mu = mus[i]
        std = stds[i]
        n_pt = n_points[i]

        cov = np.diag(std)
        
        # Sample points from multivariate normal distribution
        X.append(np.random.multivariate_normal(mu, cov, size=2*n_pt))
        Y.append(np.ones(shape=2*n_pt) * i)

    # Combine all components
    X = np.concatenate(X, axis=0)
    
    # Initial Min-Max normalization
    X -= np.min(X, axis=0)
    X /= np.max(X, axis=0)

    Y = np.concatenate(Y, axis=0)
    tot = np.concatenate((X, np.reshape(Y, shape=(-1, 1))), axis=-1)

    np.random.seed(30101990)
    np.random.shuffle(tot)
    X = tot[:, :-1]
    Y = tot[:, -1]

    # Final Min-Max normalization after shuffling
    X -= np.min(X, axis=0)
    X /= np.max(X, axis=0)

    # Split into training and testing sets (50/50)
    n_train_samples = X.shape[0]//2
    
    X_train = X[:n_train_samples]
    Y_train = Y[:n_train_samples]

    X_test = X[n_train_samples:]
    Y_test = Y[n_train_samples:]

    return X_train, Y_train, X_test, Y_test

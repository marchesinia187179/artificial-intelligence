import csv
import numpy as np


def gaussians_dataset(n_gaussian, n_points, mus, stds):
    """
    Provides a dataset made by several gaussians.

    Parameters
    ----------
    n_gaussian : int
        The number of desired gaussian components.
    n_points : list
        A list of cardinality of points (one for each gaussian).
    mus : list
        A list of means (one for each gaussian, e.g. [[1, 1], [3, 1]).
    stds : list
        A list of stds (one for each gaussian, e.g. [[1, 1], [2, 2]).

    Returns
    -------
    tuple
        a tuple like:
            data ndarray shape: (n_samples, dims).
            class ndarray shape: (n_samples,).
    """

    assert n_gaussian == len(mus) == len(stds) == len(n_points)

    X = []
    Y = []
    for i in range(0, n_gaussian):

        mu = mus[i]
        std = stds[i]
        n_pt = n_points[i]

        cov = np.diag(std)

        X.append(np.random.multivariate_normal(mu, cov, size=2*n_pt))
        Y.append(np.ones(shape=2*n_pt) * i)

    X = np.concatenate(X, axis=0)
    X -= np.min(X, axis=0)
    X /= np.max(X, axis=0)

    Y = np.concatenate(Y, axis=0)

    tot = np.concatenate((X, np.reshape(Y, shape=(-1, 1))), axis=-1)

    np.random.seed(30101990)
    np.random.shuffle(tot)
    X = tot[:, :-1]
    Y = tot[:, -1]

    # normalize X
    X -= np.min(X, axis=0)
    X /= np.max(X, axis=0)

    n_train_samples = X.shape[0]//2

    X_train = X[:n_train_samples]
    Y_train = Y[:n_train_samples]

    X_test = X[n_train_samples:]
    Y_test = Y[n_train_samples:]

    return X_train, Y_train, X_test, Y_test


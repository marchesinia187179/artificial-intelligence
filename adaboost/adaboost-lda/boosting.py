import matplotlib.pyplot as plt
import numpy as np

from utils import cmap

class LDA:

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

        self._cov_matrix /= (n_samples - self._n_classes)

    def predict(self, X_test):
        predictions = []
        for x_test in X_test:
            posteriors = []
            for idx, c in enumerate(self._classes):
                log_prior_c = np.log(self._priors[idx])
                log_likelihood = self._log_likelihood(x_test, idx)
                posteriors.append(log_likelihood + log_prior_c)

            predictions.append(self._classes[np.argmax(posteriors)])

        return np.array(predictions)

    def _log_likelihood(self, x_test, idx):
        z = x_test - self._means[idx]
        cov = self._cov_matrix + 1e-9 * np.eye(self._cov_matrix.shape[0])
        inv_cov = np.linalg.inv(cov)
        mahalanobis_dist = z.T @ inv_cov @ z

        sign, log_det = np.linalg.slogdet(2 * np.pi * cov)
        if sign <= 0:
            log_det = np.log(np.linalg.det(2 * np.pi * cov) + 1e-18)

        return -0.5 * (log_det + mahalanobis_dist)
    

class AdaBoostClassifier:
    """
    Function that models a Adaboost classifier
    """

    def __init__(self, n_learners: int, n_max_trials: int = 200):
        """
        Model constructor

        Parameters
        ----------
        n_learners: int
            number of LDA classifiers.
        """

        # initialize a few stuff
        self.n_learners = n_learners
        self.learners = []
        self.alphas = np.zeros(shape=n_learners)
        self.n_max_trials = n_max_trials

    def fit(self, X: np.ndarray, Y: np.ndarray, verbose: bool = False):
        """
        Trains the model.

        Parameters
        ----------
        X: ndarray
            features having shape (n_samples, dim).
        Y: ndarray
            class labels having shape (n_samples,).
        verbose: bool
            whether or not to visualize the learning process.
            Default is False
        """

        n, d = X.shape
        possible_labels = np.unique(Y)

        if d != 2:
            verbose = False  # only plot learning if 2 dimensional

        assert possible_labels.size == 2, 'Error: data is not binary'

        """ initialize the sample weights as equally probable """
        sample_weights = np.ones(shape=n) / n

        for l in range(self.n_learners):

            print(l)

            """ choose the indexes of 'difficult' samples. See np.random.choice
                https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.choice.html
                Pay attention to p, which indicates the probabilities that will be used during sampling."""
            cur_idx = np.random.choice(n, size=(int(0.5 * n)), replace=True, p=sample_weights)

            # extract 'difficult' samples
            cur_X = X[cur_idx]
            cur_Y = Y[cur_idx]

              # search for a weak classifier
            error = 1
            n_trials = 0
            cur_wclass = None
            y_pred = None

            # search for a 'good' weak classifier
            while error > 0.5:

                cur_wclass = LDA()

                """ train the weak classifier on the dataset subsample """
                cur_wclass.fit(cur_X, cur_Y)

                """ compute the predictions on the dataset subsample """
                y_pred = cur_wclass.predict(cur_X)

                """ according to the predicitons and labels, compute the error
                    made by the current classifier (namely, cur_wclass) """
                error = np.sum(sample_weights[cur_idx] * (y_pred != cur_Y)) / np.sum(sample_weights[cur_idx])

                n_trials += 1
                if n_trials > self.n_max_trials:
                    # initialize the sample weights again
                    sample_weights = np.ones(shape=n) / n


            """ compute the efficiency of the weak classifier """
            error = np.clip(error, 1e-16, 1 - 1e-16)
            alpha = np.log((1 - error) / error) / 2

            self.alphas[l] = alpha

            # append the learned weak classifier to the chain
            self.learners.append(cur_wclass)

            """ based on the right and wrong predictions, update sample_weights"""
            full_y_pred = cur_wclass.predict(X)
            sample_weights = sample_weights * np.exp(-alpha * Y * full_y_pred)
            denom = np.sum(sample_weights)
            if denom == 0 or np.isnan(denom):
                sample_weights = np.ones(shape=n) / n
            else:
                sample_weights = sample_weights / denom

            if verbose:
                self._plot(cur_X, y_pred, sample_weights[cur_idx],
                           self.learners[-1], l)


    def predict(self, X: np.ndarray):
        """
        Function to perform predictions over a set of samples.

        Parameters
        ----------
        X: ndarray
            examples to predict. shape: (n_examples, d).

        Returns
        -------
        ndarray
            labels for each examples. shape: (n_examples,).

        """
        num_samples = X.shape[0]

        """ fill y_pred with the predictions """
        y_pred = np.zeros(shape=num_samples)

        for i in range(num_samples):
            for j in range(self.n_learners):
                y_pred[i] += self.alphas[j] * self.learners[j].predict(X[i:i+1])[0]

        return np.where(y_pred > 0, 1, -1)


    def _plot(self, X: np.ndarray, y_pred: np.ndarray, weights: np.ndarray,
              learner: LDA, iteration: int):

        # plot decision boundary for the current weak learner
        plt.clf()

        M1, m1 = np.max(X[:, 1]), np.min(X[:, 1])
        M0, m0 = np.max(X[:, 0]), np.min(X[:, 0])

        xx, yy = np.meshgrid(
            np.linspace(m0, M0, 200),
            np.linspace(m1, M1, 200)
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        try:
            Z = learner.predict(grid)
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, alpha=0.25, cmap=cmap)
        except Exception:
            pass

        plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=weights * 50000,
                    cmap=cmap, edgecolors='k')
        plt.xlim([m0, M0])
        plt.ylim([m1, M1])
        plt.xticks([])
        plt.yticks([])
        plt.title('Iteration: {:04d}'.format(iteration))
        plt.waitforbuttonpress(timeout=0.1)

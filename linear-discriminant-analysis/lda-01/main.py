import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class LDA:
    def __init__(self):
        self.mean = None
        self.covar = None
        self.classes = None
        self.prior = None


    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape

        self.mean = np.zeros((len(self.classes), n_features))
        self.covar = np.zeros((n_features, n_features))
        self.prior = np.zeros(len(self.classes))

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[i, :] = X_c.mean(axis=0)
            self.prior[i] = X_c.shape[0] / n_samples

            diff = X_c - self.mean[i, :]
            self.covar += diff.T @ diff

        self.covar /= n_samples
    

    def predict(self, X):
        logprior = np.log(self.prior)
        posterior = self.loglikelihood(X) + logprior
        return self.classes[np.argmax(posterior, axis=1)]
        

    def loglikelihood(self, X):
        inv_cov = np.linalg.inv(self.covar)
        n_classes = len(self.classes)
        log_lh = np.zeros((X.shape[0], n_classes))

        for i in range(n_classes):
            diff = X - self.mean[i, :]
            log_lh[:, i] = -0.5 * np.sum((diff @ inv_cov) * diff, axis=1)
        
        return log_lh
    

def main():
    dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=42)
    lda = LDA()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    

if __name__ == "__main__":
    main()
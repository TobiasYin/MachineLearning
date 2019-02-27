import numpy as np


class PCA:
    def __init__(self, n_components: int):
        assert n_components >= 1
        self.n_components = n_components
        self.components_ = None

    @staticmethod
    def _f(w, X):
        return np.sum(X.dot(w) ** 2) / len(X)

    @staticmethod
    def _df(w, X):
        return 2 * (X.T.dot(X.dot(w))) / len(X)

    @staticmethod
    def _first_component(X, initial_w, eta, n_iters=1e4, epsilon=1e-8):
        w = PCA._direction(initial_w)
        for i in range(int(n_iters)):
            gradient = PCA._df(w, X)
            last_w = w
            w = w + eta * gradient
            w = PCA._direction(w)
            if (abs(PCA._f(w, X) - PCA._f(last_w, X)) < epsilon):
                break
        return w

    @staticmethod
    def _direction(w):
        return w / np.linalg.norm(w)

    @staticmethod
    def _demean(X):
        return X - np.mean(X, axis=0)

    def fit(self, X: np.ndarray, eta=0.01, n_iters=1e4, epsilon=1e-8):
        assert self.n_components <= X.shape[1]
        X_pca = self._demean(X.copy())
        res = []
        for i in range(self.n_components):
            initial_w = np.random.random(X_pca.shape[1])
            w = self._first_component(X_pca, initial_w, eta, n_iters, epsilon)
            res.append(w)
            X_pca -= X_pca.dot(w).reshape(-1, 1) * w
        self.components_ = np.array(res)

    def transform(self, X: np.ndarray):
        assert self.components_
        assert X.shape[1] == self.components_.shape[1]
        return X.dot(self.components_.T)

    def inverse_tansform(self, X: np.ndarray):
        assert X.shape[1] == self.components_.shape[0]
        return X.dot(self.components_)

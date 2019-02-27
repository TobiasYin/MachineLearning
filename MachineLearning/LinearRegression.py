import numpy as np
from .metrics import r_squared
from sklearn.preprocessing import StandardScaler


class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.interception_ = None
        self._theta = None
        self._s = None

    def fit_normal(self, X_train: np.ndarray, y_train: np.ndarray):
        assert X_train.shape[0] == y_train.shape[0]
        self._s = StandardScaler()
        self._s.fit(X_train)
        X_train = self._s.transform(X_train)
        X_b = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.coef_ = self._theta[1:]
        self.interception_ = self._theta[0]
        return self

    def fit_gd(self, X_train: np.ndarray, y_train: np.ndarray, eta=0.1, n_iters=1e4):
        assert X_train.shape[0] == y_train.shape[0]

        def dJ(theta, X_b, y):
            return (X_b.dot(theta) - y).T.dot(X_b) * 2 / len(X_b)

        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except:
                return float("inf")

        self._s = StandardScaler()
        self._s.fit(X_train)
        X_train_s = self._s.transform(X_train)

        X_b = np.hstack([np.ones((len(X_train_s), 1)), X_train_s])
        theta = np.zeros(X_b.shape[1])
        for i in range(int(n_iters)):
            gradient = dJ(theta, X_b, y_train)
            last_theta = theta
            theta = theta - eta * gradient
            if abs(J(theta, X_b, y_train) - J(last_theta, X_b, y_train)) < 1e-8:
                break
        self._theta = theta
        self.coef_ = self._theta[1:]
        self.interception_ = self._theta[0]
        return self

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, n_iters=5, t0=5, t1=50):
        assert X_train.shape[0] == y_train.shape[0]

        def dJ(theta, X_b_i, y_i):
            return X_b_i * (X_b_i.dot(theta) - y_i) * 2.

        def learning_rate(t):
            return t0 / (t + t1)

        self._s = StandardScaler()
        self._s.fit(X_train)
        X_train_s = self._s.transform(X_train)

        X_b = np.hstack([np.ones((len(X_train_s), 1)), X_train_s])
        theta = np.zeros(X_b.shape[1])
        for i in range(int(n_iters)):
            index = np.random.permutation(len(X_b))
            for j in range(len(index)):
                gradient = dJ(theta, X_b[index[j]], y_train[index[j]])
                theta = theta - learning_rate(i*len(X_b) + j) * gradient
        self._theta = theta
        self.coef_ = self._theta[1:]
        self.interception_ = self._theta[0]
        return self

    def predict(self, X_predict: np.ndarray):
        assert self._theta is not None and self.interception_ is not None and self.coef_ is not None
        assert self.coef_.shape[0] == X_predict.shape[1]
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        X_test = self._s.transform(X_test)
        y_predict = self.predict(X_test)
        return r_squared(y_test, y_predict)

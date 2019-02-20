import numpy as np
from .metrics import r_squared


class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.interception_ = None
        self._theta = None

    def fit_normal(self, X_train: np.ndarray, y_train: np.ndarray):
        assert X_train.shape[0] == y_train.shape[0]
        X_b = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.coef_ = self._theta[1:]
        self.interception_ = self._theta[0]
        return self

    def predict(self, X_predict: np.ndarray):
        assert self._theta is not None and self.interception_ is not None and self.coef_ is not None
        assert self.coef_.shape[0] == X_predict.shape[1]
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return r_squared(y_test, y_predict)

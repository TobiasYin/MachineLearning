import numpy as np
from .metrics import access_score
from sklearn.preprocessing import StandardScaler


class LogisticRegression:
    def __init__(self):
        self.coef_ = None
        self.interception_ = None
        self._theta = None
        # self._s = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, eta=0.1, n_iters=1e4):
        assert X_train.shape[0] == y_train.shape[0]

        # self._s = StandardScaler()
        # self._s.fit(X_train)
        # X_train_s = self._s.transform(X_train)

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        theta = np.zeros(X_b.shape[1])
        for i in range(int(n_iters)):
            gradient = self._dJ(theta, X_b, y_train)
            last_theta = theta
            theta = theta - eta * gradient
            if abs(self._J(theta, X_b, y_train) - self._J(last_theta, X_b, y_train)) < 1e-8:
                break
        self._theta = theta
        self.coef_ = self._theta[1:]
        self.interception_ = self._theta[0]
        return self

    @staticmethod
    def _sigmoid(t: np.ndarray):
        return 1 / (1 + np.exp(-t))

    @staticmethod
    def _dJ(theta, X_b, y):
        return X_b.T.dot(LogisticRegression._sigmoid(X_b.dot(theta))-y) / len(X_b)

    @staticmethod
    def _J(theta, X_b, y):
        y_hat = LogisticRegression._sigmoid(X_b.dot(theta))
        try:
            return - np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / len(y)
        except:
            return float("inf")

    def predict_proba(self, X_predict):
        assert self._theta is not None and self.interception_ is not None and self.coef_ is not None
        assert self.coef_.shape[0] == X_predict.shape[1]
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return self._sigmoid(X_b.dot(self._theta))

    def predict(self, X_predict: np.ndarray):
        assert self._theta is not None and self.interception_ is not None and self.coef_ is not None
        assert self.coef_.shape[0] == X_predict.shape[1]
        res = self.predict_proba(X_predict)
        return np.array(res >= 0.5, dtype=int)

    def score(self, X_test, y_test):
        # X_test = self._s.transform(X_test)
        y_predict = self.predict(X_test)
        return access_score(y_test, y_predict)

import numpy as np
from .metrics import r_squared


class SimpleLinearRegression2:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        assert x_train.ndim == 1 and y_train.ndim == 1
        assert len(x_train) == len(y_train)
        x_mean = x_train.mean()
        y_mean = y_train.mean()
        self.a_ = (x_train - x_mean).dot(y_train - y_mean) / (x_train - x_mean).dot(x_train - x_mean)
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self, x: int or float or np.ndarray) -> np.ndarray:
        assert self.a_ is not None and self.b_ is not None
        if isinstance(x, int) or isinstance(x, float):
            return np.array([self.a_ * x + self.b_])
        else:
            assert x.ndim == 1
            return self.a_ * x + self.b_

    def score(self, x_test, y_test):
        assert self.a_ is not None and self.b_ is not None
        y_predict = self.predict(x_test)
        return r_squared(y_test, y_predict)

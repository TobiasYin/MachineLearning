import numpy as np


class SimpleLinearRegression1:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        assert x_train.ndim == 1 and y_train.ndim == 1
        assert len(x_train) == len(y_train)
        x_mean = x_train.mean()
        y_mean = y_train.mean()
        self.a_ = np.sum([(i - x_mean) * (j - y_mean) for i, j in zip(x_train, y_train)]) / np.sum(
            [(i - x_mean) ** 2 for i in x_train])
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self, x: int or float or np.ndarray) -> np.ndarray:
        assert self.a_ is not None and self.b_ is not None
        if isinstance(x, int) or isinstance(x, float):
            return np.array([self.a_ * x + self.b_])
        else:
            assert x.ndim == 1
            return self.a_ * x + self.b_

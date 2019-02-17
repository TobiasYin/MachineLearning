import numpy as np


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, train_x: np.ndarray):
        assert train_x.ndim == 2
        self.mean_ = np.array([np.mean(train_x[:, i]) for i in range(train_x.shape[1])])
        self.std_ = np.array([np.std(train_x[:, i]) for i in range(train_x.shape[1])])
        return self

    def transform(self, x: np.ndarray):
        assert x.ndim == 2
        assert self.mean_ is not None and self.std_ is not None
        assert x.shape[1] == len(self.mean_)
        res = np.empty(shape=x.shape, dtype=float)
        for col in range(x.shape[1]):
            res[:, col] = (x[:, col] - self.mean_[col]) / self.std_[col]
        return res

import numpy as np
from math import sqrt
from collections import Counter


class KNNClassifier:
    def __init__(self, k: int):
        assert k >= 1
        self.k = k
        self.train_x = None
        self.train_y = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        assert self.k <= X_train.shape[0]
        assert X_train.shape[0] == y_train.shape[0]
        self.train_x = X_train
        self.train_y = y_train
        return self

    def __knn_classify(self, x: np.ndarray) -> int:
        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self.train_x]
        res = np.argsort(distances)
        topK_y = [self.train_y[i] for i in res[:self.k]]
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self.train_x is not None and self.train_y is not None
        assert self.train_x.shape[1] == x.shape[0] or self.train_x.shape[1] == x.shape[1]
        if len(x.shape) == 2:
            res = np.array([self.__knn_classify(i) for i in x])
        else:
            res = np.array([self.__knn_classify(x)])
        return res

    def score(self, test_x: np.ndarray, test_y: np.ndarray):
        res = self.predict(test_x)
        return np.sum(test_y == res) / len(test_y)

import numpy as np


def train_test_split(X: np.ndarray, y: np.ndarray, ratio=0.2, seed=None):
    assert X.shape[0] == y.shape[0]
    assert 0.0 <= ratio <= 1.0
    if seed:
        np.random.seed(seed)
    shuffle_indexes = np.random.permutation(len(X))
    size = len(X) - int(ratio * len(X))
    train_indexes, test_indexes = shuffle_indexes[:size], shuffle_indexes[size:]
    return X[train_indexes], X[test_indexes], y[train_indexes], y[test_indexes]

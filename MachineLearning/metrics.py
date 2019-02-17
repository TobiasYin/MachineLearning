import numpy as np


def access_score(y_true: np.ndarray, y_predict: np.ndarray):
    assert y_true.shape[0] == y_predict.shape[0]
    return np.sum(y_true == y_predict) / len(y_true)

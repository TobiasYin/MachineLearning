import numpy as np


def access_score(y_true: np.ndarray, y_predict: np.ndarray):
    assert y_true.shape[0] == y_predict.shape[0]
    return np.sum(y_true == y_predict) / len(y_true)


def mean_squared_error(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return (y_predict - y_true).dot(y_predict - y_true) / len(y_true)


def root_mean_squared_error(y_true, y_predict):
    return np.sqrt(mean_squared_error(y_true, y_predict))


def mean_absolute_error(y_true, y_predict):
    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)


def r_squared(y_true, y_predict):
    return 1 - np.sum((y_predict - y_true) ** 2) / np.sum((y_true.mean() - y_true) ** 2)

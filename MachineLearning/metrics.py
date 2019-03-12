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


def TP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 1))


def TN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 0))


def FP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 1))


def FN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 0))


def confusion_matrix(y_true, y_predict):
    return np.array([[TN(y_true, y_predict), FP(y_true, y_predict)], [FN(y_true, y_predict), TP(y_true, y_predict)]])


def precision_score(y_true, y_predict):
    try:
        return TP(y_true, y_predict) / (TP(y_true, y_predict) + FP(y_true, y_predict))
    except:
        return 0


def recall_score(y_true, y_predict):
    try:
        return TP(y_true, y_predict) / (TP(y_true, y_predict) + FN(y_true, y_predict))
    except:
        return 0


def f1_score(y_true, y_predict):
    preci = precision_score(y_true, y_predict)
    recall = recall_score(y_true, y_predict)
    try:
        return 2 * preci * recall / (preci + recall)
    except:
        return 0


def TPR(y_true, y_predict):
    return recall_score(y_true, y_predict)


def FPR(y_true, y_predict):
    try:
        return FP(y_true, y_predict) / (TN(y_true, y_predict) + FP(y_true, y_predict))
    except:
        return 0

import numpy as np


def accuracy(y, y_pred):
    return np.asarray((np.sum(y == y_pred)) / len(y))


def r2(y, y_pred):  # r2 score == fraction of variance unexplained
    y_mean = y.mean()
    ss_tot = ((y - y_mean) ** 2).sum()
    ss_res = ((y - y_pred) ** 2).sum()
    return np.asarray(1 - (ss_res / ss_tot))


def r2adj(y, y_pred, p):
    # adjusted r2
    y_mean = y.mean()
    ss_tot = ((y - y_mean) ** 2).sum()
    ss_res = ((y - y_pred) ** 2).sum()
    n = len(y)
    upper = ss_res / (n - p)
    lower = ss_tot / (n - 1)

    return np.asarray(1 - (upper / lower))


def mse(y, y_hat):  # mean square error
    return np.asarray(np.mean((y - y_hat) ** 2))


def corrcoef(y, y_pred):  # correlation coefficient
    return np.corrcoef(y, y_pred)[0, 1]


def gcv(y, y_pred, edf):
    rss = y - y_pred
    lower = 1 - edf / len(y)
    return np.mean((rss / lower) ** 2)

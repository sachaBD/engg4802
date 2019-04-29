import numpy as np
import pandas as pd

def backtest(data: np.ndarray, start: float, test_step_size: int):
    """
    Splits a dataset into a train and test set using backtesting with a set size of each test set and a % of data heldout to
    always be in the train set.

    data -> A numpy array of pandas dataframe of the data to split
    start -> a value in [0, 1] of the % of data to always be in the train set
    test_step_size -> The size of each test split
    """
    step = test_step_size
    start = int(start * len(data))
    splits = np.arange(start, len(data), step)

    train, test = [], []

    for test_start, test_end in zip(splits[:-1], splits[1:]):
        train += [data[:test_start]]
        test += [data[test_start:test_end]]

    return train, test


def xy_backtest(X, y, start, step_size):
    """
    Splits two dataset's into a train and test sets using backtesting with a set size of each test set and a % of data heldout to
    always be in the train set.

    :param X:
    :param y:
    :param start:
    :param step_size:
    :return:
    """
    step = step_size
    start = int(start * len(X))
    splits = np.arange(start, len(X), step)

    results = []

    for test_start, test_end in zip(splits[:-1], splits[1:]):
        results.append((X[:test_start], X[test_start:test_end], y[:test_start], y[test_start:test_end]))

    return results
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


"""
Splits time series data into a train and test set.
"""
def split_data(data : pd.DataFrame, labels : pd.DataFrame, splits = 10) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    # backtester = TimeSeriesSplit(n_splits = splits)
    # splits = backtester.split(data)

    # Calculate the split indexes
    splitsIndexes = [i * len(data) // (splits + 1) for i in range(1, splits + 1)]
    print(splitsIndexes)

    train_test_split = []

    for train in splitsIndexes:
        X_train = data.iloc[:train]
        Y_train = labels.iloc[:train]

        X_test = data.iloc[train:]
        Y_test = labels.iloc[train:]

        train_test_split.append((X_train, Y_train, X_test, Y_test))

    return train_test_split


def split(data, splits=10, offset=0, ratio=0.2):
    result = []

    for i in range(0, splits):
        start = int(offset + (i / splits) * (len(data) - offset))
        end = int(offset + ((i + 1) / splits) * (len(data) - offset))
        print(start, end)

        tempData = data[start: end]
        train, test = tempData[0: int((1 - ratio) * len(tempData))], tempData[int((1 - ratio) * len(tempData)):]

        result.append((train, test))

    return result


"""
Seperate time series data into an input and output for training and testin.
"""
def timeseries_to_supervised(dataset : pd.DataFrame, offset = 1):
	columns = [dataset.shift(i) for i in range(1, offset + 1)]

	columns.append(dataset)

	allData = np.concat(columns, axis=1)
	allData.fillna(0, inplace=True)

	return allData


"""
Generates labels of the next n samples as labels for training a time series predictor.
"""
def generate_labels():
    pass


from typing import Tuple

import numpy as np
import pandas as pd


"""
Create a set of time series features to train a tree based model for a single dimensional time series dataset.

Return:
[features, labels]
"""
def create_features(data: np.ndarray, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    # Globals
    oneHour = 60
    oneDay = 24 * oneHour
    oneWeek = oneDay * 7
    oneMonth = oneDay * 30

    featureNames = ['last_value', '2nd_last_value', '3rd_last_value']
    periods = [5, 10, 20, 30, 40, 50, 60, 100, 200, 350, 500, oneDay, 2 * oneDay, 5 * oneDay, oneWeek, oneMonth]
    featureTypes = ['mean', 'stddiv', 'median', 'min', 'max']  # 'autocorrelation'

    offset = oneMonth

    for period in periods:
        for feat in featureTypes:
            featureNames.append(str(period) + '_' + feat)

    featureExtractedData = []
    labelsData  = []

    for i ,point in enumerate(data[offset : -horizon]):

        if i % 1000 == 0:
            print(str(i / 1000), '/', len(data) / 1000)

        # The labels are the next n values
        labelsData.append(np.array(data[offset + i + 1: offset + i + horizon + 1].reshape(1, horizon)[0]))

        # Generate the features
        values = np.zeros(len(featureNames))

        values[0] = data[offset + i - 1]
        values[1] = data[offset + i - 2]
        values[2] = data[offset + i - 3]

        for j, period in enumerate(periods):
            periodData = data[offset + i - period: offset + i]
            values[3 + j * len(featureTypes) + 0] = np.mean(periodData)
            values[3 + j * len(featureTypes) + 1] = np.std(periodData)
            values[3 + j * len(featureTypes) + 2] = np.median(periodData)
            values[3 + j * len(featureTypes) + 3] = np.min(periodData)
            values[3 + j * len(featureTypes) + 4] = np.max(periodData)

        featureExtractedData.append(values)

    # Convert to numpy arrays
    featureExtractedData = np.array(featureExtractedData)
    labels = np.array(labelsData)

    return pd.DataFrame(featureExtractedData, columns = featureNames), pd.DataFrame(labels, columns = [str(i) for i in range(1, horizon + 1)])

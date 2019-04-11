import numpy as np
import pandas as pd
from copy import deepcopy


class MultiForecaster:

    """
    Forecast multiple periods ahead using a multiforecaster approach training a new model for each point in the horizon.
    The estimator given must have the .fit and .predict methods
    """
    def __init__(self, estimator, horizon : int):
        self.horizon   = horizon

        self.estimators = []

        for i in range(0, horizon):
            self.estimators.append(deepcopy(estimator))


    """
    Train the estimators based on the train data set.
    
    Inputs:
    dataset: A list of features of dimensions (n, m) where n is the number of training points and m is any integer.
    labels: A list of the labels for each datapoint. This has dimensions (n, horizon).
    """
    def fit(self, dataset : pd.DataFrame, labels : pd.DataFrame) -> None:
        for i, estimator in enumerate(self.estimators):
            estimator.fit(dataset, labels.iloc[:, i])

    """
    Get the predicts for the full horizon for each point in the dataset.
    """

    def predict(self, dataset: pd.DataFrame) -> np.ndarray:
        result = []

        for pointIndex in range(0, len(dataset)):
            tempResult = np.zeros(len(self.estimators))

            for i, estimator in enumerate(self.estimators):
                point = dataset.iloc[pointIndex: pointIndex + 1]

                tempResult[i] = estimator.predict(point)

            result.append(tempResult)

        return np.array(result)
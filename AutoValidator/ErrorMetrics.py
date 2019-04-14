import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import *

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri, numpy2ri

# Setup to parse to R
pandas2ri.activate()

# Import R libraries
ts         = ro.r('ts')
forecast   = importr('forecast')
thetaModel = importr('forecTheta')
smooth     = importr('smooth')


def calculate_all_errors(training, prediction, actual, horizon):
    errors = {}
    
    # Calculate the error given each metric
    
    # pass the data to R
    rPred    = ts(prediction, frequency = 1 )
    rActual  = ts(actual,     frequency = 1 ) 
    
    # Calculate the errors
    errors['MAE']   = #thetaModel.errorMetric(obs = rActual, forec = rPred, type = "AE", statistic = "M")[0]
    errors['RMSE']  = np.sqrt(mean_squared_error(actual, prediction))
    errors['MAPE']  = thetaModel.errorMetric(obs = rActual, forec = rPred, type = "APE", statistic = "M")[0]
    errors['sMAPE'] = thetaModel.errorMetric(obs = rActual, forec = rPred, type = "sAPE", statistic = "M")[0]
    errors['MASE']  = calculate_MASE(training, prediction, actual) 
    errors['MASE1'] = smooth.MASE(rActual, rPred, np.mean(training), digits = 5)[0]
    errors['MASE2'] = MASE(training, actual, prediction)
    errors['MEAN_MASE']: calculate_MASE(training, prediction, actual)
    errors['RW_MASE'] : calculate_rw_MASE(training, prediction, actual, horizon)
    
    return errors


def MASE(training_series, testing_series, prediction_series):
    """
    Computes the MEAN-ABSOLUTE SCALED ERROR forcast error for univariate time series prediction.
    
    See "Another look at measures of forecast accuracy", Rob J Hyndman
    
    parameters:
        training_series: the series used to train the model, 1d numpy array
        testing_series: the test series to predict, 1d numpy array or float
        prediction_series: the prediction of testing_series, 1d numpy array (same size as testing_series) or float
        absolute: "squares" to use sum of squares and root the result, "absolute" to use absolute values.
    
    """
    n = training_series.shape[0]
    d = np.abs(  np.diff(training_series) ).sum()/(n-1)
    
    errors = np.abs(testing_series - prediction_series )
    return errors.mean()/d

def my_MASE(training, test, pred):
    n = len(training)
    
    numerator = np.sum(np.abs(test - pred))
    denominator = (n / (n - 1)) * np.sum(np.abs(train[1:] - train[:-1]))
    print(numerator, denominator)
    
    q = numerator / denominator
    
    return np.mean(np.abs(q))
    

class RandomWalk():
    
    def __init__(self, horizon):
        self.horizon = horizon
    
    def fit(self, train, labels):
        self.train = labels
    
    def predict(self, data):
        comb_data = np.hstack([self.train, data])
        
        results = np.zeros((len(data), self.horizon))
        
        std = np.std(comb_data)
        
        results[:, 0] = data[:]
        
        noise = np.random.normal(0, std, (len(data), self.horizon))
        results += noise        
        
        return results.cumsum(axis=1)
    
    def __repr__(self):
        return 'Random Walk'

    
def calculate_rw_MASE(training, test, pred, horizon):
    naive_estimator = RandomWalk(horizon)
    naive_estimator.fit(training, training)
    naive_estimate = naive_estimator.predict(test[np.arange(0, len(test), horizon)]).flatten()
    
    return np.mean(np.abs(test - pred)) / np.mean(np.abs(test - naive_estimate))
    

"""
Mean absolute error.
"""
def calculate_MAE(prediction : np.ndarray, actual : np.ndarray) -> float:
#     print("actual", actual)
#     print("Prediction", prediction)
    return np.mean(np.abs(actual - prediction))



"""
Mean absolute scaled error.  
"""
def calculate_MASE(trainingData : np.ndarray, prediction : np.ndarray, actual : np.ndarray) -> float:
    assert len(prediction) == len(actual)

    # TODO: Validate this

    # Train a naive estimator as the mean estimator
    naive_estimate = np.repeat(np.mean(trainingData), len(prediction))

    # calculate errors
    naive_error     = calculate_MAE(naive_estimate, actual)
    estimator_error = calculate_MAE(prediction, actual)

    return estimator_error / naive_error


"""
Symmetric mean absolute percentage error. 

Faults: Over penalises large positive errors more then negative errors.
"""
def calculate_sMAPE(prediction : np.ndarray, actual : np.ndarray) -> float:
    assert len(prediction) == len(actual)

    return 1 / len(prediction) * np.sum(2 * np.abs(prediction - actual) / (np.abs(actual) + np.abs(prediction)))
#     return 2 / len(prediction) * np.sum( np.abs(actual - prediction) / ( np.abs(actual) + np.abs(prediction)) )


def calculate_MAPE(pred, actual):
    return np.mean(np.abs((actual - pred) / actual))
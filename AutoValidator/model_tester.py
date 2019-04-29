import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy import stats

import plotly_express as px

from sklearn.metrics import mean_squared_error
from AutoValidator.ErrorMetrics import *

import sys
sys.path.append('../utils')
from utils.progress import ProgressBar

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
    step = step_size
    start = int(start * len(X))
    splits = np.arange(start, len(X), step)
    
    results = []
    
    for test_start, test_end in zip(splits[:-1], splits[1:]):
        results.append((X[:test_start], X[test_start:test_end], y[:test_start], y[test_start:test_end]))
        
    return results

from AutoValidator.ErrorMetrics import __error_metrics__, calculate_all_errors
from utils.progress import ProgressBar

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
    splits = np.arange(start, len(data) + step, step, dtype=np.int)
    
    train, test = [], []
    
    for test_start, test_end in zip(splits[:-1], splits[1:]):
        train += [data[:test_start]]
        test += [data[test_start:test_end]]
        
    return train, test


class ModelTester():
    """
    
    """
    
    def __init__(self, dataset, model, test_step_size=10, train_holdout=0.8):
        self.horizon = test_step_size
        self.dataset = dataset
        self.train_holdout = train_holdout
        
        # Divide dataset
        self.train, self.test = backtest(dataset, train_holdout, test_step_size)
        
        self.model = model
        
        # Saves results in the form: name : array of the results over the entire testing set
        self.predictions = {} # Name : np.ndarray # This
        
        # Save the error metrics
        self.errors = pd.DataFrame(columns=__error_metrics__)
        
        self.progressBar = ProgressBar()
    

    def __repr__(self):
        return 'Tester of' + repr(model)

    
    def test_model(self):
        """
        Test the given model with the dataset. This saves the predictions and errors.
        Returns -> Predictions of the model.
        """
        self.results = {}
        
        # Count the size of the test set
        size_of_tests = sum([len(test_set) for test_set in self.test])
    
        for column in self.train[0].columns:
            cum_index = 0
            self.predictions[column] = np.zeros((len(self.test) * len(self.test[0]), self.test[0].shape[0]))

            for split_ind, (train, test) in enumerate(zip(self.train, self.test)):
                if split_ind % 10 == 0:
                    print("{:d} / {:d}".format(split_ind, len(self.train)))

                train, test = train[column], test[column]

                model.fit(train, train)
                self.predictions[column][cum_index : cum_index + test.shape[0]] = model.predict(test)
                cum_index += test.shape[0]
                                          
        return self.predictions

                                          
    def evaluate_model(self, names):
        num_cols = len(self.dataset.columns)

        metrics = {
            'RMSE' : lambda x, y: np.sqrt(mean_squared_error(x, y)),
#             'MASE' : lambda x, y: calculate_MASE(self.train[0]['window_1_0'], x, y)
        }
        num_metrics = len(metrics)

        # Create a dataframe of metric values
        col_names = []
        for name in names:
            for metric in metrics.keys():
                col_names += [name + '_' + metric]
        
        errors = pd.DataFrame(data=np.zeros((num_cols, num_metrics * len(names))), columns=col_names, index=self.dataset.columns)
        
        self.progressBar.set_length(len(names) * len(self.train[0].columns))
        for name in names:
            for col_ind, column in enumerate(self.train[0].columns):
#                 print(column)
                self.progressBar.progress()
                
                actual = self.dataset[column][int(len(self.dataset) * self.train_holdout):]
                pred = pd.DataFrame(self.results[name][column])
        
                pred = pred[pred.index % self.test[0].shape[0] == 0] # TODO: Change this so it tests all preds
                
                # Ensure both are multiples of the horizon
                pred = pred.values.flatten()
                horizon = self.test[0].shape[0]
                pred = pred[:len(pred) - len(pred) % horizon]
                actual = actual[:len(actual) - len(actual) % horizon]
                max_len = min(pred.shape[0], actual.shape[0])
                pred = pred[:max_len]
                actual = actual[:max_len]
                
                for metric_key, metric in metrics.items():
                    errors[name + '_' + metric_key][self.dataset.columns[col_ind]] = metric(pred, actual)

        for metric_key in metrics.keys():
            mets = []
            for column in errors.columns:
                if column.endswith(metric_key):
                    mets += [column]
            errors[metric_key + ' avg'] = errors[mets].mean(axis = 1)
            errors[metric_key + ' std'] = errors[mets].std(axis = 1).fillna(0)
                    
        return errors

    
    def compare_hurst(self, hurst_estimates, err_to_use='RMSE', errors=None):
        if errors is None:
            errors = self.errors
        
        err_col = []
        for col in errors.columns:
            if err_to_use in col:
                err_col += [col]

        errors[err_to_use + ' avg'] = errors[err_col].mean(axis=1)
        errors[err_to_use + ' std'] = errors[err_col].std(axis=1)
        
        plt.figure()
        plt.errorbar(x=hurst_estimates['avg'], y=errors['RMSE avg'], xerr=hurst_estimates['std'], yerr=errors['RMSE std'], fmt='o')
    
        # Linear fit
        slope, intercept, r_value, p_value, std_err = stats.linregress(hurst_estimates['avg'], errors['RMSE avg'])
        X = np.linspace(0, hurst_estimates['avg'].max() * 1.1, 10)
        plt.xlim([hurst_estimates['avg'].min() * 0.98, hurst_estimates['avg'].max() * 1.02])

        plt.plot(X, intercept + slope * X, c='black', label="y= " + str(round(slope, 2)) + "X + " + str(round(intercept, 2)) + ", R = " + str(round(r_value**2, 2)))

        plt.fill_between(X, intercept + slope * X + std_err, intercept + slope * X - std_err, facecolor='r', alpha=0.5)
        plt.legend()
    
    
    def visualise_result(self, index, figsize=(6, 6)):
        try:
            iter(index)
            indexes = index
        except TypeError as te:
            indexes = [index]
            
        fig, axes = plt.subplots(math.ceil(len(indexes)**0.5), math.ceil(len(indexes)**0.5), figsize=figsize)
        try:
            axes = axes.flatten()
        except:
            axes = [axes]
        
        for index in indexes:
            col = self.train[0].columns[index]

            pred_at = np.arange(0, len(self.predictions[col]), self.horizon, dtype=np.int)
            x = np.arange(0, len(self.predictions[col]))
            test = np.hstack([x[col] for x in self.test]).flatten()
            pred = self.predictions[col][pred_at].flatten()
            
            test, pred, x = test[:min(len(test), len(pred))], pred[:min(len(test), len(pred))], x[:min(len(test), len(pred))]
            
            axes[indexes.index(index)].plot(x, test, color='blue', label='actual')
            axes[indexes.index(index)].plot(x, pred, color='orange', label='pred')

            axes[indexes.index(index)].scatter(pred_at, self.predictions[col][pred_at, 0], color='orange')

#             pred_at = np.arange(0, len(self.predictions[col]), self.horizon, dtype=np.int)
#             axes[indexes.index(index)].plot(np.arange(0, len(self.predictions[col])), np.hstack([x.iloc[:, :] for x in self.test]).flatten(), color='blue', label='actual')
#             axes[indexes.index(index)].plot(np.arange(0, len(self.predictions[col])), self.predictions[col][pred_at].flatten(), color='orange', label='pred')
            
#             axes[indexes.index(index)].scatter(pred_at, self.predictions[col][pred_at, 0], color='orange')

            
            
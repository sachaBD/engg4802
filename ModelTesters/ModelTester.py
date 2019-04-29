import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy import stats

from AutoValidator.ErrorMetrics import *
from utils.DataPrepUtils import *
from utils.ErrorMetrics import __error_metrics__, calculate_all_errors

import sys
sys.path.append('../utils')
from utils.progress import ProgressBar


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
        return 'Tester of' + repr(self.model)


    def test_model(self):
        """
        Test the given model with the dataset. This saves the predictions and errors.
        Returns -> Predictions of the model.
        """
        self.results = {}

        # Iterate over each column
        for column in self.train[0].columns:
            # Pre-allocate memory to store the predictions
            cum_index = 0
            self.predictions[column] = np.zeros((len(self.test) * len(self.test[0]), self.test[0].shape[0]))

            for split_ind, (train, test) in enumerate(zip(self.train, self.test)):
                train, test = train[column], test[column]

                self.model.fit(train, train)
                self.predictions[column][cum_index : cum_index + test.shape[0]] = self.model.predict(test)
                cum_index += test.shape[0]

        return self.predictions


    def compare_hurst(self, hurst_estimates, err_to_use='RMSE', errors=None):
        """
        Create a graphical comparison of the Hurst Exponent vs forecasting error for each dataset

        :param hurst_estimates:
        :param err_to_use:
        :param errors:
        :return:
        """
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

        plt.plot(X, intercept + slope * X, c='black', label="y= " + str(round(slope, 2)) + "X + " + str(round(intercept, 2)) + ", R = " + str
                     (round(r_value**2, 2)))

        plt.fill_between(X, intercept + slope * X + std_err, intercept + slope * X - std_err, facecolor='r', alpha=0.5)
        plt.legend()


    def visualise_result(self, index, figsize=(6, 6)):
        try:
            iter(index)
            indexes = index
        except TypeError as te:
            indexes = [index]

        fig, axes = plt.subplots(math.ceil(len(indexes )**0.5), math.ceil(len(indexes )**0.5), figsize=figsize)
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




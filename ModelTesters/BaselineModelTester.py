# from AutoValidator.ErrorMetrics import __error_metrics__, calculate_all_errors
from utils.progress import ProgressBar
import scipy.stats as stats
import numpy as np
import pandas as pd
import math

from utils.DataPrepUtils import backtest
from utils.ErrorMetrics import *

from ModelTesters.ModelTester import ModelTester

class BaselineModelTester(ModelTester):
    """

    """

    def __init__(self, dataset, model, test_step_size=10, train_holdout=0.8):
        ModelTester.__init__(self, dataset, model, test_step_size, train_holdout)


    def __repr__(self):
        return 'BaselineTester of' + repr(self.model)


    # def compare_hurst(self, hurst_estimates, err_to_use='RMSE', errors=None, figsize=(8, 6)):
    #     """
    #
    #     :param hurst_estimates:
    #     :param err_to_use:
    #     :param errors:
    #     :return:
    #     """
    #     if errors is None:
    #         errors = self.errors
    #
    #     plt.figure(figsize=figsize)
    #     plt.errorbar(x=hurst_estimates['avg'], y=errors[err_to_use], xerr=hurst_estimates['std'], fmt='o')
    #
    #     # Linear fit
    #     slope, intercept, r_value, p_value, std_err = stats.linregress(hurst_estimates['avg'], errors[err_to_use])
    #     X = np.linspace(0, hurst_estimates['avg'].max() * 1.1, 10)
    #     plt.xlim([hurst_estimates['avg'].min() * 0.98, hurst_estimates['avg'].max() * 1.02])
    #
    #     plt.plot(X, intercept + slope * X, c='black',
    #              label="y= " + str(round(slope, 2)) + "X + " + str(round(intercept, 2)) + ", R = " + str(
    #                  round(r_value ** 2, 2)))
    #
    #     plt.fill_between(X, intercept + slope * X + std_err, intercept + slope * X - std_err, facecolor='r', alpha=0.5)
    #     plt.legend()


    def visualise_result(self, name, index, figsize=(6, 6)):
        """

        :param name:
        :param index:
        :param figsize:
        :return:
        """
        try:
            iter(index)
            indexes = index
        except TypeError as te:
            indexes = [index]

        fig, axes = plt.subplots(math.ceil(len(indexes) ** 0.5), math.ceil(len(indexes) ** 0.5), figsize=figsize)
        try:
            axes = axes.flatten()
        except:
            axes = [axes]

        for index in indexes:
            col = self.train[0].columns[index]
            axes[indexes.index(index)].plot(np.arange(0, len(self.results[name][col])),
                                            np.hstack([x.iloc[:, index] for x in self.test]), color='blue',
                                            label='actual')
            axes[indexes.index(index)].plot(np.arange(0, len(self.results[name][col])), self.results[name][col],
                                            color='orange', label='pred')

            pred_at = np.arange(0, len(self.results[name][col]), self.horizon, dtype=np.int)
            axes[indexes.index(index)].scatter(pred_at, self.results[name][col][pred_at, 0], color='orange')



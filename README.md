# ENGG4802 - Modelling a Relationship Between Renewable Forecasting Error and Statistical Predictability

## Installation
pip install numpy pandas matplotlib scipy sklearn keras rpy2 matlab lightgbm

## Hurst Calculation
1. Split dataset:
  a) Run Split_Data.ipynb to seperate the original data into n (currently 60) datasets but dividing the data into m (=15) sets then applying the specified moving averages.
2) Hurst Estimation:
  a) Run Hurst_Estimation.ipynb to estimate Hurst Exponents using matlab methods.
This produces a pandas dataframe with each column representing a different dataset. The name convention is 'window_n_m' where n is the dataset number and m is the moving average window applied. These are slightly cropped (removing 1 value in some datasets) so that each dataset has the same number of samples.

## Forecasting Methods:
All forecasting methods are implemented as a Python Class implementing the standard SKLearn regressor methods of fit(X, y) and predict(X). All forecasting methods operate on any iterable by numpy arrays or pandas dataframe are prefered.

To produce a forecast and calculate the error metrics for each method a Class inheriting from BaselineModelTester is used. This Class implements test_model(), visualise_result and compare_hurst() methods. a speed_up_calc() function is occasionally used to improve the performance of the model training and testing to speed up compuation time. The compare_hurst() method produces a linear fit between H and and the provided metric (typically MASE) for all dataset.

In each Forecasting Method file forecasts are produced for 1, 5 and 60 minute forecasting horizons. A relationship between H and MASE is calculated and show and the results (all error metrics for each dataset) are saved to 'results/forecasting_method_horizon_results.csv'.

The following error metrics are calculated:
MAE	  - Mean Absolute Error
RMSE  - Root Mean Square Error
MAPE  - Mean Absolute Percentage Error
sMAPE - Symmetric Mean Absolute Percentage Error
MASE  - Mean Absolute Scaled Error
Ignore - MASE2	MASE3	MEAN_ASE

Some methods are implemented using the R programming language 

## Notes on specific methods:


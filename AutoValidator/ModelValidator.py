class ModelTester():
    
    def __init__(self, dataset, test_step_size=10):
        self.dataset = dataset
        
        # Divide dataset
        self.train, self.test = backtest(dataset, 0.8, test_step_size)
        
        self.models = {} # Name : model
        
        # Saves results in the form: name : array of the results over the entire testing set
        self.results = {} # Name : np.ndarray # This 
    
    def add_model(self, model, name: str):
        self.models[name] = model
    
    def test_models(self):
        self.results = {}
        size_of_tests = sum([len(test_set) for test_set in self.test])
        
        for name, model in self.models.items():
            self.results[name] = np.zeros(size_of_tests)
            
            cum_test_size = 0
            for train, test in zip(self.train, self.test):
                model.fit(train, train)
                pred = model.predict(test)
                
                self.results[name][cum_test_size: cum_test_size + len(pred)] = pred
                cum_test_size += len(pred)
                
        return self.results
    
    def visulise_results(self):
        pass

# from ErrorMetrics import *
# from DataSplitting import *

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import TimeSeriesSplit

# from MutliForecaster import MultiForecaster
# from TreeBasedModel import create_features


# # Load the data
# data = pd.read_csv('data.csv')
# data = data[: 50 * 1000]


# # Define the model
# horizon = 5
# features, labels = create_features(data.values, horizon)


# # Backtest on 10 divisions of the data
# for X_train, Y_train, X_test, Y_test in split_data(features, labels):
#     print(len(X_train), len(Y_train))
#     print(len(X_test), len(Y_test))


# # Train the model
# prediction = 0
# actual = 0


# # Produce error reports
# MASE = calculate_MASE(prediction, actual)
# calcaulte_sMAPE = calcaulte_sMAPE(prediction, actual)


# # Product plots






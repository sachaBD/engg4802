from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, LSTM
import tensorflow as ft
import pandas as pd
import pickle

from DataSplitting import split_data


# ------------------------------
# Create the forecaster
# ------------------------------

inputSize = 1024

model = Sequential()
model.add(Embedding(inputSize, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# ------------------------------
# Split the data
# ------------------------------
data = pd.read_csv('data.csv')[:300]
split = split_data(data, data, 1)
X_train, Y_train, X_test, Y_test = split[0]

# ------------------------------
# Fit the data
# ------------------------------
model.fit(X_train, Y_train, batch_size=20, epochs=20)
score = model.evaluate(X_test, Y_test, batch_size=16)
predictions = model.predict(X_test)

print("Score:", score)

# ------------------------------
# Serialise the results
# ------------------------------
pickle.dump(predictions, open('results/predictions', 'wb'))
pickle.dump(split, open('results/data', 'wb'))
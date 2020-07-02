
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

datasetPath = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
columnNames = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
               'Acceleration',   'Model Year',  'Origin']
rawDataset = pd.read_csv(datasetPath, names=columnNames,
                         na_values="?", comment='\t',
                         sep=" ", skipinitialspace=True)
dataset = rawDataset.copy()
dataset = dataset.dropna()
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')

xTrain = dataset.sample(frac=.8, random_state=0)
xTest = dataset.drop(xTrain.index)
yTrain = xTrain.pop('MPG')
yTest = xTest.pop('MPG')
stats = xTrain.describe()
stats = stats.transpose()

def norm(x):
    return (x - stats['mean']) / stats['std']
normXTrain = norm(xTrain)
normXTest = norm(xTest)
print(normXTrain.dtypes)

model = keras.Sequential()
model.add(keras.layers.Dense(64, activation='relu', input_shape=[len(xTrain.keys())]))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(1))

model.compile(loss='mse',
              optimizer=keras.optimizers.RMSprop(.001),
              metrics=['mae', 'mse'])
print(model.summary())

earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(
    normXTrain, yTrain,
    epochs=1000, validation_split=.2, verbose=0,
    callbacks=[earlyStop, tfdocs.modeling.EpochDots()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
plotter.plot({'Basic': history}, metric='mae')
plt.ylim([0, 10])
plt.ylabel('MAE [MPG]')
plt.show()

loss, mae, mse = model.evaluate(normXTest, yTest, verbose=2)
print(mae)

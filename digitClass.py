# https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#0

import math
import matplotlib
from matplotlib import pyplot as plt
from tensorflow import keras

digits = keras.datasets.mnist
(xTrain, yTrain), (xTest, yTest) = digits.load_data()
xTrain = xTrain.astype('float32')
xTest = xTest.astype('float32')
xTrain /= 255
xTest /= 255
xTrain = xTrain.reshape(60000, 784)
xTest = xTest.reshape(10000, 784)
yTrain = keras.utils.to_categorical(yTrain, num_classes=10)
yTest = keras.utils.to_categorical(yTest, num_classes=10)

model = keras.models.Sequential()
model.add(keras.layers.Dense(28, activation='sigmoid', input_shape=(784,)))
# model.add(keras.layers.Dense(10, activation='sigmoid'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(xTrain, yTrain, batch_size=100, epochs=10)

loss, acc = model.evaluate(xTest, yTest)
print(acc)

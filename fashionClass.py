
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

fashion = keras.datasets.fashion_mnist
(xTrain, yTrain), (xTest, yTest) = fashion.load_data()
classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
           'Shirt', 'Sneaker', 'Bag', "Ankle boot"]
xTrain = xTrain / 255
xTest = xTest / 255

xTrain = np.expand_dims(xTrain, axis=-1)
xTest = np.expand_dims(xTest, axis=-1)

print(xTest.shape)

model = keras.Sequential()
model.add(keras.layers.Conv2D(16, 3, input_shape=(28, 28, 1), padding='same', activation='relu'))
model.add(keras.layers.Conv2D(64, 3, padding='same', activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10))
model.add(keras.layers.Softmax())

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
model.fit(xTrain, yTrain, epochs=10)

loss, acc = model.evaluate(xTest, yTest, verbose=2)
print(acc)

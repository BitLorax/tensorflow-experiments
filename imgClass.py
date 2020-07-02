
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(xTrain, yTrain), (xTest, yTest) = datasets.cifar10.load_data()
xTrain, xTest = xTrain / 255.0, xTest / 255.0

classNames = ['airplane', 'automobile', 'bird', 'cat', 'deeer', 'dog', 'frog',
              'horse', 'ship', 'truck']

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='Adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(xTrain, yTrain, epochs=10,
                    validation_data=(xTest, yTest))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([.5, 1])
plt.legend(loc='lower right')

loss, acc = model.evaluate(xTest, yTest, verbose=2)
print(acc)

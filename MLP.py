from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
from keras.utils import np_utils
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape)
# print(y_train)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

plt.figure()
plt.imshow(x_train[0])
plt.colorbar()
plt.grid(False)
plt.show()

x_train = x_train / 255.0
x_test = x_test / 255.0


plt.figure(figsize=(20, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    # keras.layers.Dense(256, activation='relu'),
    # keras.layers.Dropout(.6),
    # keras.layers.Dense(192, activation='relu'),
    # keras.layers.Dropout(.6),
    # keras.layers.Dense(64, activation='relu'),
    # keras.layers.Dropout(.6),
    keras.layers.Dense(10, activation='softmax')
])

# model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
weights = model.get_weights()
input_weight = weights[0]
input_weight = np.array(input_weight)
for i in range(10):
    weight = input_weight[:, i]
    predict = np.array(weight)
    # print(predict)
    predict = np.reshape(predict, (28, 28))
    plt.imshow(predict)
    plt.show()

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)
model.save('model.h5')
print("Model saved to disk")

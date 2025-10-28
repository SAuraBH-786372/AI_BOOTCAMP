#importing libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from keras.datasets import mnist

# Load the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Training data shape:", x_train.shape)
print("Test labels shape:", y_test.shape)
print("Sample image array:\n", x_train[0])
plt.imshow(x_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()

# Normalizing
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode labels
print(f"Before encoding label 100: {y_train[100]}")
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(f"After encoding label 100: {y_train[100]}")

# Model architecture
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

#compiling model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#training model
result=model.fit(x_train, y_train, epochs=10,batch_size=64)

#evaluating model
loss, accuracy=model.evaluate(x_train,y_train)
model.evaluate(x_test,y_test)
print(f"test loss:{loss}")
print(f"test accuracy:{accuracy}")
print(result.history)
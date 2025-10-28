# Importing libraries
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
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model (added validation split)
result = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

# Evaluate model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

# Visualize training history
plt.plot(result.history['loss'], label='Train loss', color='green')
plt.plot(result.history['val_loss'], label='Validation loss', color='blue')
plt.title('Training Loss vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(result.history['accuracy'], label='Train accuracy', color='green')
plt.plot(result.history['val_accuracy'], label='Validation accuracy', color='blue')
plt.title('Training Accuracy vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

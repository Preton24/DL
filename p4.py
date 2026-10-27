import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense

(x_train, _), (x_test, _) = mnist.load_data()      # labels not needed -> use _
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

encoder = Sequential([
    Input(shape=(784,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu')
])

decoder = Sequential([
    Input(shape=(64,)),
    Dense(128, activation='relu'),
    Dense(784, activation='sigmoid')
])

autoencoder = Sequential([encoder, decoder])
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train, epochs=10, batch_size=256,
                validation_data=(x_test, x_test))
encoded = encoder.predict(x_test[:5], verbose=0)     # 5 x 64
decoded = decoder.predict(encoded, verbose=0)        # 5 x 784

n = 5
plt.figure(figsize=(10, 6))
for i in range(n):
    plt.subplot(3, n, i + 1)                         # row 1: original
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(3, n, i + 1 + n)                     # row 2: encoded (64 = 8x8)
    plt.imshow(encoded[i].reshape(8, 8), cmap='gray')
    plt.title("Encoded 8x8")
    plt.axis('off')

    plt.subplot(3, n, i + 1 + 2 * n)                 # row 3: reconstructed
    plt.imshow(decoded[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')
plt.tight_layout()
plt.show()
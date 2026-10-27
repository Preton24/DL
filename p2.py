import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense

X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Input(shape=(2,)),
    Dense(100, activation='relu'),
    Dense(50, activation='relu'),
    Dense(1, activation='sigmoid')     # 0 or 1
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("Test loss    :", loss)
print("Test accuracy:", accuracy)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

grid = np.c_[xx.ravel(), yy.ravel()]                    # list of (x, y) points
Z = model.predict(grid, verbose=0)
Z = (Z > 0.5).astype(int).reshape(xx.shape)             # > 0.5 means class 1

plt.contourf(xx, yy, Z, alpha=0.7)                      # coloured regions
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test)       # real test points
plt.title("DNN Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
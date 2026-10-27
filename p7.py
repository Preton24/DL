import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Resizing, Rescaling, GlobalAveragePooling2D, Dense, Dropout

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, y_train = x_train[:10000], y_train[:10000]
x_test, y_test = x_test[:2000], y_test[:2000]

class_names = ["airplane", "car", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
base_model.trainable = False

model = Sequential([
    Input(shape=(32, 32, 3)),
    Resizing(96, 96),                   # MobileNetV2 needs bigger images
    Rescaling(1 / 127.5, offset=-1),    # pixels 0-255 -> -1 to 1 (what MobileNetV2 expects)
    base_model,                         # frozen, already trained
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, batch_size=32, validation_data=(x_test, y_test))

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy:", accuracy)

pred = model.predict(x_test[:10], verbose=0).argmax(axis=1)

plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i])
    plt.title("P:" + class_names[pred[i]] + "\nT:" + class_names[y_test[i, 0]], fontsize=8)
    plt.axis('off')
plt.tight_layout()
plt.show()
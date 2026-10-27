import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout

vocab_size = 10000
max_len = 200
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

model = Sequential([
    Input(shape=(max_len,)),
    Embedding(vocab_size, 64),
    Bidirectional(LSTM(64)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=3, batch_size=128,
                    validation_data=(x_test, y_test))

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy:", accuracy)

word_index = imdb.get_word_index()
index_word = {number: word for word, number in word_index.items()}

i = np.random.randint(len(x_test))
review = " ".join(index_word.get(n - 3, "?") for n in x_test[i] if n > 3)   # skip padding/special
prob = model.predict(x_test[i:i + 1], verbose=0)[0, 0]

print("Review   :", review)
print("Sentiment:", "Positive" if prob > 0.5 else "Negative")
print("Actual   :", "Positive" if y_test[i] == 1 else "Negative")

plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
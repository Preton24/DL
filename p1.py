import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense
from sklearn.manifold import TSNE

sentences = [
    "deep learning improves computer vision and language processing",
    "neural networks are powerful tools for artificial intelligence",
    "word embeddings capture meaning of words in text",
    "language models use embeddings to represent words",
]

words = " ".join(sentences).lower().split()   # all words in one list
vocab = sorted(set(words))                    # unique words
word2id = {w: i for i, w in enumerate(vocab)} # word -> number
vocab_size = len(vocab)
print("Vocabulary size:", vocab_size)

window = 2
centre = []
context = []
for i in range(len(words)):                          # i = position of centre word
    for j in range(i - window, i + window + 1):      # j = neighbours
        if j >= 0 and j < len(words) and j != i:     # stay inside text, skip itself
            centre.append(word2id[words[i]])
            context.append(word2id[words[j]])
x_train = np.array(centre)    # input  = centre word number
y_train = np.array(context)   # output = context word number
print("Number of pairs:", len(x_train))

model = Sequential([
    Input(shape=(1,)),                      # one word number in
    Embedding(vocab_size, 8),               # word -> 8 numbers (THE EMBEDDING)
    Flatten(),
    Dense(vocab_size, activation='softmax') # which word is in the context?
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(x_train, y_train, epochs=100, verbose=0)
print("Training done")

embeddings = model.layers[0].get_weights()[0]     # shape = (vocab_size, 8)
for w in vocab[:10]:
    print(w, "->", np.round(embeddings[word2id[w]], 2))


points = TSNE(n_components=2, perplexity=5, random_state=42).fit_transform(embeddings)
plt.figure(figsize=(8, 6))
plt.scatter(points[:, 0], points[:, 1])
for i, w in enumerate(vocab):
    plt.annotate(w, (points[i, 0], points[i, 1]))   # write the word next to its dot
plt.title("Skip-gram Word Embeddings")
plt.show()
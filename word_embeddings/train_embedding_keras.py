from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding

# define documents
docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!',
        'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.'
        ]
# define class
# In this case this are labels for 'good/bad' feedback classification.
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

# not sure wat is correct... but I believe we need to encode all the
# corpus together. Otherwise different words might have the same code!
# Check out an example online.
vocab = " ".join(docs)
vocab_size = len(vocab.split(' '))
print(vocab)
codes = one_hot(vocab, vocab_size)
print(codes, len(codes), vocab_size)
one_hot_encoder = {}
for c, w in zip(codes, vocab.split(' ')):
    one_hot_encoder[w] = c
encoded_docs = [[one_hot_encoder[w] for w in d.split(' ')] for d in docs]
print(encoded_docs)

# Padd the codes
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)

# Define network
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()

# Train
model.fit(padded_docs, labels, epochs=50, verbose=0)
# Evaluate
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))

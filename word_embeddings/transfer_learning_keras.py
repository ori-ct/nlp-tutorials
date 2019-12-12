"""

Following tutorial in https://stackabuse.com/python-for-nlp-movie-sentiment-analysis-using-deep-learning-in-keras/ as nlp transfer learning

"""
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer


# Load data
movie_reviews = pd.read_csv('../data/IMDB_Dataset.csv')
movie_reviews.isnull().values.any()
movie_reviews.shape
print(movie_reviews.head())
movie_reviews["review"][3]
print(movie_reviews.groupby('sentiment').count())


def remove_tags(text):
    """

    This function removes HTML tags

    """
    # pattern anything between <> characters
    TAG_RE = re.compile(r'<[^>]+>')
    return TAG_RE.sub('', text)


def preprocess_text(sen):
    """

    Helper function to preprocess text

    """
    # Remove HTML tags
    sentence = remove_tags(sen)
    # remove punctuation
    sentence = re.sub('[^A-zA-Z]', ' ', sentence)
    # single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # remove multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    return(sentence)


# Preprocess data
X = [preprocess_text(sen) for sen in list(movie_reviews['review'])]
y = movie_reviews.sentiment.apply(lambda x: 1 if x == 'positive' else 0).tolist()
print('Preprocessed example:')
print(X[3])

# Train-test
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20,
                                                    random_state=42)

# Transform data to one-hot
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Padd data to get input with same size
vocab_size = len(tokenizer.word_index)+1
maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# this loads the pretrained embedding (GloVe)
embedding_dictionary = dict()
glove_file = open('../data/glove.6B.100d.txt', encoding='utf8')
for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embedding_dictionary[word] = vector_dimensions
glove_file.close()

# create the mapping from one-hot to glove features
embedding_matrix = np.zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embedding_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

# create neural network
model = Sequential()
embedding_layer = Embedding(vocab_size,
                            100,
                            weights=[embedding_matrix],
                            input_length=maxlen,
                            trainable=False)
model.add(embedding_layer)
model.add(Flatten())
model.add(Dense(1, kernel_regularizer=l2(0.01), activation='sigmoid'))
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['acc'])
print(model.summary())

history = model.fit(X_train,
                    y_train,
                    batch_size=128,
                    epochs=10,
                    verbose=1,
                    validation_split=0.2)

acc = model.evaluate(X_test, y_test, verbose=1)
print("Test Score:", acc[0])
print("Test Accuracy:", acc[1])

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.show()

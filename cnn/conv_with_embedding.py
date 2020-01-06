import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.layers import Dense, Flatten, Embedding, GlobalMaxPool1D, Conv1D
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from dataset import get_data
from glove_keras import create_embedding_matrix
import matplotlib.pyplot as plt


def plot_convergence(history):
    all_data_metrics = list(history.history.keys())
    metrics = [item for item in all_data_metrics 
               if not item.startswith('val')]
    data = {}
    for metric in metrics:
        data[metric] = [item for item in all_data_metrics
                        if item.endswith(metric)]
    fig, ax = plt.subplots(len(metrics))
    fig.suptitle('Convergence')
    for i, metric in enumerate(metrics):
        for datapart in data[metric]:
            ax[i].plot(history.history[datapart])
        ax[i].set(xlabel='epoch', ylabel=metric)
    legend = []
    for item in data[metric]:
        aux = item.split('_')
        if len(aux) > 1:
            legend.append(aux[0])
        else:
            legend.append('train') 
    plt.legend(legend)
    plt.show()

# Get data
#df = get_data(['yelp'])  # , 'amazon', 'imdb'])
df = get_data(['yelp', 'amazon', 'imdb'])
sentences = df['sentence'].values
y = df['label'].values

# Train-test split
(sentences_train,
 sentences_test,
 y_train,
 y_test) = train_test_split(sentences,
                            y,
                            test_size=0.25,
                            random_state=1000)

# Data preparation (tokenizator)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences_train)
X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)
maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
vocab_size = len(tokenizer.word_index) + 1

pretrained_embedding = True
embedding_dim = 300
pretrained_model = {'filepath': '../data/glove_pretrained_embeddings/glove.6B.300d.txt',
                    'embedding_dim': 300}
if pretrained_embedding:
    embedding_matrix = create_embedding_matrix(tokenizer.word_index,
                                               embedding_dim,
                                               pretrained_model)

# Define model
model = Sequential()
if not pretrained_embedding:
    model.add(Embedding(input_dim=vocab_size,
                        output_dim=embedding_dim,
                        input_length=maxlen))
else:
    model.add(Embedding(input_dim=vocab_size,
                        output_dim=embedding_dim,
                        weights=[embedding_matrix],
                        input_length=maxlen,
                        trainable=True))
#model.add(Flatten())
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPool1D())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train,
                    y_train,
                    epochs=10,
                    verbose=True,
                    validation_data=(X_test, y_test),
                    batch_size=10)

loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print('-'*79)
print("Accuracy", accuracy)
plot_convergence(history)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.layers import Dense
from keras.models import Sequential
from dataset import get_data
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
df_yelp = get_data(['yelp', 'amazon', 'imdb'])
sentences = df_yelp['sentence'].values
y = df_yelp['label'].values

# Train-test split
(sentences_train,
 sentences_test,
 y_train,
 y_test) = train_test_split(sentences,
                            y,
                            test_size=0.25,
                            random_state=1000)

# Data preparation (tokenizator)
vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_test)

# Define model
input_dim = X_train.shape[1]
model = Sequential()
model.add(Dense(10, input_dim = input_dim, activation='relu'))
model.add(Dense(1, input_dim = 10, activation='sigmoid'))
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train,
                    y_train,
                    epochs=100,
                    verbose=True,
                    validation_data=(X_test, y_test),
                    batch_size=10)

loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print('-'*79)
print("Accuracy", accuracy)
plot_convergence(history)

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from dataset import get_data


df_yelp = get_data(['yelp', 'amazon', 'imdb'])
sentences = df_yelp['sentence'].values
y = df_yelp['label'].values

(sentences_train,
 sentences_test,
 y_train,
 y_test) = train_test_split(sentences,
                            y,
                            test_size=0.25,
                            random_state=1000)

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)

print("Accuracy", score)

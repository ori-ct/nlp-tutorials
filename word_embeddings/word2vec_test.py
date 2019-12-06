from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
             ['this', 'is', 'the', 'second', 'sentence'],
             ['yet', 'another', 'sentence'],
             ['one', 'more', 'sentence'],
             ['and', 'the', 'final', 'sentence']
             ]


print("Training data:")
print(sentences)

model = Word2Vec(sentences, min_count=1)
words = list(model.wv.vocab)
print(model)

print("The word 'sentence' is embedded as:")
print(model['sentence'])

X = model[model.wv.vocab]
pca = PCA(n_components=2)
viz = pca.fit_transform(X)

ax = plt.subplots()[1]
ax.scatter(viz[:, 0], viz[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
    ax.annotate(word, xy=(viz[i, 0], viz[i, 1]))
plt.show()

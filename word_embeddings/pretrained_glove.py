import gensim.downloader as api

word_vectors = api.load("glove-wiki-gigaword-100")

result = word_vectors.most_similar(positive=['woman', 'king'], negative=['man'])

print("Word-vector operation test:")
print("---------------------------")
print("(king-man) + woman = ")
print("{}: {:.4f}".format(*result[0]))

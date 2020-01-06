import numpy as np


default_glove = {'filepath': '../data/glove_pretrained_embeddings/glove.6B.50d.txt',
                 'embedding_dim': 50}

def create_embedding_matrix(max_word_index, embedding_dim, glove_model=default_glove):
    assert embedding_dim == glove_model['embedding_dim']
    vocab_size = len(max_word_index) + 1 
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    with open(glove_model['filepath']) as f:
        for line in f:
            word, *vector = line.split()
            if word in max_word_index:
                idx = max_word_index[word] 
                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]
    return embedding_matrix

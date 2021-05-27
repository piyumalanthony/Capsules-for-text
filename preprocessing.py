import pandas as pd
import numpy as np
from gensim.models import FastText
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
import pickle


def text_preprocessing(data):
    comments = data['comment']
    labels = data['label']

    comments_splitted = []
    for comment in comments:
        lines = []
        try:
            words = comment.split()
            lines += words
        except ValueError:
            continue

        comments_splitted.append(lines)
    return comments_splitted, labels


def generate_embedding_matrix(word_embedding_keydvectors_path,
                              embedding_matrix_path,
                              vocab_size,
                              embedding_size, tokenizer):
    # if embedding_type == 'fasText':
    #     word_embedding_model = FastText.load(word_embedding_path)
    # else:
    #     word_embedding_model = word2vec.Word2Vec.load(word_embedding_path)
    #
    # word_vectors = word_embedding_model.wv
    # word_vectors.save(word_embedding_keydvectors_path)
    word_vectors = KeyedVectors.load(word_embedding_keydvectors_path, mmap='r')
    print("Running the embedding generation...")
    embeddings_index = dict()
    for word, vocab_obj in word_vectors.vocab.items():
        embeddings_index[word] = word_vectors[word]

    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((vocab_size, embedding_size))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    pickle.dump(embedding_matrix, open(embedding_matrix_path, 'wb'))
    print("Successfully executed the embedding generation...")
    return embedding_matrix


def load_word_embedding_matrix(embedding_matrix_path):
    f = open(embedding_matrix_path, 'rb')
    embedding_matrix = np.array(pickle.load(f))
    return embedding_matrix

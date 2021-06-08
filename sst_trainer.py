import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import utils

# import ensemble_capsule_network
import ensemble_capsule_network
import network_test
from config import Config
# import network
from preprocessing import text_preprocessing_sst, load_word_embedding_matrix, generate_embedding_matrix, \
    generate_embedding_matrix_glove

folder_path = "D:\\deep_learning_experiments\\SST2_data"
sst_dev_data_path = folder_path + "\\dev.tsv"
sst_train_data_path = folder_path + "\\train.tsv"
sst_test_data_path = folder_path + "\\test.tsv"

glove_vectors_path = "D:\\deep_learning_experiments\\Glove_embeddings\\glove.6B.300d.txt"


word_embedding_matrix_path = 'D:\\deep_learning_experiments\\SST2_data\\word_embedding_matrix'
EMBEDDING_SIZE = 300


dev_data = pd.read_csv(sst_dev_data_path, sep='\t')
train_data = pd.read_csv(sst_train_data_path, sep='\t')
test_data = pd.read_csv(sst_test_data_path, sep='\t')
# print(df.iloc[0])
# print(test_data['label'])

all_data = pd.concat([dev_data, train_data, test_data], ignore_index=True)
# all_data['label'] = all_data['label'] - 2
print(all_data)

comments_text = text_preprocessing_sst(all_data)
t = Tokenizer()
t.fit_on_texts(comments_text)
vocab_size = len(t.word_index) + 1
print(vocab_size)

encoded_docs_train = t.texts_to_sequences(train_data['sentence'])
encoded_docs_test = t.texts_to_sequences(dev_data['sentence'])
# for i in encoded_docs:
#     print(len(i))
# zzz = lambda z: len(z)
lengths = list(map(lambda z: len(z), encoded_docs_train + encoded_docs_test))
print('###########################################')
print(lengths)

max_length = max(lengths)
padded_docs_train = pad_sequences(encoded_docs_train, maxlen=max_length, padding='post')
padded_docs_test = pad_sequences(encoded_docs_test, maxlen=max_length, padding='post')
labels_train = np.array(train_data['label'])
labels_test = np.array(dev_data['label'])
labels_train = utils.to_categorical(labels_train)
labels_test = utils.to_categorical(labels_test)
padded_docs_train = np.array(padded_docs_train)
padded_docs_test = np.array(padded_docs_test)

print("Shape of all comments: ", padded_docs_train.shape)
print("Shape of labels: ", padded_docs_test.shape)
#
# X_train, X_test, y_train, y_test = train_test_split(padded_docs, comment_labels, test_size=0.1, random_state=42,
#                                                     shuffle=True)
# print("Train lables shape: ", y_train.shape)

# generate embedding matrix
# embedding_matrix = generate_embedding_matrix_glove(glove_vectors_path, word_embedding_matrix_path, vocab_size,
#                                              EMBEDDING_SIZE, t)

# load embedding matrix
embedding_matrix = load_word_embedding_matrix(word_embedding_matrix_path)

# print(embedding_matrix[1])
config = Config(
    seq_len=max_length,
    num_classes=2,
    vocab_size=vocab_size,
    embedding_size=EMBEDDING_SIZE,
    dropout_rate=0.8,
    x_train=padded_docs_train,
    y_train=labels_train,
    x_test=padded_docs_test,
    y_test=labels_test,
    pretrain_vec=embedding_matrix)

model = ensemble_capsule_network.ensemble_capsule_network(config)
# model = network_test.get_model_from_text_layer(config)
model.fit(x=padded_docs_train, y=labels_train, validation_data=(padded_docs_test, labels_test), epochs=10)

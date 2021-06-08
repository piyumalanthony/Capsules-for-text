import numpy as np
import pandas as pd
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split, KFold

import ensemble_capsule_network
from config import Config
from preprocessing import text_preprocessing, load_word_embedding_matrix

folder_path = "D:\\deep_learning_experiments"
lankadeepa_data_path = folder_path + "\\sinhala_data\\lankadeepa_tagged_comments.csv"
gossip_lanka_data_path = folder_path + "\\sinhala_data\\gossip_lanka_tagged_comments.csv"

word_embedding_keyed_vectors_path = 'D:\\deep_learning_experiments\\word_vectors_sinhala\\keyed.kv'
word_embedding_matrix_path = 'D:\\deep_learning_experiments\\word_embedding_matrix'
EMBEDDING_SIZE = 300

lankadeepa_data = pd.read_csv(lankadeepa_data_path)[:9059]
gossipLanka_data = pd.read_csv(gossip_lanka_data_path)
gossipLanka_data = gossipLanka_data.drop(columns=['Unnamed: 3'])

word_embedding_path = folder_path

all_data = pd.concat([lankadeepa_data, gossipLanka_data], ignore_index=True)
all_data['label'] = all_data['label'] - 2
print(all_data)

comments_text, labels = text_preprocessing(all_data)
t = Tokenizer()
t.fit_on_texts(comments_text)
vocab_size = len(t.word_index) + 1
print(vocab_size)

encoded_docs = t.texts_to_sequences(comments_text)
max_length = 30
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
comment_labels = np.array(labels)
comment_labels = utils.to_categorical(comment_labels)
padded_docs = np.array(padded_docs)

print("Shape of all comments: ", padded_docs.shape)
print("Shape of labels: ", comment_labels.shape)

X_train, X_test, y_train, y_test = train_test_split(padded_docs, comment_labels, test_size=0.1, random_state=42,
                                                    shuffle=True)
print("Train lables shape: ", y_train.shape)

# generate embedding matrix
# embedding_matrix = generate_embedding_matrix(word_embedding_keyed_vectors_path, word_embedding_matrix_path, vocab_size,
#                                              EMBEDDING_SIZE, t)

# load embedding matrix
embedding_matrix = load_word_embedding_matrix(word_embedding_matrix_path)
kfold = KFold(n_splits=10, shuffle=True)
# print(embedding_matrix[1])
config = Config(
    seq_len=max_length,
    num_classes=4,
    vocab_size=vocab_size,
    embedding_size=EMBEDDING_SIZE,
    dropout_rate=0.8,
    x_train=X_train,
    y_train=y_train,
    x_test=X_test,
    y_test=y_test,
    pretrain_vec=embedding_matrix)

inputs = padded_docs
targets = comment_labels
# training_dataset = get_training_dataset(images, labels)
# validation_dataset = get_validation_dataset(images_val, labels_val)
index = 1
for train, test in kfold.split(inputs, targets):
    print("Cross validation step:", index)
    model = ensemble_capsule_network.ensemble_capsule_network(config)
    model.fit(x=inputs[train], y=targets[train], validation_data=(inputs[test], targets[test]), epochs=50,
              batch_size=config.batch_size)
    index = index + 1
    print("++++++++++++++++++++++++++++++")
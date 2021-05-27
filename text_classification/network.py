import tensorflow as tf
import tensorflow.keras.layers as layers
import keras.backend as K
from tensorflow.python.keras.regularizers import l2

from text_classification.config import Config
from text_classification.loss import Margin_loss


def get_model(self, summary=True):
    input_tokens = layers.Input(shape=(self.sequence_length,))
    embedding = layers.Embedding(self.vocab_size, self.embedding_size,
                                 weights=[self.pretrain_vec],
                                 trainable=False, embeddings_regularizer=l2(self.l2),
                                 mask_zero=True)(input_tokens)
    embedding = layers.Reshape(target_shape=(self.sequence_length, self.embedding_size, 1))(embedding)
    # embedding = layers.Lambda(lambda x: K.expand_dims(x, axis=-1))(embedding)
    conv_layer = layers.Conv2D(16, kernel_size=(self.filter_size, self.embedding_size), use_bias=False,
                               kernel_regularizer=l2(self.l2), activation=None, padding='VALID')(embedding)
    # conv_layer = layers.BatchNormalization()(conv_layer)
    # conv_layer = layers.Conv2D(32, kernel_size=(self.filter_size, self.filter_size), use_bias=False,
    #                            kernel_regularizer=l2(self.l2), activation=None, padding='same')(conv_layer)
    # conv_layer = layers.BatchNormalization()(conv_layer)
    # conv_layer = layers.Conv2D(128, kernel_size=(self.filter_size, self.filter_size), use_bias=False,
    #                            kernel_regularizer=l2(self.l2), activation=None, padding='same')(conv_layer)
    conv_layer = layers.BatchNormalization()(conv_layer)
    flatten = layers.Flatten()(conv_layer)
    # dence_layer = layers.Dense(units=256)(flatten)
    # dence_layer_activation = layers.Activation(activation='relu')(dence_layer)
    dence_layer = layers.Dense(units=128)(flatten)
    dence_layer_activation = layers.Activation(activation='relu')(dence_layer)
    # dence_layer = layers.Dense(units=64)(dence_layer_activation)
    # dence_layer_activation = layers.Activation(activation='relu')(dence_layer)
    classification_layer = layers.Dense(units=4)(dence_layer_activation)
    output_layer = layers.Activation(activation='softmax')(classification_layer)

    model = tf.keras.Model(inputs=input_tokens, outputs=output_layer, name='sample_CNN')
    # loss = Margin_loss()
    # model.add_loss(loss)

    if summary:
        model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(self.init_lr, beta_1=0.7, beta_2=0.999, amsgrad=True),
                  metrics=['accuracy'])
    return model

# w2v = tf.random.normal([30_000, 300])
# config = Config(pretrain_vec=w2v)
# 
# get_model(config)

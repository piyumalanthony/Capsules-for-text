import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.python.keras.regularizers import l2
import tensorflow.keras.backend as K

from config import Config
from routing import Routing, CapsuleNorm


def get_generic_cnn_model(self, summary=True):
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


def get_kim_cnn_model(self, summary=True):
    input_tokens = layers.Input(shape=(self.sequence_length,))
    embedding = layers.Embedding(self.vocab_size, self.embedding_size,
                                 weights=[self.pretrain_vec],
                                 trainable=True, embeddings_regularizer=l2(self.l2),
                                 mask_zero=True)(input_tokens)
    embedding = layers.Reshape(target_shape=(self.sequence_length, self.embedding_size, 1))(embedding)
    # embedding = layers.Lambda(lambda x: K.expand_dims(x, axis=-1))(embedding)
    conv_layer_1 = layers.Conv2D(128, kernel_size=(3, self.embedding_size), use_bias=False,
                                 activation=None, padding='VALID')(embedding)
    conv_layer_1 = layers.BatchNormalization()(conv_layer_1)
    conv_layer_2 = layers.Conv2D(128, kernel_size=(4, self.embedding_size), use_bias=False,
                                 activation=None, padding='VALID')(embedding)
    conv_layer_2 = layers.BatchNormalization()(conv_layer_2)
    conv_layer_3 = layers.Conv2D(128, kernel_size=(5, self.embedding_size), use_bias=False,
                                 activation=None, padding='VALID')(embedding)
    conv_layer_3 = layers.BatchNormalization()(conv_layer_3)
    concat_layer = layers.Concatenate(axis=1)([conv_layer_1, conv_layer_2, conv_layer_3])
    flatten = layers.Flatten()(concat_layer)
    dence_layer = layers.Dense(units=256)(flatten)
    dence_layer_activation = layers.Activation(activation='relu')(dence_layer)
    dence_layer = layers.Dense(units=128)(dence_layer_activation)
    dence_layer_activation = layers.Activation(activation='relu')(dence_layer)
    dence_layer = layers.Dense(units=64)(dence_layer_activation)
    dence_layer_activation = layers.Activation(activation='relu')(dence_layer)
    classification_layer = layers.Dense(units=4)(dence_layer_activation)
    output_layer = layers.Activation(activation='softmax')(classification_layer)
    output_shape = conv_layer_3.get_shape()

    model = tf.keras.Model(inputs=input_tokens, outputs=output_layer, name='sample_CNN')
    # loss = Margin_loss()
    # model.add_loss(loss)

    if summary:
        model.summary()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(self.init_lr, beta_1=0.7, beta_2=0.999, amsgrad=True),
                  metrics=['accuracy'])
    return model, output_shape


def get_capsule_network_model(self, summary=True):
    input_tokens = layers.Input(shape=(self.sequence_length,))
    embedding = layers.Embedding(self.vocab_size, self.embedding_size,
                                 weights=[self.pretrain_vec],
                                 trainable=True, embeddings_regularizer=l2(self.l2),
                                 mask_zero=True)(input_tokens)
    embedding = layers.Reshape(target_shape=(self.sequence_length, self.embedding_size, 1))(embedding)
    # embedding = layers.Lambda(lambda x: K.expand_dims(x, axis=-1))(embedding)
    conv_layer_1 = layers.Conv2D(128, kernel_size=(3, self.embedding_size), use_bias=False,
                                 activation=None, padding='VALID')(embedding)
    conv_layer_1 = layers.BatchNormalization()(conv_layer_1)
    conv_layer_2 = layers.Conv2D(128 * 8, kernel_size=(conv_layer_1.get_shape()[1], 1), use_bias=False,
                                 activation=None, padding='VALID')(conv_layer_1)
    conv_layer_2 = layers.BatchNormalization()(conv_layer_2)
    reshape_capsules = layers.Reshape(target_shape=(128, 8))(conv_layer_2)
    text_caps = Routing(num_capsule=4,
                        l2_constant=0.001,
                        dim_capsule=16,
                        routing=True,
                        num_routing=3)(reshape_capsules)
    output = CapsuleNorm()(text_caps)

    model = tf.keras.Model(inputs=input_tokens, outputs=output, name='sample_capsule')
    # loss = Margin_loss()
    # model.add_loss(loss)

    if summary:
        model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(self.init_lr, beta_1=0.7, beta_2=0.999, amsgrad=True),
                  metrics=['accuracy'])
    return model


def get_model_from_text_layer(self, summary=True):
    if self.routing:
        use_routing = True
    else:
        use_routing = False

    input_tokens = layers.Input((self.sequence_length,))
    embedding = layers.Embedding(self.vocab_size, self.embedding_size,
                                 weights=[self.pretrain_vec],
                                 trainable=False,
                                 mask_zero=True)(input_tokens)
    embedding = layers.Lambda(lambda x: K.expand_dims(x, axis=-1))(embedding)

    # non-linear gate layer
    elu_layer = layers.Conv2D(self.num_filter, kernel_size=(self.filter_size, self.embedding_size),
                              use_bias=False,
                              kernel_regularizer=l2(self.l2), activation=None)(embedding)
    elu_layer = layers.BatchNormalization()(elu_layer)
    elu_layer = layers.Activation('elu')(elu_layer)

    conv_layer = layers.Conv2D(self.num_filter, kernel_size=(self.filter_size, self.embedding_size),
                               use_bias=False,
                               kernel_regularizer=l2(self.l2), activation=None)(embedding)
    conv_layer = layers.BatchNormalization()(conv_layer)

    gate_layer = layers.Multiply()([elu_layer, conv_layer])

    # dropout
    gate_layer = layers.Dropout(self.dropout_ratio)(gate_layer)

    # convolutional capsule layer
    h_i = layers.Conv2D(32 * 8,
                        kernel_size=(K.int_shape(gate_layer)[1], 1),
                        use_bias=False,
                        kernel_regularizer=l2(self.l2), activation=None)(gate_layer)
    h_i = layers.Reshape((32, 8))(h_i)
    h_i = layers.BatchNormalization()(h_i)

    h_i = layers.Activation('relu')(h_i)

    # dropout
    h_i = layers.Dropout(self.dropout_ratio)(h_i)

    # routing algorithm
    # text_caps = Routing(num_capsule=16,
    #                     l2_constant=self.l2,
    #                     dim_capsule=10,
    #                     routing=True,
    #                     num_routing=3)(h_i)
    text_caps = Routing(num_capsule=8,
                        l2_constant=self.l2,
                        dim_capsule=12,
                        routing=True,
                        num_routing=3)(h_i)

    text_caps = Routing(num_capsule=4,
                        l2_constant=self.l2,
                        dim_capsule=16,
                        routing=True,
                        num_routing=3)(text_caps)

    output = CapsuleNorm()(text_caps)

    model = tf.keras.Model(input_tokens, output, name='text-capsnet')

    if summary:
        model.summary()

    # compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(self.init_lr, beta_1=0.7, beta_2=0.999, amsgrad=True),
                  metrics=['accuracy'])

    return model





w2v = tf.random.normal([30_000, 300])
config = Config(pretrain_vec=w2v)
print("Hi")
model = get_model_from_text_layer(config)
# print(output_shape[1])

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from tensorflow.python.keras.regularizers import l2
from sklearn.metrics import f1_score

from config import Config
from routing import Routing, CapsuleNorm



def capsule_layer(self, input_tokens, filter_ensemble_size=3):
    if self.routing:
        use_routing = True
    else:
        use_routing = False

    # input_tokens = layers.Input((self.sequence_length,))
    embedding = layers.Embedding(self.vocab_size, self.embedding_size,
                                 weights=[self.pretrain_vec],
                                 trainable=False,
                                 mask_zero=True)(input_tokens)
    embedding = layers.Lambda(lambda x: K.expand_dims(x, axis=-1))(embedding)

    # non-linear gate layer
    elu_layer = layers.Conv2D(self.num_filter, kernel_size=(filter_ensemble_size, self.embedding_size),
                              use_bias=False,
                              kernel_regularizer=l2(self.l2), activation=None)(embedding)
    elu_layer = layers.BatchNormalization()(elu_layer)
    elu_layer = layers.Activation('elu')(elu_layer)

    conv_layer = layers.Conv2D(self.num_filter, kernel_size=(filter_ensemble_size, self.embedding_size),
                               use_bias=False,
                               kernel_regularizer=l2(self.l2), activation=None)(embedding)
    conv_layer = layers.BatchNormalization()(conv_layer)

    gate_layer = layers.Multiply()([elu_layer, conv_layer])

    # dropout
    gate_layer = layers.Dropout(self.dropout_ratio)(gate_layer)

    # convolutional capsule layer
    h_i = layers.Conv2D(128 * 8,
                        kernel_size=(K.int_shape(gate_layer)[1], 1),
                        use_bias=False,
                        kernel_regularizer=l2(self.l2), activation=None)(gate_layer)
    h_i = layers.Reshape((128, 8))(h_i)
    h_i = layers.BatchNormalization()(h_i)

    h_i = layers.Activation('relu')(h_i)

    # dropout
    h_i = layers.Dropout(self.dropout_ratio)(h_i)

    # routing algorithm
    text_caps = Routing(num_capsule=16,
                        l2_constant=self.l2,
                        dim_capsule=16,
                        routing=True,
                        num_routing=3)(h_i)
    # text_caps = Routing(num_capsule=8,
    #                     l2_constant=self.l2,
    #                     dim_capsule=12,
    #                     routing=True,
    #                     num_routing=3)(text_caps)

    text_caps = Routing(num_capsule=self.num_classes,
                        l2_constant=self.l2,
                        dim_capsule=16,
                        routing=True,
                        num_routing=3)(text_caps)

    output = CapsuleNorm()(text_caps)

    # model = tf.keras.Model(input_tokens, output, name='text-capsnet')
    #
    # if summary:
    #     model.summary()
    #
    # # compile model
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=tf.keras.optimizers.Adam(self.init_lr, beta_1=0.7, beta_2=0.999, amsgrad=True),
    #               metrics=['accuracy'])

    return output


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

# @tf.function
def ensemble_capsule_network(self):
    input_tokens = layers.Input((self.sequence_length,))
    output1 = capsule_layer(self, input_tokens, filter_ensemble_size=3)
    output2 = capsule_layer(self, input_tokens, filter_ensemble_size=4)
    output3 = capsule_layer(self, input_tokens, filter_ensemble_size=5)
    output = tf.convert_to_tensor([output1, output2, output3])
    output = tf.reduce_mean(output, axis=0)

    model = tf.keras.Model(input_tokens, output, name='text-capsnet')

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(self.init_lr, beta_1=0.7, beta_2=0.999,
                                                     amsgrad=True), metrics=['accuracy'],
                  weighted_metrics=['accuracy'])
    return model


# w2v = tf.random.normal([30_000, 300])
# config = Config(pretrain_vec=w2v)
# print("Hi")
# model = ensemble_capsule_network(config)

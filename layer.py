import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K
import scipy.special as sp


def softmax_normalized(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)


def softmax_log_normalized(x, axis=-1):
    logsum = sp.logsumexp(x, axis=axis, keepdims=True)
    return K.exp(x - logsum)


def squash_v1(x, axis=-1):
    s_square_norm = K.sum(K.square(x), axis=axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_square_norm) / (0.5 + s_square_norm)
    return tf.multiply(scale, x)


def squash_v0(x, axis=-1):
    s_square_norm = K.sum(K.square(x), axis=axis, keepdims=True)
    scale = tf.divide(s_square_norm, tf.add(1, s_square_norm))
    scaled_vector = tf.divide(x, tf.sqrt(tf.add(s_square_norm, K.epsilon)))
    return tf.multiply(scale, scaled_vector)

def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors



class primary_capsules(tf.keras.layers.Layer):

    def __init__(self, num_capsules, dim_capsule, routing, strides, padding, add_bias, num_conv_filters, name):
        self.num_conv_filters = num_conv_filters
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.routing = routing
        self.strides = strides
        self.padding = padding
        self.add_bias = add_bias
        self.name = name

    def call(self, inputs, **kwargs):
        conv_layer = tf.keras.layers.Conv2D(filters=self.num_conv_filters, padding="VALID", use_bias=False, activation='relu')

import tensorflow as tf
import keras
from keras import backend as K
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


class primary_capsules(tf.keras.layers.Layer):

    def __init__(self, num_capsules, dim_capsule, routing):
        pass

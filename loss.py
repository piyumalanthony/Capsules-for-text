import tensorflow as tf


class Margin_loss(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        loss_matrix = tf.multiply(y_true, tf.square(tf.maximum(0., 0.9 - y_pred))) + tf.multiply(0.5, tf.multiply(
            (1 - y_true), tf.square(tf.maximum(0., y_pred - 0.1))))
        return tf.reduce_mean(tf.reduce_sum(loss_matrix, axis=1))


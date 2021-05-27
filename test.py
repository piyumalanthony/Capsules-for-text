import tensorflow as tf

a = tf.Variable(tf.random.normal(shape=(36, 8, 24)))
b = tf.Variable(tf.random.normal(shape=(36, 8, 24)))
print(tf.keras.backend.batch_dot(a, b, axes=[1, 2]))


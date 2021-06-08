import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np

# a = tf.Variable(tf.random.normal(shape=(36, 8, 24)))
# b = tf.Variable(tf.random.normal(shape=(36, 8, 24)))
# print(tf.keras.backend.batch_dot(a, b, axes=[1, 1]))

# a = tf.Variable(tf.random.normal(shape=(1152, 10, 16, 8)))
# b = tf.Variable(tf.random.normal(shape=(1152, 1, 8, 1)))
# print(tf.squeeze(tf.keras.backend.batch_dot(a, b, axes=[3, 2])))
import keras.backend as K

num_capsule = 10
input_num_capsule = 25
dim_capsule = 16
input_dim_capsule = 8

# a = tf.Variable(tf.random.normal(shape=(num_capsule, input_num_capsule, dim_capsule, input_dim_capsule)))
# b = tf.Variable(tf.random.normal(shape=(64, num_capsule, input_num_capsule, input_dim_capsule)))
#
# inputs_hat = K.map_fn(lambda x: K.batch_dot(x, a, [2, 3]), elems=b)
#
# print(inputs_hat)

# a = tf.Variable(tf.random.normal(shape=(64, input_num_capsule, num_capsule, dim_capsule, input_dim_capsule)))
# b = tf.Variable(tf.random.normal(shape=(64, input_num_capsule, num_capsule, input_dim_capsule, 1)))
#
# inputs_hat = tf.matmul(a,b)
#
# print(inputs_hat)

# def test_fn():
#     for i in range(10):
#         output = i * i
#     return output
#
#
# z = test_fn()
# print(z)

# for i in range(10):
#     output = i*i
#     print(output)
#
# print("Loop executed...")
# print(output)
#
# import keras.ca
# a = [[1, 2, 3], [4, 5, 6]]
# b = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
# z0 = np.asarray(a)
# z1 = np.asarray(b)
# print(z0.shape)
# print(z1.shape)
#
# m0 = tf.Variable(initial_value=z0)
# m1 = tf.Variable(initial_value=z1)
# print(m0)
# print(m1)
# e = tf.einsum('ij,jk->jk', m0, m1)
# print(e)

#
# a = np.asarray([1, 2, 3, 4])
# b = np.asarray([1, 2, 3, 4])
# c = np.asarray([1, 2, 3, 4])
#
# z = tf.convert_to_tensor([a,b,c])
# zz = tf.cast(z, dtype=tf.float32)
# m = tf.reduce_mean(zz, axis=1)
# print(m)

# zzz = tf.Variable(tf.random.normal(shape=(64, 32, 8)))
# mmm = tf.gather(zzz, indices=[[1, 2, 3], [4, 5, 6]], axis=1)
# print(mmm.shape)

zzz = tf.cast(tf.sequence_mask([1, 2, 3, 4, 3, 2], 8), dtype=tf.float32)*1e25-1e25
print(zzz)

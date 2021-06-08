import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.regularizers import l2
import tensorflow.keras.layers as layers
from tensorflow.keras import initializers

from layer import squash_v0, squash


# class Routing(tf.keras.layers.Layer):
#
#     def __init__(self, num_capsules, dim_capsule, routing_type, num_iterations=3, l2_constant=0.0001,
#                  kernel_initializer='glorot_uniform', **kwargs):
#         super(Routing, self).__init__(**kwargs)
#         print("executed.....")
#         self.num_capsules = num_capsules
#         self.dim_capsule = dim_capsule
#         self.routing_type = routing_type
#         self.num_iterations = num_iterations
#         self.l2_constant = l2_constant
#         self.kernel_initializer = tf.initializers.get(kernel_initializer)
#
#     def build(self, input_shape):
#         self.batch_size_tile = input_shape[0]
#         self.input_num_capsules = input_shape[1]
#         self.input_dim_capsule = input_shape[2]
#         # transformation matrix for capsule conversion to high level capsules
#         self.W = self.add_weight(
#             shape=[self.batch_size_tile, self.input_num_capsules, self.num_capsules, self.dim_capsule,
#                    self.input_dim_capsule],
#             initializer=self.kernel_initializer, regularizer=l2(self.l2_constant), name='routing_weights')
#         self.built = True
#
#     def call(self, inputs, **kwargs):
#         print("executed.....")
#         weighted_sum = 0
#         input_expanded = tf.expand_dims(input=inputs, axis=2)
#         input_expanded = tf.expand_dims(input=input_expanded, axis=-1)
#         input_tiled = tf.tile(input_expanded, [1, 1, self.num_capsules, 1, 1])
#
#         predicted_capsules = tf.matmul(self.W, input_tiled)
#         b = tf.zeros(shape=[self.batch_size_tile, self.input_num_capsules, self.num_capsules, 1, 1])
#         print("Done 1")
#         for i in range(self.num_iterations):
#             c = tf.nn.softmax(b, axis=2)
#             weighted_prediction = tf.multiply(c, predicted_capsules)
#             weighted_sum = tf.reduce_sum(weighted_prediction, axis=1, keepdims=True)
#             weighted_sum = squash_v0(weighted_sum, axis=-2)
#             weighted_sum_tile = tf.tile(weighted_sum, [1, self.input_num_capsules, 1, 1, 1])
#             b += tf.multiply(predicted_capsules, weighted_sum_tile)
#         activations = tf.sqrt(tf.reduce_sum(tf.square(tf.squeeze(weighted_sum), -1)))
#         return weighted_sum, activations
#
#     def compute_output_shape(self, input_shape):
#         pass
#
#     def get_config(self):
#         pass


class CapsuleNorm(layers.Layer):
    """
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """

    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1) + K.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    # def get_config(self):
    #     config = super(Length, self).get_config()
    #     return config


class Routing(layers.Layer):

    def __init__(self, num_capsule,
                 dim_capsule,
                 routing=True,
                 num_routing=3,
                 l2_constant=0.0001,
                 kernel_initializer='glorot_uniform', **kwargs):

        super(Routing, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routing = routing
        self.num_routing = num_routing
        self.l2_constant = l2_constant
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):

        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.input_num_capsule, self.num_capsule,
                                        self.input_dim_capsule, self.dim_capsule],
                                 initializer=self.kernel_initializer,
                                 regularizer=l2(self.l2_constant),
                                 name='capsule_weight',
                                 trainable=True)
        self.built = True

    def call(self, inputs, training=True):
        # print("Routing started....")
        # print(inputs.shape)
        # print(self.W.shape)
        # inputs_expand = K.expand_dims(inputs, 1)
        # inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])

        # inputs_hat.shape = [None, num_capsule, input_num_capsule, upper capsule length]
        inputs_hat = tf.einsum('ijnm,bin->bijm', self.W, inputs)
        print("weights shape:", self.W.shape)
        print("inputs shape:", inputs.shape)
        print('input hat shape:', inputs_hat.shape)
        # dynamic routing
        if self.routing:
            b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.input_num_capsule, self.num_capsule])

            for i in range(self.num_routing):
                # c shape = [batch_size, num_capsule, input_num_capsule]
                c = tf.nn.softmax(b)

                # outputs = [batch_size, num_classes, upper capsule length]
                outputs = tf.einsum('bij,bijm->bjm', c, inputs_hat)

                outputs = squash(outputs)

                if i < self.routing - 1:
                    b += tf.einsum('bjm,bijm->bij', outputs, inputs_hat)

        # static routing
        else:
            # outputs = [batch_size, num_classes, upper capsule length]
            outputs = K.sum(inputs_hat, axis=2)
            outputs = squash(outputs)
        # print("outputs shape:", outputs.shape)
        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routing': self.routing,
            'num_routing': self.num_routing,
            'l2_constant': self.l2_constant
        }
        base_config = super(Routing, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

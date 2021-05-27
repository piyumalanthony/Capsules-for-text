import tensorflow as tf
import keras.backend as K


class Routing(tf.keras.layers.Layer):

    def __init__(self, num_capsules, dim_capsule, routing_type, num_iterations=3, l2_constant=0.0001,
                 kernel_initializer='xavier', **kwargs):
        super(Routing, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.routing_type = routing_type
        self.num_iterations = num_iterations
        self.l2_constant = l2_constant
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        self.input_num_capsules = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        # transformation matrix for capsule conversion to high level capsules
        self.W = self.add_weight(
            shape=[self.num_capsules, self.input_num_capsules, self.dim_capsule, self.input_dim_capsule])

    def call(self, inputs, **kwargs):
        pass

    def compute_output_shape(self, input_shape):
        pass

    def get_config(self):
        pass

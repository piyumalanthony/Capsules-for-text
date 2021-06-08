import tensorflow.keras.layers as layers
import tensorflow as tf

from routing import Routing


def vec_transformation_by_conv(poses, input_capsule_dim, input_capsule_num, output_capsule_dim, output_capsule_num):
    u_hat_vecs = layers.Conv1D(filters=output_capsule_dim * output_capsule_num, kernel_size=input_capsule_dim)(poses)
    u_hat_vecs = tf.reshape(u_hat_vecs, shape=(-1, input_capsule_num, output_capsule_num, output_capsule_dim))
    return u_hat_vecs


class primary_capsules(layers.Layer):

    def __init__(self, num_of_capsules=16, pose_shape=16, strides=1, padding="VALID", add_bias=False,
                 kernel_initializer='glorot_uniform',
                 name="primary_caps", **kwargs):
        super().__init__(name, **kwargs)
        # self.name = name
        self.kernel_initializer = kernel_initializer
        self.num_of_capsules = num_of_capsules
        self.add_bias = add_bias
        self.padding = padding
        self.strides = strides
        self.pose_shape = pose_shape

    def build(self, input_shape):
        # Transform matrix
        # self.W = self.add_weight(shape=[self.input_num_capsule, self.num_capsule,
        #                                 self.input_dim_capsule, self.dim_capsule],
        #                          initializer=self.kernel_initializer,
        #                          regularizer=l2(self.l2_constant),
        #                          name='capsule_weight',
        #                          trainable=True)
        self.filter1 = self.add_weight(shape=[2, 300, 1, 16], initializer=self.kernel_initializer)
        self.filter2 = self.add_weight(shape=[1, 1, 16, self.num_of_capsules * self.pose_shape],
                                       initializer=self.kernel_initializer)
        self.built = True

    def call(self, inputs, **kwargs):
        inputs_shape = inputs.get_shape()

        output1 = tf.nn.conv2d(inputs, filters=self.filter1, strides=[1, 2, 1, 1], padding=self.padding)
        output2 = tf.nn.conv2d(output1, filters=self.filter2, strides=[1, 1, 1, 1], padding=self.padding)
        output2_shape = output2.get_shape()
        output = tf.reshape(output2, shape=(-1, output2_shape[1], output2_shape[2], self.num_of_capsules,
                                            self.pose_shape))
        output_shape = output.get_shape()
        print("input shape:", inputs_shape)
        print("output shape:", output_shape)
        # u_hat_vecs = vec_transformation_by_conv(output, output_shape[-1],
        #                                         output_shape[-2], 32, 20)
        # print(inputs_shape[0])
        # print(inputs_shape[1])
        return output


class conv_capsules(layers.Layer):

    def __init__(self, shape, strides, iterations=3, name='conv_capsules', **kwargs):
        super().__init__(name, **kwargs)
        self.iterations = iterations
        # self.name = name
        self.strides = strides
        self.shape = shape

    def call(self, inputs, **kwargs):
        inputs_poses_shape = inputs.get_shape()
        hk_offsets = [
            [(h_offset + k_offset) for k_offset in range(0, self.shape[0])] for h_offset in
            range(0, inputs_poses_shape[1] + 1 - self.shape[0], self.strides[1])
        ]
        wk_offsets = [
            [(w_offset + k_offset) for k_offset in range(0, self.shape[1])] for w_offset in
            range(0, inputs_poses_shape[2] + 1 - self.shape[1], self.strides[2])
        ]
        print(hk_offsets)
        print(wk_offsets)
        inputs_poses_patches = tf.transpose(
            tf.gather(
                tf.gather(
                    inputs, hk_offsets, axis=1, name='gather_poses_height_kernel'
                ), wk_offsets, axis=3, name='gather_poses_width_kernel'
            ), perm=[0, 1, 3, 2, 4, 5, 6], name='inputs_poses_patches'
        )
        patches_for_routing_shape = inputs_poses_patches.get_shape()
        print("patches_for_routing_shape overall shape:", inputs_poses_patches.get_shape())
        inputs_poses_shape = inputs_poses_patches.get_shape()
        inputs_poses_patches = tf.reshape(inputs_poses_patches, [
            -1, patches_for_routing_shape[1]*self.shape[0] * self.shape[1] * self.shape[2], inputs_poses_shape[-1]
        ])
        # print(inputs_poses_patches.get_shape())
        # poses = Routing(16, 16, num_routing=self.iterations, routing=True)(inputs_poses_patches)
        # print(poses.get_shape())
        # flatten_capsules = tf.reshape(poses, shape=(-1, patches_for_routing_shape[1], self.shape[-1], self.shape[-2]))
        # print(flatten_capsules.get_shape())
        print('patches_for_routing_shape: ', patches_for_routing_shape[1])
        return inputs_poses_patches, patches_for_routing_shape[1]

# class flatten_capsules(layers.Layer):
#
#     def __init__(self, ):
#         pass

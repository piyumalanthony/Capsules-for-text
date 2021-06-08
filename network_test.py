from tensorflow.keras import layers
import tensorflow as tf
import tensorflow.keras.backend as K

# import network
from capsule_layers import primary_capsules, conv_capsules
from config import Config
from routing import Routing, CapsuleNorm


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
    output = primary_capsules()(embedding)
    output1, conv_capsules_remaining_shape = conv_capsules([3, 1, 16, 16], [1, 1, 1, 1], iterations=3, name='conv2')(
        output)
    # output = layers.Conv1D(filters=10, kernel_size=12, padding='VALID')(embedding)
    routed_caps = Routing(2, 16)(output1)
    caps_norm  = CapsuleNorm()(routed_caps)
    # output_conv_caps = layers.Reshape(target_shape=(-1,, 16, 16))(routed_caps)

    model = tf.keras.Model(input_tokens, caps_norm, name='text-capsnet')

    if summary:
        model.summary()

    # compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(self.init_lr, beta_1=0.7, beta_2=0.999, amsgrad=True),
                  metrics=['accuracy'])

    return model


# w2v = tf.random.normal([30_000, 300])
# config = Config(pretrain_vec=w2v)
# print("Hi")
# model = get_model_from_text_layer(config)
# # print(output_shape[1])

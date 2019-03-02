from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

import tensorflow as tf

class WassersteinSimilarity(Layer):
    def __init__(self, **kwargs):
        super(WassersteinSimilarity, self).__init__(**kwargs)

    def build(self, input_shapes):
        super(WassersteinSimilarity, self).build(input_shapes)

    def call(self, inputs):
        loc, scale = inputs

        euc = tf.einsum('ij,kj->ik', loc, loc)
        scale_sqrt = tf.linalg.sqrtm(scale)
        bures = tf.linalg.trace(
            tf.linalg.sqrtm(
                tf.einsum('ijk,lkm,imn->iljn', scale_sqrt, scale, scale_sqrt)
            )
        )
        print(bures.get_shape())

        return euc + bures

    def compute_output_shape(self, input_shapes):
        loc_shape, scale_shape = input_shapes
        batch_size = loc_shape[0]

        return tf.TensorShape([batch_size, batch_size])


from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

import tensorflow as tf

class WassersteinEmbeddingScale(Layer):
    def __init__(self, dimension, **kwargs):
        self.dimension = dimension
        super(WassersteinEmbeddingScale, self).__init__(**kwargs)

    def build(self, input_shape):
        batch_size = input_shape[0]
        feature_size = int(input_shape[1])

        self.factor_weights = self.add_weight(
            name='factor_weights',
            shape=(feature_size, self.dimension, self.dimension),
            initializer='uniform',
            trainable=True
        )

        super(WassersteinEmbeddingScale, self).build(input_shape)

    def call(self, X):
        fac = tf.einsum('ij,klm->ilm', X, self.factor_weights)
        scale = tf.matmul(fac, tf.transpose(fac, perm=[0, 2, 1]))

        return scale

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]

        return tf.TensorShape([batch_size, self.dimension, self.dimension])

class WassersteinEmbeddingLocation(Layer):
    """
    Wasserstein embedding layer. Takes an input of size [batch_size,
    feature_size] and produces a Wasserstein elliptical embedding for each
    element of the batch using dense weights.
    This takes the form of a location matrix of size [batch_size, dimension] and
    a factor tensor of size [batch_size, dimension, dimension] which generates
    a scale tensor of size [batch_size, dimension, dimension] which is symmetric in
    its last two axes.
    """

    def __init__(self, dimension, **kwargs):
        self.dimension = dimension
        super(WassersteinEmbeddingLocation, self).__init__(**kwargs)

    def build(self, input_shape):
        batch_size = input_shape[0]
        feature_size = int(input_shape[1])

        self.location_weights = self.add_weight(
            name='location_weights',
            shape=[feature_size, self.dimension],
            initializer='uniform',
            trainable=True
        )

        super(WassersteinEmbeddingLocation, self).build(input_shape)

    def call(self, X):
        return tf.einsum('ij,jk->ik', X, self.location_weights)

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]

        return tf.TensorShape([batch_size, self.dimension])


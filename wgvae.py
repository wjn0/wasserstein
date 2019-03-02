import tensorflow as tf
import numpy as np

class WGAE(object):
    def __init__(self, feature_size=4, hidden_size=32, latent_size=16):
        self.N = 100
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

    # loc: N x P (matrix of location parameters)
    # fac: N x P x P (tensor of scale-generating factor parameters)
    # out: N x N (similarity matrix)
    def _outer(self, loc, fac):
        euc = tf.einsum('ij,kj->ik', loc, loc)
        scale = tf.einsum('ijk,ikl->ijl', fac, fac)
        scale_sqrt = tf.linalg.sqrtm(scale)
        bures = tf.linalg.trace(
            tf.linalg.sqrtm(
                tf.einsum('ijk,ikl,ilm->ijm', scale_sqrt, scale, scale_sqrt)
            )
        )

        return euc + bures

    def build(self, adjacency, features):
        # self.adjacency = tf.placeholder(
        #     tf.float32,
        #     name='adjancency',
        #     shape=[None, None]
        # )
        # self.features = tf.placeholder(
        #     tf.float32,
        #     name='features',
        #     shape=[None, self.feature_size]
        # )
        adjacency = tf.constant(np.random.randn(100, 100), dtype=tf.float32)
        features = tf.constant(np.random.randn(100, 4), dtype=tf.float32)

        self.adjacency = adjacency
        self.features = features
        N = tf.shape(self.adjacency)[0]

        weights = tf.Variable(
            tf.random.uniform([self.feature_size, self.hidden_size])
        )
        convolved = tf.nn.relu(
            tf.matmul(tf.matmul(self.adjacency, self.features), weights),
            name='hidden'
        )

        loc_weights = tf.Variable(
            tf.random.uniform([self.hidden_size, self.latent_size])
        )
        scale_weights = tf.Variable(
            tf.random.uniform(
                [self.hidden_size, self.latent_size, self.latent_size]
            )
        )
        
        self.z_loc = tf.matmul(
            self.adjacency, tf.matmul(convolved, loc_weights)
        )
        self.z_fac = tf.einsum(
            'ij,jkl->ikl',
            self.adjacency,
            tf.einsum('ij,jkl->ikl', convolved, scale_weights)
        )

        outer = self._outer(self.z_loc, self.z_fac)

        self.recon_loss = tf.keras.metrics.MeanSquaredError()
        self.recon_loss.update_state(
            self.adjacency,
            outer
        )
        self.train_op = tf.keras.optimizers.Adam(
            learning_rate=1e-2
        ).minimize(self.recon_loss, [weights, loc_weights, scale_weights])


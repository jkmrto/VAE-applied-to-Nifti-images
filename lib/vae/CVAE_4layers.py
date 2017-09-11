from lib.vae.CVAE import CVAE
import tensorflow as tf
import numpy as np
import lib.neural_net.kfrans_ops as ops


class CVAE_4layers(CVAE):
    def __init__(self, hyperparams, test_bool=False,
                 path_to_session=None):
        super(CVAE_4layers, self).__init__(hyperparams, test_bool=test_bool,
                                           path_to_session=path_to_session)

    def recognition(self, input_images):
        print("Son recognition module")

        with tf.variable_scope("recognition"):
            h1 = self.activation_layer(ops.conv3d(
                x=input_images,
                input_features=self.features_depth[0],
                output_features=self.features_depth[1],
                stride=self.stride,
                kernel=self.kernel_size,
                name="first_layer"))  # 28x28x1 -> 14x14x16

            self.dim_out_first_layer = tf.shape(h1)

            h2 = self.activation_layer(ops.conv3d(
                x=h1,
                input_features=self.features_depth[1],
                output_features=self.features_depth[2],
                stride=self.stride,
                kernel=self.kernel_size,
                name="second_layer"))  # 14x14x16 -> 7x7x32

            self.dim_out_second_layer = tf.shape(h2)

            h3 = self.activation_layer(ops.conv3d(
                x=h2,
                input_features=self.features_depth[2],
                output_features=self.features_depth[3],
                stride=self.stride,
                kernel=self.kernel_size,
                name="third_layer"))

            self.dim_out_third_layer = tf.shape(h3)

            h4 = self.activation_layer(ops.conv3d(
                x=h3,
                input_features=self.features_depth[3],
                output_features=self.features_depth[4],
                stride=self.stride,
                kernel=self.kernel_size,
                name="forth_layer"))

            self.dim_out_forth_layer = tf.shape(h4)

            self.input_dense_layer_dim = \
                [-1, np.array(h4.get_shape().as_list()[1:]).prod()]

            h4_flat = tf.reshape(h4, self.input_dense_layer_dim)

            w_mean = ops.dense(h4_flat, self.input_dense_layer_dim[1],
                               self.n_z, "w_mean")
            w_stddev = ops.dense(h4_flat, self.input_dense_layer_dim[1],
                                 self.n_z, "w_stddev")

        return w_mean, w_stddev

    def generation(self, z, reuse_bool):
        """

        :param z: latent layer input
        :param reuse_bool:
        :return:
        """
        with tf.variable_scope("generation", reuse=reuse_bool):
            # dense layer
            g_dense_flat_1 = self.activation_layer(ops.dense(z, self.n_z,
                                       self.input_dense_layer_dim[1],
                                       scope='z_matrix'))

            g_dense_reshape_1 = \
                tf.reshape(g_dense_flat_1, self.dim_out_third_layer)

            g4 = self.activation_layer(ops.conv3d_transpose(
                x=g_dense_reshape_1,
                output_shape=self.dim_out_second_layer,
                output_features=self.features_depth[3],
                input_features=self.features_depth[4],
                stride=self.stride,
                kernel=self.kernel_size,
                name="g4"))

            g3 = self.activation_layer(ops.conv3d_transpose(
                x=g4,
                output_shape=self.dim_out_first_layer,
                output_features=self.features_depth[2],
                input_features=self.features_depth[3],
                stride=self.stride,
                kernel=self.kernel_size,
                name="g3"))

            g2 = self.activation_layer(ops.conv3d_transpose(
                x=g3,
                output_shape=self.dim_out_first_layer,
                output_features=self.features_depth[1],
                input_features=self.features_depth[2],
                stride=self.stride,
                kernel=self.kernel_size,
                name="g2"))

            g1 = ops.conv3d_transpose(x=g2,
                                      output_shape=self.dim_in_first_layer,
                                      output_features=self.features_depth[0],
                                      input_features=self.features_depth[1],
                                      stride=self.stride,
                                      kernel=self.kernel_size,
                                      name="g1")
            g1_activated = tf.nn.sigmoid(g1)

        return g1_activated

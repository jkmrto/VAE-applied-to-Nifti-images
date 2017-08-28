from lib.vae.CVAE import CVAE
import tensorflow as tf
import numpy as np
import lib.neural_net.kfrans_ops as ops


class CVAE_3layers(CVAE):

    def __init__(self, hyperparams, test_bool=False,
                 path_to_session=None):

        super(CVAE_3layers, self).__init__(hyperparams, test_bool=test_bool,
                 path_to_session=path_to_session)

    def recognition(self, input_images):
        print("Son recognition module")

        with tf.variable_scope("recognition"):
            h1 = self.activation_layer(ops.conv3d(
                x=input_images,
                input_features=self.features_depth[0],
                output_features=self.features_depth[1],
                stride=2,
                kernel=self.kernel_size,
                name="first_layer"))  # 28x28x1 -> 14x14x16

            self.dim_out_first_layer = tf.shape(h1)

            h2 = self.activation_layer(ops.conv3d(
                x=h1,
                input_features=self.features_depth[1],
                output_features=self.features_depth[2],
                stride=2,
                kernel=self.kernel_size,
                name="second_layer"))  # 14x14x16 -> 7x7x32

            self.dim_out_second_layer = tf.shape(h2)

            h3 = self.activation_layer(ops.conv3d(
                x=h2,
                input_features=self.features_depth[2],
                output_features=self.features_depth[3],
                stride=2,
                kernel=self.kernel_size,
                name="third_layer"))  # 14x14x16 -> 7x7x32

            self.dim_out_third_layer = tf.shape(h2)

            self.input_dense_layer_dim = [-1, np.array(h3.get_shape().as_list()[1:]).prod()]

            h3_flat = tf.reshape(h3, self.input_dense_layer_dim)

            w_mean = ops.dense(h3_flat, self.input_dense_layer_dim[1],
                               self.n_z, "w_mean")
            w_stddev = ops.dense(h3_flat, self.input_dense_layer_dim[1],
                                 self.n_z, "w_stddev")

        return w_mean, w_stddev

    def generation(self, z, reuse_bool):
        with tf.variable_scope("generation", reuse=reuse_bool):
            z_develop = ops.dense(z, self.n_z, self.input_dense_layer_dim[1],
                                  scope='z_matrix')

            z_matrix = self.activation_layer(
                tf.reshape(z_develop, self.dim_out_third_layer))

            h1 = self.activation_layer(ops.conv3d_transpose(
                x=z_matrix,
                output_shape=self.dim_out_second_layer,
                output_features=self.features_depth[2],
                input_features=self.features_depth[3],
                stride=2,
                kernel=self.kernel_size,
                name="g_h1"))

            h2 = self.activation_layer(ops.conv3d_transpose(
                x=h1,
                output_shape=self.dim_out_first_layer,
                output_features=self.features_depth[1],
                input_features=self.features_depth[2],
                stride=2,
                kernel=self.kernel_size,
                name="g_h2"))

            h3 = ops.conv3d_transpose(x=h2,
                                      output_shape=self.dim_in_first_layer,
                                      output_features=self.features_depth[0],
                                      input_features=self.features_depth[1],
                                      stride=2,
                                      kernel=self.kernel_size,
                                      name="g_h3")
            h3 = tf.nn.sigmoid(h3)

        return h3
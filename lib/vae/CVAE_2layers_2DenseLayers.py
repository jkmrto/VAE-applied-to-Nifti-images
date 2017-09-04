from lib.vae.CVAE import CVAE
import tensorflow as tf
import numpy as np
import lib.neural_net.kfrans_ops as ops


class CVAE_2layers_DenseLayer(CVAE):

    def __init__(self, hyperparams, test_bool=False, meta_path=None,
                 path_to_session=None):
        super(CVAE_2layers_DenseLayer, self).__init__(
            hyperparams, test_bool=test_bool,
            path_to_session=path_to_session)

    def recognition(self, input_images):
        print("Son recognition module")

        self.middle_dense_layer_dim = 300

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
                name="second_layers"))  # 14x14x16 -> 7x7x32

            self.dim_out_second_layer = tf.shape(h2)

            self.input_dense_layer_dim = [
                -1, np.array(h2.get_shape().as_list()[1:]).prod()]

            h2_flat_layer = tf.reshape(h2, self.input_dense_layer_dim)

            dense_layer = self.activation_layer(ops.dense(
                x=h2_flat_layer,
                input_len=self.input_dense_layer_dim[1],
                output_len=self.middle_dense_layer_dim,
                scope="Middle_Dense_Layer",
                with_w=False,
                reuse=False))

            w_mean = ops.dense(dense_layer, self.middle_dense_layer_dim,
                               self.n_z, "w_mean")
            w_stddev = ops.dense(dense_layer, self.middle_dense_layer_dim,
                                 self.n_z, "w_stddev")

        return w_mean, w_stddev

    def generation(self, z, reuse_bool):
        with tf.variable_scope("generation", reuse=reuse_bool):
            dense_first_layer = self.activation_layer(
                ops.dense(x=z, input_len=self.n_z,
                          output_len=self.middle_dense_layer_dim,
                          scope='z_first_dense_layer'))

            dense_second_layer = self.activation_layer(
                ops.dense(x=dense_first_layer,
                          input_len=self.middle_dense_layer_dim,
                          output_len=self.input_dense_layer_dim[1],
                          scope='z_second_dense_layer'))

            z_matrix = self.activation_layer(
                tf.reshape(dense_second_layer, self.dim_out_second_layer))

            h1 = self.activation_layer(ops.conv3d_transpose(
                x=z_matrix,
                output_shape=self.dim_out_first_layer,
                output_features=self.features_depth[1],
                input_features=self.features_depth[2],
                stride=self.stride,
                kernel=self.kernel_size,
                name="g_h1"))

            h2 = ops.conv3d_transpose(x=h1,
                                      output_shape=self.dim_in_first_layer,
                                      output_features=self.features_depth[0],
                                      input_features=self.features_depth[1],
                                      stride=self.stride,
                                      kernel=self.kernel_size,
                                      name="g_h2")
            h2 = tf.nn.sigmoid(h2)

        return h2

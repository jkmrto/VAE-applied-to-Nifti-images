import tensorflow as tf
from lib import session_helper
from cvae_hub import cvae_3d
from lib import kfrans_ops
import numpy as np
from lib import loss_function as loss
from lib.test_over_segmenting_regions import load_regions_segmented
from lib.math_utils import sample_gaussian


class cvae_two_layers(cvae_3d):

    def __init__(self):
        super().__init__()

        self.n_z = 100
        self.batchsize = 100
        self.input_shape = [34, 42, 41]
        self.filter_per_layer = [16, 32]
        self.stride = 4
        self.learning_rate = 0.001
        self.kernel_size = 8
        self.activation = kfrans_ops.lrelu

    def _build_graph(self):

        self.x_in = tf.placeholder(tf.float32, [None, self.input_shape[0],
                                                self.input_shape[1],
                                                self.input_shape[2]])

        x_with_depth_layer = tf.reshape(self.x_in, [-1,
                                                    self.input_shape[0],
                                                    self.input_shape[1],
                                                    self.input_shape[2],
                                                    1])
        h1 = self.activation(kfrans_ops.conv3d(
            x=x_with_depth_layer,
            input_features=1,
            output_features=self.filter_per_layer[0],
            stride=2,
            name="first_layer",
            kernel_size=self.kernel_size))  # 28x28x1 -> 14x14x16

        h2 = self.activation(kfrans_ops.conv3d(
            x=h1,
            input_features=self.filter_per_layer[0],
            output_features=self.filter_per_layer[1],
            stride=self.stride,
            name="second_layers",
            kernel_size=self.kernel_size))  # 14x14x16 -> 7x7x32

        print(h2.get_shape().as_list())
        h2_size = h2.get_shape().as_list()
        h2_size_with_undefined_first_layer = [-1]
        h2_size_with_undefined_first_layer.extend(h2_size)
        total_size = np.array(h2_size)[1:].prod()
        print(total_size)

        h2_flat = tf.reshape(h2, [-1, total_size])

        z_mean = kfrans_ops.dense(h2_flat, input_len=total_size,
                                  output_len=self.n_z,
                                  scope="w_mean")
        z_stddev = kfrans_ops.dense(h2_flat, input_len=total_size,
                                    output_len=self.n_z,
                                    scope="w_stddev")

        # samples = tf.random_normal([None, self.n_z], 0, 1,
        #                           dtype=tf.float32)

        z = sample_gaussian(z_mean, z_stddev)
        # z = z_mean + (z_stddev * samples)

        z_develop_flatten = kfrans_ops.dense(z, input_len=self.n_z,
                                             output_len=total_size,
                                             scope='z_matrix')
        print(z_develop_flatten.get_shape())
        z_develop_matrix = tf.reshape(z_develop_flatten,
                                                 tf.shape(h2))

        print(z_develop_matrix.get_shape())
        g1 = self.activation(kfrans_ops.conv3d_transpose(
            x=z_develop_matrix,
            output_shape=tf.shape(h1),
            input_features=self.filter_per_layer[-1],
            output_features=self.filter_per_layer[-2],
            stride=self.stride,
            name="g_h1"))

        print(g1.get_shape())
        g2 = self.activation(kfrans_ops.conv3d_transpose(
            x=g1,
            output_shape=tf.shape(x_with_depth_layer),
            input_features=self.filter_per_layer[0],
            output_features=1,
            stride=self.stride,
            name="g_h2"))

        #images_out = tf.nn.sigmoid(g2)

        x_in_flatten = tf.reshape(self.x_in,
                                  [-1, np.array(self.input_shape).prod()])
        print(x_in_flatten.get_shape())
        print(g2.get_shape())
        x_out_flatten = tf.reshape(g2, tf.shape(x_in_flatten))

        generation_loss = tf.reduce_mean(-tf.reduce_sum(
            x_in_flatten * tf.log(1e-8 + x_out_flatten) + (1 - x_in_flatten) *
            tf.log(1e-8 + 1 - x_out_flatten), 1))

        latent_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(z_mean) +
                                               tf.square(z_stddev)
                                               - tf.log(tf.square(z_stddev)) - 1, 1))
        self.cost = tf.reduce_mean(generation_loss + latent_loss)

       # cost = self.__build_cost_estimate(x_out_flatten, x_in_flatten, z_mean, z_stddev)

        global_step = tf.Variable(0, trainable=False)
    #    with tf.name_scope("Adam_optimizer"):
    #        optimizer = tf.train.AdamOptimizer(self.learning_rate)
    #        tvars = tf.trainable_variables()
    #        print(tvars)
    #        grads_and_vars = optimizer.compute_gradients(cost, tvars)
    #        clipped = [(tf.clip_by_value(grad, -5, 5), tvar)  # gradient clipping
    #                   for grad, tvar in grads_and_vars]
    #        train_op = optimizer.apply_gradients(clipped, global_step=global_step,
     #                                            name="minimize_cost")

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        return self.x_in, z_mean, z_stddev, g2, self.cost, global_step, \
               generation_loss, latent_loss

    def __build_cost_estimate(self, x_reconstructed, x_in, z_mean, z_log_sigma):

        # reconstruction loss: mismatch b/w x & x_reconstructed
        # binary cross-entropy -- assumes x & p(x|z) are iid Bernoullis
        rec_loss = loss.crossEntropy(x_reconstructed, x_in)

        # Kullback-Leibler divergence: mismatch b/w approximate vs. imposed/true posterior
        kl_loss = loss.kullbackLeibler(z_mean, z_log_sigma)

     #   with tf.name_scope("l2_regularization"):
     #       regularizers = [tf.nn.l2_loss(var) for var in self.session.graph.get_collection(
     #           "trainable_variables") if "weights" in var.name]
     #       l2_reg = self.hyper_params['lambda_l2_reg'] * tf.add_n(regularizers)

        with tf.name_scope("cost"):
            # average over minibatch
            cost = tf.reduce_mean(rec_loss + kl_loss, name="vae_cost")
           # cost += l2_reg

        return cost



regions_used = "three"
list_regions = session_helper.select_regions_to_evaluate(regions_used)
region = 3
region_segmented = load_regions_segmented(list_regions)[3]
print(region_segmented.shape)
cvae = cvae_two_layers()
print("training")
cvae.train(x_in=region_segmented)

import os
import numpy as np
from lib import kfrans_ops
import tensorflow as tf
from lib import loss_function as loss
from lib import session_helper
from lib.aux_functionalities.functions import get_batch_from_samples_unsupervised_3d
from lib.aux_functionalities.os_aux import create_directories
from lib.test_over_segmenting_regions import load_regions_segmented
from settings import path_to_project
from lib.math_utils import sample_gaussian


class cvae_3d():
    """
    Juan Carlos Martinez de la Torre
    """

    RESTORE_KEY = "restore"

    def __init__(self, architecture=None, hyperparams=None, meta_graph=None,
                 path_to_session=None, test_bool=False):

        """(Re)build a symmetric VAE model with given:

            * architecture (list of nodes per encoder layer); e.g.
               [1000, 500, 250, 10] specifies a VAE with 1000-D inputs, 10-D latents,
               & end-to-end architecture [1000, 500, 250, 10, 250, 500, 1000]

            * hyperparameters (optional dictionary of updates to `DEFAULTS`)
        """

        self.session = tf.Session()
        self.hyper_params = hyperparams
        self.path_session_folder = path_to_session

        self.n_hidden = 500
        self.n_z = 1000
        self.batchsize = 100
        self.input_shape = [34, 42, 41]
        self.filter_per_layer = [16, 32, 64]
        self.stride = 2
        self.learning_rate = 0.00001
        self.kernel_size = 2
        self.activation = tf.nn.sigmoid


        # summaryWritter under testing

        logs_path = os.path.join(path_to_project, "cvae_logs")

        if not meta_graph:  # new model
            self.architecture = architecture

            if test_bool:
                print("Hyperparamers indicated: " + str(self.hyper_params))

            # path_to_session should be indicated if we want to create data
            # associated to the session such as the logs, and metagraphs
            # generated by sensor flow. It it is just a test session, in order
            # to test a feature, it is not necessary to indicate the path
            if None is not self.path_session_folder:
                self.init_session_folders()

            # build graph
            handles = self._build_graph()
            for handle in handles:
                tf.add_to_collection(cvae_3d.RESTORE_KEY, handle)
            self.session.run(tf.global_variables_initializer())

        else:  # restore saved model
            tf.train.import_meta_graph(meta_graph + ".meta").restore(
                self.session, meta_graph)
            handles = self.session.graph.get_collection_ref(cvae_3d.RESTORE_KEY)

        self.x_in, self.z_mean, self.z_stddev, self.images_out, self.cost,\
            self.global_step, self.generation_loss, self.latent_loss= handles[0:8]


        self.writer = tf.summary.FileWriter(logs_path, graph=self.session.graph)

    def init_session_folders(self):
        """
        This method will create inside the "out" folder a folder with the datetime
        of the execution of the neural net and with, with 3 folders inside it
        :return:
        """
        self.path_to_images = os.path.join(self.path_session_folder, "images")
        self.path_to_logs = os.path.join(self.path_session_folder, "logs")
        self.path_to_meta = os.path.join(self.path_session_folder, "meta")
        self.path_to_grad_desc_error = os.path.join(self.path_to_logs,
                                                    "DescGradError")

        create_directories([self.path_session_folder, self.path_to_images,
                            self.path_to_logs, self.path_to_meta])

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

        h3 = self.activation(kfrans_ops.conv3d(
            x=h2,
            input_features=self.filter_per_layer[1],
            output_features=self.filter_per_layer[2],
            stride=self.stride,
            name="three_layers",
            kernel_size=self.kernel_size))

        print(h3.get_shape().as_list())
        h3_size = h2.get_shape().as_list()
        total_size = np.array(h3_size)[1:].prod()
        print(total_size)

        h3_flat = tf.reshape(h3, [-1, total_size])

        z_mean = kfrans_ops.dense(h3_flat, input_len=total_size,
                                  output_len=self.n_z,
                                  scope="w_mean")
        z_stddev = kfrans_ops.dense(h3_flat, input_len=total_size,
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
                                      tf.shape(h3))

        print(z_develop_matrix.get_shape())
        g1 = self.activation(kfrans_ops.conv3d_transpose(
            x=z_develop_matrix,
            output_shape=tf.shape(h2),
            input_features=self.filter_per_layer[-1],
            output_features=self.filter_per_layer[-2],
            stride=self.stride,
            name="g_h1"))

        print(g1.get_shape())
        g2 = self.activation(kfrans_ops.conv3d_transpose(
            x=g1,
            output_shape=tf.shape(h1),
            input_features=self.filter_per_layer[-2],
            output_features=self.filter_per_layer[-3],
            stride=self.stride,
            name="g_h2"))

        g3 = self.activation(kfrans_ops.conv3d_transpose(
            x=g2,
            output_shape=tf.shape(x_with_depth_layer),
            input_features=self.filter_per_layer[-3],
            output_features=1,
            stride=self.stride,
            name="g_h3"))

        # images_out = tf.nn.sigmoid(g2)

        x_in_flatten = tf.reshape(self.x_in,
                                  [-1, np.array(self.input_shape).prod()])
        print(x_in_flatten.get_shape())
        print(g2.get_shape())
        x_out_flatten = tf.reshape(g3, tf.shape(x_in_flatten))

        generation_loss = tf.reduce_mean(-tf.reduce_sum(
            x_in_flatten * tf.log(1e-8 + x_out_flatten) + (1 - x_in_flatten) *
            tf.log(1e-8 + 1 - x_out_flatten), 1))

        latent_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(z_mean) +
                                                         tf.square(z_stddev)
                                                         - tf.log(
            tf.square(z_stddev)) - 1, 1))
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

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.cost)

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

    def train(self, x_in, bool_save_meta=False, max_iter=1000):

        n_samples = x_in.shape[0]
        # train
        if bool_save_meta:
            saver = tf.train.Saver(max_to_keep=2)

        i = 0
        while True:

            batch = get_batch_from_samples_unsupervised_3d(
                x_in, 16)

            feed_dict = {self.x_in: batch}
            fetches = [self.cost, self.generation_loss, self.latent_loss]

            [cost, generation_loss, latent_loss] =  \
                self.session.run(fetches, feed_dict=feed_dict)

            # dumb hack to print cost every epoch
            if i % 20 == 0:
                print("iter {0}, error {1}, GEN: {2}, LATENT {3}".format(
                    i, cost, generation_loss, latent_loss))

            if bool_save_meta:
                saver.save(self.session, os.getcwd() + "/training/train",
                            global_step=i)
            i = i +1

            if i >= max_iter:
                self.writer.close()
                break


regions_used = "three"
list_regions = session_helper.select_regions_to_evaluate(regions_used)
region = 3
region_segmented = load_regions_segmented(list_regions)[3]
print(region_segmented.shape)
cvae = cvae_3d()
print("training")
cvae.train(x_in=region_segmented)



class cvae_three_layers(cvae_3d):

    def __init__(self):
        super().__init__([16, 32, 64]
)

        self.n_z = 100
        self.batchsize = 100
        self.input_shape = [34, 42, 41]
        self.stride = 2
        self.learning_rate = 0.001
        self.kernel_size = 8
        self.activation = kfrans_ops.lrelu




regions_used = "three"
list_regions = session_helper.select_regions_to_evaluate(regions_used)
region = 3
region_segmented = load_regions_segmented(list_regions)[3]
print(region_segmented.shape)
cvae = cvae_three_layers()
print("training")
cvae.train(x_in=region_segmented)

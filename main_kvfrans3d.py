import os
import nibabel as nib
import numpy as np
import tensorflow as tf
from scipy.misc import imsave as ims
from lib import session_helper
from lib.test_over_segmenting_regions import load_regions_segmented
import lib.kfrans_ops as ops
from lib.aux_functionalities.functions import \
    get_batch_from_samples_unsupervised_3d
from lib import utils

import settings

path_to_nii_output = os.path.join(settings.path_to_project, "test_over_cvae")


def from_3d_image_to_nifti_file(path_to_save, image3d):
    img = nib.Nifti1Image(image3d, np.eye(4))
    img.to_filename("{}.nii".format(path_to_save))

bool_save_meta = False

class LatentAttention():
    def __init__(self, hyperparams):

        self.session = tf.Session()

        self.n_z = hyperparams['latent_layer_dim']
        self.lambda_l2_reg = hyperparams['lambda_l2_regularization']
        self.learning_rate = hyperparams['learning_rate']
        self.features_depth = hyperparams['features_depth']
        self.image_shape = hyperparams['image_shape']
        self.total_size = hyperparams['total_size']
        self.kernel_size = hyperparams['kernel_size']
        self.activation_layer = hyperparams['activation_layer']

        self.dim_in_first_layer = None
        self.dim_out_first_layer = None
        self.dim_out_second_layer = None
        self.input_dense_layer_dim = None

        self.decay_rate = tf.placeholder_with_default(
            float(hyperparams['decay_rate']), shape=[],  name="decay_rate")

        self.images = tf.placeholder(tf.float32, [None, self.total_size])
        image_matrix = tf.reshape(self.images,
                                [-1, self.image_shape[0], self.image_shape[1],
                                self.image_shape[2], 1])
        self.dim_in_first_layer = tf.shape(image_matrix)

        z_mean, z_stddev = self.recognition(image_matrix)
        samples = tf.random_normal(tf.shape(z_mean), 0, 1,
                                   dtype=tf.float32)
        guessed_z = z_mean + (z_stddev * samples)

        self.generated_images = self.generation(guessed_z)
        generated_flat = tf.reshape(self.generated_images,
                                    tf.shape(self.images))

        self.cost = self.__cost_calculation(generated_flat, z_mean, z_stddev)

        self.global_step = tf.Variable(0, trainable=False)

        self.temp_learning_rate = self.learning_rate * \
            tf.exp(- tf.multiply(tf.cast(self.global_step, tf.float32),
                                 self.decay_rate))

        self.optimizer = tf.train.AdamOptimizer(
            self.temp_learning_rate).minimize(
            self.cost, global_step=self.global_step)

        self.session.run(tf.initialize_all_variables())


    def __cost_calculation(self, images_reconstructed, z_mean, z_stddev):
        self.generation_loss = -tf.reduce_sum(
            self.images * tf.log(1e-8 + images_reconstructed) + (
                1 - self.images) * tf.log(1e-8 + 1 - images_reconstructed), 1)

        self.latent_loss = 0.5 * tf.reduce_sum(
            tf.square(z_mean) + tf.square(z_stddev) - tf.log(
                tf.square(z_stddev)) - 1, 1)
        cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
        # self

        if self.lambda_l2_reg != 0:
            with tf.name_scope("l2_regularization"):
                regularizers = [tf.nn.l2_loss(var) for var in
                                self.session.graph.get_collection(
                                    "trainable_variables") if
                                "weights" in var.name]

                l2_reg = self.lambda_l2_reg * tf.add_n(regularizers)

                cost += l2_reg


        return cost

    # encoder
    def recognition(self, input_images):
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
                name="second_layers"))  # 14x14x16 -> 7x7x32

            self.dim_out_second_layer = tf.shape(h2)

            self.input_dense_layer_dim = [
                -1, np.array(h2.get_shape().as_list()[1:]).prod()]
            h2_flat = tf.reshape(h2, self.input_dense_layer_dim)

            w_mean = ops.dense(h2_flat, self.input_dense_layer_dim[1], self.n_z,
                               "w_mean")
            w_stddev = ops.dense(h2_flat, self.input_dense_layer_dim[1],
                                 self.n_z,
                                 "w_stddev")

        return w_mean, w_stddev

    # decoder
    def generation(self, z):
        with tf.variable_scope("generation"):
            z_develop = ops.dense(z, self.n_z, self.input_dense_layer_dim[1],
                                  scope='z_matrix')

            z_matrix = self.activation_layer(
                tf.reshape(z_develop, self.dim_out_second_layer))

            h1 = self.activation_layer(ops.conv3d_transpose(
                x=z_matrix,
                output_shape=self.dim_out_first_layer,
                output_features=self.features_depth[1],
                input_features=self.features_depth[2],
                stride=2,
                kernel=self.kernel_size,
                name="g_h1"))

            h2 = ops.conv3d_transpose(x=h1,
                                      output_shape=self.dim_in_first_layer,
                                      output_features=self.features_depth[0],
                                      input_features=self.features_depth[1],
                                      stride=2,
                                      kernel=self.kernel_size,
                                      name="g_h2")
            h2 = tf.nn.sigmoid(h2)

        return h2

    def train(self, X, n_iters=1000, batchsize=10):

        for iter in range(n_iters):

            batch_images = get_batch_from_samples_unsupervised_3d(
                X, batch_size=batchsize)
            batch_flat = np.reshape(batch_images,
                               [batch_images.shape[0], self.total_size])

            feed_dict = {self.images: batch_flat}
            _, gen_loss, lat_loss, global_step, learning_rate = \
                self.session.run(
                    (self.optimizer, self.generation_loss, self.latent_loss,
                     self.global_step, self.temp_learning_rate),
                    feed_dict=feed_dict)
            # dumb hack to print cost every epoch

            if iter % 10  == 0:
                print("iter %d: genloss %f latloss %f learning_rate %f"  % (
                    iter, np.mean(gen_loss), np.mean(lat_loss), learning_rate))

            if iter % 10 == 0:
                feed_dict = {self.images:batch_flat[0:2, :]}
                generated_test = self.session.run(self.generated_images[1, :],
                                                  feed_dict=feed_dict)

                image_3d = np.reshape(generated_test, self.image_shape)
                image_3d = image_3d.astype(float)
                file_path = os.path.join(path_to_nii_output,
                                         "epoc_{}".format(iter))
                from_3d_image_to_nifti_file(file_path, image_3d)


regions_used = "all"
region_selected = 40
list_regions = session_helper.select_regions_to_evaluate(regions_used)
train_images = load_regions_segmented(list_regions)[region_selected]

hyperparams = {}
hyperparams['latent_layer_dim'] = 100
hyperparams['kernel_size'] = 5
hyperparams['features_depth'] = [1, 16, 32]
hyperparams['image_shape'] = train_images.shape[1:]
hyperparams['activation_layer'] = ops.lrelu
hyperparams['total_size'] = np.array(train_images.shape[1:]).prod()
hyperparams['decay_rate'] = 0.0002
hyperparams['learning_rate'] = 0.001
hyperparams['lambda_l2_regularization'] = 0.0001

model = LatentAttention(hyperparams=hyperparams)
model.train(X=train_images, n_iters=1000, batchsize=50)

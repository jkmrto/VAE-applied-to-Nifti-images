import os
import nibabel as nib
import numpy as np
import tensorflow as tf
from scipy.misc import imsave as ims
from lib import session_helper
from lib.test_over_segmenting_regions import load_regions_segmented
import lib.kfrans_ops as ops
from lib import utils

import settings

path_to_nii_output = os.path.join(settings.path_to_project, "test_over_cvae")

def from_3d_image_to_nifti_file(path_to_save, image3d):
    img = nib.Nifti1Image(image3d, np.eye(4))
    img.to_filename("{}.nii".format(path_to_save))

bool_save_meta = False
regions_used = "three"
list_regions = session_helper.select_regions_to_evaluate(regions_used)
region = 3
kernel = 5
train_images = load_regions_segmented(list_regions)[3]
input_shape = [34, 42, 41]
total_size = 34 * 42 * 41
activation_layer = ops.lrelu

class LatentAttention():
    def __init__(self):


        self.train_images = train_images.reshape(
            train_images.shape[0], 34, 42, 41, 1)
        self.n_samples = self.train_images.shape[0]

        self.n_hidden = 500
        self.n_z = 1000
        self.batchsize = 100

        self.images = tf.placeholder(tf.float32, [None, total_size])
        image_matrix = tf.reshape(self.images, [self.batchsize, 34, 42, 41, 1])

        z_mean, z_stddev = self.recognition(image_matrix)
        samples = tf.random_normal([self.batchsize, self.n_z], 0, 1, dtype=tf.float32)
        guessed_z = z_mean + (z_stddev * samples)

        self.generated_images = self.generation(guessed_z)
        generated_flat = tf.reshape(self.generated_images, [self.batchsize, total_size])

        self.generation_loss = -tf.reduce_sum(self.images * tf.log(1e-8 + generated_flat) + (1-self.images) * tf.log(1e-8 + 1 - generated_flat),1)

        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1)
        self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)

        self.optimizer = tf.train.AdamOptimizer(0.00001).minimize(self.cost)

    # encoder
    def recognition(self, input_images):
        with tf.variable_scope("recognition"):
            h1 = activation_layer(ops.conv3d(
                x=input_images,
                input_features=1,
                output_features=16,
                stride=2,
                kernel=kernel,
                name="first_layer")) # 28x28x1 -> 14x14x16
            print(h1.shape)
            h2 = activation_layer(ops.conv3d(
                x=h1,
                input_features=16,
                output_features=32,
                stride=2,
                kernel=kernel,
                name="second_layers")) # 14x14x16 -> 7x7x32

            print(h2.shape)
            h2_flat = tf.reshape(h2, [self.batchsize, 9*11*11*32])

            w_mean = ops.dense(h2_flat, 9 * 11 * 11 * 32, self.n_z, "w_mean")
            w_stddev = ops.dense(h2_flat, 9 * 11 * 11 * 32, self.n_z, "w_stddev")

        return w_mean, w_stddev

    # decoder
    def generation(self, z):
        with tf.variable_scope("generation"):
            z_develop = ops.dense(z, self.n_z, 9 * 11 * 11 * 32, scope='z_matrix')
            z_matrix = tf.nn.relu(tf.reshape(z_develop, [-1, 9, 11, 11, 32]))
            h1 = activation_layer(ops.conv3d_transpose(
                x = z_matrix,
                output_shape = [self.batchsize, 17, 21, 21, 16],
                output_features=16,
                input_features=32,
                stride=2,
                kernel=kernel,
                name="g_h1"))

            h2 = ops.conv3d_transpose(x = h1,
                                      output_shape = [self.batchsize, 34, 42, 41, 1],
                                      output_features=1,
                                      input_features=16,
                                      stride=2,
                                      kernel=kernel,
                                      name="g_h2")
            h2 = tf.nn.sigmoid(h2)

        print("hola")
        return h2

    def train(self):

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for epoch in range(100):
                for idx in range(int(self.n_samples / self.batchsize)):
                    batch_images = train_images[0:self.batchsize, :, :, :]
                    batch = np.reshape(batch_images, [batch_images.shape[0], 34*42*41])

                    _, gen_loss, lat_loss = sess.run((self.optimizer, self.generation_loss, self.latent_loss), feed_dict={self.images: batch})
                    # dumb hack to print cost every epoch
                    if idx % (self.n_samples - 3) == 0:
                       print ("epoch %d: genloss %f latloss %f" % (epoch, np.mean(gen_loss), np.mean(lat_loss)))

                if epoch % 10 == 0:
                    generated_test = sess.run(self.generated_images, feed_dict={ self.images:batch})
                    generated_test = generated_test[0,:]
                    image_3d = np.reshape(generated_test, [34, 42, 41])
                    image_3d = image_3d.astype(float)
                    file_path = os.path.join(path_to_nii_output,
                                             "epoc_{}".format(epoch))
                    from_3d_image_to_nifti_file(file_path, image_3d)

model = LatentAttention()
model.train()

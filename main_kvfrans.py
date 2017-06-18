import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
from scipy.misc import imsave as ims
import utils as utils
import ops as ops


bool_save_meta = False

class LatentAttention():
    def __init__(self):
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.train_images = self.mnist.train.images.reshape(
            self.mnist.train.images.shape[0],28,28,1)
        self.n_samples = self.train_images.shape[0]

        self.n_hidden = 500
        self.n_z = 20
        self.batchsize = 100

        self.images = tf.placeholder(tf.float32, [None, 784])
        image_matrix = tf.reshape(self.images, [self.batchsize, 28, 28, 1])

        z_mean, z_stddev = self.recognition(image_matrix)
        samples = tf.random_normal([self.batchsize,self.n_z],0,1,dtype=tf.float32)
        guessed_z = z_mean + (z_stddev * samples)

        self.generated_images = self.generation(guessed_z)
        generated_flat = tf.reshape(self.generated_images, [self.batchsize, 28*28])

        self.generation_loss = -tf.reduce_sum(self.images * tf.log(1e-8 + generated_flat) + (1-self.images) * tf.log(1e-8 + 1 - generated_flat),1)

        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1)
        self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)

        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)

    # encoder
    def recognition(self, input_images):
        with tf.variable_scope("recognition"):
            h1 = ops.lrelu(ops.conv2d(
                x=input_images,
                inputdepth=1,
                n_filters=16,
                stride=2,
                name="first_layer")) # 28x28x1 -> 14x14x16
            h2 = ops.lrelu(ops.conv2d(
                x=h1,
                inputdepth=16,
                n_filters=32,
                stride=2,
                name="second_layers")) # 14x14x16 -> 7x7x32

            h2_flat = tf.reshape(h2, [self.batchsize, 7*7*32])

            w_mean = ops.dense(h2_flat, 7*7*32, self.n_z, "w_mean")
            w_stddev = ops.dense(h2_flat, 7*7*32, self.n_z, "w_stddev")

        return w_mean, w_stddev

    # decoder
    def generation(self, z):
        with tf.variable_scope("generation"):
            z_develop = ops.dense(z, self.n_z, 7*7*32, scope='z_matrix')
            z_matrix = tf.nn.relu(tf.reshape(z_develop,
                                             [-1, 7, 7, 32]))
            h1 = tf.nn.relu(ops.conv2d_transpose(
                z_matrix,
                [self.batchsize, 14, 14, 16],
                stride=2,
                name="g_h1"))

            h2 = ops.conv2d_transpose(h1,
                                      [self.batchsize, 28, 28, 1],
                                      stride=2,
                                      name="g_h2")
            h2 = tf.nn.sigmoid(h2)

        print("hola")
        return h2

    def train(self):
        visualization = self.mnist.train.next_batch(self.batchsize)[0]
        reshaped_vis = visualization.reshape(self.batchsize,28,28)
        ims("results/base.jpg",utils.merge(reshaped_vis[:64],[8,8]))
        # train
        if bool_save_meta:
            saver = tf.train.Saver(max_to_keep=2)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for epoch in range(10):
                for idx in range(int(self.n_samples / self.batchsize)):
                    batch = self.mnist.train.next_batch(self.batchsize)[0]
                    _, gen_loss, lat_loss = sess.run((self.optimizer, self.generation_loss, self.latent_loss), feed_dict={self.images: batch})
                    # dumb hack to print cost every epoch
                    if idx % (self.n_samples - 3) == 0:
                       print ("epoch %d: genloss %f latloss %f" % (epoch, np.mean(gen_loss), np.mean(lat_loss)))

                       if bool_save_meta:
                            saver.save(sess, os.getcwd()+"/training/train",global_step=epoch)

                       generated_test = sess.run(self.generated_images, feed_dict={self.images: visualization})
                       generated_test = generated_test.reshape(self.batchsize,28,28)
                       ims("results/"+str(epoch)+".jpg",utils.merge(generated_test[:64],[8,8]))


model = LatentAttention()
model.train()

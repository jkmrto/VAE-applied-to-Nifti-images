#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import os
from scipy.misc import imsave as ims
from utils import *
import ops as ops

#Servidor
stackdir = '/mnt/datos/home/compartido/Datos/ADNI/ADNI1Screening_1.5T_normalized_segmented_non_smoothed/stacks'
workdir='/home/andres/python/tensorflow'

# PC Despacho
#stackdir = '/media/Datos/imagenes/ADNI/ADNI1Screening_1.5T_normalized_segmented_non_smoothed/stacks'
#workdir='/home/andres/Dropbox/DeepLearning/tensorflow/ADNI'

#Carga imagenes
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
image_matrix = mnist.train.images.reshape(mnist.train.images.shape[0],28,28,1)
n_samples = mnist.train.images.shape[0]
n_hidden = 10
n_z = 20
batchsize = 100
del mnist


x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x,[-1, 28, 28, 1])



# Encoder
def recognition(input_images):
    with tf.variable_scope("recognition",reuse=True):
        h1 = ops.lrelu(ops.conv2d(input_images, 1, 16, "d_h1")) # 28x28x1 -> 14x14x16
        h2 = ops.lrelu(ops.conv2d(h1, 16, 32, "d_h2")) # 14x14x16 -> 7x7x32
        h2_flat = tf.reshape(h2,[batchsize, 7*7*32])

        w_mean = ops.dense(h2_flat, 7*7*32, n_z, "w_mean")
        w_stddev = ops.dense(h2_flat, 7*7*32, n_z, "w_stddev")

    return w_mean, w_stddev

# decoder
def generation(z):
    with tf.variable_scope("generation",reuse=True):
        z_develop = tf.nn.dense(z, n_z, 7*7*32, scope='z_matrix')
        z_matrix = tf.nn.relu(tf.reshape(z_develop, [batchsize, 7, 7, 32]))
        h1 = tf.nn.relu(ops.conv2d_transpose(z_matrix, [batchsize, 14, 14, 16], "g_h1"))
        h2 = ops.conv2d_transpose(h1, [batchsize, 28, 28, 1], "g_h2")
        h2 = tf.nn.sigmoid(h2)

    return h2
    

z_mean, z_stddev = recognition(x_image)
samples = tf.random_normal([batchsize,n_z],0,1,dtype=tf.float32)
guessed_z = z_mean + (z_stddev * samples)

generated_images = generation(guessed_z)

generated_flat = tf.reshape(generated_images, [batchsize, 28*28])

generation_loss = -tf.reduce_sum(x * tf.log(1e-8 + generated_flat) + (1-x) * tf.log(1e-8 + 1 - generated_flat),1)
latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1)
cost = tf.reduce_mean(generation_loss + latent_loss)
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

def train():
        visualization = mnist.train.next_batch(batchsize)[0]
        reshaped_vis = visualization.reshape(batchsize,28,28)
        ims("results/base.jpg",merge(reshaped_vis[:64],[8,8]))
        # train
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
                        saver.save(sess, os.getcwd()+"/training/train",global_step=epoch)
                        generated_test = sess.run(self.generated_images, feed_dict={self.images: visualization})
                        generated_test = generated_test.reshape(self.batchsize,28,28)
                        ims("results/"+str(epoch)+".jpg",merge(generated_test[:64],[8,8]))
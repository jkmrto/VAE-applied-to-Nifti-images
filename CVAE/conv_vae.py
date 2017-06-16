#!/usr/bin/env python

"""
@author: Dan Salo, Jan 2017

Purpose: Implement Convolutional Variational Autoencoder for Semi-Supervision with partially-labeled MNIST dataset.
MNIST Dataset will be downloaded and batched automatically.

"""

from tensorbase.base import Model, Layers
from scipy.misc import imsave
from libs.utils import corrupt
import sys
import tensorflow as tf
import numpy as np
import math


# Global Dictionary of Flags
flags = {
    'data_directory': 'MNIST_data/',
    'save_directory': 'summaries/',
    'model_directory': 'conv_vae/',
    'restore': False,
    'restore_file': 'start.ckpt',
    'datasets': 'MNIST',
    'image_dim': 28,
    'hidden_size': 10,
    'num_classes': 10,
    'batch_size': 100,
    'display_step': 200,
    'weight_decay': 1e-6,
    'learning_rate': 0.001,
    'corrupt_rate':0.9,
    'epochs': 1,
    'run_num': 10,
}


class ConvVae(Model):
    def __init__(self, flags_input, run_num):
        super().__init__(flags_input, run_num)
        #tf.reset_default_graph()
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        
        
    def _data(self):
        """ Define data I/O """
        #tf.reset_default_graph()
        self.x = tf.placeholder(tf.float32, [None, flags['image_dim'], flags['image_dim'], 1], name='x')
        self.epsilon = tf.placeholder(tf.float32, [None, flags['hidden_size']], name='epsilon')
        self.keep_prob = tf.placeholder(tf.float32)

            
    def _summaries(self):
        """ Define summaries for Tensorboard """
        tf.summary.scalar("Total_Loss", self.cost)
        tf.summary.scalar("Reconstruction_Loss", self.recon)
        tf.summary.scalar("VAE_Loss", self.vae)
        tf.summary.scalar("Weight_Decay_Loss", self.weight)
        tf.summary.histogram("Mean", self.mean)
        tf.summary.histogram("Stddev", self.stddev)
        tf.summary.image("x", self.x)
        tf.summary.image("x_hat", self.x_hat)

    def _encoder(self, x):
        """Define q(z|x) network"""
        encoder = Layers(x)
        encoder.conv2d(5, 64)
        encoder.maxpool()
        encoder.conv2d(3, 64)
        encoder.conv2d(3, 64)
        encoder.conv2d(3, 128, stride=2)
        encoder.conv2d(3, 128)
        encoder.conv2d(1, 64)
        encoder.conv2d(1, self.flags['hidden_size'] * 2, activation_fn=None)
        encoder.avgpool(globe=True)
        return encoder.get_output()

    def _decoder(self, z):
        """ Define p(x|z) network"""
        if z is None:
            mean = None
            stddev = None
            input_sample = self.epsilon
        else:
            z = tf.reshape(z, [-1, self.flags['hidden_size'] * 2])
            print(z.get_shape())
            mean, stddev = tf.split(z,2,1)
            stddev = tf.sqrt(tf.exp(stddev))
            input_sample = mean + self.epsilon * stddev
        decoder = Layers(tf.expand_dims(tf.expand_dims(input_sample, 1), 1))
        decoder.deconv2d(3, 128, padding='VALID')
        decoder.deconv2d(3, 128, padding='VALID', stride=2)
        decoder.deconv2d(3, 64, stride=2)
        decoder.deconv2d(3, 64, stride=2)
        decoder.deconv2d(5, 1, activation_fn=tf.nn.tanh, s_value=None)
        return decoder.get_output(), mean, stddev

    def _network(self):
        """ Define network """
        with tf.variable_scope("model", reuse=None):
            #self.xc=tf.nn.dropout(self.x,self.keep_prob)
            self.xc=corrupt(self.x)
            self.latent = self._encoder(x=self.xc)
            self.x_hat, self.mean, self.stddev = self._decoder(z=self.latent)
        with tf.variable_scope("model", reuse=True):
            self.x_gen, _, _ = self._decoder(z=None)

    def _optimizer(self):
        """ Define losses and initialize optimizer """
        epsilon = 1e-8
        const = 1/(self.flags['batch_size'] * self.flags['image_dim'] * self.flags['image_dim'])
        self.recon = const * tf.reduce_sum(tf.squared_difference(self.x, self.x_hat))
        self.vae = const * -0.5 * tf.reduce_sum(1.0 - tf.square(self.mean) - tf.square(self.stddev) + 2.0 * tf.log(self.stddev + epsilon))
        self.weight = self.flags['weight_decay'] * tf.add_n(tf.get_collection('weight_losses'))
        self.cost = tf.reduce_sum(self.vae + self.recon + self.weight)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.flags['learning_rate']).minimize(self.cost)

    def _generate_train_batch(self):
        """ Generate a training batch of images """
        #self.train_batch_y, self.train_batch_x = self.data.next_train_batch(self.flags['batch_size'])
        self.train_batch_x=np.reshape(self.batches[self.bi],[self.flags['batch_size'],28,28,1])
        self.norm = np.random.standard_normal([self.flags['batch_size'], self.flags['hidden_size']])
    
    def _generate_batches(self,data,num):
        """
        Return a total of `num` samples from the array `data`. 
        """
        data_batches={}
        idx = np.arange(0, len(data))  # get all possible indexes
        np.random.shuffle(idx)  # shuffle indexes
        idx_n=np.array([idx[i:i + num] for i in range(0, len(idx), num)])
        
        for j in range(0,idx_n.shape[0]):
            data_batches[j] = np.array([data[i,:] for i in idx_n[j,:]])  # get list of `num` random samples
        return data_batches


    def _run_train_iter(self):
        """ Run training iteration"""
        summary, _ = self.sess.run([self.merged, self.optimizer],
                                        feed_dict={self.x: self.train_batch_x, self.epsilon: self.norm})
        return summary

    def _run_train_metrics_iter(self):
        """ Run training iteration and also calculate metrics """
        summary, self.loss, self.x_recon, _ =\
            self.sess.run([self.merged, self.cost, self.x_hat, self.optimizer],
                          feed_dict={self.x: self.train_batch_x, self.epsilon: self.norm})
        return summary

    def _record_train_metrics(self):
        """ Record training metrics """
        for j in range(1):
            imsave(self.flags['restore_directory'] + 'x_' + str(self.step) + '.png', np.squeeze(self.train_batch_x[j]))
            imsave(self.flags['restore_directory'] + 'x_recon_' + str(self.step) + '.png', np.squeeze(self.x_recon[j]))
        self.print_log("Batch Number: " + str(self.step) + ", Image Loss= " + "{:.6f}".format(self.loss))


    def encode(self, xi):
        feed_dict = {self.x: xi, self.keep_prob: 1.0}
        lat=self.sess.run(self.latent, feed_dict=feed_dict)
        return lat
    
    def decode(self, zi):
        #norm = np.random.standard_normal([zi.shape[0], self.flags['hidden_size']])
        feed_dict = {self.latent: zi, self.epsilon: 1e-8*np.zeros((1,flags['hidden_size']))}
        recon=self.sess.run(self.x_hat, feed_dict=feed_dict)
        return recon
    

    def train(self, data):
        """ Train the autoencoder """
        self.print_log('Learning Rate: %d' % self.flags['learning_rate'])
        #iters = self.flags['epochs'] * data.shape[0]
        iters=self.flags['epochs'] 
        self.print_log('Iterations: %d' % iters)
        print('Corruption: %1.2f'% self.flags['corrupt_rate'])
        self.batches=self._generate_batches(data, self.flags['batch_size'])
        imsave(self.flags['restore_directory'] + 'orig' + '.png', data[5,:].reshape(28,28))
        for i in range(iters):
            self.bi=0
            for self.bi in range(0,len(self.batches)):
                self._generate_train_batch()
                if self.bi % self.flags['display_step'] != 0:
                    summary = self._run_train_metrics_iter()
                    print('Batch number: %d, Loss=%1.6f'%(self.bi,self.loss))                     
                else:
                    summary = self._run_train_metrics_iter()
                    self._record_train_metrics()
                    lat=self.encode(data[5,:].reshape(1,28,28,1))
                    rec= self.decode(lat)
                    imsave(self.flags['restore_directory'] + 'recon' + '.png', rec.reshape(28,28))
                self._record_training_step(summary)
        self._save_model(section=i)
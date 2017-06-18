import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.misc import imsave as ims
import utils as utils
import ops as cops
from tensorflow.python.framework import ops
from tensorflow.contrib.layers.python.layers.layers import batch_norm

# Global Dictionary of Flags
flags = {
    'hidden_size': 10,
    'batch_size': 100,
    'display_step': 200,
    'weight_decay': 1e-6,
    'learning_rate': 0.001,
    'epochs': 50,
    'datadim':784,
}

class CNNVAE():
    '''Class to define convolutional variational autoencoder in Tensorflow'''
    def __init__(self, flags_input):
        # Reset default graph
        ops.reset_default_graph()
        # Capture parameters
        self.flags=flags_input
        self.n_z = flags['hidden_size']
        self.batchsize = flags['batch_size']
        self.epsilon = tf.placeholder(tf.float32, [None, flags['hidden_size']], name='epsilon')
        self.is_training=tf.placeholder(tf.bool, name='is_training')
        
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.001
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)
        
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, [None, flags_input['datadim']])  
            self.X = tf.placeholder(tf.float32, [None, flags_input['datadim']])  
        self.images=self.X
        
        
        self.image_matrix = tf.reshape(self.X,[-1, 28, 28, 1])
        self.z_mean, self.z_stddev = self.recognition(self.image_matrix)
        
        samples = tf.random_normal([tf.shape(self.X)[0],self.n_z],0,1,dtype=tf.float32)
        
        self.guessed_z = self.z_mean + (self.z_stddev * samples)
        self.generated_images = self.generation(self.guessed_z) 
              
        generated_flat = tf.reshape(self.generated_images, [tf.shape(self.X)[0], 28*28])

        self.generation_loss = -tf.reduce_sum(self.images * tf.log(1e-8 + generated_flat) + (1-self.images) * tf.log(1e-8 + 1 - generated_flat),1)

        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(self.z_mean) + tf.square(self.z_stddev) - tf.log(tf.square(self.z_stddev)) - 1,1)
        self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost,global_step=global_step)
         # Init session
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        
    # encoder
    def recognition(self, images):
        with tf.variable_scope("recognition"):
            h1 = cops.lrelu(cops.conv2d(images, 1, 4, 1, "d_h1")) # 28x28x1 -> 28x28x4
            #h1 = batch_norm(h1, epsilon=0.001, decay=.99, is_training=self.is_training) 
            h1= tf.nn.max_pool(h1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='hp1')
            h2 = cops.lrelu(cops.conv2d(h1, 4, 8, 1, "d_h2")) # 28x28x4 -> 28x28x8
            h2= tf.nn.max_pool(h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='hp2')
            h3 = cops.lrelu(cops.conv2d(h2, 8, 16, 1, "d_h3")) # 28x28x8 -> 14x14x16
            h3= tf.nn.max_pool(h3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='hp3')
            h4 = cops.lrelu(cops.conv2d(h3, 16, 32, 1, "d_h4")) # 14x14x16 -> 7x7x32
            h4= tf.nn.avg_pool(h4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='hp4')

            h4_flat = tf.reshape(h4,[tf.shape(images)[0], 7*7*32])
    
            self.w_mean = cops.dense(h4_flat, 7*7*32, self.n_z, "w_mean")
            self.w_stddev = cops.dense(h4_flat, 7*7*32, self.n_z, "w_stddev")

        return self.w_mean, self.w_stddev
    
    
    # decoder
    def generation(self, z):
        with tf.variable_scope("generation"):
            if z is None:
                z = self.epsilon
            else:
                z_develop = cops.dense(z, self.n_z, 7*7*32, scope='z_matrix')
                z_matrix = tf.nn.relu(tf.reshape(z_develop, [flags['batch_size'], 7, 7, 32]))
                #z_matrix = batch_norm(z_matrix, epsilon=0.001, decay=.99, is_training=self.is_training) 
                z_matrix= tf.nn.max_unpool(z_matrix, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='hp1')
                h1 = tf.nn.relu(cops.conv2d_transpose(z_matrix, [flags['batch_size'], 14, 14, 16], 2, "g_h1"))
                h1 = tf.nn.max_unpool(h1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='hp1')
                h2 = tf.nn.relu(cops.conv2d_transpose(h1, [flags['batch_size'], 28, 28, 8], 2, "g_h2"))
                h2 = tf.nn.max_unpool(h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='hp1')

                h3 = tf.nn.relu(cops.conv2d_transpose(h2, [flags['batch_size'], 28, 28, 4], 1, "g_h3"))
                
                self.h4 = cops.conv2d_transpose(h3, [flags['batch_size'], 28, 28, 1], 1, "g_h4")
                
                self.h4 = tf.nn.sigmoid(self.h4)
    
            return self.h4

        
    def generate_batches(self, data, num):
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
    
    def encode(self,data):
        return self.sess.run([self.z_mean, self.z_stddev], feed_dict={self.X: data, self.is_training: False})
    
    def decode(self,latent):
        return self.sess.run(self.generated_images, feed_dict={self.guessed_z: latent, self.is_training: False})
        
    def train(self,traindata):
        self.batches=self.generate_batches(traindata, self.flags['batch_size'])
        batch_vis=self.batches[0]
        reshaped_vis=batch_vis.reshape(batch_vis.shape[0],28,28)
        ims("results/orig.jpg",utils.merge(reshaped_vis[:64],[8,8]))
        # train
        saver = tf.train.Saver(max_to_keep=2)
        #with tf.Session() as sess:
        self.sess.run(tf.global_variables_initializer())
        step=0
        for epoch in range(flags['epochs']):
            for idx in range(0,len(self.batches)):
                batch = self.batches[idx]
                _, gen_loss, lat_loss = self.sess.run((self.optimizer, self.generation_loss, self.latent_loss), feed_dict={self.images: batch, self.is_training: True})
                # dumb hack to print cost every epoch
                if step % self.flags['display_step'] == 0:
                    print ("epoch %d: genloss %f latloss %f" % (epoch, np.mean(gen_loss), np.mean(lat_loss)))
                    #saver.save(self.sess, os.getcwd()+"/training",global_step=epoch)
                    generated_test = self.sess.run(self.generated_images, feed_dict={self.images: batch_vis, self.is_training: False})
                    generated_test = generated_test.reshape(self.flags['batch_size'],28,28)
                    #ims("results/"+str(epoch)+".jpg",utils.merge(generated_test[:64],[8,8]))
                    ims("results/"+"reconstructed"+".jpg",utils.merge(generated_test[:64],[8,8]))
                step=step+1
                

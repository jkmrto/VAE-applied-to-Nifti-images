import tensorflow as tf
import numpy as np
import pandas as pd
import time
import pickle
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

class MultilayerPerceptron:
    '''Class to define Multilayer Perceptron Neural Networks architectures such as 
       Autoencoder in Tensorflow'''
    def __init__(self, layersize, activation, learning_rate=0.05, summaries_dir='/tmp/tfboard'):
        ''' Generate a multilayer perceptron network according to the specification and 
            initialize the computational graph.'''
        assert(len(layersize)-1==len(activation), 
		    'Activation function list must be one less than number of layers.')
        # Reset default graph
        ops.reset_default_graph()
        # Capture parameters
        self.learning_rate=learning_rate
        self.summaries_dir=summaries_dir+'/'+time.strftime('%d.%m-%a-%H:%M:%S',time.localtime())
        # Define the computation graph for an Autoencoder
        with tf.name_scope('inputs'):
            self.X = tf.placeholder("float", [None, layersize[0]])
        inputs = self.X
        # Iterate through specification to generate the multilayer perceptron network
        for i in range(len(layersize)-1):
            with tf.name_scope('layer_'+str(i)):
                n_input = layersize[i]
                n_hidden_layer = layersize[i+1]
                # Init weights and biases
                weights = tf.Variable(tf.random_normal([n_input, n_hidden_layer])*0.001, 
				name='weights')
                biases  = tf.Variable(tf.random_normal([n_hidden_layer])*0.001, name='biases')
                # Create layer with weights, biases and given activation
                layer = tf.add(tf.matmul(inputs, weights), biases)
                tf.summary.histogram('pre-activation-'+activation[i].__name__, layer)
                layer = activation[i](layer)
                # Current outputs are the input for the next layer
                inputs = layer
                tf.summary.histogram('post-activation-'+activation[i].__name__, inputs)
        self.nn = layer
        # Define loss and optimizer
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.square(tf.subtract(self.nn, self.X)))
            tf.summary.scalar("training_loss", self.loss)
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(
			learning_rate=self.learning_rate).minimize(self.loss)
        with tf.name_scope('anomalyscore'):
            self.anomalyscore = tf.reduce_mean(tf.abs(tf.subtract(self.nn, self.X)), 1)
        with tf.name_scope('crossentropy'):
            self.xentropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			                                        logits=self.nn, labels=self.X))
            tf.summary.scalar("cross_entropy", self.xentropy)
        # Init session
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        # Configure logs
        tf.gfile.MakeDirs(summaries_dir)
        self.merged = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.summaries_dir, self.sess.graph)


    def train(self, data, nsteps=100): 
        ''' Train the Neural Network using to the data.'''
        lossdev = []
        scoredev = []
        # Training cycle
        steps = np.linspace(0,len(data), num=nsteps, dtype=np.int)
        for it,(step1,step2) in enumerate(zip(steps[0:-1],steps[1:])):
            c = self.sess.run([self.optimizer, self.loss], 
				feed_dict={self.X: data[step1:step2,:]})  
            s = self.sess.run(self.nn, 
				feed_dict={self.X: data[step1:step1+1,:]})
            l,ts = self.sess.run([self.loss, self.merged], 
				feed_dict={self.X: data[step1:step1+1,:]})
            scoredev.append(s[0])
            lossdev.append(l)
            print('.', end='')
            self.summary_writer.add_summary(ts, it)
        print
        return lossdev

    def predict(self, data):
        '''Predict outcome for data'''
        return self.sess.run(self.nn, feed_dict={self.X: data})
        
    def score(self, data):
        '''Compute anomaly score based on reconstruction error.'''
        return self.sess.run(self.anomalyscore, feed_dict={self.X: data})


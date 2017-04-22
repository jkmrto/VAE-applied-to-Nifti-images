# -*- coding: utf-8 -*-
import os

import numpy as np
import scipy.io as sio
import tensorflow as tf

np.random.seed(0)
tf.set_random_seed(0)


stackdir = '/home/andres/Dropbox/DeepLearning/challenge'
workdir='/home/andres/Dropbox/DeepLearning/challenge'

f = sio.loadmat(os.path.join(stackdir, 'stack_train.mat'))
f.keys() # Muestra las claves del diccionario file. 
trainlabels = f['labels'].flatten()
train_data = f['data_train'].astype(np.float32)
del f
f = sio.loadmat(os.path.join(stackdir, 'stack_test.mat'))
f.keys() # Muestra las claves del diccionario file. 
test_data = f['data_test'].astype(np.float32)
del f
n_samples=train_data.shape[0]
data_dim=train_data.shape[1]

import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
train_data_mnist=mnist.train.images
n_samples = train_data.shape[0]
data_dim_mnist=train_data_mnist.shape[1]
del mnist




def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)
def next_batch(data, num):
    import numpy as np 
    """
    Return a total of `num` samples from the array `data`. 
    """
    idx = np.arange(0, len(data))  # get all possible indexes
    np.random.shuffle(idx)  # shuffle indexes
    idx = idx[0:num]  # use only `num` random indexes
    data_shuffle = [data[i] for i in idx]  # get list of `num` random samples

    return np.asarray(data_shuffle)  # get back numpy array


def train(data,network_architecture, learning_rate=0.001,
          batch_size=10, training_epochs=10, display_step=5):
    vae = VariationalAutoencoder.VariationalAutoencoder(network_architecture, 
                                 learning_rate=learning_rate, 
                                 batch_size=batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs= next_batch(data,batch_size)

            # Fit training using batch data
            cost = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), 
                  "cost=", "{:.9f}".format(avg_cost))
    return vae

network_architecture = dict(n_hidden_recog_1=500, # 1st layer encoder neurons
         n_hidden_recog_2=500, # 2nd layer encoder neurons
         n_hidden_gener_1=500, # 1st layer decoder neurons
         n_hidden_gener_2=500, # 2nd layer decoder neurons
         n_input=data_dim, # MNIST data input (img shape: 28*28)
         n_z=20)  # dimensionality of latent space

vae = train(train_data,network_architecture, training_epochs=75)
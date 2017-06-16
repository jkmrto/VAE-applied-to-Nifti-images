# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.misc import imsave as ims
import utils as utils
import ops as ops
from CNNVAE import CNNVAE

flags = {
    'hidden_size': 10,
    'batch_size': 100,
    'display_step': 200,
    'weight_decay': 1e-6,
    'learning_rate': 0.001,
    'epochs': 10,
    'datadim':784,
}

def load_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets("./MNIST_data")

def reconstruct_image(vimg):
    DIM=vimg.shape[1]
    return vimg.reshape(np.sqrt(DIM).astype(np.int32),np.sqrt(DIM).astype(np.int32))


model = CNNVAE(flags)

mnist=load_mnist()
train_data=mnist.train.images
test_data=mnist.test.images
train_labels=mnist.train.labels
test_labels=mnist.test.labels


model.train(train_data)

test=test_data[5,:].reshape(1,-1)
latt=np.zeros((100,10))
lat=np.asarray(model.encode(test)).reshape(1,20)
latt[0,:]=lat[0,0:10].reshape(1,10)

rec=model.decode(latt)
rec=rec[0,:,:].reshape(28,28)
plt.figure(1)
plt.imshow(test.reshape(28,28),cmap='gray')
plt.figure(2)
plt.imshow(rec,cmap='gray')



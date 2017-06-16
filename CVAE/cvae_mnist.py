# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import math
from conv_vae import ConvVae
from tensorbase.data import Mnist
from tensorbase.base import Model
import matplotlib.pyplot as plt




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
    'run_num': 1,
}

def load_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets("./MNIST_data")

def reconstruct_image(vimg):
    DIM=vimg.shape[1]
    return vimg.reshape(np.sqrt(DIM).astype(np.int32),np.sqrt(DIM).astype(np.int32))


mnist=load_mnist()
train_data=mnist.train.images
test_data=mnist.test.images
train_labels=mnist.train.labels
test_labels=mnist.test.labels

flags['seed'] = np.random.randint(1, 1000, 1)[0]
flags['run_num'] = 10
model_vae = ConvVae(flags, run_num=flags['run_num'])
model_vae.train(train_data)

ai=train_data[1,:].reshape(1,28,28,1)
lat=model_vae.encode(ai)
rec=model_vae.decode(lat)

# Visualize decoded

fig, axs = plt.subplots(5,10, figsize=(15, 6))
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()
idx=np.random.randint(test_labels.shape[0],size=50)
k=0
for i in idx:
    lat = model_vae.encode(test_data[i,:].reshape(1,28,28,1))
    norm = np.random.standard_normal([1, flags['hidden_size']])
    rec,m,std=model_vae.decode(lat)
    image_decoded=rec.reshape(28,28)
    #print('label=%d '%(train_labels[i]))   
    axs[k].imshow(image_decoded,cmap='gray')
    axs[k].set_title(str(test_labels[i]))
    k=k+1
    
    
# Show in latent space
#import pylab
#cm = pylab.get_cmap('jet')
#z=np.zeros((len(idx),10))
#k=0
#for i in idx:   
#    mus,_ = v.encode(test_data[i,:].reshape(1,DIM))
#    z[k,:]=mus
#    color = cm(1.*test_labels[k]/len(np.unique(test_labels[idx])))
#    #cgen = (cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS))
#    plt.scatter(z[k,0],z[k,1],z[k,2],c=color)
#    plt.annotate(str(test_labels[k]), (z[k,0],z[k,1]))
#    k=k+1    
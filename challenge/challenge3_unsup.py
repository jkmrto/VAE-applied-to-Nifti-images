# -*- coding: utf-8 -*-
import os

import VAE
import numpy as np
import scipy.io as sio
import tensorflow as tf
from sklearn import svm
from sklearn.preprocessing import normalize

from lib.CV_utils import cvpartition

np.random.seed(0)
tf.set_random_seed(0)


stackdir = '/home/andres/Dropbox/DeepLearning/challenge/data'
workdir='/home/andres/Dropbox/DeepLearning/challenge'

DIM = 429
DIM_LAT= 20
ARCHITECTURE = [DIM, # Input dimension
                1000, 500, 100, # intermediate encoding
                DIM_LAT] # latent space dims
                # 50]
# (and symmetrically back out again)

HYPERPARAMS = {
    "batch_size": 20,
    "learning_rate": 0.001,
    "dropout": 0.9,
    "lambda_l2_reg": 0.00001,
    "nonlinearity": tf.nn.elu,
    "squashing": tf.nn.sigmoid
}

MAX_ITER = 2000#2**16
MAX_EPOCHS = np.inf

LOG_DIR = "./log"
METAGRAPH_DIR = "./out"
PLOTS_DIR = "./png"



def load_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets("./mnist_data")

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

def reconstruct_image(vimg):
    DIM=vimg.shape[1]
    return vimg.reshape(np.sqrt(DIM).astype(np.int32),np.sqrt(DIM).astype(np.int32))
    

f = sio.loadmat(os.path.join(stackdir, 'stack_train.mat'))
f.keys() # Muestra las claves del diccionario file. 
trainlabels = f['labels'].flatten()
train_data = f['data_train'].astype(np.float32)
train_data = normalize(train_data, norm='l1', axis=0)
del f
f = sio.loadmat(os.path.join(stackdir, 'stack_test.mat'))
f.keys() # Muestra las claves del diccionario file. 
test_data = f['data_test'].astype(np.float32)
del f
n_samples=train_data.shape[0]
data_dim=train_data.shape[1]


# El dataset original se divide en training + test
trainidx,testidx=cvpartition(np.arange(1,n_samples),10)
tf.reset_default_graph()

v = VAE.VAE(ARCHITECTURE, HYPERPARAMS, log_dir=LOG_DIR)    
v.train(train_data[trainidx[i],:], max_iter=MAX_ITER, max_epochs=MAX_EPOCHS, cross_validate=False,
            verbose=False, save=True, outdir=METAGRAPH_DIR, plots_outdir=PLOTS_DIR,
            plot_latent_over_time=False)


for i in range(len(trainidx)):
    tf.reset_default_graph()
    v = VAE.VAE(ARCHITECTURE, HYPERPARAMS, log_dir=LOG_DIR)    
    v.train(train_data[trainidx[i],:], max_iter=MAX_ITER, max_epochs=MAX_EPOCHS, cross_validate=False,
            verbose=False, save=True, outdir=METAGRAPH_DIR, plots_outdir=PLOTS_DIR,
            plot_latent_over_time=False)
    print("Trained!")
    
    z=np.zeros((len(idx),DIM_LAT))   # Latent space
    k=0
    for j in trainidx:
        mus,_ = v.encode(traindata[j,:].reshape(1,DIM))
        z_train[k,:]=mus
        k=k+1
    k=0
    for j in testidx:
        mus,_ = v.encode(traindata[j,:].reshape(1,DIM))
        z_test[k,:]=mus
        k=k+1
        
    # Train multiclass SVC
    clf = svm.SVC(decision_function_shape='ovr')
    clf.fit(z_train, trainlabels[trainidx]) 
    p=clf.predict(z_test[testidx])
        





    
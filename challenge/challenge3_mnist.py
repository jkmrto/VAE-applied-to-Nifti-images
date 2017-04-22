# -*- coding: utf-8 -*-
import os

import VAE
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tensorflow as tf
from sklearn.preprocessing import normalize

from lib.CV_utils import cvpartition

np.random.seed(0)
tf.set_random_seed(0)


stackdir = '/home/andres/Dropbox/DeepLearning/challenge/data'
workdir='/home/andres/Dropbox/DeepLearning/challenge'

DIM = 784
DIM_LAT= 3
ARCHITECTURE = [DIM, # Input dimension
                500, 500, 100, # intermediate encoding
                DIM_LAT] # latent space dims
                # 50]
# (and symmetrically back out again)

HYPERPARAMS = {
    "batch_size": 100,
    "learning_rate": 0.001,
    "dropout": 0.9,
    "lambda_l2_reg": 0.00001,
    "nonlinearity": tf.nn.elu,
    "squashing": tf.nn.sigmoid
}

MAX_ITER = 3000#2**16
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

mnist=load_mnist()
train_data=mnist.train.images
test_data=mnist.test.images
train_labels=mnist.train.labels
test_labels=mnist.test.labels
n_samples = train_data.shape[0]
data_dim=train_data.shape[1]
del mnist

#train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])


tf.reset_default_graph()
v = VAE.VAE(ARCHITECTURE, HYPERPARAMS, log_dir=LOG_DIR)
# El dataset original se divide en training + test=validation
trainidx,testidx = cvpartition(np.arange(1,n_samples),10)


v.train(train_data, max_iter=MAX_ITER, max_epochs=MAX_EPOCHS, cross_validate=False,
        verbose=False, save=True, outdir=METAGRAPH_DIR, plots_outdir=PLOTS_DIR,
        plot_latent_over_time=False)
print("Trained!")



#clf = svm.SVC(decision_function_shape='ovr')
#clf.fit(z, TDl) 
#p=clf.predict(z)

# Visualize decoded

fig, axs = plt.subplots(5,10, figsize=(15, 6))
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()
idx=np.random.randint(test_labels.shape[0],size=50)

k=0
for i in idx:
    mus,_ = v.encode(test_data[i,:].reshape(1,DIM))
    d=v.decode(mus)
    image_decoded=reconstruct_image(d)
    #print('label=%d '%(train_labels[i]))   
    axs[k].imshow(image_decoded,cmap='gray')
    axs[k].set_title(str(test_labels[i]))
    k=k+1


# Show in latent space
import pylab
cm = pylab.get_cmap('jet')
z=np.zeros((len(idx),DIM_LAT))
k=0
for i in idx:   
    mus,_ = v.encode(test_data[i,:].reshape(1,DIM))
    z[k,:]=mus
    color = cm(1.*test_labels[k]/len(np.unique(test_labels[idx])))
    #cgen = (cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS))
    plt.scatter(z[k,0],z[k,1],z[k,2],c=color)
    plt.annotate(str(test_labels[k]), (z[k,0],z[k,1]))
    k=k+1



    
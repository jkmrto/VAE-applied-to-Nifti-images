#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 11:07:35 2017

@author: Andres
"""
#%reset
import tensorflow as tf
import numpy as np
import nibabel as nib
import os
import scipy.io as sio
import math



stackdir = '/home/andres/Dropbox/DeepLearning/challenge/data'
workdir='/home/andres/Dropbox/DeepLearning/challenge'


def standarize(A):
    mu=np.mean(A, axis=0)
    sigma=np.std(A, axis=0)
    return mu, sigma, ((A - mu) / (sigma+1E-20))
    
def create_image(x):
    sqz=(np.ceil(np.sqrt(len(x)))**2).astype(np.int32)
    imv=np.zeros((1,sqz))
    imv[0,0:len(x)]=x;
    return imv.flatten().reshape(np.sqrt(sqz).astype(np.int32),np.sqrt(sqz).astype(np.int32))
    
f = sio.loadmat(os.path.join(stackdir, 'stack_train.mat'))
f.keys() # Muestra las claves del diccionario file. 
trainlabels = f['labels'].flatten()
train_data = f['data_train'].astype(np.float32)
mu, sigma, train_data = standarize(train_data)
del f
f = sio.loadmat(os.path.join(stackdir, 'stack_test.mat'))
f.keys() # Muestra las claves del diccionario file. 
test_data = f['data_test'].astype(np.float32)
test_data = standarize(test_data)
del f
n_samples=train_data.shape[0]
data_dim=train_data.shape[1]


stack_train=np.zeros((train_data.shape[0],21,21))
for i in range(train_data.shape[0]):
    stack_train[i,:,:]=create_image(train_data[i,:])
    
imshape = np.array(stack_train.shape[1:]).astype(np.int32)


n_classes = 4

label_arr = np.zeros((len(trainlabels), n_classes))
for i in range(n_classes):
    label_arr[:,i] = trainlabels==i

sess = tf.InteractiveSession(config=tf.ConfigProto(intra_op_parallelism_threads=4))


x = tf.placeholder(tf.float32, shape=[None, imshape[0], imshape[1]]) # none to accept any batch size, 784 = 28x28
y_ = tf.placeholder(tf.float32, shape=[None, n_classes])


def weight_variable(shape, name=None):
  initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
  initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
  return tf.Variable(initial, name=name)
  
  
def conv2d(x, W, name=None):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)

def max_pool_2x2(x, name=None):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name=name)


  
x_image = tf.reshape(x, [-1,imshape[0], imshape[1], 1])

W_conv1 = weight_variable([4, 4, 1, 6], name='Wconv1')
b_conv1 = bias_variable([6], name='bconv1')


h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, name='relu1') + b_conv1)
h_pool1 = max_pool_2x2(h_conv1, name='pool1')


W_conv2 = weight_variable([5, 5, 6, 12], name='Wconv2')
b_conv2 = bias_variable([12], name='bconv2')

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, name='relu2') + b_conv2)
h_pool2 = max_pool_2x2(h_conv2, name='pool2')

W_conv3 = weight_variable([6, 6, 12, 24], name='Wconv3')
b_conv3 = bias_variable([24], name='bconv3')

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, name='relu3') + b_conv3)
h_pool3 = max_pool_2x2(h_conv3, name='pool3')

finalsize = np.ceil(np.array(list(imshape))/4).astype(int)

#
W_fc1 = weight_variable([finalsize[0] * finalsize[1] * 24, 1024], name='fc1')
b_fc1 = bias_variable([1024], name='bfc1')
#
h_pool2_flat = tf.reshape(h_pool3, [-1, finalsize[0] * finalsize[1] * 24], name='pool2f')
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name='fcrelu')
#
#
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
W_fc2 = weight_variable([1024, n_classes])
b_fc2 = bias_variable([n_classes])
#
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# learning rate decay
max_learning_rate = 0.003
min_learning_rate = 0.0001
decay_speed = 2000.0
#lr = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)


from sklearn.metrics import confusion_matrix
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
confmat = tf.contrib.metrics.confusion_matrix(tf.argmax(y_conv,1), tf.argmax(y_,1), num_classes=n_classes)

sess.run(tf.global_variables_initializer())
print("Training...")



from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(n_splits=10)
skfint = StratifiedKFold(n_splits=5) 
accuracies = []
firstiter=True

for train, test in skf.split(stack_train, trainlabels):
    sess.run(tf.global_variables_initializer())
    trset = stack_train[train,:,:]
    labelsplit = trainlabels[train]
    trlabel = label_arr[train,:]
    accTraining = []
    for i in range(100):
        lr = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
        print("Iteration %d:, learning rate=%1.4f" %(i,lr))
        for trainint, testint in skfint.split(trset, labelsplit):
            train_step.run(feed_dict={x:trset[testint,:], y_: trlabel[testint,:], keep_prob: 0.75})
            train_step.run(feed_dict={x:np.fliplr(trset[testint,:]), y_: trlabel[testint,:], keep_prob: 0.75})
        acc = accuracy.eval(feed_dict={x:trset, y_: trlabel, keep_prob:0.75})
        print(acc)
        accTraining.append(acc)
    if firstiter:
        cmat = [confmat.eval(feed_dict={x: stack_train[test,:,:], y_: label_arr[test,:], keep_prob: 1.0}).flatten()]
        firstiter = False
    else: 
        cmat.append(confmat.eval(feed_dict={x: stack_train[test,:,:], y_: label_arr[test,:], keep_prob: 1.0}).flatten())
    accIter = accuracy.eval(feed_dict={x: stack_train[test,:,:], y_: label_arr[test,:], keep_prob: 1.0})
    accuracies.append(accIter)
    print("test accuracy %g"%accIter)

conf = np.array(cmat).sum(axis=0).reshape((n_classes, n_classes))
print(conf)
print("Final accuracy $ %1.3f \\pm %1.3f $"%(np.array(accuracies).mean(), np.array(accuracies).std()))


#%% VISUALIZATION
#import os
#import matplotlib.pyplot as plt
#os.chdir('/home/pakitochus/Investigacion/Funciones/Python/pyhacks')
#import montage as mt
#wconv1_value = W_conv1.eval()
#wconv2_value = W_conv2.eval()
#os.chdir('/home/pakitochus/Investigacion/Pubs/2017/IWINAC 2017/filters')

#for i in range(8):
#    mt.montage(wconv1_value[:,:,:,0,i], cmap='viridis')
#    plt.savefig('wconv1_filter'+str(i)+'_woSWEDD.eps')
#    plt.close()

#for i in range(16):
#    mt.montage(wconv2_value[:,:,:,0,i], cmap='viridis')
#    plt.savefig('wconv2_filter'+str(i)+'_woSWEDD.eps')
#    plt.close()



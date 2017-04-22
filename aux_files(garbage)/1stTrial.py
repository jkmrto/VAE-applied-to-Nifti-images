#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 11:07:35 2017

@author: pakitochus
"""
#%reset
import tensorflow as tf
import numpy as np
import nibabel as nib
import os

dbdir = '/home/pakitochus/Investigacion/Databases/parkinson'
normdir = '/home/pakitochus/Investigacion/Funciones/Python/pyhacks'
os.chdir(dbdir)
#imsample = nib.load(os.listdir()[0])
#imshape = imsample.shape


import scipy.io as sio
f = sio.loadmat(os.path.join(dbdir, 'PPMI301stack_all.mat'))
f.keys() # Muestra las claves del diccionario file. 
labels = f['labels'].flatten()
stack_all = f['stack_all']
imshape = f['tamano'].flatten().astype(np.int32)
del f

os.chdir(normdir)
import normalization as norm
os.chdir(dbdir)


def recortar_im(stack, th_perc=0.1):
    eq = [[2, 1], [2, 0], [0, 0]]
    N = stack.shape[0]
    Imean = stack.mean(axis=0)
    thval = th_perc*Imean.max()
    Ithresholded = Imean>(th_perc*Imean.max())
    ndim = len(stack.shape)-1
    minidx=np.zeros(ndim,dtype=np.int)
    maxidx=np.zeros(ndim,dtype=np.int)
    for ax in range(ndim):
        filtered_lst = [idx for idx,y in enumerate(Ithresholded.any(axis=eq[ax][0]).any(axis=eq[ax][1])) if y > thval]
        minidx[ax] = min(filtered_lst)
        maxidx[ax] = max(filtered_lst)
    newstack = stack[:, minidx[0]:maxidx[0], minidx[1]:maxidx[1], minidx[2]:maxidx[2]]
    return newstack
        
stack_all = stack_all[labels!=1,:,:,:]
labels = labels[labels!=1]>0

stack_norm = recortar_im(stack_all, 0.35)
imshape = np.array(stack_norm.shape[1:]).astype(np.int32)

#%% 
#stack_norm = norm.norm_int(stack_all)

n_classes = 2

label_arr = np.zeros((len(labels), n_classes))
for i in range(n_classes):
    label_arr[:,i] = labels==i

sess = tf.InteractiveSession(config=tf.ConfigProto(intra_op_parallelism_threads=8))


x = tf.placeholder(tf.float32, shape=[None, imshape[0], imshape[1], imshape[2]]) # none to accept any batch size, 784 = 28x28
y_ = tf.placeholder(tf.float32, shape=[None, n_classes])


def weight_variable(shape, name=None):
  initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
  initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
  return tf.Variable(initial, name=name)
  
  
def conv3d(x, W, name=None):
  return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME', name=name)

def max_pool_2x2x2(x, name=None):
  return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1],
                        strides=[1, 2, 2, 2, 1], padding='SAME', name=name)
  
x_image = tf.reshape(x, [-1,imshape[0], imshape[1], imshape[2], 1])

W_conv1 = weight_variable([5, 5, 5, 1, 8], name='Wconv1')
b_conv1 = bias_variable([8], name='bconv1')


h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1, name='relu1') + b_conv1)
h_pool1 = max_pool_2x2x2(h_conv1, name='pool1')


W_conv2 = weight_variable([5, 5, 5, 8, 16], name='Wconv2')
b_conv2 = bias_variable([16], name='bconv2')

h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2, name='relu2') + b_conv2)
h_pool2 = max_pool_2x2x2(h_conv2, name='pool2')


finalsize = np.ceil(np.array(list(imshape))/4).astype(int)

#x = tf.placeholder(tf.float32, shape=[None, 784])
#y = tf.placeholder(tf.float32, shape=[None, 10])
#
#W_h1 = tf.Variable(tf.random_normal([784, 512]))
#h1 = tf.nn.sigmoid(tf.matmul(x, W_h1))
#
#W_out = tf.Variable(tf.random_normal([512, 10]))
#y_ = tf.matmul(h1, W_out)

#
W_fc1 = weight_variable([finalsize[0] * finalsize[1] * finalsize[2] * 16, 4000], name='fc1')
b_fc1 = bias_variable([4000], name='bfc1')
#
h_pool2_flat = tf.reshape(h_pool2, [-1, finalsize[0] * finalsize[1] * finalsize[2] * 16], name='pool2f')
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name='fcrelu')
#
#
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
W_fc2 = weight_variable([4000, n_classes])
b_fc2 = bias_variable([n_classes])
#
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


#error = tf.reduce_mean(tf.nn.l2_loss(y_-y_conv))
#train_step = tf.train.GradientDescentOptimizer(0.1).minimize(error)
#cross_entropy = tf.reduce_mean(tf.nn.softmax(y_conv, y_))
#cross_entropy = tf.reduce_sum(- y_ * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)) - (1 - y_) * tf.log(tf.clip_by_value(1 - y_conv, 1e-10, 1.0)), 1)
#loss = tf.reduce_mean(cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
#correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
#from sklearn.model_selection import StratifiedKFold
#skf = StratifiedKFold(n_splits=10)
#
#with tf.Session() as s:
#    for train, test in skf.split(stack_norm, labels):
##        sess.run(tf.global_variables_initializer())
#        trset = stack_norm[train,:,:]
#        labelsplit = labels[train]
#        trlabel = label_arr[train,:]
#        s.run(tf.initialize_all_variables())
#
#        for i in range(10000):
#            s.run(train_step, feed_dict={x:trset, y_: trlabel, keep_prob: 0.5})
#    #        s.run(train_step, feed_dict={x: batch_x, y: batch_y})
#    
#            if i % 10 == 0:
#                train_accuracy = accuracy.eval(feed_dict={x: trset[testint,:,:], y_: trlabel[testint,:]})
#                print('step {0}, training accuracy {1}'.format(i, train_accuracy))

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
confmat = tf.contrib.metrics.confusion_matrix(tf.argmax(y_conv,1), tf.argmax(y_,1), num_classes=n_classes)
sess.run(tf.global_variables_initializer())



from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10)
skfint = StratifiedKFold(n_splits=6) 
accuracies = []
firstiter=True
for train, test in skf.split(stack_norm, labels):
    sess.run(tf.global_variables_initializer())
    trset = stack_norm[train,:,:]
    labelsplit = labels[train]
    trlabel = label_arr[train,:]
    accTraining = []
    for i in range(100):
#    train_step.run(feed_dict={x:trset, y_: trlabel, keep_prob:0.5})
#    train_step.run(feed_dict={x:np.fliplr(trset), y_: trlabel, keep_prob:0.5})
        for trainint, testint in skfint.split(trset, labelsplit):
            train_step.run(feed_dict={x:trset[testint,:,:], y_: trlabel[testint,:], keep_prob: 0.5})
            train_step.run(feed_dict={x:np.fliplr(trset[testint,:,:]), y_: trlabel[testint,:], keep_prob: 0.5})
        acc = accuracy.eval(feed_dict={x:trset, y_: trlabel, keep_prob:0.5})
        print(acc)
        accTraining.append(acc)
#        if
    if firstiter:
        cmat = [confmat.eval(feed_dict={x: stack_norm[test,:,:,:], y_: label_arr[test,:], keep_prob: 1.0}).flatten()]
        firstiter = False
    else: 
        cmat.append(confmat.eval(feed_dict={x: stack_norm[test,:,:,:], y_: label_arr[test,:], keep_prob: 1.0}).flatten())
    accIter = accuracy.eval(feed_dict={x: stack_norm[test,:,:,:], y_: label_arr[test,:], keep_prob: 1.0})
    accuracies.append(accIter)
    print("test accuracy %g"%accIter)

conf = np.array(cmat).sum(axis=0).reshape((n_classes, n_classes))
print(conf)
print("Final accuracy $ %1.3f \\pm %1.3f $"%(np.array(accuracies).mean(), np.array(accuracies).std()))


#%% VISUALIZATION
import os
import matplotlib.pyplot as plt
os.chdir('/home/pakitochus/Investigacion/Funciones/Python/pyhacks')
import montage as mt
wconv1_value = W_conv1.eval()
wconv2_value = W_conv2.eval()
os.chdir('/home/pakitochus/Investigacion/Pubs/2017/IWINAC 2017/filters')

for i in range(8):
    mt.montage(wconv1_value[:,:,:,0,i], cmap='viridis')
    plt.savefig('wconv1_filter'+str(i)+'_woSWEDD.eps')
    plt.close()

for i in range(16):
    mt.montage(wconv2_value[:,:,:,0,i], cmap='viridis')
    plt.savefig('wconv2_filter'+str(i)+'_woSWEDD.eps')
    plt.close()











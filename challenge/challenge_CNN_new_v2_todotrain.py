#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 11:07:35 2017

@author: Andres
"""
#%reset
import math
import os

import numpy as np
import scipy.io as sio
import tensorflow as tf

from lib.CV_utils import show_confusion_matrix

tf.set_random_seed(0)




stackdir = '/home/andres/Dropbox/DeepLearning/challenge/data'
workdir='/home/andres/Dropbox/DeepLearning/challenge'
n_classes=4


def load_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets("./mnist_data")
    
def standarize(A):
    mu=np.mean(A, axis=0)
    sigma=np.std(A, axis=0)
    return mu, sigma, ((A - mu) / (sigma+1E-20))
    
def create_image(x):
    sqz=(np.ceil(np.sqrt(len(x)))**2).astype(np.int32)
    imv=np.zeros((1,sqz))
    imv[0,0:len(x)]=x;
    return imv.flatten().reshape(np.sqrt(sqz).astype(np.int32),np.sqrt(sqz).astype(np.int32))
    
def write_csv(filename, predictions_test):
    class_names= ['HC', 'MCI', 'cMCI', 'AD']
    with open(filename, 'w') as f:
        #print('SUB_ID' + ',' + 'Diagnosis')
        f.write('SUB_ID' + ',' + 'Diagnosis\n')
        subject= 0
        for x in predictions_test.flatten():
           subject= subject + 1
           subject_str= 'TEST_' + '{:03d}'.format(subject)
           #print(subject_str + "," + class_names[x])
           f.write(subject_str + "," + class_names[x]+'\n')    
    return True


f = sio.loadmat(os.path.join(stackdir, 'stack_train.mat'))
f.keys() # Muestra las claves del diccionario file. 
trainlabels = f['labels'].flatten()
train_data = f['data_train'].astype(np.float32)
mu, sigma, train_data = standarize(train_data)
del f
f = sio.loadmat(os.path.join(stackdir, 'stack_test.mat'))
f.keys() # Muestra las claves del diccionario file. 
test_data = f['data_test'].astype(np.float32)
test_data=((test_data - mu) / (sigma+1E-20))
del f
n_samples=train_data.shape[0]
data_dim=train_data.shape[1]


stack_train=np.zeros((train_data.shape[0],21,21,1))
stack_test=np.zeros((test_data.shape[0],21,21,1))

for i in range(train_data.shape[0]):
    stack_train[i,:,:,0]=create_image(train_data[i,:])
for i in range(test_data.shape[0]):
    stack_test[i,:,:,0]=create_image(test_data[i,:])

imshape = np.array(stack_train.shape[1:]).astype(np.int32)


label_arr = np.zeros((len(trainlabels), n_classes))
for i in range(n_classes):
    label_arr[:,i] = trainlabels==i


sess = tf.InteractiveSession(config=tf.ConfigProto(intra_op_parallelism_threads=4))

# input X: 21x21 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 21, 21,1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 4])
# variable learning rate
lr = tf.placeholder(tf.float32)
# Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
pkeep = tf.placeholder(tf.float32)

# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
K = 32  # first convolutional layer output depth
L = 64  # second convolutional layer output depth
M = 128  # third convolutional layer
N = 200  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1))  # 6x6 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))

W4 = tf.Variable(tf.truncated_normal([6 * 6 * M, N], stddev=0.1))
B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
W5 = tf.Variable(tf.truncated_normal([N, 4], stddev=0.1))
B5 = tf.Variable(tf.constant(0.1, tf.float32, [4]))

# The model
stride = 1  # output is 21x21
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2  # output is ceil(float(in_height) / float(stride)) -> 11x11
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 2  # output is 6x6
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 6*6 * M])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
YY4 = tf.nn.dropout(Y4, pkeep)
Ylogits = tf.matmul(YY4, W5) + B5
Y = tf.nn.softmax(Ylogits)

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

confmat = tf.contrib.metrics.confusion_matrix(tf.argmax(Y,1), tf.argmax(Y_,1), num_classes=4)

# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(n_splits=10,random_state=0)
skfint = StratifiedKFold(n_splits=5,random_state=0) 
accuracies = []
firstiter=True
max_learning_rate = 0.001
min_learning_rate = 0.0001
decay_speed =2000.0

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for train, test in skf.split(stack_train, trainlabels):
    sess.run(tf.global_variables_initializer())
    trset = stack_train[train,:,:]
    labelsplit = trainlabels[train]
    trlabel = label_arr[train,:]
    accTraining = []
    #batch_X, batch_Y = mnist.train.next_batch(100)
    for i in range(100):
        lrate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
        print("Iteration %d:, learning rate=%1.4f" %(i,lrate))
        sess.run(train_step, {X:trset, Y_: trlabel, lr: lrate, pkeep: 0.9})
        sess.run(train_step, {X:np.fliplr(trset), Y_: trlabel, lr: lrate, pkeep: 0.9})
        acc, c = sess.run([accuracy, cross_entropy], {X:stack_train[test,:,:] , Y_: label_arr[test,:], pkeep: 1.0})
        print(acc)
        accTraining.append(acc)
    if firstiter:
        cmat = sess.run([confmat], {X: stack_train[test,:,:], Y_: label_arr[test,:], pkeep: 1.0})
        firstiter = False
    else: 
        cmat.append(sess.run([confmat], {X: stack_train[test,:,:], Y_: label_arr[test,:], pkeep: 1.0}))
    accIter = sess.run([accuracy], {X: stack_train[test,:,:], Y_: label_arr[test,:], pkeep: 1.0})
    accIter=accIter[0];
    accuracies.append(accIter)
    print("test accuracy %1.3f"%accIter)

    
conf = np.array(cmat).sum(axis=0).reshape((4, 4))
print(conf)
print("Final accuracy %1.3f \\pm %1.3f $"%(np.array(accuracies).mean(), np.array(accuracies).std()))
show_confusion_matrix(conf, ['HC', 'MCI', 'cMCI', 'AD'])

# Con todo el dataset de training completo
print('*********** CON TODO EL DATASET DE TRAINING **********')
for i in range(500):
    lrate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
    print("Iteration %d:, learning rate=%1.4f" %(i,lrate))
    sess.run(train_step, {X:stack_train, Y_: label_arr, lr: lrate, pkeep: 0.9})
    sess.run(train_step, {X:np.fliplr(stack_train), Y_: label_arr, lr: lrate, pkeep: 0.9})
    acc, c = sess.run([accuracy, cross_entropy], {X:stack_train , Y_: label_arr, pkeep: 1.0})
    print(acc)
#Test
plabels = sess.run([tf.argmax(Y,1)], {X:stack_test, pkeep: 1.0})
write_csv('result_CNN', plabels)
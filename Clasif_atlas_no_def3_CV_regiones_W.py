#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
import numpy as np
import nibabel as nib
import os
import scipy.io as sio
import h5py
from scipy import stats
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import svm


stackdir = '/mnt/datos/home/compartido/Datos/ADNI/ADNI1Screening_1.5T_normalized_segmented_non_smoothed/stacks'
stackdir2 = '/media/Datos/imagenes/ADNI/DB_ADNI_MRI_PET_PREPROCESSED/stacks'
workdir='/home/andres/python/tensorflow/CNN_Ensemble'

#stackdir = '/media/Datos/imagenes/ADNI/ADNI1Screening_1.5T_normalized_segmented_non_smoothed/stacks'
#workdir='/home/andres/Dropbox/DeepLearning/python/tensorflow/ADNI'
#workdir='/run/user/1000/gvfs/sftp:host=biosip1/home/andres/python/tensorflow/CNN_Ensemble'


# Regiones
frontal_lobe_val=np.array([3,4,7,8,23,24,9,10,27,28,31,32])
parietal_lobe_val=np.array([59,60,61,62,67,68,35,36])
occipital_lobe_val=np.array([49,50,51,52,53,54])
temporal_lobe_val=np.array([81,82,83,84,85,86,87,88,89,90,55,56,37,38,39,40])
cerebellum_val=np.array([91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108])
vermis_val=np.array([109,110,111,112,113,114,115,116])

lobes_mask_val=np.concatenate((frontal_lobe_val,parietal_lobe_val,occipital_lobe_val,temporal_lobe_val),axis=0)
#lobes_mask_val_cerebellum=np.concatenate((lobes_mask_val,cerebellum_val),axis=0);
#lobes_mask_val=np.arange(1,90)

imsize_MRI=np.array([121,145,121])
train_iter=100;
# Load atlas
# Cambio al directorio de los stacks
os.chdir(workdir)
print("Loading atlas...")
img = nib.load('ratlas116_MRI.nii')
img_data = img.get_data()
atlasdata=img_data.flatten()
bckvoxels=np.where(atlasdata!=0)
#atlasdata=atlasdata[bckvoxels]
vals=np.unique(atlasdata)
reg_idx={}     # reg_idx contiene los indices de cada region
for i in range(max(vals)+1):
    reg_idx[i]=np.where(atlasdata==i)

def carga_datos_matlab(filename):
    with h5py.File(filename) as f:
        data = [f[element[0]][:] for element in f['rank']]
    return data

def dict_to_mat(d):
    regs=[k for k in d.keys()]
    mat=np.zeros((len(d),len(d[regs[0]])))
    k=0
    for i in regs:
        mat[k,:]=d[i]
        k=k+1
    return mat

# Batches index
# Split data into batches of batch_size
def generate_batches(X,batch_size):            
    l=np.arange(0,X.shape[0])
    batch_idx=[l[i:i+batch_size] for i in range(0,len(l),batch_size)]
    batch_idx=np.array(batch_idx)
    return batch_idx  

def standarize(A):
    mu=np.mean(A, axis=0)
    sigma=np.std(A, axis=0)
    return mu, sigma, ((A - mu) / (sigma+1E-20))

def perf_measure(l,a):
    cm = confusion_matrix(l,a)
    #####from confusion matrix calculate accuracy
    total=sum(sum(cm))
    acc=(cm[0,0]+cm[1,1])/total
    sens = cm[0,0]/(cm[0,0]+cm[0,1])
    spec = cm[1,1]/(cm[1,0]+cm[1,1])
    return acc,sens,spec 

def recortar_region(stack, region, imgsize, reg_idx,thval=0):
    # Se genera la mascara a partir del atlas
    mask_atlas=np.zeros(imgsize.prod()) 
    mask_atlas[reg_idx[region]]=1
    mask_atlas=mask_atlas.reshape(imgsize[0],imgsize[1],imgsize[2])
    eq = [[2, 1], [2, 0], [0, 0]]
    ndim = len(mask_atlas.shape)
    minidx=np.zeros(ndim,dtype=np.int)
    maxidx=np.zeros(ndim,dtype=np.int)
    for ax in range(ndim):
        filtered_lst = [idx for idx,y in enumerate(mask_atlas.any(axis=eq[ax][0]).any(axis=eq[ax][1])) if y > thval]
        minidx[ax] = min(filtered_lst)
        maxidx[ax] = max(filtered_lst)
    stack_masked=np.zeros((stack.shape[0],abs(minidx[0]-maxidx[0]),abs(minidx[1]-maxidx[1]),abs(minidx[2]-maxidx[2])))
        
    for i in range(stack_masked.shape[0]):    
        print('.',end='')
        image=np.zeros(imgsize.prod())
        image[reg_idx[0]]=stack[:,i]
        image=image.reshape(imgsize[0],imgsize[1],imgsize[2])
        stack_masked[i,:,:,:] = image[minidx[0]:maxidx[0], minidx[1]:maxidx[1], minidx[2]:maxidx[2]]
    print('')    
    return stack_masked
    
# Cambio al directorio de los stacks y carga del stack de matlab
os.chdir(stackdir)
#f=h5py.File('stack_NORAD_GM.mat','r')
#variables=f.items() 
#labels=f['labels'].flatten();
#stack=f["stack_NORAD_GM"];
#nobck_idx=f["nobck_idx"]; 
#imgsize=f["imgsize"].astype(np.int32)


#f = sio.loadmat(os.path.join(stackdir, 'stack_NORMCI_W.mat'))
#f.keys() # Muestra las claves del diccionario file. 

f = h5py.File('stack_MCIsMCIc_W.mat')
labels = f["labels"]; labels=np.array(labels).flatten()
stack = f["stack_MCIsMCIc_W"]; stack=np.array(stack).astype(np.float32).T;
imgsize = f["imgsize"]; imgsize=np.array(imgsize).flatten().astype(np.int32)
del f

os.chdir(workdir)

# Define SVM linear voter
clf=svm.SVC(kernel='linear')


# Change labels to 0, 1
ul=np.unique(labels)
l=labels
l[l==ul[0]]=0
l[l==ul[1]]=1
labels==l

# Define CNN and start TF session
n_classes = 2    # Number of classes (for classification)
label_arr = np.zeros((labels.shape[0], n_classes))    # Labels array for FCN
for i in range(n_classes):
    label_arr[:,i] = labels==i     # Generate binary labels array
         

# CNN definitions
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


# Start TF session
# Aqui bucle de regiones
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
skf = StratifiedKFold(n_splits=10,random_state=0)
#skf2=StratifiedShuffleSplit(n_splits=5, test_size=None, random_state=0)
skf2=StratifiedKFold(n_splits=5, random_state=0)


gpu_options = tf.GPUOptions(allow_growth=True)
#sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
#sess = tf.InteractiveSession(config=tf.ConfigProto(intra_op_parallelism_threads=4))

s=0 # Cross validation counter
clasmat_s={}
clasmat_conv_s={}
labels_s={}
W1={}; W2={}
perf_ensemble=np.zeros((10,3))
train_acc_region = np.zeros((117,10))
test_acc_region= np.zeros((117,10))
traintest_region=np.zeros((117,10,train_iter,2))
acc_test_ensemble=np.zeros((10))
for train, test in skf.split(np.zeros(len(labels)), labels):
    clasmat={}; clasmat_conv={}
    for region in lobes_mask_val:
        with tf.Graph().as_default(), tf.Session() as sess:
            #gpu_options = tf.GPUOptions(intra_op_parallelism_threads=4)
            #sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
            # Extract region from stack
            stack_masked=recortar_region(stack, region, imsize_MRI, reg_idx)
            imshape=np.array((stack_masked.shape[1:4]))
            # TF placeholders according to region shape
            x = tf.placeholder(tf.float32, shape=[None, imshape[0], imshape[1], imshape[2]]) 
            y_ = tf.placeholder(tf.float32, shape=[None, n_classes])
            print("Setting up CNN...")  
            x_image = tf.reshape(x, [-1,imshape[0], imshape[1], imshape[2], 1])
            # First CNN layer
            W_conv1 = weight_variable([8, 8, 8, 1, 8], name='Wconv1')
            b_conv1 = bias_variable([8], name='bconv1') 
            #
            h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1, name='relu1') + b_conv1)
            h_pool1 = max_pool_2x2x2(h_conv1, name='pool1')
            # Second CNN layer
            W_conv2 = weight_variable([8, 8, 8, 8, 16], name='Wconv2')
            b_conv2 = bias_variable([16], name='bconv2')
            # 
            h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2, name='relu2') + b_conv2)
            h_pool2 = max_pool_2x2x2(h_conv2, name='pool2')
            finalsize = np.ceil(np.array(list(imshape))/4).astype(int)

            # FC Layer
            W_fc1 = weight_variable([finalsize[0] * finalsize[1] * finalsize[2] * 16, 4000], name='fc1')
            b_fc1 = bias_variable([4000], name='bfc1')
            #
            h_pool2_flat = tf.reshape(h_pool2, [-1, finalsize[0] * finalsize[1] * finalsize[2] * 16], name='pool2f')
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name='fcrelu')
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
            #
            W_fc2 = weight_variable([4000, n_classes])
            b_fc2 = bias_variable([n_classes])
            #
            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
            clas=tf.argmax(y_conv,1)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            firstiter=True
            sess.run(tf.global_variables_initializer())
            trset = stack_masked[train,:,:,:]
            labelsplit = labels[train]
            trlabel = label_arr[train,:]
            print('Region %d, CV iteration %d'%(region,s+1))
            print('------------------------------')
            for i in range(train_iter):
                #batch_idx=generate_batches(trset,50)
                print('Region %d, Step %d '%(region,i+1),end='')
                #print('Training iteration ',i+1)   
                # Train using batches to avoid exhaust memory
                for btrain, btest in skf2.split(trset, labelsplit):
                    train_step.run(feed_dict={x: trset[btest,:,:,:], y_: trlabel[btest,:], keep_prob: 0.5})
                    # Data Augmentation :-)
                    train_step.run(feed_dict={x:np.fliplr(trset[btest,:,:,:]), y_: trlabel[btest,:], keep_prob: 0.5})
                    print('.', end='')
        
                #train_step.run(feed_dict={x: trset, y_: trlabel, keep_prob: 0.5})
                #train_step.run(feed_dict={x:np.fliplr(trset), y_: trlabel, keep_prob: 0.5})

                # Evaluate using both training and test
                acc = accuracy.eval(feed_dict={x:trset, y_: trlabel, keep_prob: 1.0})
                acc_test =  accuracy.eval(feed_dict={x:stack_masked[test,:,:,:], y_: label_arr[test,:], keep_prob: 1.0})
                print(' Training acc=%1.3f, Test acc=%1.3f' %(acc,acc_test))
                traintest_region[region,s,i,:]=np.hstack((acc,acc_test))
                #train_acc.append(acc)
            train_acc_region[region,s]=acc
            acc =  accuracy.eval(feed_dict={x:stack_masked[test,:,:,:], y_: label_arr[test,:], keep_prob: 1.0})
            clasmat[region]=clas.eval(feed_dict={x:stack_masked[test,:,:,:], y_: label_arr[test,:], keep_prob: 1.0})		               
            clasmat_conv[region]=y_conv.eval(feed_dict={x:stack_masked[test,:,:,:], y_: label_arr[test,:], keep_prob: 1.0})            
            print("s=%d, Region=%d, Test accuracy=%1.3f" %(s+1,region,acc))
            test_acc_region[region,s]=acc 
            W1[region,s]=W_conv1.eval(session=sess)
            W2[region,s]=W_conv2.eval(session=sess) 
        sess.close()
        del sess
    # Votacion por mayoria
    clasmat_a=dict_to_mat(clasmat)
    labels_s[s]=labels[test]
    #np.savez('clasmat.npz',clasmat=clasmat,clasmat_a=clasmat_a,clasmat_conv=clasmat_conv,labels_s=labels_s,s_last=s)
    [clas_ens,kk]=stats.mode(clasmat_a,0)
    perf_ensemble[s,:]=perf_measure(labels_s[s],clas_ens.flatten())
    clasmat_s[s]=clasmat
    clasmat_conv_s[s]=clasmat_conv
    s=s+1
    np.savez_compressed('results_MCIsMICc_W_partial.npz',perf_ensemble=perf_ensemble,labels_s=labels_s,clasmat=clasmat_s,clasmat_conv=clasmat_conv_s, W1=W1, W2=W2)
np.savez_compressed('results_MCIsMCIc_W.npz',perf_ensemble=perf_ensemble,labels_s=labels_s,clasmat=clasmat_s,clasmat_conv=clasmat_conv_s, W1=W1, W2=W2)
print("Final accuracy acc= %1.3f, sens=%1.3f, spec=%1.3f"%(perf_ensemble.mean(0)[0],perf_ensemble.mean(0)[1],perf_ensemble.mean(0)[2]))
#data=np.load('mat.npz')
#data.keys()




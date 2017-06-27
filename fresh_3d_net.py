import os
from tensorflow.python import debug as tf_debug
import tensorflow as tf
import tensorflow as tf
from lib import session_helper
from lib import kfrans_ops
import numpy as np
from lib import loss_function as loss
from lib.test_over_segmenting_regions import load_regions_segmented
from lib.math_utils import sample_gaussian
from lib import session_helper
from lib.aux_functionalities.functions import \
    get_batch_from_samples_unsupervised_3d
from lib.aux_functionalities.os_aux import create_directories
from lib.test_over_segmenting_regions import load_regions_segmented
from settings import path_to_project
from settings import path_to_project

n_hidden = 500
n_z = 1000
batchsize = 32
input_shape = [34, 42, 41]
stride = 2
latent_features = 16
learning_rate = 2000000000
kernel_size = 5
# activation = tf.nn.relu

x_in = tf.placeholder(np.float32, [None, 34, 42, 41, 1])
totalsize = 34 * 42 * 41

with tf.variable_scope("layer_conv"):
    w_1 = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, kernel_size,
                                        1, latent_features], stddev=0.05, mean=1), name="weights",
                      trainable=True)
    b_1 = tf.Variable(tf.truncated_normal([latent_features], stddev=0.05, mean=0.0), name="biases",
                      trainable=True)

    conv = tf.add(tf.nn.conv3d(x_in, w_1, strides=[1, stride, stride, stride, 1],
                        padding='SAME'), b_1)

    conv_activate = tf.nn.sigmoid(conv, name="activation")

with tf.variable_scope("layer_deconv"):
    w_2 = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, kernel_size,
                                        1, latent_features], stddev=0.05, mean=0.5), name="weights",
                      trainable=True)

    b_2 = tf.Variable(tf.truncated_normal([1], stddev=0.05, mean=0.0),
                      name="biases", trainable=True)

    output_shape = tf.shape(x_in)
    deconv = tf.add(tf.nn.conv3d_transpose(conv_activate, w_2,
                                    output_shape=output_shape,
                                    strides=[1, stride, stride, stride, 1],
                                    padding='SAME'), b_2)

    deconv_activate = tf.nn.sigmoid(deconv, "activation")

with tf.variable_scope("evaluation_layer"):
    out_flat = tf.reshape(deconv_activate, [-1, totalsize], name="out_flat")
    in_flat = tf.reshape(x_in, [-1, totalsize], name="in_flat")

    cost = tf.reduce_mean(-tf.reduce_sum(
        in_flat * tf.log(1e-3 + out_flat) + (1 - in_flat) *
       tf.log(1e-3 + 1 - out_flat), 1))

    #cost = tf.reduce_mean(tf.pow(tf.subtract(in_flat, out_flat), 2))

    optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost)

regions_used = "three"
list_regions = session_helper.select_regions_to_evaluate(regions_used)
region = 3
region_segmented = load_regions_segmented(list_regions)[3]
print(region_segmented.shape)
print("training")

region_segmented = np.reshape(region_segmented,
                              [region_segmented.shape[0], 34, 42, 41, 1],
                              "F")

region_segmented = region_segmented + \
                   np.random.normal(loc=0, scale=0.001, size=[138, 34, 42, 41,1])

init_op = tf.initialize_all_variables()
session = tf.Session()

logs_path = os.path.join(path_to_project, "cvae_logs")
writer = tf.summary.FileWriter(logs_path, graph=session.graph)

#session = tf_debug.LocalCLIDebugWrapperSession(session)
#session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

session.run(init_op)

for i in range(1, 1000, 1):
    feed_dict = {x_in: region_segmented}
    fetches = [cost]

    [cost_output] = session.run(fetches, feed_dict=feed_dict)

    if i % 20 == 0:
        print("iter {0}, error {1}".format(i, cost_output))

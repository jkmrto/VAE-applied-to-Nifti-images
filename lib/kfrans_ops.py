import numpy as np
import tensorflow as tf


class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                    initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                    initializer=tf.random_normal_initializer(1., 0.02))

                batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed


# standard convolution layer
def conv2d(x, inputdepth, n_filters, stride, name):
    with tf.variable_scope(name):
        w = tf.get_variable("w", [5, 5, inputdepth, n_filters],
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [n_filters],
                            initializer=tf.constant_initializer(0.0))
        return tf.nn.conv2d(
            input=x,
            filter=w,
            strides=[1, stride, stride,1],
            padding="SAME") + b


# standard convolution layer
def conv3d(x, input_features, output_features, stride,  name, kernel_size=5):
    with tf.variable_scope(name):
        w = tf.get_variable("w", [kernel_size, kernel_size, kernel_size,
                                  input_features, output_features],
                            initializer=tf.random_normal_initializer(stddev=0.05))

        b = tf.get_variable("b", [output_features],
                            initializer=tf.random_normal_initializer(stddev=0.05))

        conv = tf.nn.conv3d(x, w, strides=[1,stride, stride, stride, 1],
                            padding='SAME') + b

        return conv


def conv2d_transpose(x, outputShape, stride, name):
    with tf.variable_scope(name):
        # h, w, out, in2
        w = tf.get_variable("w",[5,5, outputShape[-1], x.get_shape()[-1]], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b",[outputShape[-1]], initializer=tf.truncated_normal_initializer(stddev=0.02))
        convt = tf.nn.conv2d_transpose(x, w, output_shape=outputShape, strides=[1,stride,stride,1])
        return convt


def conv3d_transpose(x, output_shape, input_features, output_features,
                     name, stride=2, kernel_size=5):

    with tf.variable_scope(name):

        w = tf.get_variable("w", [kernel_size, kernel_size, kernel_size,
                                  output_features, input_features],
                            initializer=tf.random_normal_initializer(stddev=0.05),
                            trainable=True)

        b = tf.get_variable("b", [output_features],
                            initializer=tf.random_normal_initializer(stddev=0.05),
                            trainable=True)

        convt = tf.nn.conv3d_transpose(x, w,
                                       output_shape=output_shape,
                                       strides=[1, stride, stride, stride, 1],
                                       padding='SAME') + b
        return convt


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d"):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return deconv


# leaky reLu unit
def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


# fully-conected layer
def dense(x, input_len, output_len, scope=None, with_w=False):
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [input_len, output_len],
                                 tf.float32, initializer=
                                 tf.random_normal_initializer(stddev=0.05))
        bias = tf.get_variable("bias", [output_len],
                               initializer=tf.random_normal_initializer(stddev=0.05))
        if with_w:
            return tf.matmul(x, matrix) + bias, matrix, bias
        else:
            return tf.matmul(x, matrix) + bias


def wbVars_conv3d_net(output_shape, input_features, output_features,
                      tride, filter_dim):
    """Helper to initialize weights and biases, via He's adaptation
    of Xavier init for ReLUs: https://arxiv.org/abs/1502.01852
    """
    # (int, int) -> (tf.Variable, tf.Variable)
    stddev = tf.cast((2 / fan_in) ** 0.5, tf.float32)

    initial_w = tf.random_normal([fan_in, fan_out], stddev=stddev)
    initial_b = tf.zeros([fan_out])

    return (tf.Variable(initial_w, trainable=True, name="weights"),
            tf.Variable(initial_b, trainable=True, name="biases"))


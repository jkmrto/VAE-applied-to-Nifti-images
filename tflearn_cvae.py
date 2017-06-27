"""
3d sparse autoencoder applied to ShapeNet dataset.
Usage:
    autoencoder3d.py [options]
    autoencoder3d.py train <data> [<checkpoint_id>] [options]
    autoencoder3d.py encode <checkpoint_id> <data> <out> [options]
Options:
    --gpu-id=<gpu_id>   GPU id [default: 0]
    --sparsity          whether or not to enforce sparsity
    --latent=<dims>     Dimensions in latent representation [default: 64]
    --sp-weight=<wgt>   Sparsity weight [default: 0.001]
    --sp-level=<level>  Sparsity level [default: 0.05]
    --epochs=<epochs>   Number of epochs to run for [default: 10]
    --conv              Use 3d convolutional layers
"""
from lib.test_over_segmenting_regions import load_regions_segmented
from lib import session_helper
import sys
import numpy as np
import matplotlib
import tensorflow as tf
if sys.platform == 'darwin':
    matplotlib.use('macosx')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io
import tflearn
import tensorflow as tf
import docopt
from tflearn.layers.conv import conv_3d, max_pool_3d, conv_3d_transpose

import time

input_shape = [34, 42, 41]
total_size = 34 * 42 * 41
THRESHOLD = 1500 / 27000.
DEFAULT_SPARSITY_LEVEL = 0.05
DEFAULT_SPARSITY_WEIGHT = 0.01
ID_ = str(time.time())[-5:]
CKPT_FORMAT = './checkpoints/{id}/checkpoint_{id}.ckpt'


def kl_divergence(p_hat, p=DEFAULT_SPARSITY_LEVEL):
    return p * tf.log(p) - p * tf.log(p_hat) + \
           (1 - p) * tf.log(1 - p) - (1 - p) * tf.log(1 - p_hat)


def train(
        X=None,
        gpu_id=0,
        sparsity=False,
        latent=64,
        num_filters=4,
        filter_size=2,
        sparsity_level=DEFAULT_SPARSITY_LEVEL,
        sparsity_weight=DEFAULT_SPARSITY_WEIGHT,
        epochs=10,
        checkpoint=None,
        is_training=True):
    assert checkpoint is not None or X is not None, \
        'Either data to train on or model to restore is required.'
    print(' * [INFO] Using GPU %s' % gpu_id)
    print(' * [INFO]', 'Using' if sparsity else 'Not using', 'sparsity')
    print(' * [INFO] Latent dimensions: %d' % latent)
    # Building the encoder

    encoder = tflearn.input_data(shape=[None, 34, 42, 41])
    encoder = tf.reshape(encoder, shape=[-1, 34, 42, 41, 1])

    encoder = conv_3d(encoder, num_filters, filter_size,
                      activation=tf.nn.sigmoid)
    encoder = tflearn.fully_connected(encoder, latent,
                                      activation=tf.nn.sigmoid)

    avg_activations = tf.reduce_mean(encoder, axis=1)
    div = tf.reduce_mean(kl_divergence(avg_activations, sparsity_level))

    # Building the decoder


    decoder = tflearn.fully_connected(encoder, (total_size) * num_filters,
                                  activation=tf.nn.sigmoid)
    decoder = tflearn.reshape(decoder, [-1, 34, 42, 41, num_filters])
    decoder = conv_3d_transpose(decoder, 1, filter_size, [34, 42, 41],
                            activation=tf.nn.sigmoid)

    decoder = tf.reshape(decoder, [-1, 34, 42, 41])

    def sparsity_loss(y_pred, y_true):
        return tf.reduce_mean(tf.square(y_pred - y_true)) + \
               sparsity_weight * div


# Regression, with mean square error
    net = tflearn.regression(decoder, optimizer='adam', learning_rate=1e-4,
                             loss=sparsity_loss if sparsity else 'mean_square',
                             metric=None)

# Training the auto encoder
    model = tflearn.DNN(net, tensorboard_verbose=0)
    encoding_model = tflearn.DNN(encoder, session=model.session)
    saver = tf.train.Saver()
    checkpoint_path = CKPT_FORMAT.format(id=checkpoint or ID_)

    if is_training:
        model.fit(X, X, n_epoch=epochs, run_id="auto_encoder", batch_size=100)
    #           saver.save(encoding_model.session, checkpoint_path)
    else:
        pass
# saver.restore(encoding_model.models, checkpoint_path)

    return {
        'model': model,
        'encoding_model': encoding_model
    }


def evaluate(model, X_test, reconstructed_data=None):
    if reconstructed_data is None:
        print(" * [INFO] Reconstructing test set")
        reconstructed_data = model.predict(X_test)

    print(" * [INFO] Checking accuracy for reconstruction...")
    threshold = THRESHOLD * np.prod(X_test.shape[1:])
    differences = [np.sum(np.abs(x - x_pred)) for x, x_pred in
                   zip(X_test, reconstructed_data)]
    correct = float(len([diff for diff in differences if diff < threshold]))
    accuracy = correct / X_test.shape[0]
    print(' * [INFO] Reconstruction accuracy: %f (avg: %f)' % (
        accuracy, np.mean(differences)))
    return accuracy


if __name__ == '__main__':
    arguments = docopt.docopt(__doc__)
    data_path = arguments['<data>']
    checkpoint = arguments['<checkpoint_id>']
    out = arguments['<out>']
    sparsity_weight = float(arguments['--sp-weight'])
    sparsity_level = float(arguments['--sp-level'])
    gpu_id = arguments['--gpu-id']
    sparsity = arguments['--sparsity']
    latent = int(arguments['--latent'])
    epochs = int(arguments['--epochs'])
    conv = bool(arguments['--conv'])
    is_training = arguments['train']

    regions_used = "three"
    list_regions = session_helper.select_regions_to_evaluate(regions_used)
    region = 3
    region_segmented = load_regions_segmented(list_regions)[3]
    print(region_segmented.shape)

    models = train(region_segmented, gpu_id, sparsity, latent)

    evaluate(models, region_segmented)

import tensorflow as tf

import settings
from lib.math_utils import sample_gaussian
from lib.mnist import mnist_functions
from lib.mnist.mnist_plot import plotSubset
from lib.vae import VAE

ARCHITECTURE = [28**2, # 784 pixels
                500, 500, # intermediate encoding
                10]# latent space dims
                # 50]
# (and symmetrically back out again)

HYPERPARAMS = {
    "batch_size": 128,
    "learning_rate": 5E-4,
    "dropout": 0.9,
    "lambda_l2_reg": 1E-5,
    "nonlinearity": tf.nn.elu,
    "squashing": tf.nn.sigmoid
}

path_save = settings.path_to_project + "/meta/170403_2233_vae_784_500_300_200_100_10-10000"

sess = tf.Session()
new_saver = tf.train.import_meta_graph(path_save + '.meta')
new_saver.restore(sess, path_save)

v = VAE.VAE(ARCHITECTURE, HYPERPARAMS, log_dir=settings.LOG_DIR, meta_graph=path_save)
mnist = mnist_functions.load_mnist()
mnist_aux = mnist.train._images[500:600, :]
print(len(v.encode(mnist_aux)))

code = v.encode(mnist_aux)  # [mu, sigma]
code_sample = sample_gaussian(code[0], code[1])
salida = v.decode(code_sample)

plotSubset(mnist_aux, salida, n = 20, img_name="prueba.png")

sess.close()



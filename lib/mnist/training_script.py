import tensorflow as tf
from lib.aux_functionalities import os_aux
import settings
from lib.mnist import mnist_functions
from lib.vae import VAE

IMG_DIM = 28;

ARCHITECTURE = [IMG_DIM**2, # 784 pixels
                500, 500, 100, # intermediate encoding
                10]# latent space dims
                # 50]
# (and symmetrically back out again)

HYPERPARAMS = {
    "batch_size": 128,
    "learning_rate": 5E-6,
    "dropout": 0.9,
    "lambda_l2_reg": 1E-5,
    "nonlinearity": tf.nn.elu,
    "squashing": tf.nn.sigmoid
}

def main(to_reload=None):
    mnist = mnist_functions.load_mnist()
    mnist = mnist.train._images[0:50000, :]

    if to_reload:  # restore
        v = VAE.VAE(ARCHITECTURE, HYPERPARAMS, meta_graph=to_reload)
        print("Loaded!")

    else:  # train
        v = VAE.VAE(ARCHITECTURE, HYPERPARAMS, log_dir=settings.LOG_DIR)

        v.train(mnist, max_iter = 1000, max_epochs=settings.MAX_EPOCHS, cross_validate=False,
                verbose=True, save_bool=True, sufix_file_saver_name="mnist_")
        print("Trained!")

 #       all_plots(v, mnist)


if __name__ == "__main__":
    tf.reset_default_graph()
    os_aux.create_directories(settings.List_of_dir)
    main()



import tensorflow as tf
import numpy as np
from lib.aux_functionalities import os_aux
import settings
from lib.mri import mri_atlas
from lib.vae import VAE
from lib.mri import stack_NORAD
from lib import utils

os_aux.create_directories(settings.List_of_dir)


super_region_name = 'frontal_lobe_val'
region_selected = 8
region_voxels_index = mri_atlas.load_atlas_mri()[region_selected]
#region_voxels_index = mri_atlas.get_super_region_to_voxels()[region_name]
dict_norad = stack_NORAD.get_gm_stack()  # 'stack' 'voxel_index' 'labels'

# Los voxeles dados por el mapa en las regions vienen ya referidos a su posicion sin background
# Por lo tanto, en una red neuronal no convolucional no hace falta reconstruir la imagen,
# simplemente debemos coger los voxeles, en funcion de cada region
# dict_norad ->  (417, x 438574)

region_voxels_values = dict_norad['stack'][:, region_voxels_index]
region_voxels_values, max_denormalize = utils.normalize_array(region_voxels_values)

#from lib.mnist import mnist_functions
#mnist = mnist_functions.load_mnist()
#mnist_aux = mnist.train._images

input = len(region_voxels_index)

ARCHITECTURE = [input, 1500, 1000, 500, 100]

HYPERPARAMS = {
    "batch_size": 16,
    "learning_rate": 2E-6,
    "dropout": 0.9,
    "lambda_l2_reg": 1E-5,
    "nonlinearity": tf.nn.elu,
    "squashing": tf.nn.sigmoid,
}

v = VAE.VAE(ARCHITECTURE, HYPERPARAMS, log_dir=settings.LOG_DIR)


sufix = 'region_' + str(region_selected) + "_"

v.train(region_voxels_values, max_iter=100,
        save_bool=True, sufix_files_generated=sufix, iters_to_save=1000, iters_to_show_error=100)
print("Trained!")

#       all_plots(v, mnist)


#if __name__ == "__main__":
#    tf.reset_default_graph()
#
#    main()

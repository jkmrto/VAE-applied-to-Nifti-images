import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from lib.utils.os_aux import create_directories
from lib.data_loader.pet_loader import load_pet_data_flat
from lib.vae import VAE
import tensorflow as tf
import settings

region_selected = 3
after_input_architecture = [1000, 500, 100]  # no include hidden layer

# VAE SETTINGS
# Net Configuration
hyperparams = {
    "batch_size": 64,
    "learning_rate": 1E-5,
    "dropout": 0.90,
    "lambda_l2_reg": 1E-5,
    "nonlinearity": tf.nn.elu,
    "squashing": tf.nn.sigmoid,
}

# Vae session cofiguration
session_conf = {
    "bool_normalized": True,
    "max_iter": 100,
    "save_meta_bool": False,
    "show_error_iter": 10,
}

dic_regions_to_flatten_voxels_pet, patient_labels, n_samples = \
    load_pet_data = load_pet_data_flat([region_selected])

region_voxels_values = dic_regions_to_flatten_voxels_pet[region_selected]
architecture = [region_voxels_values.shape[1]]
architecture.extend(after_input_architecture)

path_session_folder = os.path.join(settings.path_to_general_out_folder,
                                   "vae_tensorboard_example")
tf.reset_default_graph()
v = VAE.VAE(architecture, hyperparams,
            path_to_session=path_session_folder,
            generate_tensorboard=True)

#tensorboard --logdir=run1:/media/jkmrto/Multimedia/TFM/VAN-applied-to-Nifti-images/out/vae_tensorboard_example/tb/ --port 6006
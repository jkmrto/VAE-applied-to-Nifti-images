import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import settings
from lib.vae import CVAE
import lib.neural_net.kfrans_ops as ops
import numpy as np
from datetime import datetime
from lib.data_loader.pet_loader import load_pet_regions_segmented
from lib.utils.os_aux import create_directories

"""
THis script has as objective train an CVAE over the 3D images
of a region, printing the evolution of the reconstruction made.
So We can observe how the resolution and the patterns over the
reconstructred image got better
"""


explicit_iter_per_region = {
    73: 300}

# CURRENTLY THIS SCRIP ONLY WORKS WITH PET IMAGES
# Settings

region_selected = 40

hyperparams = {'latent_layer_dim': 100,
               'kernel_size': 5,
               'activation_layer': ops.lrelu,
               'features_depth': [1, 64, 128],
               'decay_rate': 0.0025,
               'learning_rate': 0.001,
               'lambda_l2_regularization': 0.0001}

session_conf = {'bool_normalized': False,
                'n_iters': 1500,
                "batch_size": 16,
                "show_error_iter": 100}

# Path settings
own_datetime = datetime.now().strftime(r"%d_%m_%_Y_%H:%M")
session_prefix_name = "reconstruction_evolution_region_{}".format(region_selected)
session_name = session_prefix_name + "_" + own_datetime

path_to_session = os.path.join(settings.path_to_general_out_folder, session_name)
create_directories([path_to_session])

# Loading Data
list_regions = [region_selected]
region_to_img_dict = load_pet_regions_segmented(list_regions,
                                                bool_logs=False)
train_cube_images = region_to_img_dict[region_selected]

model = CVAE.CVAE(hyperparams, path_to_session=path_to_session)
out = model.train(X=train_cube_images,
    n_iters=session_conf["n_iters"],
    batchsize=session_conf["batch_size"],
    iter_show_error=session_conf["show_error_iter"],
    tempSGD_3dimages=True,
    save_bool=False,
    break_if_nan_error_value=True)

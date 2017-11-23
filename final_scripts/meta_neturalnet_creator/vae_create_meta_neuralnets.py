import os
import sys
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import settings

import lib.neural_net.kfrans_ops as ops
from lib import session_helper
from lib.data_loader.pet_loader import load_pet_data_flat
from lib.delete_pre_final_meta_data import delete_simple_session
from lib.over_regions_lib.vae_over_regions import \
    execute_saving_meta_without_cv


# Loading Data
regions_used = "all"
#list_regions = range(1, 117, 1)
list_regions = session_helper.select_regions_to_evaluate(regions_used)
dic_regions_to_flatten_voxels_pet, patient_labels, n_samples = \
    load_pet_data_flat(list_regions, bool_logs=False)

# Iters per region
max_iters = 2000
explicit_iter_per_region = {}

# VAE SETTINGS
# Net Configuration
hyperparams = {
    "batch_size": 64,
    "learning_rate": 0.0001,
    "dropout": 0.90,
    "lambda_l2_reg": 1E-5,
    "nonlinearity": tf.nn.elu,
    "squashing": tf.nn.sigmoid,
}

# Vae session cofiguration
session_conf = {
    "bool_normalized": True,
    "max_iter": max_iters,
    "save_meta_bool": True,
    "show_error_iter": 100,
    "after_input_architecture": [1500, 1000, 500], # no include hidden layer
}

path_to_session = execute_saving_meta_without_cv(
    dic_region_to_flat_voxels = dic_regions_to_flatten_voxels_pet,
    hyperparams = hyperparams,
    session_conf = session_conf,
    after_input_architecture = session_conf["after_input_architecture"],
    list_regions = list_regions,
    path_to_root = settings.path_to_general_out_folder,
    explicit_iter_per_region = explicit_iter_per_region,
    session_prefix_name="vae_2000iters")


session_to_clean_meta_folder = os.path.join(path_to_session, "meta")
delete_simple_session(session_to_clean_meta_folder=session_to_clean_meta_folder)
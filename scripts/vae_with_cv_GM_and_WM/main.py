from datetime import datetime
from scripts.vae_with_cv_GM_and_WM import vae_over_regions
from lib.aux_functionalities.os_aux import create_directories
from lib.mri import stack_NORAD
import tensorflow as tf
from lib import session_helper as session
from lib.cv_utils import generate_and_store_train_and_test_index
from scripts.vae_with_cv_GM_and_WM import session_settings

cv_rate = 0.8



# Loading the stack of images
dict_norad_gm = stack_NORAD.get_gm_stack()
dict_norad_wm = stack_NORAD.get_wm_stack()

# TEST AND TRAIN SPLIT DATASET
# The indexes generated point to the same images of both stack WM and GM
# so we just need to make cv over one of the stack, and extend the
# results to the other stack
generate_and_store_train_and_test_index(dict_norad_gm['stack'], cv_rate,
                                        session_settings.path_cv_folder)

hyperparams = {
    "batch_size": 16,
    "learning_rate": 5E-6,
    "dropout": 0.9,
    "lambda_l2_reg": 1E-5,
    "nonlinearity": tf.nn.elu,
    "squashing": tf.nn.sigmoid,
}

# Neural net architecture
after_input_architecture = [1000, 800, 500, 100]

# SESSION CONFIGURATION
session_conf = {
    "cv_rate": 0.6,
    "bool_normalized": True,
    "regions_used": "all",
    "max_iter": 2000,
}

# Selecting the GM folder
path_to_root = session_settings.path_GM_folder
path_cv_index_folder = session_settings.path_cv_folder

vae_over_regions.execute(dict_norad_gm, hyperparams, session_conf,
                         after_input_architecture,
                         path_to_root,
                         path_to_cv_index_folder=path_cv_index_folder)


# Selecting the WM folder
path_to_root = session_settings.path_WM_folder
path_cv_index_folder = session_settings.path_cv_folder

vae_over_regions.execute(dict_norad_wm, hyperparams, session_conf,
                         after_input_architecture,
                         path_to_root,
                         path_to_cv_index_folder=path_cv_index_folder)
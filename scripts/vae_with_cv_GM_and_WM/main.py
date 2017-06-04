import tensorflow as tf

from lib import cv_utils
from lib.data_loader import MRI_stack_NORAD
from scripts.vae_with_cv_GM_and_WM import session_settings

cv_rate = 0.8

# Loading the stack of images
dict_norad_gm = MRI_stack_NORAD.get_gm_stack()
dict_norad_wm = MRI_stack_NORAD.get_wm_stack()

# TEST AND TRAIN SPLIT DATASET
# The indexes generated point to the same images of both stack WM and GM
# so we just need to make cv over one of the stack, and extend the
# results to the other stack

n_folds = 10
cv_utils.generate_k_fold(session_settings.path_kfolds_folder,
                         dict_norad_gm['stack'], n_folds)

for k_fold_index in range(1, 11, 1):
    train_index, test_index = cv_utils.get_train_and_test_index_from_k_fold(
        session_settings.path_kfolds_folder, 1, n_folds)

    print(len(train_index))
    print(len(test_index))

# Selecting the GM folder
path_to_root_GM = session_settings.path_kfolds_GM_folder
path_to_root_WM = session_settings.path_kfolds_WM_folder

hyperparams = {
    "batch_size": 16,
    "learning_rate": 5E-6,
    "dropout": 0.9,
    "lambda_l2_reg": 1E-5,
    "nonlinearity": tf.nn.elu,
    "squashing": tf.nn.sigmoid,
}

# Neural net architecture
after_input_architecture = [1000, 500, 100]

# SESSION CONFIGURATION
session_conf = {
    "bool_normalized": True,
    "regions_used": "three",
    "max_iter": 150,
    "save_meta_bool": False,
}
#for k_fold_index in range(1,11,1):
#    vae_over_regions_kfolds.execute(dict_norad_gm, hyperparams, session_conf,
#                             after_input_architecture,
#                            path_to_root_GM, k_fold_index, n_folds)

#    path_to_root = session_settings.path_kfolds_GM_folder

#    vae_over_regions_kfolds.execute(dict_norad_wm, hyperparams, session_conf,
#                             after_input_architecture,
#                            path_to_root_WM, k_fold_index, n_folds)
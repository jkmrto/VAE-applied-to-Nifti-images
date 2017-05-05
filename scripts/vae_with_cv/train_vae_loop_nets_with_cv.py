import tensorflow as tf
from lib.aux_functionalities.os_aux import create_directories
from lib.aux_functionalities.functions import generate_session_descriptor
from lib import region_selector_hub
from datetime import datetime
from lib.aux_functionalities import functions
import settings
import os
from lib.mri import mri_atlas
from lib.vae import VAE
from lib.mri import stack_NORAD
from lib import utils
import numpy as np


def init_session_folders(architecture):
    """
    This method will create inside the "out" folder a folder with the datetime
    of the execution of the neural net and with, with 3 folders inside it
    :return:
    """
    own_datetime = datetime.now().strftime(r"%d_%m_%_Y_%H:%M")
    path_to_general_out_folder = os.path.join(settings.path_to_project, "out")
    iden_session = own_datetime + " arch: " + "_".join(map(str, (architecture)))
    path_session_folder = os.path.join(path_to_general_out_folder,
                                       iden_session)

    path_to_images = os.path.join(path_session_folder, "images")
    path_to_logs = os.path.join(path_session_folder, "logs")
    path_to_meta = os.path.join(path_session_folder, "meta")
    path_to_cv = os.path.join(path_session_folder, "cv")
    path_to_grad_desc_error = os.path.join(path_to_logs, "DescGradError")
    path_to_grad_desc_error_images = os.path.join(path_to_images,
                                                  "DescGradError")

    create_directories([path_session_folder, path_to_images,
                        path_to_logs, path_to_meta, path_to_grad_desc_error,
                        path_to_grad_desc_error_images, path_to_cv])

    return path_session_folder, path_to_grad_desc_error, \
           path_to_grad_desc_error_images, path_to_cv


def generate_and_store_train_and_test_index(stack, cv_rate, path_to_cv):
    train_index = np.random.choice(range(stack.shape[0]),
                                   int(cv_rate * stack.shape[0]), replace=False)
    train_index.sort()
    test_index = [index for index in range(0, stack.shape[0], 1) if
                  index not in train_index]

    np.savetxt(os.path.join(path_to_cv, "train_index_to_stack.csv"), train_index,
               delimiter=',')
    np.savetxt(os.path.join(path_to_cv, "test_index_to_stack.csv"), test_index,
               delimiter=',')

    return train_index  # only returning train_index because it an unsupervesid learning method


def plot_grad_desc_error_per_region(path_to_grad_desc_error, region_selected,
                                    path_to_grad_desc_error_images):
    path_to_grad_desc_error_region_log = os.path.join(
        path_to_grad_desc_error, "region_{}.log".format(region_selected))
    path_to_grad_desc_error_region_image = os.path.join(
        path_to_grad_desc_error_images, "region_{}.png".format(region_selected))

    functions.plot_x_y_from_file_with_title(
        "Region {}".format(region_selected), path_to_grad_desc_error_region_log,
        path_to_grad_desc_error_region_image)


# SESSION CONFIGURATION
HYPERPARAMS = {
    "batch_size": 16,
    "learning_rate": 5E-6,
    "dropout": 0.9,
    "lambda_l2_reg": 1E-5,
    "nonlinearity": tf.nn.elu,
    "squashing": tf.nn.sigmoid,
}

cv_rate = 0.6  # cv_rate training data and (1 - cv_rate) test data
bool_normalized = True
max_denormalize = 1
regions_used = "all"
max_iter = 1500

region_voxels_index_per_region = mri_atlas.load_atlas_mri()

after_input_architecture = [1000, 800, 500, 100]
path_session_folder, path_to_grad_desc_error, \
path_to_grad_desc_error_images, path_to_cv = init_session_folders(
    after_input_architecture)

# region_voxels_index = mri_atlas.get_super_region_to_voxels()[region_name]
dict_norad = stack_NORAD.get_gm_stack()  # 'stack' 'voxel_index' 'labels'

train_index = generate_and_store_train_and_test_index(dict_norad['stack'],
                                                      cv_rate, path_to_cv)


# SESSION DESCRIPTOR CREATION
session_descriptor_data = {"voxeles normalized": str(bool_normalized),
                           "max_iter": max_iter,
                           "voxels normalized by": str(max_denormalize),
                           "architecture:": "input_" + "_".join(
                               str(x) for x in after_input_architecture),
                           "regions used": str(regions_used)}
session_descriptor_data.update(HYPERPARAMS)
generate_session_descriptor(path_session_folder, session_descriptor_data)


# LIST REGIONS SELECTION
list_regions = region_selector_hub.select_regions_to_evaluate(regions_used)

# LOOP OVER REGIONS
for region_selected in list_regions:
    print("Region Nº {} selected".format(region_selected))
    voxels_index = region_voxels_index_per_region[region_selected]
    # We map the voxels indexes to his voxels value,
    # which is stored in the Stacked previously loaded

    # First map and the normalize to the unit the value of the pixels
    region_voxels_values = dict_norad['stack'][:, voxels_index]
    region_voxels_values = region_voxels_values[train_index, :]

    if bool_normalized: region_voxels_values, max_denormalize = \
        utils.normalize_array(region_voxels_values)

    architecture = [region_voxels_values.shape[1]]
    architecture.extend(after_input_architecture)
    tf.reset_default_graph()
    v = VAE.VAE(architecture, HYPERPARAMS, path_to_session=path_session_folder)

    region_suffix = 'region_' + str(region_selected)

    v.train(region_voxels_values, max_iter=max_iter,
            save_bool=True, suffix_files_generated=region_suffix,
            iter_to_save=500, iters_to_show_error=100)

    # Script para pintar
    print("Region {} Trained!".format(region_selected))

    plot_grad_desc_error_per_region(path_to_grad_desc_error, region_selected,
                                    path_to_grad_desc_error_images)



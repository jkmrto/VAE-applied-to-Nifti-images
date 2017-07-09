import os
from datetime import datetime

import tensorflow as tf

import settings
from lib import session_helper
from lib import utils
from lib.aux_functionalities.os_aux import create_directories
from lib.data_loader import MRI_stack_NORAD
from lib.data_loader import mri_atlas
from lib.session_helper import generate_session_descriptor
from lib.session_helper import plot_grad_desc_error_per_region
from lib.utils.cv_utils import generate_and_store_train_and_test_index
from lib.vae import VAE


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
regions_used = "all"
max_iter = 1500

region_voxels_index_per_region = mri_atlas.load_atlas_mri()

after_input_architecture = [1000, 800, 500, 100]
path_session_folder, path_to_grad_desc_error, \
path_to_grad_desc_error_images, path_to_cv = init_session_folders(
    after_input_architecture)

# region_voxels_index = mri_atlas.get_super_region_to_voxels()[region_name]
dict_norad = MRI_stack_NORAD.get_gm_stack()  # 'stack' 'voxel_index' 'labels'

train_index = generate_and_store_train_and_test_index(dict_norad['stack'],
                                                      cv_rate, path_to_cv)


# SESSION DESCRIPTOR CREATION
session_descriptor_data = {"voxeles normalized": str(bool_normalized),
                           "max_iter": max_iter,
                           "voxels normalized by": str(bool_normalized),
                           "architecture:": "input_" + "_".join(
                               str(x) for x in after_input_architecture),
                           "regions used": str(regions_used)}
session_descriptor_data.update(HYPERPARAMS)
generate_session_descriptor(path_session_folder, session_descriptor_data)


# LIST REGIONS SELECTION
list_regions = session_helper.select_regions_to_evaluate(regions_used)

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



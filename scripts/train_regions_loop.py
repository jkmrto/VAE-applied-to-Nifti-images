import tensorflow as tf
from lib.aux_functionalities.os_aux import create_directories
from datetime import datetime
import settings
import os
from lib.mri import mri_atlas
from lib.vae import VAE
from lib.mri import stack_NORAD
from lib import utils

# Hyperparameters and architecture for all the regions
HYPERPARAMS = {
    "batch_size": 16,
    "learning_rate": 5E-5,
    "dropout": 0.9,
    "lambda_l2_reg": 1E-5,
    "nonlinearity": tf.nn.elu,
    "squashing": tf.nn.sigmoid,
}

# region_voxels_index = mri_atlas.get_super_region_to_voxels()[region_name]
dict_norad = stack_NORAD.get_gm_stack()  # 'stack' 'voxel_index' 'labels'

list_regions = settings.list_regions_evaluated


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
    path_to_grad_desc_error = os.path.join(path_to_logs, "DescGradError")

    create_directories([path_session_folder, path_to_images,
                        path_to_logs, path_to_meta, path_to_grad_desc_error])

    return path_session_folder

architecture = [1000, 800, 500, 100]
path_session_folder = init_session_folders(architecture)

for region_selected in list_regions:
    # Here we have the index of the voxels of the selected region
    print("Region Nº {}".format(region_selected))
    region_voxels_index = mri_atlas.load_atlas_mri()[region_selected]
    # We map the voxels indexes to his voxels value, which is stored in the Stacked previously loaded
    # First map and the normalize to the unit the value of the pixels
    region_voxels_values = dict_norad['stack'][:, region_voxels_index]
    region_voxels_values, max_denormalize = utils.normalize_array(region_voxels_values)

    architecture = [len(region_voxels_index), 1000, 800, 500, 100]
    tf.reset_default_graph()
    v = VAE.VAE(architecture, HYPERPARAMS, path_to_session=path_session_folder)

    region_suffix = 'region_' + str(region_selected) + "_"

    v.train(region_voxels_values, max_iter=100,
            save_bool=True, suffix_files_generated=region_suffix, iter_to_save=100, iters_to_show_error=100)
    print("Trained!")

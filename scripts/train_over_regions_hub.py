#import tensorflow as tf
from lib.aux_functionalities.os_aux import create_directories
from datetime import datetime
from lib.aux_functionalities import functions
import settings
from lib import session_helper as session
import os
from lib.data_loader import mri_atlas
from lib.vae import VAE
from lib.data_loader import MRI_stack_NORAD
from lib import utils


def init_session_folders(architecture, folder_prefix=""):
    """
    This method will create inside the "out" folder a folder with the datetime
    of the execution of the neural net and with, with 3 folders inside it
    :return:
    """
    own_datetime = datetime.now().strftime(r"%d_%m_%_Y_%H:%M")
    path_to_general_out_folder = os.path.join(settings.path_to_project, "out")
    iden_session = folder_prefix + "_" + own_datetime + " arch: " + "_".join(map(str, (architecture)))
    path_session_folder = os.path.join(path_to_general_out_folder,
                                       iden_session)

    path_to_images = os.path.join(path_session_folder, "images")
    path_to_logs = os.path.join(path_session_folder, "logs")
    path_to_meta = os.path.join(path_session_folder, "meta")
    path_to_grad_desc_error = os.path.join(path_to_logs, "DescGradError")
    path_to_grad_desc_error_images = os.path.join(path_to_images,
                                                  "DescGradError")

    create_directories([path_session_folder, path_to_images,
                        path_to_logs, path_to_meta, path_to_grad_desc_error,
                        path_to_grad_desc_error_images])

    return path_session_folder, path_to_grad_desc_error, \
           path_to_grad_desc_error_images


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
def run_session(dict_norad, region_voxels_index_per_region, vae_hyperparams, session_settings):
    dict_norad = MRI_stack_NORAD.get_gm_stack()  # 'stack' 'voxel_index' 'labels'
    region_voxels_index_per_region = mri_atlas.load_atlas_mri()

    path_session_folder, path_to_grad_desc_error, \
    path_to_grad_desc_error_images = init_session_folders(session_settings["after_input_architecture"])

    # SESSION DESCRIPTRO
    session_descriptor = {}
    session_descriptor.update(session_settings)
    session_descriptor.update(vae_hyperparams)
    session.generate_session_descriptor(path_session_folder, session_descriptor)

    list_regions = session.select_regions_to_evaluate(session_settings["regions_used"])

    if session_settings["bool_norm_truncate"]:
        dict_norad['stack'][dict_norad['stack'] < 0] = 0
        dict_norad['stack'][dict_norad['stack'] > 1] = 1

    for region_selected in list_regions:
        print("Region Nº {} selected".format(region_selected))
        voxels_index = region_voxels_index_per_region[region_selected]
        region_voxels_values = dict_norad['stack'][:, voxels_index]

        if session_settings["bool_normalized_per_region"]:
            region_voxels_values, max_denormalize = \
                utils.normalize_array(region_voxels_values)

        architecture = [region_voxels_values.shape[1]]
        architecture.extend(session_settings["after_input_architecture"])

  #      tf.reset_default_graph()
        v = VAE.VAE(architecture, vae_hyperparams, path_to_session=path_session_folder)

        region_suffix = 'region_' + str(region_selected)

        v.train(region_voxels_values, max_iter=session_settings["max_iter"],
                save_bool=True, suffix_files_generated=region_suffix,
                iter_to_save=session_settings["iter_to_save"],
                iters_to_show_error=session_settings["iter_to_show_error"],
                bool_log_grad_desc_error=session_settings["bool_log_grad_desc_error"])

        # Script para pintar
        print("Region {} Trained!".format(region_selected))

        if session_settings["bool_log_grad_desc_error"]:
            plot_grad_desc_error_per_region(path_to_grad_desc_error, region_selected,
                                            path_to_grad_desc_error_images)

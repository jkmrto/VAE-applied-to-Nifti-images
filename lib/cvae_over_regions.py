import os
import tensorflow as tf
from lib.data_loader import mri_atlas
from lib import session_helper as session
from datetime import datetime
from lib.data_loader import MRI_stack_NORAD
from lib import cv_utils
from lib import utils
from lib.vae import VAE
from lib import session_helper
from lib.test_over_segmenting_regions import load_regions_segmented
from lib.aux_functionalities.os_aux import create_directories
from scripts.vae_with_cv_GM_and_WM import session_settings
import lib.kfrans_ops as ops
from lib.session_helper import generate_session_descriptor
import numpy as np
import main_kvfrans3d


def init_session_folders(architecture, path_to_root):
    """
    This method will create inside the "out" folder a folder with the datetime
    of the execution of the neural net and with, with 3 folders inside it
    :return:
    """
    own_datetime = datetime.now().strftime(r"%d_%m_%_Y_%H:%M")
    iden_session = own_datetime + " arch: " + "_".join(map(str, (architecture)))
    path_session_folder = os.path.join(path_to_root,
                                       iden_session)

    path_to_images = os.path.join(path_session_folder, session.folder_images)
    path_to_logs = os.path.join(path_session_folder, session.folder_log)
    path_to_meta = os.path.join(path_session_folder, session.folder_meta)
    path_to_encoding_out = os.path.join(path_session_folder,
                                        session.folder_encoding_out)
    path_to_encoding_out_test = os.path.join(path_session_folder,
                                             session.folder_encoding_out_test)
    path_to_encoding_out_train = os.path.join(path_session_folder,
                                              session.folder_encoding_out_train)
    path_to_grad_desc_error = os.path.join(path_to_logs, "DescGradError")
    path_to_grad_desc_error_images = os.path.join(path_to_images,
                                                  "DescGradError")

    create_directories([path_session_folder, path_to_images,
                        path_to_logs, path_to_meta, path_to_grad_desc_error,
                        path_to_grad_desc_error_images, path_to_encoding_out,
                        path_to_encoding_out, path_to_encoding_out_test,
                        path_to_encoding_out_train])

    return path_session_folder, path_to_grad_desc_error, \
           path_to_grad_desc_error_images, path_to_encoding_out_test, \
           path_to_encoding_out_train


def auto_execute():
    # Autoenconder configuration
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
        "regions_used": "68to117",
        "max_iter": 2000,
    }

    # Selecting the GM folder
    # dict_norad = stack_NORAD.get_gm_stack()
    # path_to_root = session_settings.path_GM_folder
    # path_cv_index_folder = session_settings.path_cv_folder

    # Selecting the wM folder
    dict_norad = MRI_stack_NORAD.get_wm_stack()
    path_to_root = session_settings.path_WM_folder
    path_cv_index_folder = session_settings.path_cv_folder

    execute(dict_norad, hyperparams, session_conf, after_input_architecture,
            path_to_root, path_to_cv_index_folder=path_cv_index_folder)


def execute(voxels_values, hyperparams, session_conf, after_input_architecture,
            list_regions, path_to_root):
    per_region_results = {}

    region_voxels_index_per_region = mri_atlas.load_atlas_mri()

    path_session_folder, path_to_grad_desc_error, \
    path_to_grad_desc_error_images, \
    path_to_encoding_out_test, path_to_encoding_out_train \
        = init_session_folders(after_input_architecture, path_to_root)

    # LOOP OVER REGIONS
    for region_selected in list_regions:
        print("Region Nº {} selected".format(region_selected))
        voxels_index = region_voxels_index_per_region[region_selected]

        # First map and the normalize to the unit the value of the pixels
        # filtering voxels per region
        region_voxels_values_train = voxels_values['train'][:, voxels_index]
        region_voxels_values_test = voxels_values['test'][:, voxels_index]

        if session_conf['bool_normalized']:
            region_voxels_values_train, max_denormalize = \
                utils.normalize_array(region_voxels_values_train)
            region_voxels_values_test = region_voxels_values_test / max_denormalize

        architecture = [region_voxels_values_train.shape[1]]
        architecture.extend(after_input_architecture)

        tf.reset_default_graph()
        v = VAE.VAE(architecture, hyperparams,
                    path_to_session=path_session_folder)

        region_suffix = 'region_' + str(region_selected)

        v.train(region_voxels_values_train,
                max_iter=session_conf["max_iter"],
                save_bool=session_conf['save_meta_bool'],
                suffix_files_generated=region_suffix,
                iter_to_save=500,
                iters_to_show_error=session_conf['show_error_iter'])

        # Script para pintar
        print("Region {} Trained!".format(region_selected))

        #      session.plot_grad_desc_error_per_region(path_to_grad_desc_error,
        #                                              region_selected,
        #                                              path_to_grad_desc_error_images)

        # ENCODING PHASE
        # Encoding samples for the next step
        train_output = v.encode(region_voxels_values_train)
        test_output = v.encode(region_voxels_values_test)

        session.save_encoding_output_per_region(path_to_encoding_out_train,
                                                train_output, region_selected)
        session.save_encoding_output_per_region(path_to_encoding_out_test,
                                                test_output, region_selected)

        per_region_results[str(region_selected)] = {}

        per_region_results[str(region_selected)]['train_output'] = train_output
        per_region_results[str(region_selected)]['test_output'] = test_output

    return per_region_results


def execute_without_any_logs(region_images_dict, hyperparams, session_conf,
                             list_regions, path_to_root=None):
    """

    :param voxels_values:
    :param hyperparams:
    :param session_conf:
    :param after_input_architecture:
    :param list_regions:
    :param path_to_root:
    :return:
    """


    per_region_results = {}

    # LOOP OVER REGIONS
    for region_selected in list_regions:
        print("Region Nº {} selected".format(region_selected))
        # Selecting the cubes_images of the region selected
        region_cube_images = region_images_dict[region_selected]

        # Updating hyperparmas due to the dimensions of the cubes of the region
        hyperparams['image_shape'] = region_cube_images.shape[1:]
        hyperparams['total_size'] = np.array(region_cube_images.shape[1:]).prod()


        region_cube_images_train = region_cube_images
        #region_cube_images_test = region_cube_images['test']

        if session_conf['bool_normalized']:
            region_voxels_values_train, max_denormalize = \
                utils.normalize_array(region_voxels_values_train)
            region_voxels_values_test = region_voxels_values_test / max_denormalize

        tf.reset_default_graph()
        model = main_kvfrans3d.LatentAttention(hyperparams)

        region_suffix = 'region_' + str(region_selected)

        model.train(X=region_cube_images_train, n_iters=session_conf['n_iters'],
                    batchsize=session_conf["batch_size"])

        # Script para pintar
        print("Region {} Trained!".format(region_selected))

        # ENCODING PHASE
        # Encoding samples for the next step
        train_output = model.encode(region_cube_images_train)
        #test_output = v.encode(region_voxels_values_test)

        per_region_results[str(region_selected)] = {}

        per_region_results[str(region_selected)]['train_output'] = train_output
      #  per_region_results[str(region_selected)]['test_output'] = test_output

    return per_region_results


def auto_execute_without_logs():
    regions_used = "three"
    region_selected = 38
    list_regions = session_helper.select_regions_to_evaluate(regions_used)
    train_images = load_regions_segmented(list_regions, bool_logs=False)

    hyperparams = {}
    hyperparams['latent_layer_dim'] = 100
    hyperparams['kernel_size'] = 5
    hyperparams['activation_layer'] = ops.lrelu
    hyperparams['features_depth'] = [1, 16, 32]
    hyperparams['decay_rate'] = 0.0002
    hyperparams['learning_rate'] = 0.001
    hyperparams['lambda_l2_regularization'] = 0.0001

    session_conf = {'bool_normalized': False,
                    'n_iters': 50,
                    "batch_size": 16}

    results = execute_without_any_logs(train_images, hyperparams, session_conf,
                                       list_regions, path_to_root=None)

auto_execute_without_logs()
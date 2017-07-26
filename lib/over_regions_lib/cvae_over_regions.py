import os
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
from lib.utils import output_utils
import lib.neural_net.kfrans_ops as ops
import settings
from lib import session_helper
from lib import utils
from lib.aux_functionalities.os_aux import create_directories
from lib.data_loader.pet_loader import load_pet_regions_segmented
from lib.utils import cv_utils
from lib.vae import CVAE


def get_adequate_number_iterations(region_selected, explicit_iter_per_region,
                                   predefined_iters):
    if region_selected in explicit_iter_per_region.keys():
        if explicit_iter_per_region[region_selected] < predefined_iters:
            max_train_iter = explicit_iter_per_region[region_selected]
        else:
            max_train_iter = predefined_iters
    else:
        max_train_iter = predefined_iters

    return max_train_iter


def execute_saving_meta_graph_without_any_cv(region_cubes_dict, hyperparams,
                                             session_conf, list_regions,
                                             path_to_root,
                                             session_prefix="",
                                             explicit_iter_per_region={}):

    own_datetime = datetime.now().strftime(r"%d_%m_%_Y_%H:%M")
    session_name = session_prefix + "_" + own_datetime

    path_to_session = os.path.join(path_to_root, session_name)
    create_directories([path_to_session])

    #Session description issues
    session_descriptor = {}
    session_descriptor['VAE hyperparameters'] = hyperparams
    session_descriptor['VAE session configuration'] = session_conf
    path_session_description_file = os.path.join(path_to_session,
                                                 "session_description.txt")

    file_session_descriptor = open(path_session_description_file, "w")
    output_utils.print_recursive_dict(session_descriptor,
                                      file=file_session_descriptor)
    file_session_descriptor.close()


    # LOOP OVER REGIONS
    for region_selected in list_regions:
        print("Region Nº {} selected".format(region_selected))

        # Selecting the cubes_images of the region selected
        train_cube_images = region_cubes_dict[region_selected]

        # Updating hyperparameters
        hyperparams['image_shape'] = train_cube_images.shape[1:]

        # Currently the normalization is not inclueded
        if session_conf['bool_normalized']:
            train_cube_images[train_cube_images < 0] = 0
            train_cube_images[train_cube_images > 1] = 1

        tf.reset_default_graph()
        model = CVAE.CVAE(hyperparams, path_to_session=path_to_session)

        max_train_iter = get_adequate_number_iterations(region_selected,
                                                        explicit_iter_per_region,
                                                        predefined_iters=
                                                        session_conf["n_iters"])
        print("Numbers Iters requered {}".format(max_train_iter))
        out = model.train(X=train_cube_images,
                          n_iters=max_train_iter,
                          batchsize=session_conf["batch_size"],
                          iter_show_error=session_conf["show_error_iter"],
                          tempSGD_3dimages=True,
                          save_bool=True,
                          break_if_nan_error_value=True,
                          suffix_files_generated="region_{}".format(
                              region_selected))

        if out == -1:
            print("Region {} Training process failed!"
                  "SGD doesnt converge".format(region_selected))
            print("Exiting, readjust parameter for that region")
            sys.exit(0)

        elif out == 0:
            print("Region {} Correctly Trained and Saved!".format(
                region_selected))

    return path_to_session


def auto_execute_saving_meta_graph_without_any_cv(hyperparams=None,
                                                  session_conf=None):
    regions_used = "all"
    list_regions = session_helper.select_regions_to_evaluate(regions_used)
    region_to_img_dict = load_pet_regions_segmented(list_regions,
                                                    bool_logs=False)

    # Autoenconder configuration
    if hyperparams is None:
        hyperparams = {'latent_layer_dim': 100,
                       'kernel_size': 5,
                       'activation_layer': ops.lrelu,
                       'features_depth': [1, 16, 32],
                        'decay_rate': 0.0002,
                        'learning_rate': 0.001,
                        'lambda_l2_regularization': 0.0001}

    if session_conf is None:
    # SESSION CONFIGURATION
        session_conf = {'bool_normalized': False,
                       'n_iters': 100,
                        "batch_size": 16,
                        "show_error_iter": 10}

    execute_saving_meta_graph_without_any_cv(
        region_cubes_dict=region_to_img_dict,
        hyperparams=hyperparams,
        session_conf=session_conf,
        list_regions=list_regions,
        path_to_root=settings.path_to_general_out_folder,
        session_prefix="test_saving_meta_PET")

#auto_execute_saving_meta_graph_without_any_cv()


def execute_without_any_logs(region_train_cubes_dict, hyperparams, session_conf,
                             list_regions, path_to_root=None,
                             region_test_cubes_dict=None,
                             explicit_iter_per_region=[]):
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
    regions_whose_net_not_converge = []

    # LOOP OVER REGIONS
    for region_selected in list_regions:
        print("Region Nº {} selected".format(region_selected))

        # Selecting the cubes_images of the region selected
        train_cube_images = region_train_cubes_dict[region_selected]

        # If cv is requested
        test_cube_images = None
        if region_test_cubes_dict is not None:
            test_cube_images = region_test_cubes_dict[region_selected]

        # Updating hyperparameters due to the dimensions of the cubes of the region
        hyperparams['image_shape'] = train_cube_images.shape[1:]
        hyperparams['total_size'] = np.array(train_cube_images.shape[1:]).prod()

        # Currently the normalization is not inclueded
        if session_conf['bool_normalized']:
            region_voxels_values_train, max_denormalize = \
                utils.normalize_array(region_voxels_values_train)
            region_voxels_values_test = region_voxels_values_test / max_denormalize

        tf.reset_default_graph()
        model = CVAE.CVAE(hyperparams)

        max_train_iter = get_adequate_number_iterations(region_selected,
                                                        explicit_iter_per_region,
                                                        predefined_iters=
                                                        session_conf["n_iters"])

        out = model.train(X=train_cube_images,
                          n_iters=max_train_iter,
                          batchsize=session_conf["batch_size"],
                          iter_show_error=session_conf["show_error_iter"],
                          tempSGD_3dimages=False,
                          save_bool=False,
                          break_if_nan_error_value=True)

        if out == -1:
            print("Region {} Training process failed!"
                  "SGD doesnt converge".format(region_selected))
            regions_whose_net_not_converge.append(region_selected)

        elif out == 0:
            print("Region {} Correctly Trained!".format(region_selected))

            # ENCODING PHASE
            # Encoding samples for the next step
            train_output = model.encode(train_cube_images)
            test_output = model.encode(test_cube_images)

            per_region_results[region_selected] = {}

            per_region_results[region_selected]['train_output'] = train_output
            per_region_results[region_selected]['test_output'] = test_output

    return per_region_results, regions_whose_net_not_converge


def auto_execute_without_logs_over_one_region():
    regions_used = "three"
    list_regions = session_helper.select_regions_to_evaluate(regions_used)
    region_to_img_dict = load_pet_regions_segmented(list_regions,
                                                    bool_logs=False)

    n_folds = 10
    n_samples = region_to_img_dict[1].shape[0]
    k_fold_dict = cv_utils.generate_k_folder_in_dict(n_samples=n_samples,
                                                     n_folds=n_folds)

    reg_to_group_to_images_dict = cv_utils.restructure_dictionary_based_on_cv_index_3dimages(
        dict_train_test_index=k_fold_dict[0],
        region_to_img_dict=region_to_img_dict)

    hyperparams = {'latent_layer_dim': 20,
                   'kernel_size': 5,
                   'activation_layer': ops.lrelu,
                   'features_depth': [1, 16, 32],
                   'decay_rate': 0.0002,
                   'learning_rate': 0.001,
                   'lambda_l2_regularization': 0.0001}

    session_conf = {'bool_normalized': False,
                    'n_iters': 50,
                    "batch_size": 16,
                    "show_error_iter": 10}

    results = execute_without_any_logs(
        region_train_cubes_dict=reg_to_group_to_images_dict['train'],
        hyperparams=hyperparams,
        session_conf=session_conf,
        path_to_root=None,
        list_regions=list_regions,
        region_test_cubes_dict=reg_to_group_to_images_dict['test'])

    return results

# out = auto_execute_without_logs_over_one_region()

import os
import sys
from datetime import datetime
import numpy as np
import tensorflow as tf
from lib import session_helper as session
from lib import utils
from lib.utils.os_aux import create_directories
from lib.vae import CVAE
from lib.vae import CVAE_helper


def execute_saving_meta_graph_without_any_cv(region_cubes_dict, hyperparams,
                                             session_conf, list_regions,
                                             path_to_root,
                                             session_prefix_name="",
                                             explicit_iter_per_region={}):

    own_datetime = datetime.now().strftime(r"%d_%m_%_Y_%H:%M")
    session_name = session_prefix_name + "_" + own_datetime

    path_to_session = os.path.join(path_to_root, session_name)
    create_directories([path_to_session])

    session.generate_predefined_session_descriptor(
        path_session_folder= path_to_session,
        vae_hyperparameters= hyperparams,
        configuration=session_conf
    )

    #LOOP OVER REGIONS
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

        max_train_iter = session.get_adequate_number_iterations(
                                                        region_selected,
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
                          suffix_files_generated=
                          "region_{}".format(region_selected),
                          final_dump_comparison=
                          session_conf["final_dump_comparison"],
                          final_dump_samples_to_compare=
                          session_conf["final_dump_samples_to_compare"],
                          final_dump_planes_per_axis_to_show_in_compare=
                          session_conf["final_dump_planes_per_axis_to_show_in_compare"][region_selected])

        if out == -1:
            print("Region {} Training process failed!"
                  "SGD doesnt converge".format(region_selected))
            print("Exiting, readjust parameter for that region")
            sys.exit(0)

        elif out == 0:
            print("Region {} Correctly Trained and Saved!".format(
                region_selected))

    return path_to_session


def execute_without_any_logs(region_train_cubes_dict, hyperparams, session_conf,
                             list_regions, path_to_root=None,
                             region_test_cubes_dict=None,
                             explicit_iter_per_region={}):
    """
    :param region_train_cubes_dict:
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

        max_train_iter = session.get_adequate_number_iterations(
            region_selected, explicit_iter_per_region, session_conf["n_iters"])

        tf.reset_default_graph()

        CVAE_model = CVAE_helper.select_model(hyperparams["cvae_model"])
        model = CVAE_model.cvae_net(hyperparams)
        model.generate_meta_net()
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




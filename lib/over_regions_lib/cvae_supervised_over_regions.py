import os
from datetime import datetime
from lib.aux_functionalities.os_aux import create_directories
from lib.utils import output_utils
import tensorflow as tf
from lib.vae import CVAE_supervised
from lib import session_helper
import sys


def execute_saving_meta_graph_without_any_cv(region_cubes_dict,
                                             labels,
                                             hyperparams,
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
        model = CVAE_supervised.CVAE(hyperparams, path_to_session=path_to_session)

        max_train_iter = session_helper.get_adequate_number_iterations(
                                                        region_selected,
                                                        explicit_iter_per_region,
                                                        predefined_iters=
                                                        session_conf["n_iters"])
        print("Numbers Iters requered {}".format(max_train_iter))
        out = model.train(X=train_cube_images,
                          Y=labels,
                          n_iters=max_train_iter,
                          batchsize=session_conf["batch_size"],
                          iter_show_error=session_conf["show_error_iter"],
                          tempSGD_3dimages=True,
                          save_bool=True,
                          break_if_nan_error_value=True,
                          suffix_files_generated="region_{}".format(
                              region_selected),
                          max_to_keep=session_conf['max_to_keep'])

        if out == -1:
            print("Region {} Training process failed!"
                  "SGD doesnt converge".format(region_selected))
            print("Exiting, readjust parameter for that region")
            sys.exit(0)

        elif out == 0:
            print("Region {} Correctly Trained and Saved!".format(
                region_selected))

    return path_to_session

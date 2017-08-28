import os
from lib.utils import output_utils
import numpy
import numpy as np

import settings
from lib.utils import functions
from lib.utils.functions import print_dictionary

# Encoding results file
region_mean_file = "region_{}_mean.txt"
region_desv_file = "region_{}_desv.txt"


def print_session_description(path_to_file, session_descriptor):
    file = open(path_to_file, 'w')
    for key, value in session_descriptor.items():
        file.write("{0}: {1}\n".format(str(key), str(value)))
    file.close()


def generate_session_descriptor(path_session_folder, session_descriptor_data):
    path_to_file_session_descriptor = \
        os.path.join(path_session_folder, "session_descriptor.txt")
    print_dictionary(path_to_file_session_descriptor, session_descriptor_data)


def generate_predefined_session_descriptor(path_session_folder,
                                           vae_hyperparameters,
                                           configuration):
    #Session description issues
    session_descriptor = {}
    session_descriptor['VAE hyperparameters'] = vae_hyperparameters
    session_descriptor['VAE session configuration'] = configuration
    path_session_description_file = os.path.join(path_session_folder,
                                                 "session_description.txt")

    file_session_descriptor = open(path_session_description_file, "w")
    output_utils.print_recursive_dict(session_descriptor,
                                      file=file_session_descriptor)
    file_session_descriptor.close()


def select_regions_to_evaluate(regions_used):
    list_regions = []
    if regions_used == "all":
        list_regions = range(1, 117, 1)
    elif regions_used == "most_important":
        list_regions = settings.list_regions_evaluated
    elif regions_used == "three":
        list_regions = [1, 2, 3]
    elif regions_used == "68to117":
        list_regions = range(68, 117, 1)
    elif regions_used == "one":
        list_regions = [1]
    elif regions_used == "four":
        list_regions = [1,2,3,4]

    return list_regions


def plot_grad_desc_error_per_region(path_to_grad_desc_error, region_selected,
                                    path_to_grad_desc_error_images):
    path_to_grad_desc_error_region_log = os.path.join(
        path_to_grad_desc_error, "region_{}.log".format(region_selected))
    path_to_grad_desc_error_region_image = os.path.join(
        path_to_grad_desc_error_images, "region_{}.png".format(region_selected))

    functions.plot_x_y_from_file_with_title(
        "Region {}".format(region_selected), path_to_grad_desc_error_region_log,
        path_to_grad_desc_error_region_image)


def save_encoding_output_per_region(path_to_encoding_storage_folder,
                                    code_data, region_selected):
    encoding_mean_file_saver = os.path.join(path_to_encoding_storage_folder,
                                            region_mean_file.format(
                                                region_selected))
    encoding_desv_file_saver = os.path.join(path_to_encoding_storage_folder,
                                            region_desv_file.format(
                                                region_selected))

    numpy.savetxt(encoding_mean_file_saver, code_data[0], delimiter=',')
    numpy.savetxt(encoding_desv_file_saver, code_data[1], delimiter=',')


def load_out_encoding_per_region(path_to_vae_session_encoding, region_selected):
    path_to_test_out = os.path.join(path_to_vae_session_encoding, "test_out")
    path_to_train_out = os.path.join(path_to_vae_session_encoding, "train_out")

    test_out = load_encoding_per_folder(path_to_test_out, region_selected)
    train_out = load_encoding_per_folder(path_to_train_out, region_selected)

    return test_out, train_out


def load_encoding_per_folder(path_to_out, region_selected):
    path_to_out_mean = os.path.join(path_to_out,
                                    region_mean_file.format(region_selected))
    path_to_out_desv = os.path.join(path_to_out,
                                    region_desv_file.format(region_selected))

    out = {}
    out['means'] = np.genfromtxt(path_to_out_mean, delimiter=",")
    out['desvs'] = np.genfromtxt(path_to_out_desv, delimiter=",")

    return out


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


def validate_threshold(threshold):
    """
    This function is on charge of guarantee that the threshold
    indicated it is between 0 and 1
    :param threshold:
    :return:
    """

    if threshold is not None:
        if threshold > 1:
            return None
        elif threshold < 0:
            return None

    return threshold
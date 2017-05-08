import os
import settings
import numpy
from lib.aux_functionalities import functions
from lib.aux_functionalities.functions import print_dictionary
import numpy as np

folder_meta = "meta"
folder_images = "images"
folder_post_encoding = "post_encoding"
folder_encoding_out = "encoding_out"
folder_encoding_out_train = os.path.join(folder_encoding_out, "train_out")
folder_encoding_out_test = os.path.join(folder_encoding_out, "test_out")
folder_log = "logs"
fodler_cv = "cv"

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


def select_regions_to_evaluate(regions_used):
    list_regions = []
    if regions_used == "all":
        list_regions = range(1, 117, 1)
    elif regions_used == "most important":
        list_regions = settings.list_regions_evaluated
    elif regions_used == "three":
        list_regions = [1, 2, 3]
    elif regions_used == "68to117":
        list_regions = range(68, 117, 1)

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

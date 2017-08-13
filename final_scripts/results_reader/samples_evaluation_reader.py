import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import settings
import json
from matplotlib import pyplot as plt
from lib.utils. os_aux import create_directories
from lib.data_loader import PET_stack_NORAD
import numpy as np
from lib.utils import evaluation_utils as eval
import copy


def truncate_based_on_evaluation_method(values, method):

    if isinstance(values, list):
        values = np.array(values)

    if method == "SVM":
        values = truncate_over_max_min_values(values, 1, -1)
    elif method == "SMV":
        values = truncate_over_max_min_values(values, 1, 0)
    elif method == "CMV":
        values = truncate_over_max_min_values(values, 1, -1)

    return values


def assign_binary_labels_based_on_threshold_and_method(values, method):

        if isinstance(values, list):
            values = np.array(values)

        if method == "SVM":
            labels_obtained = assign_binary_labels_based_on_threshold(values, 0)
        elif method == "SMV":
            labels_obtained = assign_binary_labels_based_on_threshold(values, 0.5)
        elif method == "CMV":
            labels_obtained = assign_binary_labels_based_on_threshold(values, 0)

        return labels_obtained


def assign_binary_labels_based_on_threshold(scores, threshold):
    aux_scores = copy.deepcopy(scores)

    aux_scores[aux_scores <= threshold] = 0
    aux_scores[aux_scores > threshold] = 1

    return aux_scores


def truncate_over_max_min_values(values, max, min):
    values[values > max] = max
    values[values < min] = min
    return values


swap_variable = "Kernel"
images_used = "PET"
session_name = "CVAE_PET_session_swap_kernel_PET"
images_folder = "images"
test_evaluation_file = "test_scores_evaluation_per_sample.log"
full_evaluation_file = "full_scores_evaluation_per_sample.log"
patient_labels = PET_stack_NORAD.load_patients_labels()

path_to_images_folder = os.path.join(settings.path_to_general_out_folder,
                                            session_name, images_folder)

path_to_full_evaluation_file = os.path.join(settings.path_to_general_out_folder,
                                            session_name, full_evaluation_file)

path_to_test_evaluation_file = os.path.join(settings.path_to_general_out_folder,
                                            session_name, test_evaluation_file)

create_directories([path_to_images_folder])

full_evaluation_file = open(path_to_full_evaluation_file)
test_evaluation_file = open(path_to_test_evaluation_file)

full_evaluation_dic_container = json.load(fp=full_evaluation_file)
test_evaluation_dic_container = json.load(fp=test_evaluation_file)


evaluation_methods_list = list(full_evaluation_dic_container.keys())
swap_variables_list = list(full_evaluation_dic_container[evaluation_methods_list[0]].keys())
kfold_list = list(full_evaluation_dic_container[evaluation_methods_list[0]][swap_variables_list[0]].keys())
example_out = full_evaluation_dic_container[evaluation_methods_list[0]][swap_variables_list[0]][kfold_list[0]]


for method in evaluation_methods_list:
    plt.figure()
    plt.title("Method used {}".format(method))
    example_out = test_evaluation_dic_container[method][
        swap_variables_list[0]]

    example_out = truncate_based_on_evaluation_method(example_out, method)

    example_out_labeled = \
        assign_binary_labels_based_on_threshold_and_method(example_out, method)

    tp, fn, tn, fp = eval.get_classification_masks_over_labels(
        true_labels=patient_labels,
        obtained_labels=example_out_labeled)

    x = list(range(0, len(example_out), 1))
    x = np.array(x)
    example_out = example_out[:, 0]

    plt.scatter(x[tp], example_out[tp], color='dodgerblue', label="tp: {}".format(sum(tp)))
    plt.scatter(x[fp], example_out[fp], color='blue', label="fp: {}".format(sum(fp)))
    plt.scatter(x[fn], example_out[fn], color='red', label="fn: {}".format(sum(fn)))
    plt.scatter(x[tn], example_out[tn], color='salmon', label="tn: {}".format(sum(tn)))
    plt.legend()
    plt.savefig(os.path.join(path_to_images_folder,
                             "Evaluation Methdod:{0}, {1}: {2}.png").format(
                             method, swap_variable, swap_variables_list[0]))

full_evaluation_file.close()
test_evaluation_file.close()


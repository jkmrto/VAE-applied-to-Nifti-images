import numpy as np
import os
import matplotlib.pyplot as plt

from lib.aux_functionalities.os_aux import create_directories


def get_batch_from_samples(X, Y, batch_size):
    index = np.random.choice(range(X.shape[0]), batch_size, replace=False)
    index = index.tolist()
    return X[index, :], Y[index]


def get_batch_from_samples_unsupervised(X, batch_size):
    index = np.random.choice(range(X.shape[0]), batch_size, replace=False)
    index = index.tolist()
    return X[index, :]


def print_dictionary(path_to_file, dictionary):
    file = open(path_to_file, 'w')
    for key, value in dictionary.items():
        file.write("{0}: {1}\n".format(str(key), str(value)))
    file.close()


def print_session_description(path_to_file, session_descriptor):
    file = open(path_to_file, 'w')
    for key, value in session_descriptor.items():
        file.write("{0}: {1}\n".format(str(key), str(value)))
    file.close()


def generate_session_descriptor(path_session_folder, session_descriptor_data):
    path_to_file_session_descriptor = \
        os.path.join(path_session_folder, "session_descriptor.txt")
    print_dictionary(path_to_file_session_descriptor, session_descriptor_data)


def plot_x_y_from_file_with_title(
        graph_title, path_to_log, path_where_to_save_png):
    """
    The file passed should be in csv formant,
    :param graph_title:
    :param path_to_log:
    :param path_where_to_save_png:
    :return:
    """
    file = open(path_to_log, 'r')

    iter = []
    error = []
    for line in file:
        [iter_aux, error_aux] = file.readline().split(",")
        iter.append(int(iter_aux))
        error.append(float(error_aux))

    plt.figure()
    plt.plot(error)
    plt.title(graph_title)
    plt.savefig(path_where_to_save_png, dpi = 200)


def assign_binary_labels_based_on_threshold(scores, threshold):
    scores[scores < threshold] = 0
    scores[scores > threshold] = 1

    return  scores


import numpy as np
import os
import matplotlib.pyplot as plt

from lib.aux_functionalities.os_aux import create_directories


def get_batch_from_samples(X, Y, batch_size):
    """
    It is mandatory batch_size > X.shape[0] && batch_size > Y.shape[0]
    :param X:
    :param Y:
    :param batch_size:
    :return:
    """
    index = np.random.choice(range(X.shape[0]), batch_size, replace=False)
    index = index.tolist()
    return X[index, :], Y[index]


def get_batch_from_samples_unsupervised(X, batch_size):
    index = np.random.choice(range(X.shape[0]), batch_size, replace=False)
    index = index.tolist()
    return X[index, :]


def get_batch_from_samples_unsupervised_3d(X, batch_size):
    index = np.random.choice(range(X.shape[0]), batch_size, replace=False)
    index = index.tolist()
    return X[index, :, :, :]


def print_dictionary(path_to_file, dictionary):
    file = open(path_to_file, 'w')
    for key, value in dictionary.items():
        file.write("{0}: {1}\n".format(str(key), str(value)))
    file.close()


def load_csv_file_iter_to_error(path_to_log):

    file = open(path_to_log, 'r')

    iter = []
    error = []
    for line in file:
        [iter_aux, error_aux] = file.readline().split(",")
        iter.append(int(iter_aux))
        error.append(float(error_aux))

    file.close()
    return iter, error


def plot_x_y_from_file_with_title(
        graph_title, path_to_log, path_where_to_save_png):
    """
    The file passed should be in csv formant,
    :param graph_title:
    :param path_to_log:
    :param path_where_to_save_png:
    :return:
    """
    iter, error = load_csv_file_iter_to_error(path_to_log)
    plt.figure()
    plt.plot(error)
    plt.title(graph_title)
    plt.savefig(path_where_to_save_png, dpi = 200)

def print_dictionary(path_to_file, dictionary):
    file = open(path_to_file, 'w')
    for key, value in dictionary.items():
        file.write("{0}: {1}\n".format(str(key), str(value)))
    file.close()
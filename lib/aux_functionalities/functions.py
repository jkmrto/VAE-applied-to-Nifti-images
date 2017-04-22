import numpy as np
import os
from lib.aux_functionalities.os_aux import create_directories


def get_batch_from_samples(X, Y, batch_size):
    index = np.random.choice(range(X.shape[0]), batch_size, replace=False)
    index = index.tolist()
    return X[index, :], Y[index]


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
    print_session_description(path_to_file_session_descriptor, session_descriptor_data)

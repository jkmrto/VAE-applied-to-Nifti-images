import os
import numpy
from lib.mri.stack_NORAD import load_patients_labels

train_index_file = "train_index_to_stack.csv"
test_index_file = "test_index_to_stack.csv"


def get_train_and_test_index_from_files(path):
    train_index_path = os.path.join(path, train_index_file)
    test_index_path = os.path.join(path, test_index_file)

    train_index = numpy.genfromtxt(train_index_path).astype(int).tolist()
    test_index = numpy.genfromtxt(test_index_path).astype(int).tolist()

    return train_index, test_index


def get_label_per_patient(path_to_cv_folder):
    patients_labels = load_patients_labels()  # 417x1
    train_index, test_index = get_train_and_test_index_from_files(
        path_to_cv_folder)
    Y_train = patients_labels[train_index]
    Y_test = patients_labels[test_index]

    return Y_train, Y_test
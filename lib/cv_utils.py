import os
import numpy as np
from lib.mri.stack_NORAD import load_patients_labels

train_index_file = "train_index_to_stack.csv"
test_index_file = "test_index_to_stack.csv"
k_folds_files_template = "index_fold_{}.csv"


def get_train_and_test_index_from_files(path):
    train_index_path = os.path.join(path, train_index_file)
    test_index_path = os.path.join(path, test_index_file)

    train_index = np.genfromtxt(train_index_path).astype(int).tolist()
    test_index = np.genfromtxt(test_index_path).astype(int).tolist()

    return train_index, test_index


def get_label_per_patient(path_to_cv_folder):
    patients_labels = load_patients_labels()  # 417x1
    train_index, test_index = get_train_and_test_index_from_files(
        path_to_cv_folder)

    Y_train = patients_labels[train_index]
    Y_test = patients_labels[test_index]

    return Y_train, Y_test


def generate_and_store_train_and_test_index(stack, cv_rate, path_to_cv):
    """This functions stores the index to the stack images.
    This approaches allows us to point with the indes, avoiding store all
    the data again.
    """
    train_index = np.random.choice(range(stack.shape[0]),
                                   int(cv_rate * stack.shape[0]), replace=False)
    train_index.sort()
    test_index = [index for index in range(0, stack.shape[0], 1) if
                  index not in train_index]

    np.savetxt(os.path.join(path_to_cv, "train_index_to_stack.csv"),
               train_index, delimiter=',')
    np.savetxt(os.path.join(path_to_cv, "test_index_to_stack.csv"),
               test_index, delimiter=',')

    return train_index, test_index


def generate_k_fold(path_to_kfold_folder, stack, n_folds):
    index = np.random.choice(range(stack.shape[0]), stack.shape[0],
                             replace=False)
    n_over_samples = index.shape[0] % n_folds
    over_samples = index[-n_over_samples:]

    samples_per_fold = int(index.shape[0] / n_folds)

    if n_over_samples > 0:
        index_matrix = np.reshape(index[0:-n_over_samples],
                              (n_folds, samples_per_fold))
    else:
        index_matrix = np.reshape(index, (n_folds, samples_per_fold))

    for i in range(0, n_folds, 1):
        if i < n_over_samples:
            out = np.append(index_matrix[i, :], over_samples[i])
        else:
            out = index_matrix[i, :]

        np.savetxt(os.path.join(path_to_kfold_folder,
                                k_folds_files_template.format(i +1 )), out,
                   delimiter=',')


def get_train_and_test_index_from_k_fold(path_to_kfold_folder, test_fold, n_folds):
    train_index = np.empty(shape=0)
    test_index = np.empty(shape=0)

    #   path_to_kfold
    for i in range(1, n_folds + 1, 1):
        path_to_file = os.path.join(path_to_kfold_folder, k_folds_files_template.format(i))
        file_loaded = np.genfromtxt(path_to_file).astype(int).tolist()
        if i == test_fold:
            test_index = np.append(test_index, file_loaded).astype(int).tolist()
        else:
            train_index = np.append(train_index, file_loaded).astype(int).tolist()

    train_index.sort()
    test_index.sort()

    return train_index, test_index

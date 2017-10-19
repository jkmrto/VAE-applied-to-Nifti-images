import os
import numpy as np
from lib.data_loader.MRI_stack_NORAD import load_patients_labels

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


def generate_k_folder_in_dict(n_samples, n_folds):
    """

    :param n_samples:
    :param n_folds:
    :return: k_fold_dict[fold_index]["train"|"test"] -> label_index
    """
    index = np.random.choice(range(n_samples),n_samples,
                             replace=False)
    n_over_samples = index.shape[0] % n_folds
    over_samples = index[-n_over_samples:]

    samples_per_fold = int(index.shape[0] / n_folds)

    if n_over_samples > 0:
        index_matrix = np.reshape(index[0:-n_over_samples],
                              (n_folds, samples_per_fold))
    else:
        index_matrix = np.reshape(index, (n_folds, samples_per_fold))

    k_fold_dict = {}
    for test_fold in range(0, n_folds , 1):

        train_index = np.empty(shape=0)
        test_index = np.empty(shape=0)

        for i in range(0, n_folds, 1):
            if i == test_fold:
                test_index = np.append(test_index, index_matrix[i, :]).astype(int).tolist()
                if i < n_over_samples:
                    test_index = np.append(test_index, over_samples[i])
            else:
                train_index = np.append(train_index, index_matrix[i,:]).astype(int).tolist()
                if i < n_over_samples:
                    train_index = np.append(train_index, over_samples[i])

        train_index.sort()
        test_index.sort()

        k_fold_dict[test_fold] = {}
        k_fold_dict[test_fold]['train'] = train_index
        k_fold_dict[test_fold]['test'] = test_index

    return k_fold_dict


def test_over_generate_k_folder_in_dict():
    kfold_dict = generate_k_folder_in_dict(n_samples=103, n_folds=10)
    print(kfold_dict)

def generate_k_fold(path_to_kfold_folder, n_samples, n_folds):
    index = np.random.choice(range(n_samples),n_samples,
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


def restructure_dictionary_based_on_cv_index_3dimages(
        dict_train_test_index, region_to_img_dict):
    """

    :param dict_train_test_index: Dic["train"|"test"] -> index_samples
    :param region_to_img_dict:  Dic[region] -> image 3d values
    :return: Dic["train"|"test][region] -> image 3d values
    """

    test_index = dict_train_test_index['test']
    train_index = dict_train_test_index['train']

    restructure_output = {}
    restructure_output["test"] = {}
    restructure_output["train"] = {}
    for region in region_to_img_dict.keys():
        restructure_output["test"][region] = region_to_img_dict[region][test_index,:,:,:]
        restructure_output["train"][region] = region_to_img_dict[region][train_index,:,:,:]

    return restructure_output


def restructure_dictionary_based_on_cv(
        dict_train_test_index, region_to_img_dict):
    """

    :param dict_train_test_index: Dic["train"|"test"] -> index_samples
    :param region_to_img_dict:  Dic[region] -> image 3d values
    :return: Dic["train"|"test][region] -> image 3d values
    """

    test_index = dict_train_test_index['test']
    train_index = dict_train_test_index['train']

    restructure_output = {"test": {}, "train": {}}

    for region in region_to_img_dict.keys():
        restructure_output["test"][region] = \
            region_to_img_dict[region][test_index, :]
        restructure_output["train"][region] =\
            region_to_img_dict[region][train_index, :]

    return restructure_output


def get_test_and_train_labels_from_kfold_dict_entry(k_fold_entry, patient_labels):
    """
    :param k_fold_entry: dict[test|train] -> Type:array "index"
    :param patient_labels: Type: array "labels"
    :return:
    """
    train_index = k_fold_entry["train"]
    test_index = k_fold_entry["test"]

    Y_train = patient_labels[train_index]
    Y_test = patient_labels[test_index]
    Y_train = np.row_stack(Y_train)
    Y_test = np.row_stack(Y_test)

    return Y_train, Y_test
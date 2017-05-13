import tensorflow as tf
import os
from lib import cv_utils
from lib.mri import stack_NORAD
from lib import session_helper as session
from scripts.vae_with_kfolds import session_settings
from scripts.vae_with_kfolds import vae_over_regions_kfolds
from lib.mri.stack_NORAD import load_patients_labels
from lib import svm_utils
from lib.evaluation_utils import simple_evaluation_output
import numpy as np
import csv


# OUTPUT FODLER
k_fold_output_file = os.path.join(session_settings.path_kfolds_session_folder,
                                  "k_fold_output.csv")
file = open(k_fold_output_file, "w")


# Selecting the GM folder
path_to_root_GM = session_settings.path_GM_folder
path_to_root_WM = session_settings.path_WM_folder
# Loading the stack of images
dict_norad_gm = stack_NORAD.get_gm_stack()
dict_norad_wm = stack_NORAD.get_wm_stack()
patient_labels = load_patients_labels()

n_folds = 10
bool_test = True
cv_utils.generate_k_fold(session_settings.path_kfolds_folder,
                         dict_norad_gm['stack'], n_folds)

# LIST REGIONS SELECTION
regions_used = "three"
list_regions = session.select_regions_to_evaluate(regions_used)

hyperparams = {
    "batch_size": 16,
    "learning_rate": 1E-5,
    "dropout": 0.9,
    "lambda_l2_reg": 1E-5,
    "nonlinearity": tf.nn.elu,
    "squashing": tf.nn.sigmoid,
}

# Neural net architecture
after_input_architecture = [1000, 500, 100]

# SESSION CONFIGURATION
session_conf = {
    "bool_normalized": True,
    "max_iter": 100,
    "save_meta_bool": False,
}

k_folds_results_train = []
k_folds_results_test = []
for k_fold_index in range(1, n_folds, 1):
    vae_output = {}

    train_index, test_index = cv_utils.get_train_and_test_index_from_k_fold(
        session_settings.path_kfolds_folder, k_fold_index, n_folds)

    Y_train = patient_labels[train_index]
    Y_test = patient_labels[test_index]
    Y_train = np.row_stack(Y_train)
    Y_test = np.row_stack(Y_test)

    print("Kfold {} Selected".format(k_fold_index))
    print("Number test samples {}".format(len(test_index)))
    print("Number train samples {}".format(len(train_index)))

    voxels_values = {}
    voxels_values['train'] = dict_norad_gm['stack'][train_index, :]
    voxels_values['test'] = dict_norad_gm['stack'][test_index, :]

    print("Train over GM regions")
    vae_output['gm'] = vae_over_regions_kfolds.execute(voxels_values, hyperparams, session_conf,
                                    after_input_architecture,
                                    path_to_root_GM, list_regions)

    voxels_values = {}
    voxels_values['train'] = dict_norad_wm['stack'][train_index, :]
    voxels_values['test'] = dict_norad_wm['stack'][test_index, :]

    print("Train over WM regions")
    vae_output['wm'] = vae_over_regions_kfolds.execute(voxels_values, hyperparams, session_conf,
                                    after_input_architecture,
                                    path_to_root_WM, list_regions)

    train_score_matriz = np.zeros((len(train_index), len(list_regions)))
    test_score_matriz = np.zeros((len(test_index), len(list_regions)))

    print("SVM step")

    i = 0
    dic_region_to_matriz_pos = {}

    for region_selected in list_regions:
        dic_region_to_matriz_pos[str(region_selected)] = i

        print("region {} selected".format(region_selected))
        train_output_wm = vae_output['wm'][str(region_selected)]['train_output']
        test_output_wm = vae_output['wm'][str(region_selected)]['test_output']

        train_output_gm = vae_output['gm'][str(region_selected)]['train_output']
        test_output_gm = vae_output['gm'][str(region_selected)]['test_output']

        train_means_gm = train_output_wm[0]
        test_means_gm = test_output_wm[0]

        train_means_wm = train_output_gm[0]
        test_means_wm = test_output_gm[0]

        wm_and_gm_train_data = np.concatenate((train_means_gm, train_means_wm),
                                              axis=1)
        wm_and_gm_test_data = np.concatenate((test_means_gm, test_means_wm),
                                             axis=1)

        print(wm_and_gm_train_data.shape)
        print(wm_and_gm_test_data.shape)
        train_score, test_score = svm_utils.fit_svm_and_get_decision_for_requiered_data(
            wm_and_gm_train_data, Y_train, wm_and_gm_test_data)

        train_score_matriz[:, i] = train_score
        test_score_matriz[:, i] = test_score

        if bool_test:
            print("TEST SVM SCORE REGION {}".format(region_selected))
            print(train_score.shape)
            print(Y_train.shape)
            test_train_score = np.hstack((np.row_stack(train_score), np.row_stack(Y_train)))
            test_test_score = np.hstack((np.row_stack(test_score), np.row_stack(Y_test)))
            print(test_train_score)
            print(test_test_score)

        i += 1



    print("Diccionario de regions utilizadas")
    print(dic_region_to_matriz_pos)
    # majority vote

    means_train = np.row_stack(train_score_matriz.mean(axis=1))
    means_test = np.row_stack(test_score_matriz.mean(axis=1))

    if bool_test:
        print("TEST OVER FINAL RESULTS")
        test_train_score = np.hstack(
            (np.row_stack(means_train), np.row_stack(Y_train)))
        test_test_score = np.hstack(
            (np.row_stack(means_test), np.row_stack(Y_test)))
        print(test_train_score)
        print(test_test_score)

    threshold, output_dic_train = simple_evaluation_output(means_train, Y_train)
    threshold, output_dic_test = simple_evaluation_output(means_test, Y_test, threshold)

    print("Output kfolds nº {} test samples".format(k_fold_index))
    print(output_dic_test)

    print("Output kfolds nº {} train samples".format(k_fold_index))
    print(output_dic_train)

    k_folds_results_train.append(output_dic_train)
    k_folds_results_test.append(output_dic_test)

writer = csv.DictWriter(file, delimiter=',', fieldnames=list(k_folds_results_test[0].keys()))
writer.writeheader()
for row in k_folds_results_test:
    writer.writerow(row)
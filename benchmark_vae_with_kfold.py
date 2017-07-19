import os
import tarfile
from datetime import datetime
from shutil import copyfile

import numpy as np
import tensorflow as tf
from lib.evaluation_utils import get_average_over_metrics

import settings
from lib import evaluation_utils
from lib import output_utils
from lib import svm_utils
from lib.aux_functionalities.os_aux import create_directories
from lib.data_loader.nifti_regions_loader import \
    load_mri_data_flat, load_pet_data_flat
from lib.neural_net import leaky_net_utils
from lib.utils import cv_utils
from scripts.vae_with_kfolds import session_settings
from scripts.vae_with_kfolds import vae_over_regions_kfolds

"""
Still need to test MRI with this script,
and completely use in a long run server
"""

images_used = "MRI"
#images_used = "PET"

# Meta settings.
n_folds = 3
bool_test = False
bool_log_svm_output = True
regions_used = "three"
#regions_used = "three"
#list_regions = session.select_regions_to_evaluate(regions_used)
list_regions = [1, 2]
# VAE SETTINGS
# Net Configuration
hyperparams_vae = {
    "batch_size": 16,
    "learning_rate": 1E-5,
    "dropout": 0.9,
    "lambda_l2_reg": 1E-5,
    "nonlinearity": tf.nn.elu,
    "squashing": tf.nn.sigmoid,
}

# Vae session cofiguration
vae_session_conf = {
    "bool_normalized": True,
    "max_iter": 20,
    "save_meta_bool": False,
    "show_error_iter": 10,
    "after_input_architecture": [1000, 500, 100],
}

# DECISION NET CONFIGURATION
decision_net_session_conf = {
    "decision_net_tries": 1,
    "field_to_select_try": "area under the curve",
    "max_iter": 50,
    "threshould_prefixed_to_0.5": True,
}

HYPERPARAMS_decision_net = {
    "batch_size": 50,
    "learning_rate": 1E-6,
    "lambda_l2_reg": 0.000001,
    "dropout": 1,
    "nonlinearity": tf.nn.relu,
}

session_datetime = datetime.now().isoformat()
print("Time session init: {}".format(session_datetime))

# OUTPUT SETTINGS

# OUTPUT: List of dictionaries
complex_majority_vote_k_folds_results_train = []
complex_majority_vote_k_folds_results_test = []

simple_majority_vote_k_folds_results_train = []
simple_majority_vote_k_folds_results_test = []

decision_net_k_folds_results_train = []
decision_net_vote_k_folds_results_test = []

svm_weighted_regions_k_folds_results_train = []
svm_weighted_regions_k_folds_results_test = []
svm_weighted_regions_k_folds_coefs = []


# OUTPUT: Folder initialization
Kfolds_folder = "Kfolds index"
session_name = "Full_classification_session_with_k-folds"
path_session_folder = os.path.join(
    settings.path_to_general_out_folder,
    session_name)

path_kfolds_folder = os.path.join(path_session_folder, Kfolds_folder)
create_directories([path_session_folder,path_kfolds_folder])


# OUTPUT: Files initialization
k_fold_output_file_simple_majority_vote = os.path.join(
    path_session_folder,
    "k_fold_output_simple_majority_vote.csv")

k_fold_output_file_complex_majority_vote = os.path.join(
    path_session_folder,
    "k_fold_output_complex_majority_vote.csv")

k_fold_output_file_decision_net = os.path.join(
    session_settings.path_kfolds_session_folder,
    "k_fold_output_decision_net.csv")

k_fold_output_file_weighted_svm = os.path.join(
    path_session_folder,
    "k_fold_output_weighted_svm.csv")

k_fold_output_file_coefs_weighted_svm = os.path.join(
    path_session_folder,
    "k_fold_output_coefs_weighted_svm.csv")

k_fold_output_path_session_description = os.path.join(
    path_session_folder,
    "session_description.csv")

k_fold_output_resume_path = os.path.join(
    path_session_folder,
    "resume.csv")

tar_file_main_output_path = os.path.join(
    path_session_folder,
    "main_out_session_{}.tar.gz".format(session_datetime))

tar_file_main_output_path_replica = os.path.join(
    path_session_folder,
    "last_session_main_out.tar.gz".format(session_datetime))

list_paths_files_to_store = [k_fold_output_file_simple_majority_vote,
                             k_fold_output_file_complex_majority_vote,
                             k_fold_output_file_decision_net,
                             k_fold_output_resume_path,
                             k_fold_output_path_session_description,
                             k_fold_output_file_weighted_svm,
                             k_fold_output_file_coefs_weighted_svm]

# Session descriptor elaboration
session_descritpr = {}
session_descriptor = {"Images Used": images_used}
session_descriptor['meta settings'] = {"n_folds": n_folds,
                                       "bool_test": bool_test,
                                       "bool_log_svm_output": bool_log_svm_output,
                                       "regions_used": regions_used}
session_descriptor['VAE'] = {}
session_descriptor['Decision net'] = {}
session_descriptor['VAE']["net configuration"] = hyperparams_vae
session_descriptor['VAE']["net configuration"]["depth_conv_layer"] = \
    "input_" + "_".join(
        str(x) for x in vae_session_conf["after_input_architecture"])
session_descriptor['VAE']["session configuration"] = vae_session_conf
session_descriptor['Decision net']["net configuration"] = \
    HYPERPARAMS_decision_net
session_descriptor['Decision net']["net configuration"]['architecture'] = \
    "[nºregions, nºregions/2, 1]"
session_descriptor['Decision net']['session_conf'] = decision_net_session_conf

file_session_descriptor = open(k_fold_output_path_session_description, "w")
output_utils.print_recursive_dict(session_descriptor,
                                  file=file_session_descriptor)
file_session_descriptor.close()

n_samples = 0
if images_used == "PET":
    dic_regions_to_flatten_voxels_pet, patient_labels, n_samples = \
        load_pet_data = load_pet_data_flat(list_regions)
elif images_used == "MRI":
    dic_regions_to_flatten_voxels_mri_gm, dic_regions_to_flatten_voxels_mri_wm, \
        patient_labels, n_samples = load_mri_data_flat(list_regions)

# Load regions index and create kfolds folder

k_fold_dict = cv_utils.generate_k_folder_in_dict(n_samples, n_folds)

# Main Loop
for k_fold_index in range(0, n_folds, 1):
    vae_output = {}

    if images_used == "MRI":
        reg_to_group_to_images_dict_mri_gm = \
            cv_utils.restructure_dictionary_based_on_cv_index_flat_images(
                dict_train_test_index=k_fold_dict[k_fold_index],
                region_to_img_dict=dic_regions_to_flatten_voxels_mri_gm)

        reg_to_group_to_images_dict_mri_wm = \
            cv_utils.restructure_dictionary_based_on_cv_index_flat_images(
                dict_train_test_index=k_fold_dict[k_fold_index],
                region_to_img_dict=dic_regions_to_flatten_voxels_mri_wm)

    if images_used == "PET":
        reg_to_group_to_images_dict_pet = \
            cv_utils.restructure_dictionary_based_on_cv_index_flat_images(
                dict_train_test_index=k_fold_dict[k_fold_index],
                region_to_img_dict=dic_regions_to_flatten_voxels_pet)

    Y_train = patient_labels[k_fold_dict[k_fold_index]["train"]]
    Y_test = patient_labels[k_fold_dict[k_fold_index]["test"]]
    Y_train = np.row_stack(Y_train)
    Y_test = np.row_stack(Y_test)

    print("Kfold {} Selected".format(k_fold_index))
    print("Number test samples {}".format(len(k_fold_dict[k_fold_index]["test"])))
    print("Number train samples {}".format(len(k_fold_dict[k_fold_index]["train"])))

    if images_used == "MRI":

        print("Training MRI GM regions")
        vae_output['gm'] = vae_over_regions_kfolds.execute_without_any_logs(
            region_to_flat_voxels_train_dict=reg_to_group_to_images_dict_mri_gm["train"],
            hyperparams=hyperparams_vae,
            session_conf=vae_session_conf,
            list_regions=list_regions,
            path_to_root=None,
            region_to_flat_voxels_test_dict=reg_to_group_to_images_dict_mri_gm["test"],
            explicit_iter_per_region=[]
        )

        print("Training MRI WM regions")
        vae_output['wm'] = vae_over_regions_kfolds.execute_without_any_logs(
            region_to_flat_voxels_train_dict=reg_to_group_to_images_dict_mri_wm["train"],
            hyperparams=hyperparams_vae,
            session_conf=vae_session_conf,
            list_regions=list_regions,
            path_to_root=None,
            region_to_flat_voxels_test_dict=reg_to_group_to_images_dict_mri_wm["test"],
            explicit_iter_per_region=[]
        )

        #[patient x region]
        train_score_matriz, test_score_matriz = svm_utils.svm_mri_over_vae_output(
            vae_output, Y_train, Y_test, list_regions, bool_test=bool_test)

    if images_used == "PET":

        print("Training PET regions")
        vae_output = vae_over_regions_kfolds.execute_without_any_logs(
            region_to_flat_voxels_train_dict=reg_to_group_to_images_dict_pet["train"],
            hyperparams=hyperparams_vae,
            session_conf=vae_session_conf,
            list_regions=list_regions,
            path_to_root=None,
            region_to_flat_voxels_test_dict=reg_to_group_to_images_dict_pet["test"],
            explicit_iter_per_region=[]
        )

        train_score_matriz, test_score_matriz = svm_utils.svm_pet_over_vae_output(
            vae_output, Y_train, Y_test, list_regions, bool_test=bool_test)

    if bool_log_svm_output:
        suffix = "log_svm_output_kfold_" + str(k_fold_index) + "_"
        path_to_log_svm_output = os.path.join(path_session_folder, suffix)
        Y_train_matrix_unicolumn = np.reshape(Y_train, [Y_train.flatten().shape[0], 1])
        Y_test_matrix_unicolumn = np.reshape(Y_test, [Y_test.flatten().shape[0], 1])

        log_svm_out_train_matrix = out = np.concatenate(
            [Y_train_matrix_unicolumn, train_score_matriz], axis=1)
        log_svm_out_test_matrix = out = np.concatenate(
            [Y_test_matrix_unicolumn, test_score_matriz],axis=1)

        np.savetxt(path_to_log_svm_output + "train.csv", X=log_svm_out_train_matrix,
                   fmt="%.2f", delimiter=",")
        np.savetxt(path_to_log_svm_output + "test.csv", X=log_svm_out_test_matrix,
                   fmt="%.2f", delimiter=",")

    data = {}
    data["test"] = {}
    data["train"] = {}
    data["test"]["data"] = test_score_matriz
    data["test"]["label"] = Y_test
    data["train"]["data"] = train_score_matriz
    data["train"]["label"] = Y_train

    if bool_test:
        print("\nMatriz svm scores -> shapes, After svm before qualifying")
        print("train matriz [patients x region]: " + str(
            train_score_matriz.shape))
        print("test matriz scores [patient x region]: " + str(
            test_score_matriz.shape))

    # COMPLEX MAJORITY VOTE

    complex_output_dic_test, complex_output_dic_train = \
        evaluation_utils.complex_majority_vote_evaluation(data,
                                                          bool_test=bool_test)

    # Adding results to kfolds output
    complex_majority_vote_k_folds_results_train.append(complex_output_dic_train)
    complex_majority_vote_k_folds_results_test.append(complex_output_dic_test)

    if bool_test:
        print("\nMatriz svm scores -> shapes, after complex majority vote")
        print("train matriz [patients x regions]: " + str(train_score_matriz.shape))
        print("test matriz scores [patients x regions]: " + str(
        test_score_matriz.shape))

    # SIMPLE MAJORITY VOTE

    simple_output_dic_train, simple_output_dic_test = \
        evaluation_utils.simple_majority_vote(
            train_score_matriz, test_score_matriz, Y_train, Y_test,
            bool_test=False)

    print("Output kfolds nº {}".format(k_fold_index))
    print("Simple Majority Vote Test: " + str(simple_output_dic_test))
    print("Simple Majority Vote Train: " + str(simple_output_dic_train))

    simple_majority_vote_k_folds_results_train.append(simple_output_dic_train)
    simple_majority_vote_k_folds_results_test.append(simple_output_dic_test)

    # SVM weighted REGIONS RESULTS
    print("DECISION WEIGHTING SVM OUTPUTS")
    # The score matriz is in regions per patient, we should transpose it
    # in the svm process

    weighted_output_dic_test, weighted_output_dic_train, \
    aux_dic_regions_weight_coefs = \
        evaluation_utils.weighted_svm_decision_evaluation(data, list_regions,
                                                          bool_test=bool_test)

    svm_weighted_regions_k_folds_results_train.append(weighted_output_dic_train)
    svm_weighted_regions_k_folds_results_test.append(weighted_output_dic_test)
    svm_weighted_regions_k_folds_coefs.append(aux_dic_regions_weight_coefs)

    # DECISION NEURAL NET
    print("Decision neural net step")

    # train score matriz [patients x regions]
    input_layer_size = train_score_matriz.shape[1]
    architecture = [input_layer_size, int(input_layer_size / 2), 1]

    net_train_dic, net_test_dic = \
        leaky_net_utils.train_leaky_neural_net_various_tries_over_svm_output(
        decision_net_session_conf, architecture, HYPERPARAMS_decision_net,
        train_score_matriz, test_score_matriz, Y_train, Y_test, bool_test=False)

    decision_net_k_folds_results_train.append(net_train_dic)
    decision_net_vote_k_folds_results_test.append(net_test_dic)

# Outputs files
output_utils.print_dictionary_with_header(
    k_fold_output_file_simple_majority_vote,
    simple_majority_vote_k_folds_results_test)

output_utils.print_dictionary_with_header(
    k_fold_output_file_complex_majority_vote,
    complex_majority_vote_k_folds_results_test)

output_utils.print_dictionary_with_header(
    k_fold_output_file_decision_net,
    decision_net_vote_k_folds_results_test)

output_utils.print_dictionary_with_header(
    k_fold_output_file_weighted_svm,
    svm_weighted_regions_k_folds_results_test)

output_utils.print_dictionary_with_header(
    k_fold_output_file_coefs_weighted_svm,
    svm_weighted_regions_k_folds_coefs
)

resume_list_dicts = []
simple_majority_vote = get_average_over_metrics(
    simple_majority_vote_k_folds_results_test)
complex_majority_vote = get_average_over_metrics(
    complex_majority_vote_k_folds_results_test)

svm_weighted = get_average_over_metrics(
    svm_weighted_regions_k_folds_results_test)

decision_net = get_average_over_metrics(decision_net_vote_k_folds_results_test)

# Add kind of decission session
temp_simple_majority_vote = {"decision step": "Simple majority vote"}
temp_simple_majority_vote.update(simple_majority_vote)

temp_complex_majority_vote = {"decision step": "Complex majority vote"}
temp_complex_majority_vote.update(complex_majority_vote)

temp_decision_net = {"decision step": "Decision neural net"}
temp_decision_net.update(decision_net)

temp_svm_weighted = {"decision step": "Weighted SVM"}
temp_svm_weighted.update(svm_weighted)

output_utils.print_dictionary_with_header(
    k_fold_output_resume_path,
    [temp_simple_majority_vote, temp_complex_majority_vote, temp_decision_net,
     temp_svm_weighted])

# Tarfile to group the results
tar = tarfile.open(tar_file_main_output_path, "w:gz")
for file in list_paths_files_to_store:
    tar.add(file)
tar.close()
copyfile(tar_file_main_output_path, tar_file_main_output_path_replica)


# Weighted SVM  Coefs Gotten: {'3': 1.056914793220729, '1': 0.9768996145437621, '2': 1.1293619260635606}

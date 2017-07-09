import os
import tarfile
from copy import deepcopy
from datetime import datetime

import tensorflow as tf
from lib.evaluation_utils import get_average_over_metrics

from lib import evaluation_utils
from lib import output_utils
from lib import session_helper as session
from lib import svm_utils
from lib.neural_net import leaky_net_utils
from lib.utils import cv_utils
from lib.utils.cv_utils import get_test_and_train_labels_from_kfold_dict_entry, generate_k_folder_in_dict
from nifti_regions_loader import \
    load_mri_data_flat, load_pet_data_flat
from scripts.vae_sweep_over_features import loop_latent_layer_session_settings
from scripts.vae_with_kfolds import vae_over_regions_kfolds

session_datetime = datetime.now().isoformat()
print("Time session init: {}".format(session_datetime))

# Meta settings.
images_used = "PET"
#images_used = "MRI"
n_folds = 2
bool_test = False
regions_used = "three"

# Vae settings
# Net Configuration
middle_architecture = [200, 150]
# latent_code_dim_list = [5, 10 ,15]
latent_code_dim_list = [2, 5, 8, 10, 25, 50, 75, 100, 125, 150, 175, 200, 225,
                        250, 275, 300, 325, 350, 375, 400]
# latent_code_dim_list = [100]
list_regions = session.select_regions_to_evaluate(regions_used)

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
    "max_iter": 10,
    "save_meta_bool": False,
    "show_error_iter": 10,
}

# DECISION NET CONFIGURATION
decision_net_session_conf = {
    "decision_net_tries": 1,
    "field_to_select_try": "area under the curve",
    "max_iter": 10,
    "threshould_prefixed_to_0.5": True,
}

HYPERPARAMS_decision_net = {
    "batch_size": 32,
    "learning_rate": 1E-5,
    "lambda_l2_reg": 0.000001,
    "dropout": 0.9,
    "nonlinearity": tf.nn.relu,
}

# OUTPUT: Files initialization
loop_output_file_simple_majority_vote = os.path.join(
    loop_latent_layer_session_settings.path_kfolds_session_folder,
    "loop_output_simple_majority_vote.csv")

loop_output_file_complex_majority_vote = os.path.join(
    loop_latent_layer_session_settings.path_kfolds_session_folder,
    "loop_output_complex_majority_vote.csv")

loop_output_file_decision_net = os.path.join(
    loop_latent_layer_session_settings.path_kfolds_session_folder,
    "loop_output_decision_net.csv")

loop_output_file_weighted_svm = os.path.join(
    loop_latent_layer_session_settings.path_kfolds_session_folder,
    "loop_output_weighted_svm.csv")

loop_output_path_session_description = os.path.join(
    loop_latent_layer_session_settings.path_kfolds_session_folder,
    "session_description.csv")

tar_file_main_output_path = os.path.join(
    loop_latent_layer_session_settings.path_kfolds_session_folder,
    "loop_over_dim_out_session_{}.tar.gz".format(session_datetime))

list_paths_files_to_store = [loop_output_file_simple_majority_vote,
                             loop_output_file_complex_majority_vote,
                             loop_output_file_decision_net,
                             loop_output_file_weighted_svm,
                             loop_output_path_session_description]

# SESSION DESCRIPTOR ELLABORATION
session_descriptor = {}
session_descriptor['meta settings'] = {"n_folds": n_folds,
                                       "bool_test": bool_test,
                                       "regions_used": regions_used,
                                       "loop_over_latent_dim":
                                           str(latent_code_dim_list)}
session_descriptor['VAE'] = {}
session_descriptor['Decision net'] = {}
session_descriptor['VAE']["net configuration"] = hyperparams_vae
session_descriptor['VAE']["net configuration"][
    "architecture"] = "input_" + "_".join(
    str(x) for x in middle_architecture)
session_descriptor['VAE']["session configuration"] = vae_session_conf
session_descriptor['Decision net'][
    "net configuration"] = HYPERPARAMS_decision_net
session_descriptor['Decision net']["net configuration"]['architecture'] = \
    "[nºregions, nºregions/2, 1]"
session_descriptor['Decision net']['session_conf'] = decision_net_session_conf

file_session_descriptor = open(loop_output_path_session_description, "w")
output_utils.print_recursive_dict(session_descriptor,
                                  file=file_session_descriptor)
file_session_descriptor.close()

# Loading data
n_samples = 0
if images_used == "PET":
    dic_regions_to_flatten_voxels_pet, patient_labels, n_samples = \
        load_pet_data = load_pet_data_flat(list_regions)
elif images_used == "MRI":
    dic_regions_to_flatten_voxels_mri_gm, dic_regions_to_flatten_voxels_mri_wm, \
        patient_labels, n_samples = load_mri_data_flat(list_regions)

list_averages_svm_weighted = []
list_averages_simple_majority_vote = []
list_averages_decision_net = []
list_averages_complex_majority_vote = []

for latent_dim in latent_code_dim_list:

    print( "Evaluating the system with a latent code of {} dim".format(latent_dim))

    temp_architecture = deepcopy(middle_architecture)
    temp_architecture.extend([latent_dim])
    vae_session_conf["after_input_architecture"] = temp_architecture

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

    k_fold_dict = generate_k_folder_in_dict(
        n_samples, n_folds)

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

        Y_train, Y_test = get_test_and_train_labels_from_kfold_dict_entry(
            k_fold_entry=k_fold_dict[k_fold_index],
            patient_labels=patient_labels)

        print("Kfold {} Selected".format(k_fold_index))
        print("Number test samples {}".format(len(Y_test)))
        print("Number train samples {}".format(len(Y_train)))

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

        data = {}
        data["test"] = {}
        data["train"] = {}
        data["test"]["data"] = test_score_matriz
        data["test"]["label"] = Y_test
        data["train"]["data"] = train_score_matriz
        data["train"]["label"] = Y_train

        if bool_test:
            print("\nMatriz svm scores -> shapes, before complex majority vote")
            print("train matriz [patients x region]: " + str(
                train_score_matriz.shape))
            print("test matriz scores [patient x region]: " + str(
                test_score_matriz.shape))

        # COMPLEX MAJORITY VOTE

        complex_output_dic_test, complex_output_dic_train = \
            evaluation_utils.complex_majority_vote_evaluation(data, bool_test=bool_test)

        # Adding results to kfolds output
        complex_majority_vote_k_folds_results_train.append(
            complex_output_dic_train)
        complex_majority_vote_k_folds_results_test.append(
            complex_output_dic_test)

        if bool_test:
            print("\nMatriz svm scores -> shapes, after complex majority vote")
            print("train matriz [patients x regions]: " + str(
                train_score_matriz.shape))
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

        simple_majority_vote_k_folds_results_train.append(
            simple_output_dic_train)
        simple_majority_vote_k_folds_results_test.append(simple_output_dic_test)

        # SVM weighted REGIONS RESULTS
        print("DECISION WEIGHTING SVM OUTPUTS")
        # The score matriz is in regions per patient, we should transpose it
        # in the svm process

        weighted_output_dic_test, weighted_output_dic_train, \
        aux_dic_regions_weight_coefs = \
            evaluation_utils.weighted_svm_decision_evaluation(data,
                                                              list_regions,
                                                              bool_test=bool_test)

        svm_weighted_regions_k_folds_results_train.append(
            weighted_output_dic_train)
        svm_weighted_regions_k_folds_results_test.append(
            weighted_output_dic_test)
        svm_weighted_regions_k_folds_coefs.append(aux_dic_regions_weight_coefs)

        # DECISION NEURAL NET
        print("Decision neural net step")

        # train score matriz [patients x regions]
        input_layer_size = train_score_matriz.shape[1]
        architecture = [input_layer_size, int(input_layer_size / 2), 1]

        net_train_dic, net_test_dic = \
            leaky_net_utils.train_leaky_neural_net_various_tries_over_svm_output(
                decision_net_session_conf, architecture,
                HYPERPARAMS_decision_net,
                train_score_matriz, test_score_matriz, Y_train, Y_test,
                bool_test=False)

        decision_net_k_folds_results_train.append(net_train_dic)
        decision_net_vote_k_folds_results_test.append(net_test_dic)

    # GET AVERAGE RESULTS OVER METRICS
    extra_field = {"latent_dim": str(latent_dim)}

    averages_simple_majority_vote = get_average_over_metrics(
        simple_majority_vote_k_folds_results_test)
    averages_simple_majority_vote.update(extra_field)

    averages_complex_majority_vote = get_average_over_metrics(
        complex_majority_vote_k_folds_results_test)
    averages_complex_majority_vote.update(extra_field)

    averages_svm_weighted = get_average_over_metrics(
        svm_weighted_regions_k_folds_results_test)
    averages_svm_weighted.update(extra_field)

    averages_decision_net = get_average_over_metrics(
        decision_net_vote_k_folds_results_test)
    averages_decision_net.update(extra_field)

    list_averages_svm_weighted.append(averages_svm_weighted)
    list_averages_simple_majority_vote.append(averages_simple_majority_vote)
    list_averages_decision_net.append(averages_decision_net)
    list_averages_complex_majority_vote.append(averages_complex_majority_vote)

# Outputs files
# simple majority
output_utils.print_dictionary_with_header(
    loop_output_file_simple_majority_vote,
    list_averages_simple_majority_vote)
# complex majority
output_utils.print_dictionary_with_header(
    loop_output_file_complex_majority_vote,
    list_averages_complex_majority_vote)

output_utils.print_dictionary_with_header(
    loop_output_file_decision_net,
    list_averages_decision_net)

output_utils.print_dictionary_with_header(
    loop_output_file_weighted_svm,
    list_averages_svm_weighted)

# Tarfile to group the results
tar = tarfile.open(tar_file_main_output_path, "w:gz")
for file in list_paths_files_to_store:
    tar.add(file)
tar.close()

import tensorflow as tf
import os
from lib.cv_utils import get_test_and_train_labels_from_kfold_dict_entry, generate_k_folder_in_dict
from lib import cv_utils
from lib import session_helper as session
from scripts.vae_sweep_over_features import loop_latent_layer_session_settings
from scripts.vae_with_kfolds import vae_over_regions_kfolds
from lib import svm_utils
from lib.evaluation_utils import get_average_over_metrics
from lib import evaluation_utils
from lib import output_utils
from copy import deepcopy
import numpy as np
from lib import cvae_over_regions
import tarfile
from datetime import datetime
from nifti_regions_loader import \
    load_mri_data_flat, load_pet_data_flat
from lib.neural_net.leaky_relu_decision_net import DecisionNeuralNet as \
    DecisionNeuralNet_leaky_relu_3layers_with_sigmoid
from lib.neural_net.decision_neural_net import DecisionNeuralNet
from lib.neural_net import leaky_net_utils
import lib.neural_net.kfrans_ops as ops
from settings import explicit_iter_per_region
from nifti_regions_loader import \
    load_pet_data_3d, load_mri_data_3d

session_datetime = datetime.now().isoformat()
print("Time session init: {}".format(session_datetime))

# Meta settings.
images_used = "PET"
#images_used = "MRI"
n_folds = 2
bool_test = False
regions_used = "three"
list_regions = session.select_regions_to_evaluate(regions_used)


# Vae settings
# Net Configuration
kernel_list = [2, 5]

hyperparams = {
               "latent_layer_dim": 10,
               'activation_layer': ops.lrelu,
               'features_depth': [1, 2, 4],
               'decay_rate': 0.002,
               'learning_rate': 0.001,
               'lambda_l2_regularization': 0.0001}

# Vae session cofiguration
cvae_session_conf = {
    "batch_size": 8,
    "bool_normalized": False,
    "n_iters": 10,
    "save_meta_bool": False,
    "show_error_iter": 1,
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
                                       "loop_over_kernel":
                                           str(kernel_list)}
session_descriptor['VAE'] = {}
session_descriptor['Decision net'] = {}
session_descriptor['VAE']["net configuration"] = hyperparams
session_descriptor['VAE']["session configuration"] = cvae_session_conf
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
n_samples=0
if images_used == "PET":
    region_to_3dimg_dict_pet, patient_labels, n_samples = \
        load_pet_data_3d(list_regions)

elif images_used == "MRI":
    region_to_3dimg_dict_mri_gm, region_to_3dimg_dict_mri_wm,\
    patient_labels, n_samples = load_mri_data_3d(list_regions)

list_averages_svm_weighted = []
list_averages_simple_majority_vote = []
list_averages_decision_net = []
list_averages_complex_majority_vote = []

for kernel in kernel_list:

    print( "Evaluating the system with a latent code of {} dim".format(kernel))

    hyperparams["kernel_size"] = kernel

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
                    region_to_img_dict=region_to_3dimg_dict_mri_gm)

            reg_to_group_to_images_dict_mri_wm = \
                cv_utils.restructure_dictionary_based_on_cv_index_flat_images(
                    dict_train_test_index=k_fold_dict[k_fold_index],
                    region_to_img_dict=region_to_3dimg_dict_mri_wm)

        if images_used == "PET":
            reg_to_group_to_images_dict_pet = \
                cv_utils.restructure_dictionary_based_on_cv_index_flat_images(
                    dict_train_test_index=k_fold_dict[k_fold_index],
                    region_to_img_dict=region_to_3dimg_dict_pet)

        Y_train, Y_test = get_test_and_train_labels_from_kfold_dict_entry(
            k_fold_entry=k_fold_dict[k_fold_index],
            patient_labels=patient_labels)

        print("Kfold {} Selected".format(k_fold_index))
        print("Number test samples {}".format(len(Y_test)))
        print("Number train samples {}".format(len(Y_train)))

        if images_used == "MRI":
            print("Training MRI regions over GM")
            vae_output["gm"] = cvae_over_regions.execute_without_any_logs(
                region_train_cubes_dict=reg_to_group_to_images_dict_mri_gm["train"],
                hyperparams=hyperparams,
                session_conf=cvae_session_conf,
                list_regions=list_regions,
                path_to_root=None,
                region_test_cubes_dict=reg_to_group_to_images_dict_mri_gm["test"],
                explicit_iter_per_region=explicit_iter_per_region
            )

            print("Training MRI regions over GM")
            vae_output["wm"] = cvae_over_regions.execute_without_any_logs(
                region_train_cubes_dict=reg_to_group_to_images_dict_mri_wm["train"],
                hyperparams=hyperparams,
                session_conf=cvae_session_conf,
                list_regions=list_regions,
                path_to_root=None,
                region_test_cubes_dict=reg_to_group_to_images_dict_mri_wm["test"],
                explicit_iter_per_region=explicit_iter_per_region
            )

            # [patient x region]
            train_score_matriz, test_score_matriz = svm_utils.svm_mri_over_vae_output(
                vae_output, Y_train, Y_test, list_regions, bool_test=bool_test)

        if images_used == "PET":
            print("Train over regions")
            vae_output = cvae_over_regions.execute_without_any_logs(
                region_train_cubes_dict=reg_to_group_to_images_dict_pet["train"],
                hyperparams=hyperparams,
                session_conf=cvae_session_conf,
                list_regions=list_regions,
                path_to_root=None,
                region_test_cubes_dict=reg_to_group_to_images_dict_pet["test"],
                explicit_iter_per_region=explicit_iter_per_region
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
    extra_field = {"kernel size": str(kernel)}

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

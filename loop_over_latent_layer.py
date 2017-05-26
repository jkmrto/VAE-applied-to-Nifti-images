import tensorflow as tf
import os
from lib import cv_utils
from lib.mri import stack_NORAD
from lib import session_helper as session
from scripts.vae_loop_dim import session_settings
from scripts.vae_with_kfolds import vae_over_regions_kfolds
from lib.mri.stack_NORAD import load_patients_labels
from lib import svm_utils
from lib.evaluation_utils import simple_evaluation_output
from lib.evaluation_utils import get_average_over_metrics
from lib import evaluation_utils
from lib import output_utils
from shutil import copyfile
import numpy as np
import tarfile
from datetime import datetime
from lib.neural_net.leaky_relu_decision_net import DecisionNeuralNet as \
    DecisionNeuralNet_leaky_relu_3layers_with_sigmoid
from lib.neural_net.decision_neural_net import DecisionNeuralNet
from lib.neural_net import leaky_net_utils



session_datetime = datetime.now().isoformat()
print("Time session init: {}".format(session_datetime))

# Meta settings.
n_folds = 3
bool_test = False
regions_used = "most_important"

# Vae settings
# Net Configuration
middle_architecture = [1000, 500]
#latent_code_dim_list = [5, 10 ,15]
latent_code_dim_list = [2, 5, 8, 10, 25, 50, 75, 100, 125, 150, 175, 200]
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
    "max_iter": 100,
    "save_meta_bool": False,
    "show_error_iter": 10,
}

# DECISION NET CONFIGURATION
decision_net_session_conf = {
    "decision_net_tries": 1,
    "field_to_select_try": "area under the curve",
    "max_iter": 100,
    "threshould_prefixed_to_0.5": True,
}

HYPERPARAMS_decision_net = {
    "batch_size": 200,
    "learning_rate": 1E-6,
    "lambda_l2_reg": 0.000001,
    "dropout": 1,
    "nonlinearity": tf.nn.relu,
}

# Selecting the GM folder
path_to_root_GM = session_settings.path_GM_folder
path_to_root_WM = session_settings.path_WM_folder
# Loading the stack of images
dict_norad_gm = stack_NORAD.get_gm_stack()
dict_norad_wm = stack_NORAD.get_wm_stack()
patient_labels = load_patients_labels()

# OUTPUT: Files initialization
loop_output_file_simple_majority_vote = os.path.join(
    session_settings.path_kfolds_session_folder,
    "loop_output_simple_majority_vote.csv")

loop_output_file_complex_majority_vote = os.path.join(
    session_settings.path_kfolds_session_folder,
    "loop_output_complex_majority_vote.csv")

loop_output_file_decision_net = os.path.join(
    session_settings.path_kfolds_session_folder,
    "loop_output_decision_net.csv")

loop_output_file_weighted_svm = os.path.join(
    session_settings.path_kfolds_session_folder,
    "loop_output_weighted_svm.csv")

loop_output_path_session_description = os.path.join(
    session_settings.path_kfolds_session_folder,
    "session_description.csv")


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

list_averages_svm_weighted = []
list_averages_simple_majority_vote = []
list_averages_decision_net = []
list_averages_complex_majority_vote = []

for latent_dim in latent_code_dim_list:
    print("Evaluating the system with a latent code of {} dim".format(latent_dim))

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

    cv_utils.generate_k_fold(session_settings.path_kfolds_folder,
                             dict_norad_gm['stack'], n_folds)

    for k_fold_index in range(1, n_folds + 1, 1):
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
        middle_architecture.extend([latent_dim])
        vae_output['gm'] = vae_over_regions_kfolds.execute(voxels_values,
                                                           hyperparams_vae,
                                                           vae_session_conf,
                                                           middle_architecture,
                                                           path_to_root_GM,
                                                           list_regions)

        voxels_values = {}
        voxels_values['train'] = dict_norad_wm['stack'][train_index, :]
        voxels_values['test'] = dict_norad_wm['stack'][test_index, :]

        print("Train over WM regions")
        vae_output['wm'] = vae_over_regions_kfolds.execute(voxels_values,
                                                           hyperparams_vae,
                                                           vae_session_conf,
                                                           middle_architecture,
                                                           path_to_root_WM,
                                                           list_regions)
        train_score_matriz, test_score_matriz = svm_utils.svm_over_vae_output(
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
            evaluation_utils.complex_majority_vote_evaluation(data,
                                                              bool_test=bool_test)

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


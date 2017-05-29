import tensorflow as tf
import os
from lib import cv_utils
from lib.mri import stack_NORAD
from lib import session_helper as session
from scripts.vae_sweep_over_features import \
    loop_svm_minimum_error_session_settings as session_settings
from scripts.vae_with_kfolds import vae_over_regions_kfolds
from lib.mri.stack_NORAD import load_patients_labels
from lib import svm_utils
from lib.evaluation_utils import simple_evaluation_output
from lib.evaluation_utils import get_average_over_metrics
from lib import evaluation_utils
from lib import output_utils
from copy import deepcopy
from shutil import copyfile
import numpy as np
import tarfile
from datetime import datetime
from lib.neural_net.leaky_relu_decision_net import DecisionNeuralNet as \
    DecisionNeuralNet_leaky_relu_3layers_with_sigmoid
from lib.neural_net.decision_neural_net import DecisionNeuralNet
from lib.neural_net import leaky_net_utils


def format_output_data(dict_per_svm_error_kfold_outputs):

    list_averages_results = []
    for key, value in dict_per_svm_error_kfold_outputs.items():
        extra_field = {"svm_minimun_error": str(key)}

        temp_averages_over_kfold_results = get_average_over_metrics(
            value)

        temp_averages_over_kfold_results.update(extra_field)

        list_averages_results.append(temp_averages_over_kfold_results)

    return list_averages_results

# Default SVM error 1e-3
list_svm_errors = [0.1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.00025, 0.001,
                   0.00005, 0.00001]

#list_svm_errors = [0.1, 0.01, 0.001]

session_datetime = datetime.now().isoformat()
print("Time session init: {}".format(session_datetime))

# Meta settings.
n_folds = 10
bool_test = False
regions_used = "most_important"
#regions_used = "three"

# Vae settings
# Net Configuration
after_architecture = [1000, 500, 100]
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
    "decision_net_tries": 20,
    "field_to_select_try": "area under the curve",
    "max_iter": 100,
    "threshould_prefixed_to_0.5": True,
}

HYPERPARAMS_decision_net = {
    "batch_size": 200,
    "learning_rate": 1E-5,
    "lambda_l2_reg": 0.000001,
    "dropout": 0.9,
    "nonlinearity": tf.nn.relu,
}

# Loading the stack of images
dict_norad_gm = stack_NORAD.get_gm_stack()
dict_norad_wm = stack_NORAD.get_wm_stack()
patient_labels = load_patients_labels()


# OUTPUT: Files initialization
loop_output_file_simple_majority_vote = os.path.join(
    session_settings.path_session,
    "loop_output_simple_majority_vote.csv")

loop_output_file_complex_majority_vote = os.path.join(
    session_settings.path_session,
    "loop_output_complex_majority_vote.csv")

loop_output_file_decision_net = os.path.join(
    session_settings.path_session,
    "loop_output_decision_net.csv")

loop_output_file_weighted_svm = os.path.join(
    session_settings.path_session,
    "loop_output_weighted_svm.csv")

loop_output_path_session_description = os.path.join(
    session_settings.path_session,
    "session_description.csv")

tar_file_main_output_path = os.path.join(
    session_settings.path_session,
    "loop_over_svm_minimum_error_session_{}.tar.gz".format(session_datetime))

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
                                       "svm_over_minimum_error":
                                           str(list_svm_errors)}
session_descriptor['VAE'] = {}
session_descriptor['Decision net'] = {}
session_descriptor['VAE']["net configuration"] = hyperparams_vae
session_descriptor['VAE']["net configuration"][
    "architecture"] = "input_" + "_".join(
    str(x) for x in after_architecture)
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

cm_vote_per_svm_error_k_folds_results_train = {}
cm_vote_per_svm_error_k_folds_results_test = {}

sm_vote_per_svm_error_k_folds_results_train = {}
sm_vote_per_svm_error_k_folds_results_test = {}

nn_per_svm_error_k_folds_results_train = {}
nn_per_svm_error_k_folds_results_test = {}

svmw_per_svm_error_folds_results_train = {}
svmw_per_svm_error_k_folds_results_test = {}

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
    vae_output['gm'] = vae_over_regions_kfolds.execute_without_any_logs(voxels_values,
                                                       hyperparams_vae,
                                                       vae_session_conf,
                                                       after_architecture,
                                                       list_regions)

    voxels_values = {}
    voxels_values['train'] = dict_norad_wm['stack'][train_index, :]
    voxels_values['test'] = dict_norad_wm['stack'][test_index, :]

    print("Train over WM regions")
    vae_output['wm'] = vae_over_regions_kfolds.execute_without_any_logs(voxels_values,
                                                       hyperparams_vae,
                                                       vae_session_conf,
                                                       after_architecture,
                                                       list_regions)

    for svm_minimun_error in list_svm_errors:

        cm_vote_per_svm_error_k_folds_results_train[str(svm_minimun_error)] = []
        cm_vote_per_svm_error_k_folds_results_test[str(svm_minimun_error)] = []

        sm_vote_per_svm_error_k_folds_results_train[str(svm_minimun_error)] = []
        sm_vote_per_svm_error_k_folds_results_test[str(svm_minimun_error)] = []

        nn_per_svm_error_k_folds_results_train[str(svm_minimun_error)] = []
        nn_per_svm_error_k_folds_results_test[str(svm_minimun_error)] = []

        svmw_per_svm_error_folds_results_train[str(svm_minimun_error)] = []
        svmw_per_svm_error_k_folds_results_test[str(svm_minimun_error)] = []

        train_score_matriz, test_score_matriz = svm_utils.svm_over_vae_output(
            vae_output, Y_train, Y_test, list_regions, bool_test=bool_test,
            minimum_training_svm_error=svm_minimun_error)

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
        cm_vote_per_svm_error_k_folds_results_train[str(svm_minimun_error)].append(
            complex_output_dic_train)
        cm_vote_per_svm_error_k_folds_results_test[str(svm_minimun_error)].append(
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

        sm_vote_per_svm_error_k_folds_results_train[str(svm_minimun_error)].append(
            simple_output_dic_train)
        sm_vote_per_svm_error_k_folds_results_test[str(svm_minimun_error)].append(simple_output_dic_test)

        # SVM weighted REGIONS RESULTS
        print("DECISION WEIGHTING SVM OUTPUTS")
        # The score matriz is in regions per patient, we should transpose it
        # in the svm process

        weighted_output_dic_test, weighted_output_dic_train, \
        aux_dic_regions_weight_coefs = \
            evaluation_utils.weighted_svm_decision_evaluation(data,
                                                              list_regions,
                                                              bool_test=bool_test)

        svmw_per_svm_error_folds_results_train[str(svm_minimun_error)].append(
            weighted_output_dic_train)
        svmw_per_svm_error_k_folds_results_test[str(svm_minimun_error)].append(
            weighted_output_dic_test)

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

        nn_per_svm_error_k_folds_results_train[str(svm_minimun_error)].append(net_train_dic)
        nn_per_svm_error_k_folds_results_test[str(svm_minimun_error)].append(net_test_dic)


averages_sm = format_output_data(sm_vote_per_svm_error_k_folds_results_test)

averages_cm = format_output_data(cm_vote_per_svm_error_k_folds_results_test)

averages_nn = format_output_data(nn_per_svm_error_k_folds_results_test)

averages_svmw = format_output_data(svmw_per_svm_error_k_folds_results_test)

# Outputs files
# simple majority
output_utils.print_dictionary_with_header(
    loop_output_file_simple_majority_vote,
    averages_sm)
# complex majority
output_utils.print_dictionary_with_header(
    loop_output_file_complex_majority_vote,
    averages_cm)

output_utils.print_dictionary_with_header(
    loop_output_file_decision_net,
    averages_nn)

output_utils.print_dictionary_with_header(
    loop_output_file_weighted_svm,
    averages_svmw)

# Tarfile to group the results
tar = tarfile.open(tar_file_main_output_path, "w:gz")
for file in list_paths_files_to_store:
    tar.add(file)
tar.close()

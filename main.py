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
from lib.evaluation_utils import get_average_over_metrics
from lib import evaluation_utils
from lib import output_utils
from shutil import copyfile
import numpy as np
import tarfile
from datetime import datetime
from lib.neural_net.manual_layer_decision_net import DecisionNeuralNet as \
    DecisionNeuralNet_leaky_relu_3layers_with_sigmoid
from lib.neural_net.decision_neural_net import DecisionNeuralNet

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

# OUTPUT: Files initialization
k_fold_output_file_simple_majority_vote = os.path.join(
    session_settings.path_kfolds_session_folder,
    "k_fold_output_simple_majority_vote.csv")

k_fold_output_file_complex_majority_vote = os.path.join(
    session_settings.path_kfolds_session_folder,
    "k_fold_output_complex_majority_vote.csv")

k_fold_output_file_decision_net = os.path.join(
    session_settings.path_kfolds_session_folder,
    "k_fold_output_decision_net.csv")

k_fold_output_path_session_description = os.path.join(
    session_settings.path_kfolds_session_folder,
    "session_description.csv")

k_fold_output_resume_path = os.path.join(
    session_settings.path_kfolds_session_folder,
    "resume.csv")

tar_file_main_output_path = os.path.join(
    session_settings.path_kfolds_session_folder,
    "main_out_session_{}.tar.gz".format(session_datetime))

tar_file_main_output_path_replica = os.path.join(
    session_settings.path_kfolds_session_folder,
    "last_session_main_out.tar.gz".format(session_datetime))

list_paths_files_to_store = [k_fold_output_file_simple_majority_vote,
                             k_fold_output_file_complex_majority_vote,
                             k_fold_output_file_decision_net,
                             k_fold_output_resume_path,
                             k_fold_output_path_session_description]

# Selecting the GM folder
path_to_root_GM = session_settings.path_GM_folder
path_to_root_WM = session_settings.path_WM_folder
# Loading the stack of images
dict_norad_gm = stack_NORAD.get_gm_stack()
dict_norad_wm = stack_NORAD.get_wm_stack()
patient_labels = load_patients_labels()

# Meta settings.
n_folds = 10
bool_test = False
regions_used = "most_important"

# Vae settings
# Net Configuration
after_input_architecture = [500, 200, 50]

hyperparams_vae = {
    "batch_size": 16,
    "learning_rate": 1E-5,
    "dropout": 0.7,
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
    "decision_net_tries": 10,
    "field_to_select_try": "area under the curve",
    "max_iter": 100,
    "threshould_prefixed_to_0.5": True,
}

HYPERPARAMS_decision_net = {
    "batch_size": 200,
    "learning_rate": 1E-4,
    "lambda_l2_reg": 0.000001,
    "dropout": 1,
    "nonlinearity": tf.nn.relu,
}

# Session descriptor elaboration
session_descriptor = {}
session_descriptor['meta settings'] = {"n_folds": n_folds,
                                       "bool_test": bool_test,
                                       "regions_used": regions_used}
session_descriptor['VAE'] = {}
session_descriptor['Decision net'] = {}
session_descriptor['VAE']["net configuration"] = hyperparams_vae
session_descriptor['VAE']["net configuration"][
    "architecture"] = "input_" + "_".join(
    str(x) for x in after_input_architecture)
session_descriptor['VAE']["session configuration"] = vae_session_conf
session_descriptor['Decision net'][
    "net configuration"] = HYPERPARAMS_decision_net
session_descriptor['Decision net']["net configuration"]['architecture'] = \
    "[nºregions, nºregions/2, 1]"
session_descriptor['Decision net']['session_conf'] = decision_net_session_conf

file_session_descriptor = open(k_fold_output_path_session_description, "w")
output_utils.print_recursive_dict(session_descriptor,
                                  file=file_session_descriptor)
file_session_descriptor.close()

# Load regions index and create kfolds folder
list_regions = session.select_regions_to_evaluate(regions_used)
cv_utils.generate_k_fold(session_settings.path_kfolds_folder,
                         dict_norad_gm['stack'], n_folds)

# Main Loop
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
    vae_output['gm'] = vae_over_regions_kfolds.execute(voxels_values,
                                                       hyperparams_vae,
                                                       vae_session_conf,
                                                       after_input_architecture,
                                                       path_to_root_GM,
                                                       list_regions)

    voxels_values = {}
    voxels_values['train'] = dict_norad_wm['stack'][train_index, :]
    voxels_values['test'] = dict_norad_wm['stack'][test_index, :]

    print("Train over WM regions")
    vae_output['wm'] = vae_over_regions_kfolds.execute(voxels_values,
                                                       hyperparams_vae,
                                                       vae_session_conf,
                                                       after_input_architecture,
                                                       path_to_root_WM,
                                                       list_regions)

    train_score_matriz = np.zeros((len(train_index), len(list_regions)))
    test_score_matriz = np.zeros((len(test_index), len(list_regions)))

    i = 0
    dic_region_to_matriz_pos = {}

    for region_selected in list_regions:
        dic_region_to_matriz_pos[str(region_selected)] = i

        print("SVM step")
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
        if bool_test:
            print("Shape wm+gm train data post encoder")
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
            test_train_score = np.hstack(
                (np.row_stack(train_score), np.row_stack(Y_train)))
            test_test_score = np.hstack(
                (np.row_stack(test_score), np.row_stack(Y_test)))
            print(test_train_score)
            print(test_test_score)

        i += 1

    if bool_test:
        print("Diccionario de regions utilizadas")
        print(dic_region_to_matriz_pos)

    # complex majority vote
    complex_means_train = np.row_stack(train_score_matriz.mean(axis=1))
    complex_means_test = np.row_stack(test_score_matriz.mean(axis=1))

    if bool_test:
        print("TEST OVER FINAL RESULTS")
        test_train_score = np.hstack(
            (np.row_stack(complex_means_train), np.row_stack(Y_train)))
        test_test_score = np.hstack(
            (np.row_stack(complex_means_test), np.row_stack(Y_test)))
        print(test_train_score)
        print(test_test_score)

    # COMPLEX MAJORITY VOTE
    threshold = 0
    _, complex_output_dic_train = simple_evaluation_output(complex_means_train,
                                                           Y_train, threshold,
                                                           bool_test=bool_test)
    _, complex_output_dic_test = simple_evaluation_output(complex_means_test,
                                                          Y_test, threshold,
                                                          bool_test=bool_test)

    print("Output kfolds nº {}.".format(k_fold_index))
    print("Complex Majority Vote Test: " + str(complex_output_dic_test))
    print("Complex Majority Vote Train: " + str(complex_output_dic_train))

    complex_majority_vote_k_folds_results_train.append(complex_output_dic_train)
    complex_majority_vote_k_folds_results_test.append(complex_output_dic_test)

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

    # DECISION NEURAL NET
    print("Decision neural net step")

    # train score matriz [patients x regions]
    input_layer_size = train_score_matriz.shape[1]
    architecture = [input_layer_size, int(input_layer_size / 2), 1]

    temp_results_per_try_test = []
    temp_results_per_try_train = []
    for i in range(1, decision_net_session_conf['decision_net_tries'] + 1, 1):
        print("Neural net try: {}".format(i))
        tf.reset_default_graph()
        v = DecisionNeuralNet_leaky_relu_3layers_with_sigmoid(
            architecture=architecture,
            hyperparams=HYPERPARAMS_decision_net,
            bool_test=bool_test)

        v.train(train_score_matriz, Y_train,
                max_iter=decision_net_session_conf['max_iter'],
                iter_to_show_error=10)
        print("Net Trained")

        # Test net created
        score_train = v.forward_propagation(train_score_matriz)[0]
        score_test = v.forward_propagation(test_score_matriz)[0]

        # Fixing the threeshold in function of the test evaluation or
        # prefixed it to 0.5, the last door of the neural net is a sigmoid

        if decision_net_session_conf["threshould_prefixed_to_0.5"]:

            threshold, decision_net_dic_train = simple_evaluation_output(
                score_train,
                Y_train, thresholds_establised=0.5,
                bool_test=bool_test)
        else:

            threshold, decision_net_dic_train = simple_evaluation_output(
                score_train,
                Y_train, bool_test=bool_test)

        _, decision_net_dic_test = simple_evaluation_output(
            score_test, Y_test, thresholds_establised=0.5,
            bool_test=bool_test)

        temp_results_per_try_test.append(decision_net_dic_test)
        temp_results_per_try_train.append(decision_net_dic_train)

    try_selected_test_dic = sorted(temp_results_per_try_test,
                                   key=lambda results: results[
                                       decision_net_session_conf[
                                           'field_to_select_try']],
                                   reverse=True)[0]

    try_selected_train_dic = sorted(temp_results_per_try_train,
                                    key=lambda results: results[
                                        decision_net_session_conf[
                                            'field_to_select_try']],
                                    reverse=True)[0]

    if bool_test:
        print("Results over 10 tries over decision neural net:")
        for i in range(0, len(temp_results_per_try_test), 1):
            print("Test: " + str(temp_results_per_try_test[i]))
            print("Train: " + str(temp_results_per_try_train[i]))

    print("Intent selected:")
    print("Decision Neural Net Test: " + str(try_selected_test_dic))
    print("Decision Neural Net Train: " + str(try_selected_train_dic))

    decision_net_k_folds_results_train.append(try_selected_train_dic)
    decision_net_vote_k_folds_results_test.append(try_selected_test_dic)

output_utils.print_dictionary_with_header(
    k_fold_output_file_simple_majority_vote,
    simple_majority_vote_k_folds_results_test)

output_utils.print_dictionary_with_header(
    k_fold_output_file_complex_majority_vote,
    complex_majority_vote_k_folds_results_test)

output_utils.print_dictionary_with_header(
    k_fold_output_file_decision_net,
    decision_net_vote_k_folds_results_test)

resume_list_dicts = []
simple_majority_vote = get_average_over_metrics(
    simple_majority_vote_k_folds_results_test)
complex_majority_vote = get_average_over_metrics(
    complex_majority_vote_k_folds_results_test)
decision_net = get_average_over_metrics(decision_net_vote_k_folds_results_test)

temp_simple_majority_vote = {"decision step": "Simple majority vote"}
temp_simple_majority_vote.update(simple_majority_vote)

temp_complex_majority_vote = {"decision step": "Complex majority vote"}
temp_complex_majority_vote.update(complex_majority_vote)

temp_decision_net = {"decision step": "Decision neural net"}
temp_decision_net.update(decision_net)

output_utils.print_dictionary_with_header(
    k_fold_output_resume_path,
    [temp_simple_majority_vote, temp_complex_majority_vote, temp_decision_net])

# Tarfile to group the results
tar = tarfile.open(tar_file_main_output_path, "w:gz")
for file in list_paths_files_to_store:
    tar.add(file)
tar.close()
copyfile(tar_file_main_output_path, tar_file_main_output_path_replica)

import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from lib.svm_utils import load_svm_output_score
from matplotlib import pyplot as plt
from sklearn.metrics import log_loss

import settings as set
from lib.neural_net.leaky_relu_decision_net import DecisionNeuralNet
from lib.utils.cv_utils import get_label_per_patient
from lib.utils.functions import load_csv_file_iter_to_error
from lib.utils.os_aux import create_directories

TYPE_SESSION_DECISION = "neural_net"
TEST_OUTPUT_FOLDER = "test_out"
TRAIN_OUTPUT_FOLDER = "train_out"
CURVE_ROC_FILE_NAME = "curva_roc.png"
RESUME_FILE_NAME = "resume.txt"
FILE_TRUE_TO_GET_PER_PATIENT = "results.txt"


def plot_test_error_vs_train_error(path_to_image, path_to_train_error_log,
                                   path_to_test_error_log):
    train_iter, train_error = load_csv_file_iter_to_error(
        path_to_train_error_log)
    test_iter, test_error = load_csv_file_iter_to_error(path_to_test_error_log)

    plt.figure()
    p1, = plt.plot(train_iter, train_error)
    p2, = plt.plot(test_iter, test_error)
    plt.legend([p1, p2], ["Train error", "Test error"])
    plt.savefig(path_to_image, dpi=100)


def cross_entropy(labels, scores, eps=1e-15):
    error = labels * np.log(np.array(scores) + eps) + (1 - np.array(
        labels)) * np.log(1 - np.array(scores) + eps)

    return error


def rectificate_values(array, min, max):
    array[array < min] = min
    array[array > max] = max

    return array


def init_session_folders(iden_session, svm_test_name):
    path_to_main_session = os.path.join(set.path_to_general_out_folder,
                                        iden_session)

    path_to_cv_folder = os.path.join(path_to_main_session,
                                     set.main_cv_vae_session)

    path_to_svm_test = os.path.join(path_to_main_session,
                                    set.main_test_folder_autoencoder_session,
                                    svm_test_name)

    path_file_svm_train_score = os.path.join(path_to_svm_test,
                                             set.svm_folder_name_train_out,
                                             set.svm_file_name_scores_file_name)

    path_file_svm_test_score = os.path.join(path_to_svm_test,
                                            set.svm_folder_name_test_out,
                                            set.svm_file_name_scores_file_name)

    datetime_str = datetime.now().strftime(r"%Y_%m_%d-%H:%M")
    # str_architecture = "_".join(map(str, architecture))
    root_session_folder_name = TYPE_SESSION_DECISION + "-" + datetime_str
    # Decision folders tree
    path_decision_session_folder = os.path.join(path_to_svm_test,
                                                root_session_folder_name)

    path_to_images = os.path.join(path_decision_session_folder, "images")
    path_to_logs = os.path.join(path_decision_session_folder, "logs")
    path_to_meta = os.path.join(path_decision_session_folder, "meta")
    path_to_post_train = os.path.join(path_decision_session_folder,
                                      "post_train")
    path_to_train_out = os.path.join(path_to_post_train, TRAIN_OUTPUT_FOLDER)
    path_to_test_out = os.path.join(path_to_post_train, TEST_OUTPUT_FOLDER)
    path_to_train_error_log = os.path.join(path_to_logs, "train_error.log")
    path_to_test_error_log = os.path.join(path_to_logs, "test_error.log")
    path_to_grad_error_image = os.path.join(path_to_images,
                                            "train_test_grad_error.png")

    create_directories([path_decision_session_folder,
                        path_to_logs,
                        path_to_images,
                        path_to_meta,
                        path_to_post_train])

    return (path_decision_session_folder, path_file_svm_train_score,
            path_file_svm_test_score, path_to_cv_folder,
            path_to_grad_error_image, path_to_train_error_log,
            path_to_test_error_log, path_to_train_out, path_to_test_out)


idi_session = "05_05_2017_08:19 arch: 1000_800_500_100"
test_name = "svm"
max_iter = 50000
iter_show_save_info = 1000

HYPERPARAMS = {
    "batch_size": 200,
    "learning_rate": 2.5E-6,
    "dropout": 1,
    "lambda_l2_reg": 0.000001,
    "nonlinearity": tf.nn.elu,
    "squashing": tf.nn.sigmoid,
}

path_decision_session_folder, path_file_train_score, path_file_test_score, \
path_to_cv_folder, path_to_grad_error_image, path_to_train_error_log, \
path_to_test_error_log, path_to_train_out, path_to_test_out = \
    init_session_folders(idi_session, test_name)

# LOADING DATA
X_train = load_svm_output_score(path_file_train_score)['data_normalize']
X_test = load_svm_output_score(path_file_test_score)['data_normalize']
Y_train, Y_test = get_label_per_patient(path_to_cv_folder)

# DIMENSIONING THE NET
input_layer_size = X_train.shape[1]
architecture = [input_layer_size, int(input_layer_size / 2), 1]
session = tf.Session()

v = DecisionNeuralNet(architecture, HYPERPARAMS,
                      root_path=path_decision_session_folder)

train_error_log = open(path_to_train_error_log, "w")
test_error_log = open(path_to_test_error_log, "w")

print("Start training by step process. Max iter = {}".format(max_iter))
for iter in range(0, max_iter, 1):
    # one step training
    v.train_by_step(X_train, Y_train, max_meta_to_keep=0,
                    step_to_save=iter_show_save_info)
    # error for the new net

    if iter % iter_show_save_info == 0:
        score_train = v.forward_propagation(X_train)[0]
        score_test = v.forward_propagation(X_test)[0]

        score_train = rectificate_values(score_train, 0, 1)
        score_test = rectificate_values(score_test, 0, 1)

        train_cost = log_loss(Y_train, score_train) / len(Y_train)
        test_cost = log_loss(Y_test, score_test) / len(Y_test)

        train_cost_manual = cross_entropy(Y_train, score_train) / len(Y_train)
        test_cost_manual = cross_entropy(Y_test, score_test) / len(Y_test)

        train_error_log.write("{0},{1}\n".format(iter, train_cost))
        train_error_log.flush()
        test_error_log.write("{0},{1}\n".format(iter, test_cost))
        test_error_log.flush()

        print("Iter: {}".format(iter))
        print("Error per sample, train samples error: {0}, "
              "test samples error: {1}".format(train_cost, test_cost))

        print("MANUAL ENTROPY: Error per sample, train samples error: {0}, "
              "test samples error: {1}\n".format(train_cost, test_cost))

train_error_log.close()
test_error_log.close()

plot_test_error_vs_train_error(path_to_grad_error_image,
                               path_to_train_error_log,
                               path_to_test_error_log)

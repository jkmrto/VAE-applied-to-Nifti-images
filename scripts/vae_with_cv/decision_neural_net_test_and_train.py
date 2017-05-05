import os
import settings as set
import numpy as np
from lib.aux_functionalities.os_aux import create_directories
from lib.svm_hub import load_svm_output_score
from lib.neural_net.decision_neural_net import DecisionNeuralNet
import tensorflow as tf
from lib.aux_functionalities import functions
from datetime import datetime
from lib.aux_functionalities.functions import print_dictionary
from lib.cv_hub import get_label_per_patient
from sklearn import metrics
from matplotlib import pyplot as plt

TYPE_SESSION_DECISION = "neural_net"
TEST_OUTPUT_FOLDER = "test_out"
TRAIN_OUTPUT_FOLDER = "train_out"
CURVE_ROC_FILE_NAME = "curva_roc.png"
RESUME_FILE_NAME = "resume.txt"
FILE_TRUE_TO_GET_PER_PATIENT = "results.txt"


def plot_grad_desc_error(path_to_grad_desc_error_log,
                         path_to_grad_desc_error_images):
    functions.plot_x_y_from_file_with_title(
        "Error descenso gradiente", path_to_grad_desc_error_log,
        path_to_grad_desc_error_images)


def evaluation_output(path_to_resume_file, path_to_roc_png,
                      path_to_results_file, y_obtained,  y_test):

    results = np.concatenate((y_test, y_obtained))

    precision = metrics.average_precision_score(y_test, y_obtained)
    auc = metrics.roc_auc_score(y_test, y_obtained)
    output_dic = {"precision": precision,

                  "area under the curve": auc}
    print_dictionary(path_to_resume_file, output_dic)
    [fpr, tpr, thresholds] = metrics.roc_curve(y_test, y_obtained)
    np.savetxt(path_to_results_file, results, delimiter=',')

    plt.figure()
    plt.plot(fpr, tpr, linestyle='--')
    plt.savefig(path_to_roc_png)


def init_session_folders(iden_session, svm_test_name):
    score_file_name = "patient_score_per_region.log"

    path_to_main_session = os.path.join(set.path_to_general_out_folder,
                                        iden_session)

    path_to_cv_folder = os.path.join(path_to_main_session,
                                     set.main_cv_vae_session)

    path_to_svm_test = os.path.join(path_to_main_session,
                                    set.main_test_folder_autoencoder_session,
                                    svm_test_name)

    path_file_train_score = os.path.join(path_to_svm_test,
                                         set.svm_folder_name_train_out,
                                         set.svm_file_name_scores_file_name)

    path_file_test_score = os.path.join(path_to_svm_test,
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
    path_to_grad_desc_error_log = os.path.join(path_to_logs, "grad_error.log")
    path_to_grad_desc_error_image = os.path.join(path_to_images,
                                                 "grad_error.png")

    create_directories([path_decision_session_folder,
                        path_to_logs,
                        path_to_images,
                        path_to_meta,
                        path_to_post_train,
                        path_to_train_out,
                        path_to_test_out])

    return (path_decision_session_folder, path_file_train_score,
            path_file_test_score, path_to_cv_folder,
            path_to_grad_desc_error_image, path_to_grad_desc_error_log,
            path_to_train_out, path_to_test_out)


# Session configuration
idi_session = "05_05_2017_08:19 arch: 1000_800_500_100"
test_name = "svm"
max_iter = 20000

HYPERPARAMS = {
    "batch_size": 200,
    "learning_rate": 5E-6,
    "dropout": 1,
    "lambda_l2_reg": 0.000001,
    "nonlinearity": tf.nn.relu,
    "squashing": tf.nn.sigmoid,
}

path_decision_session_folder, path_file_train_score, path_file_test_score, \
path_to_cv_folder, path_to_grad_desc_error_image, path_to_grad_desc_error_log, \
path_to_train_out, path_to_test_out = \
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
v.train(X_train, Y_train, max_iter=max_iter,
        path_to_grad_error_log_file_name=path_to_grad_desc_error_log)
# print(np.concatenate((y_true, y_obtained), axis=1))1
# svm_score = data['data_normalize']  # 417x42
plot_grad_desc_error(path_to_grad_desc_error_log, path_to_grad_desc_error_image)

# TEST TIME
score_train = v.forward_propagation(X_train)[0]
score_test = v.forward_propagation(X_test)[0]

evaluation_output(os.path.join(path_to_train_out, RESUME_FILE_NAME),
                  os.path.join(path_to_train_out, CURVE_ROC_FILE_NAME),
                  os.path.join(path_to_train_out, FILE_TRUE_TO_GET_PER_PATIENT),
                  score_train, Y_train)

evaluation_output(os.path.join(path_to_test_out, RESUME_FILE_NAME),
                  os.path.join(path_to_test_out, CURVE_ROC_FILE_NAME),
                  os.path.join(path_to_test_out, FILE_TRUE_TO_GET_PER_PATIENT),
                  score_test, Y_test)

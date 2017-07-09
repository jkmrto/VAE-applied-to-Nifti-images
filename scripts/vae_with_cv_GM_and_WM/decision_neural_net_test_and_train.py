import os
from datetime import datetime

import tensorflow as tf
from lib.evaluation_utils import evaluation_output
from lib.svm_utils import load_svm_output_score

import settings as set
from lib.aux_functionalities import functions
from lib.aux_functionalities.os_aux import create_directories
from lib.neural_net.decision_neural_net import DecisionNeuralNet
from lib.session_helper import generate_session_descriptor
from lib.utils.cv_utils import get_label_per_patient
from scripts.vae_with_cv_GM_and_WM import session_settings as main_settings
from scripts.vae_with_cv_GM_and_WM import svm_session_settings as svm_settings

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


def init_session_folders(svm_session_folder_name, kind_svm_session="combined"):
    path_to_svm_session = os.path.join(main_settings.path_post_encoding_folder,
                                       svm_session_folder_name)

    path_to_root = os.path.join(path_to_svm_session,
                                svm_settings.folder_post_svm)

    own_datetime = datetime.now().strftime(r"%d_%m_%_Y_%H:%M")
    iden_session = "DecisionNeuralNet_" + own_datetime
    path_decision_session_folder = os.path.join(path_to_root,
                                           iden_session)
    # Till here we have the folder to the session

    if kind_svm_session == "combined":
        path_file_train_score = os.path.join(path_to_svm_session,
            svm_settings.folder_combined_wm_and_gm_code_data_as_input_to_svm,
            set.svm_folder_name_train_out, set.svm_file_name_scores_file_name)

        path_file_test_score = os.path.join(path_to_svm_session,
            svm_settings.folder_combined_wm_and_gm_code_data_as_input_to_svm,
            set.svm_folder_name_test_out, set.svm_file_name_scores_file_name)

    # Defining own folders
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

    create_directories([path_to_root,
                        path_decision_session_folder,
                        path_to_logs,
                        path_to_images,
                        path_to_meta,
                        path_to_post_train,
                        path_to_train_out,
                        path_to_test_out])

    return (path_decision_session_folder, path_file_train_score,
            path_file_test_score,
            path_to_grad_desc_error_image, path_to_grad_desc_error_log,
            path_to_train_out, path_to_test_out)


# Session configuration
svm_session_folder_name = "bueno_svm_08_05_2017_20:02"
max_iter = 1500

# info session:
net_used = "relu net"

HYPERPARAMS = {
    "batch_size": 200,
    "learning_rate": 1E-4,
    "lambda_l2_reg": 0.000001,
    "dropout": 1,
    "nonlinearity": tf.nn.relu,
}

path_decision_session_folder, path_file_train_score, path_file_test_score, \
path_to_grad_desc_error_image, path_to_grad_desc_error_log, \
ath_to_train_out, path_to_test_out = \
    init_session_folders(svm_session_folder_name)

# LOADING DATA
X_train = load_svm_output_score(path_file_train_score)['data_normalize']
X_test = load_svm_output_score(path_file_test_score)['data_normalize']
Y_train, Y_test = get_label_per_patient(main_settings.path_cv_folder)

# DIMENSIONING THE NET
input_layer_size = X_train.shape[1]
architecture = [input_layer_size, int(input_layer_size / 2), 1]
session = tf.Session()

v = DecisionNeuralNet(architecture, HYPERPARAMS,
                      root_path=path_decision_session_folder)
# v = DecisionNeuralNet_leaky_relu_3layers(architecture, HYPERPARAMS,
#                      root_path=path_decision_session_folder)
v.train(X_train, Y_train, max_iter=max_iter,
        path_to_grad_error_log_file_name=path_to_grad_desc_error_log)
# print(np.concatenate((y_true, y_obtained), axis=1))1
# svm_score = data['data_normalize']  # 417x42
plot_grad_desc_error(path_to_grad_desc_error_log, path_to_grad_desc_error_image)

# TEST TIME
score_train = v.forward_propagation(X_train)[0]
score_test = v.forward_propagation(X_test)[0]

threshold = evaluation_output(os.path.join(path_to_train_out, RESUME_FILE_NAME),
                              os.path.join(path_to_train_out,
                                           CURVE_ROC_FILE_NAME),
                              os.path.join(path_to_train_out,
                                           FILE_TRUE_TO_GET_PER_PATIENT),
                              score_train, Y_train)

evaluation_output(os.path.join(path_to_test_out, RESUME_FILE_NAME),
                  os.path.join(path_to_test_out, CURVE_ROC_FILE_NAME),
                  os.path.join(path_to_test_out, FILE_TRUE_TO_GET_PER_PATIENT),
                  score_test, Y_test, thresholds_establised=threshold)

# SESSION DESCRIPTOR CREATION
session_descriptor_data = {"max_iter": max_iter,
                           "net used": net_used,
                           "architecture:": "_".join(
                               str(x) for x in architecture)}
session_descriptor_data.update(HYPERPARAMS)
generate_session_descriptor(path_decision_session_folder,
                            session_descriptor_data)

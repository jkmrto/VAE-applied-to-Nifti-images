import os
from datetime import datetime

import tensorflow as tf
from lib.evaluation_utils import evaluation_output
from lib.svm_utils import load_svm_output_score

import settings as set
from lib.aux_functionalities import functions
from lib.aux_functionalities.os_aux import create_directories
from lib.neural_net.decision_neural_net import DecisionNeuralNet
from lib.utils.cv_utils import get_label_per_patient

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


def init_session_folders(iden_session, svm_test_name):

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
svm_step_name = "svm"
max_iter = 50000

# info session:
net_used = "relu net"

HYPERPARAMS = {
    "batch_size": 200,
    "learning_rate": 1E-6,
    "lambda_l2_reg": 0.000001,
    "dropout": 1,
    "nonlinearity": tf.nn.relu,
}


path_decision_session_folder, path_file_train_score, path_file_test_score, \
path_to_cv_folder, path_to_grad_desc_error_image, path_to_grad_desc_error_log, \
path_to_train_out, path_to_test_out = \
    init_session_folders(idi_session, svm_step_name)

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
#v = DecisionNeuralNet_leaky_relu_3layers(architecture, HYPERPARAMS,
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
                  os.path.join(path_to_train_out, CURVE_ROC_FILE_NAME),
                  os.path.join(path_to_train_out, FILE_TRUE_TO_GET_PER_PATIENT),
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
session.generate_session_descriptor(path_decision_session_folder, session_descriptor_data)
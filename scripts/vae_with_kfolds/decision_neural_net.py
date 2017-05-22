import os
import settings as set
import numpy as np
from lib.aux_functionalities.os_aux import create_directories
from lib.svm_utils import load_svm_output_score
from lib.session_helper import generate_session_descriptor
from lib.neural_net.decision_neural_net import DecisionNeuralNet
from scripts.vae_with_cv_GM_and_WM import svm_session_settings as svm_settings
import tensorflow as tf
from lib.aux_functionalities import functions
from datetime import datetime
from lib.evaluation_utils import evaluation_output
from lib.cv_utils import get_label_per_patient
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
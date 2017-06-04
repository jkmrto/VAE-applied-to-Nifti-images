import os
import settings
import numpy as np
from lib.data_loader.MRI_stack_NORAD import load_patients_labels
from lib.aux_functionalities.os_aux import create_directories
from lib.svm_utils import load_svm_output_score
from lib.neural_net.decision_neural_net import DecisionNeuralNet
import tensorflow as tf
from lib.aux_functionalities import functions
from datetime import datetime
from sklearn.model_selection import train_test_split

TYPE_SESSION_DECISION = "neural_net"


def plot_grad_desc_error(path_to_grad_desc_error_log,
                         path_to_grad_desc_error_images):
    functions.plot_x_y_from_file_with_title(
        "Error descenso gradiente", path_to_grad_desc_error_log,
        path_to_grad_desc_error_images)


def init_session_folders(iden_session, svm_test_name):
    output_project_folder = "out"
    main_test_folder_autoencoder_session = "post_train"
    score_file_name = "patient_score_per_region.log"

    path_to_svm_test = os.path.join(settings.path_to_project,
                                    output_project_folder,
                                    iden_session,
                                    main_test_folder_autoencoder_session,
                                    svm_test_name)

    path_score_file = os.path.join(path_to_svm_test, score_file_name)

    datetime_str = datetime.now().strftime(r"%Y_%m_%d-%H:%M")
    # str_architecture = "_".join(map(str, architecture))
    root_session_folder_name = TYPE_SESSION_DECISION + "-" + datetime_str
    # Decision folders tree
    path_decision_session_folder = os.path.join(path_to_svm_test,
                                                root_session_folder_name)
    path_cross_validation_data_folder = os.path.join(
        path_decision_session_folder, "cv_data")
    path_to_images = os.path.join(path_decision_session_folder, "images")
    path_to_logs = os.path.join(path_decision_session_folder, "logs")
    path_to_meta = os.path.join(path_decision_session_folder, "meta")
    path_to_grad_desc_error_log = os.path.join(path_to_logs, "grad_error.log")
    path_to_grad_desc_error_image = os.path.join(path_to_images,
                                                 "grad_error.png")

    create_directories([path_decision_session_folder,
                        path_cross_validation_data_folder,
                        path_to_logs,
                        path_to_images,
                        path_to_meta])

    return (path_decision_session_folder, path_score_file,
            path_cross_validation_data_folder,
            path_to_grad_desc_error_image, path_to_grad_desc_error_log)


# iden_session = "02_05_2017_21:09 arch: 1000_800_500_100_2"
idi_session = "03_05_2017_08:12 arch: 1000_800_500_100"
test_name = "svm"
max_iter = 10000

# Neural net configuration
HYPERPARAMS = {
    "batch_size": 200,
    "learning_rate": 2.5E-6,
    "dropout": 1,
    "lambda_l2_reg": 0.000001,
    "nonlinearity": tf.nn.relu,
    "squashing": tf.nn.sigmoid,
}

path_decision_session_folder, path_score_file, \
path_cross_validation_data_folder, path_to_grad_desc_error_image, \
path_to_grad_desc_error_log = init_session_folders(idi_session, test_name)

data = load_svm_output_score(path_score_file)

patients_labels = load_patients_labels()  # 417x1
svm_score = data['data_normalize']  # 417x42

X_train, X_test, y_train, y_test = train_test_split(
    svm_score, patients_labels, test_size=0.4)

input_layer_size = svm_score.shape[1]
architecture = [input_layer_size, int(input_layer_size / 2), 1]

session = tf.Session()

# Storing the data after separate it in the cross validation
np.savetxt(path_cross_validation_data_folder + "/X_train.csv", X_train,
           delimiter=',')
np.savetxt(path_cross_validation_data_folder + "/X_test.csv", X_test,
           delimiter=',')
np.savetxt(path_cross_validation_data_folder + "/y_train.csv", y_train,
           delimiter=',')
np.savetxt(path_cross_validation_data_folder + "/y_test.csv", y_test,
           delimiter=',')

v = DecisionNeuralNet(architecture, HYPERPARAMS,
                      root_path=path_decision_session_folder)
v.train(X_train, y_train, max_iter=max_iter,
        path_to_grad_error_log_file_name=path_to_grad_desc_error_log)  # print("New iteration:")
# print(np.concatenate((y_true, y_obtained), axis=1))1

plot_grad_desc_error(path_to_grad_desc_error_log,
                     path_to_grad_desc_error_image)

import os
import settings
import numpy as np
from lib.mri.stack_NORAD import load_patients_labels
from lib.aux_functionalities.os_aux import create_directories
from scripts.svm_score_aux_functions import load_svm_output_score
from lib.neural_net.decision_neural_net import DecisionNeuralNet
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split

iden_session = "02_05_2017_21:09 arch: 1000_800_500_100_2"
svm_test_name = "test_svm_important"

# Neural net configuration
HYPERPARAMS = {
    "batch_size": 200,
    "learning_rate": 5E-6,
    "dropout": 1,
    "lambda_l2_reg": 0.00001,
    "nonlinearity": tf.nn.relu,
    "squashing": tf.nn.sigmoid,
}

max_iter = 100000
output_project_folder = "out"
main_test_folder_autoencoder_session = "post_train"

score_file_name = "patient_score_per_region.log"
path_to_svm_test = os.path.join(settings.path_to_project,
                                output_project_folder,
                                iden_session,
                                main_test_folder_autoencoder_session,
                                svm_test_name)

path_score_file = os.path.join(path_to_svm_test, score_file_name)

data = load_svm_output_score(path_score_file)
patients_labels = load_patients_labels()  # 417x1
svm_score = data['data_normalize']  # 417x42
X_train, X_test, y_train, y_test = train_test_split(
    svm_score, patients_labels, test_size=0.4)

input_layer_size = svm_score.shape[1]
architecture = [input_layer_size, int(input_layer_size / 2), 1]

type_session_evaluation = "neural_net"
datetime_str = datetime.now().strftime(r"%Y_%m_%d-%H:%M")
str_architecture = "_".join(map(str, architecture))
root_session_folder_name = type_session_evaluation + "-" + datetime_str + \
                           "-" + str_architecture
path_session_folder = os.path.join(path_to_svm_test,
                                   root_session_folder_name)
path_cross_validation_data_folder = os.path.join(path_session_folder,
                                                 "cv_data")
create_directories([path_session_folder, path_cross_validation_data_folder])
session = tf.Session()

# Storing the data after separate it in the cross validation
np.savetxt(path_cross_validation_data_folder + "/X_train.csv", X_train, delimiter=',')
np.savetxt(path_cross_validation_data_folder + "/X_test.csv", X_test, delimiter=',')
np.savetxt(path_cross_validation_data_folder + "/y_train.csv", y_train, delimiter=',')
np.savetxt(path_cross_validation_data_folder + "/y_test.csv", y_test, delimiter=',')


v = DecisionNeuralNet(architecture, HYPERPARAMS, root_path=path_session_folder)
v.train(X_train, y_train, max_iter=max_iter)  # print("New iteration:")
# print(np.concatenate((y_true, y_obtained), axis=1))

import os

import tensorflow as tf
from lib.evaluation_utils import evaluation_output
from lib.svm_utils import load_svm_output_score

import settings as set
from lib.neural_net.leaky_relu_decision_net import DecisionNeuralNet
from lib.utils.cv_utils import get_label_per_patient
from lib.utils.os_aux import create_directories

TYPE_SESSION_DECISION = "neural_net"
TEST_OUTPUT_FOLDER = "test_out"
TRAIN_OUTPUT_FOLDER = "train_out"
CURVE_ROC_FILE_NAME = "curva_roc.png"
RESUME_FILE_NAME = "resume.txt"
FILE_TRUE_TO_GET_PER_PATIENT = "results.txt"


def init_session_folders(iden_session, svm_test_name, decision_net_folder):
    path_to_main_session = os.path.join(set.path_to_general_out_folder,
                                        iden_session)
    # CV main session folder, necessary to extract labels per index
    path_to_cv_folder = os.path.join(path_to_main_session,
                                     set.main_cv_vae_session)
    # SVM main folder
    path_to_svm_test = os.path.join(path_to_main_session,
                                    set.main_test_folder_autoencoder_session,
                                    svm_test_name)
    # SVM output
    path_svm_train_score = os.path.join(path_to_svm_test,
                                        set.svm_folder_name_train_out,
                                        set.svm_file_name_scores_file_name)
    # SVM output
    path_svm_test_score = os.path.join(path_to_svm_test,
                                       set.svm_folder_name_test_out,
                                       set.svm_file_name_scores_file_name)

    path_decision_session_folder = os.path.join(path_to_svm_test,
                                                decision_net_folder)

    path_to_post_train = os.path.join(path_decision_session_folder,
                                      "post_train")
    path_to_meta = os.path.join(path_decision_session_folder, "meta")

    # Folder where store the data generated in this test, such as roc curve
    # accuracy
    path_to_train_out = os.path.join(path_to_post_train, TRAIN_OUTPUT_FOLDER)
    path_to_test_out = os.path.join(path_to_post_train, TEST_OUTPUT_FOLDER)

    create_directories([path_to_post_train,
                        path_to_train_out,
                        path_to_test_out])

    return (path_decision_session_folder, path_svm_train_score,
            path_svm_test_score, path_to_cv_folder, path_to_meta,
            path_to_train_out, path_to_test_out)


idi_session = "05_05_2017_08:19 arch: 1000_800_500_100"
svm_test_name = "svm"
iter_to_meta_load = "900"
decision_net_name = "neural_net-2017_05_06-18:35"

path_decision_session_folder, path_svm_train_score, path_svm_test_score, \
path_to_cv_folder, path_to_meta, path_to_train_out, path_to_test_out = \
    init_session_folders(idi_session, svm_test_name, decision_net_name)

# LOADING DATA
X_train = load_svm_output_score(path_svm_train_score)['data_normalize']
X_test = load_svm_output_score(path_svm_test_score)['data_normalize']
Y_train, Y_test = get_label_per_patient(path_to_cv_folder)

tf.reset_default_graph()
sess = tf.Session()
metafile = os.path.join(path_to_meta, "-{}.meta".format(iter_to_meta_load))
savefile = os.path.join(path_to_meta, "-{}".format(iter_to_meta_load))
new_saver = tf.train.import_meta_graph(metafile)
new_saver.restore(sess, savefile)

v = DecisionNeuralNet(root_path=path_decision_session_folder,
                      meta_graph=savefile)

score_train = v.forward_propagation(X_train)[0]
score_test = v.forward_propagation(X_test)[0]

# Train evaluation
threshold = evaluation_output(os.path.join(path_to_train_out, RESUME_FILE_NAME),
                              os.path.join(path_to_train_out,
                                           CURVE_ROC_FILE_NAME),
                              os.path.join(path_to_train_out,
                                           FILE_TRUE_TO_GET_PER_PATIENT),
                              score_train, Y_train)

# Test evaluation
evaluation_output(os.path.join(path_to_test_out, RESUME_FILE_NAME),
                  os.path.join(path_to_test_out, CURVE_ROC_FILE_NAME),
                  os.path.join(path_to_test_out, FILE_TRUE_TO_GET_PER_PATIENT),
                  score_test, Y_test, thresholds_establised=threshold)

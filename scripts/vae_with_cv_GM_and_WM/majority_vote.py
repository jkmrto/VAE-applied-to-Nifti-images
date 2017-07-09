import os
from datetime import datetime

import numpy as np
from lib.evaluation_utils import evaluation_output
from lib.svm_utils import load_svm_output_score

import settings as set
from lib.aux_functionalities import functions
from lib.aux_functionalities.os_aux import create_directories
from lib.utils.cv_utils import get_label_per_patient
from scripts.vae_with_cv_GM_and_WM import session_settings as main_settings

TYPE_SESSION_DECISION = "majority_vote"
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
from scripts.vae_with_cv_GM_and_WM import svm_session_settings as svm_settings


def init_session_folders(svm_session_folder_name, kind_svm_session="combined"):
    path_to_svm_session = os.path.join(main_settings.path_post_encoding_folder,
                                       svm_session_folder_name)

    path_to_root = os.path.join(path_to_svm_session,
                                svm_settings.folder_post_svm)

    own_datetime = datetime.now().strftime(r"%d_%m_%_Y_%H:%M")
    iden_session = TYPE_SESSION_DECISION + "_" + own_datetime
    path_decision_session_folder = os.path.join(path_to_root, iden_session)
    # Till here we have the folder to the session

    path_file_train_score = ""
    path_file_test_score = ""
    if kind_svm_session == "combined":
        path_file_train_score = os.path.join(path_to_svm_session,
            svm_settings.folder_combined_wm_and_gm_code_data_as_input_to_svm,
            set.svm_folder_name_train_out, set.svm_file_name_scores_file_name)

        path_file_test_score = os.path.join(path_to_svm_session,
            svm_settings.folder_combined_wm_and_gm_code_data_as_input_to_svm,
            set.svm_folder_name_test_out, set.svm_file_name_scores_file_name)

    path_to_train_out = os.path.join(path_decision_session_folder,
                                     TRAIN_OUTPUT_FOLDER)
    path_to_test_out = os.path.join(path_decision_session_folder,
                                    TEST_OUTPUT_FOLDER)


    create_directories([path_to_root,
                        path_decision_session_folder,
                        path_to_train_out,
                        path_to_test_out])

    return (path_file_train_score, path_file_test_score,
            path_to_train_out, path_to_test_out)


# SVM Session configuration

svm_session_folder_name = "bueno_svm_08_05_2017_20:02"

# Loading folders
path_file_train_score, path_file_test_score, \
path_to_train_out, path_to_test_out = \
    init_session_folders(svm_session_folder_name, kind_svm_session="combined")

X_train = load_svm_output_score(path_file_train_score)['raw']
X_test = load_svm_output_score(path_file_test_score)['raw']
Y_train, Y_test = get_label_per_patient(main_settings.path_cv_folder)

means_train = np.row_stack(X_train.mean(axis=1))
means_test = np.row_stack(X_test.mean(axis=1))

threshold = evaluation_output(os.path.join(path_to_train_out, RESUME_FILE_NAME),
                              os.path.join(path_to_train_out,
                                           CURVE_ROC_FILE_NAME),
                              os.path.join(path_to_train_out,
                                           FILE_TRUE_TO_GET_PER_PATIENT),
                              means_train, Y_train)

evaluation_output(os.path.join(path_to_test_out, RESUME_FILE_NAME),
                  os.path.join(path_to_test_out, CURVE_ROC_FILE_NAME),
                  os.path.join(path_to_test_out, FILE_TRUE_TO_GET_PER_PATIENT),
                  means_test, Y_test, thresholds_establised=threshold)
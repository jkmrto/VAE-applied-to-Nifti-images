import os
import numpy as np
from lib import session_helper as session
from datetime import datetime
from lib import svm_utils
from lib import cv_utils
from lib.aux_functionalities.os_aux import create_directories
from lib.aux_functionalities.os_aux import create_directories
from scripts.vae_with_cv_GM_and_WM import session_settings
from lib import session_helper as session
from scripts.vae_with_cv_GM_and_WM import session_settings
from lib.mri import stack_NORAD

folder_combined_wm_and_gm_code_data_as_input_to_svm = \
    "output_combining_wm_and_gm_code_data_as_input_to_svm"

folder_one_svm_for_gm_and_other_to_wm = \
    "output_no_combining_gm_wm,_evaluating_separately"

folder_gm_single_output = "gm_output"

folder_wm_single_output = "wm_output"

per_reg_acc = "per_reg_acc.log"

scores = "scores.log"


def generate_log_files_for_svm_execution(path_to_storage_folder):
    path_to_train_results_folder = os.path.join(path_to_storage_folder,
                                                "train_out")
    path_to_test_results_folder = os.path.join(path_to_storage_folder,
                                               "test_out")
    create_directories([path_to_train_results_folder,
                        path_to_test_results_folder])

    path_to_train_scores = os.path.join(path_to_train_results_folder,
                                        "scores.log")
    path_to_test_scores = os.path.join(path_to_test_results_folder,
                                       "scores.log")

    path_to_train_per_reg_acc = os.path.join(path_to_train_results_folder,
                                             per_reg_acc)
    path_to_test_per_reg_acc = os.path.join(path_to_test_results_folder,
                                            per_reg_acc)
    log_hub = {}
    log_hub['train'] = {}
    log_hub["test"] = {}

    log_hub["train"]["scores"] = open(path_to_train_scores, "w")
    log_hub["test"]["scores"] = open(path_to_test_scores, "w")
    log_hub["train"]["per_reg_acc"] = open(path_to_train_per_reg_acc, "w")
    log_hub["test"]["per_reg_acc"] = open(path_to_test_per_reg_acc, "w")

    return log_hub


def execute_svm_over_data(train_data, test_data, train_Y, test_Y, log_hub,
                          reg_select):

    train_score, test_score = svm_utils.fit_svm_and_get_decision_for_requiered_data(
        train_data, train_Y, test_data)

    svm_utils.per_region_evaluation(train_score, train_Y,
                                    log_hub["train"]["per_reg_acc"], reg_select)
    svm_utils.per_region_evaluation(test_score, test_Y,
                                    log_hub["test"]["per_reg_acc"], reg_select)

    svm_utils.log_scores(train_score, log_hub["train"]["scores"], reg_select)
    svm_utils.log_scores(test_score, log_hub["test"]["scores"], reg_select)


def close_logs_file_per_svm_execution(log_hub):
    # train logs files
    log_hub["train"]["scores"].close()
    log_hub["train"]["per_reg_acc"].close()

    # test log files
    log_hub["test"]["scores"].close()
    log_hub["test"]["per_reg_acc"].close()


def init_svm_session_folders(path_to_post_encoding_folder, GM_session_id,
                             WM_session_id):
    # SVM session main folder
    own_datetime = datetime.now().strftime(r"%d_%m_%_Y_%H:%M")
    iden_session = "svm_" + own_datetime
    path_svm_session_folder = os.path.join(path_to_post_encoding_folder,
                                           iden_session)

    # Output folders
    path_to_combined_output = os.path.join(path_svm_session_folder,
                                           folder_combined_wm_and_gm_code_data_as_input_to_svm)

    path_to_separate_output = os.path.join(path_svm_session_folder,
                                           folder_one_svm_for_gm_and_other_to_wm)

    path_to_separate_gm_output = os.path.join(path_to_separate_output,
                                              folder_gm_single_output)
    path_to_separata_wm_output = os.path.join(path_to_separate_output,
                                              folder_wm_single_output)

    # Path towards the data to be loaded
    path_GM_session_encoding = os.path.join(session_settings.path_GM_folder,
                                            GM_session_id,
                                            session.folder_encoding_out)
    path_WM_session_encoding = os.path.join(session_settings.path_WM_folder,
                                            WM_session_id,
                                            session.folder_encoding_out)

    create_directories([path_svm_session_folder, path_to_combined_output,
                        path_to_separate_output,
                        path_to_separata_wm_output, path_to_separate_gm_output])

    return (
        path_GM_session_encoding, path_WM_session_encoding,
        path_to_combined_output,
        path_to_separata_wm_output, path_to_separate_gm_output)


path_to_post_encoding_folder = session_settings.path_post_encoding_folder
session_vae_GM = "bueno_08_05_2017_08:06 arch: 1000_800_500_100"
session_vae_WM = "08_05_2017_10:49 arch: 1000_800_500_100"


path_GM_session_encoding, path_WM_session_encoding, path_to_combined_output, \
path_to_separata_wm_output, path_to_separate_gm_output = init_svm_session_folders(
    path_to_post_encoding_folder, session_vae_GM,
    session_vae_WM)


# Process for the combining output
log_hub_combined = generate_log_files_for_svm_execution(
    path_to_combined_output)
log_hub_wm = generate_log_files_for_svm_execution(
    path_to_separata_wm_output)
log_hub_gm = generate_log_files_for_svm_execution(
    path_to_separate_gm_output)

# Loading labels
Y_train, Y_test = cv_utils.get_label_per_patient(
    session_settings.path_cv_folder)

list_regions = session.select_regions_to_evaluate("all")
for region_selected in list_regions:
    # Init svm session
    print("Region {} selected".format(region_selected))
    # Loading data
    wm_test_out, wm_train_out = session.load_out_encoding_per_region(
        path_WM_session_encoding, region_selected)
    gm_test_out, gm_train_out = session.load_out_encoding_per_region(
        path_GM_session_encoding, region_selected)

    # Extracting the mean as main data
    wm_train_data = wm_train_out['means']
    wm_test_data = wm_test_out['means']

    gm_test_data = gm_test_out['means']
    gm_train_data = gm_train_out['means']

    wm_and_gm_train_data = np.concatenate((wm_train_data, gm_train_data), axis=1)
    wm_and_gm_test_data = np.concatenate((wm_test_data, gm_test_data), axis=1)


    print("Executing SVM per grey and whiter matter combined")
    execute_svm_over_data(wm_and_gm_train_data, wm_and_gm_test_data, Y_train,
                          Y_test, log_hub_combined, region_selected)

    print("Executing SVM per white matter")
    execute_svm_over_data(wm_train_data, wm_test_data, Y_train, Y_test,
                          log_hub_wm, region_selected)

    print("Executing SVM per grey matter")
    execute_svm_over_data(gm_train_data, gm_test_data, Y_train, Y_test,
                          log_hub_gm, region_selected)

    print("Region {} process ended".format(region_selected))

close_logs_file_per_svm_execution(log_hub_combined)
close_logs_file_per_svm_execution(log_hub_wm)
close_logs_file_per_svm_execution(log_hub_gm)

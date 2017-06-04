import tensorflow as tf
from lib import svm_utils
from lib.aux_functionalities.os_aux import create_directories
import os
from lib.data_loader import mri_atlas
import settings
from lib.vae import VAE
from lib.data_loader import MRI_stack_NORAD
from sklearn.model_selection import train_test_split

from lib import utils
from lib.aux_functionalities import functions

# SVM CONFIGURATION
# iden_session = "02_05_2017_21:09 arch: 1000_800_500_100_2"
# iden_session = "03_05_2017_08:12 arch: 1000_800_500_100"
iden_session = "05_05_2017_08:19 arch: 1000_800_500_100"

test_name = "svm"
regions_used = "all"
#regions_used = "three"
iter_to_meta_load = 1500

# PATH INITIALIZATION
path_to_session = os.path.join(settings.path_to_general_out_folder,
                               iden_session)
path_to_cv = os.path.join(path_to_session, "cv")
path_to_meta_folder = os.path.join(path_to_session, "meta")
path_to_main_test = os.path.join(path_to_session, "post_train")
path_to_particular_test = os.path.join(path_to_main_test, test_name)

path_to_train_results_folder = os.path.join(path_to_particular_test,
                                            "train_out")
path_to_test_results_folder = os.path.join(path_to_particular_test, "test_out")

create_directories([path_to_main_test, path_to_particular_test,
                    path_to_train_results_folder, path_to_test_results_folder])

path_to_train_scores = os.path.join(path_to_train_results_folder,
                                    "scores.log")
path_to_test_scores = os.path.join(path_to_test_results_folder,
                                   "scores.log")

path_to_train_per_reg_acc = os.path.join(path_to_train_results_folder,
                                         "per_reg_acc.log")
path_to_test_per_reg_acc = os.path.join(path_to_test_results_folder,
                                        "per_reg_acc.log")

train_scores_file = open(path_to_train_scores, "w")
test_scores_file = open(path_to_test_scores, "w")
train_per_reg_acc = open(path_to_train_per_reg_acc, "w")
test_per_reg_acc = open(path_to_test_per_reg_acc, "w")


# LOADING INDEX TO CV
train_index, test_index = svm_utils.get_train_and_test_index_from_files(path_to_cv)


# LOADING THE DATA
dict_norad = MRI_stack_NORAD.get_gm_stack()  # 'stack' 'voxel_index' 'labels'
# Loading Labels
region_voxels_label = dict_norad['labels']
Y_train = region_voxels_label[train_index]
Y_test = region_voxels_label[test_index]
# Loading data, values per voxels
stack = dict_norad['stack']
stack_train = stack[train_index, :]
stack_test = stack[test_index, :]

# SELECTING REGIONS TO BE EVALUATED
list_regions = region_selector_hub.select_regions_to_evaluate(regions_used)

# LOOP OVER REGIONS
for reg_select in list_regions:
    region_voxels_index = mri_atlas.load_atlas_mri()[reg_select]
    region_voxels_values_train = stack_train[:, region_voxels_index]
    region_voxels_values_test = stack_test[:, region_voxels_index]
    X_train, max_denormalize_train = utils.normalize_array(
        region_voxels_values_train)
    X_test, max_denormalize_test = utils.normalize_array(
        region_voxels_values_test)

    suffix = 'region_' + str(reg_select)
    savefile = os.path.join(path_to_meta_folder,
                            suffix + "-{}".format(iter_to_meta_load))
    metafile = os.path.join(path_to_meta_folder,
                            suffix + "-{}.meta".format(iter_to_meta_load))
    print("Loading the file {}".format(metafile))

    tf.reset_default_graph()
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph(metafile)
    new_saver.restore(sess, savefile)

    # Hyperparameters and architecture is not used in loading setup
    v = VAE.VAE(meta_graph=savefile)

    print("Coding training data")
    code_train = v.encode(X_train)  # [mu, sigma]
    code_test = v.encode(X_test)  # [mu, sigma]

    score_train, score_test = svm_utils.fit_svm_and_get_decision_for_requiered_data(
        code_train[0], Y_train, code_test[0])

    svm_utils.per_region_evaluation(score_train, Y_train, train_per_reg_acc, reg_select)
    svm_utils.per_region_evaluation(score_test, Y_test, test_per_reg_acc, reg_select)

    svm_utils.log_scores(score_train, train_scores_file, reg_select)
    svm_utils.log_scores(score_test, test_scores_file, reg_select)

train_scores_file.close()
test_scores_file.close()
train_per_reg_acc.close()
test_per_reg_acc.close()

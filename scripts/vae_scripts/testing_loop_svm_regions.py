import os

import tensorflow as tf

import settings
from lib import session_helper
from lib import utils
from lib.aux_functionalities.os_aux import create_directories
from lib.data_loader import MRI_stack_NORAD
from lib.data_loader import mri_atlas
from lib.utils import svm_utils
from lib.vae import VAE

# SVM CONFIGURATION
# iden_session = "02_05_2017_21:09 arch: 1000_800_500_100_2"
iden_session = "03_05_2017_08:12 arch: 1000_800_500_100"
test_name = "svm"
regions_used = "all"
#regions_used = "most important"
iter_to_meta_load = 5000

path_to_session = os.path.join(settings.path_to_general_out_folder,
                               iden_session)
path_to_meta_folder = os.path.join(path_to_session, "meta")
path_to_main_test = os.path.join(path_to_session, "post_train")
path_to_particular_test = os.path.join(path_to_main_test, test_name)
create_directories([path_to_main_test, path_to_particular_test])

score_file = open(path_to_particular_test + "/patient_score_per_region.log",
                  "w")
per_region_accuracy_file = open(os.path.join(path_to_particular_test,
                                             "per_region_accuracy.log"), "w")

# LOADING DATA
dict_norad = MRI_stack_NORAD.get_gm_stack()  # 'stack' 'voxel_index' 'labels'
atlas_mri = mri_atlas.load_atlas_mri()
region_voxels_label = dict_norad['labels']
Y_train = region_voxels_label

# SELECTING REGIONS TO BE EVALUATED
list_regions = session_helper.select_regions_to_evaluate(regions_used)

for region_selected in list_regions:
    region_voxels_index = atlas_mri[region_selected]
    region_voxels_values = dict_norad['stack'][:, region_voxels_index]
    region_voxels_values, max_denormalize = utils.normalize_array(
        region_voxels_values)

    X_train = region_voxels_values

    suffix = 'region_' + str(region_selected)
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

    # Fitting SVM
    score, _ = svm_utils.fit_svm_and_get_decision_for_requiered_data(
        code_train[0], Y_train, code_train[0])

    svm_utils.per_region_evaluation(score, Y_train, per_region_accuracy_file,
                                    region_selected)

    svm_utils.log_scores(score, score_file, region_selected)

score_file.close()
per_region_accuracy_file.close()

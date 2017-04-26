import tensorflow as tf
from sklearn import svm
from lib.aux_functionalities.os_aux import create_directories
import os
from sklearn.metrics import confusion_matrix
from lib.mri import mri_atlas
import settings
from lib.vae import VAE
from lib.mri import stack_NORAD
from sklearn.model_selection import train_test_split
from sklearn import metrics
from lib import utils
from lib.aux_functionalities import os_aux

# SVM CONFIGURATION
architecture = [1000, 800, 500, 100]
test_name = "second_test"
regions_used = "all"

HYPERPARAMS = {
    "batch_size": 128,
    "learning_rate": 5E-4,
    "dropout": 0.9,
    "lambda_l2_reg": 1E-5,
    "nonlinearity": tf.nn.elu,
    "squashing": tf.nn.sigmoid
}


dict_norad = stack_NORAD.get_gm_stack()  # 'stack' 'voxel_index' 'labels'

iden_session = "25_04_2017_20:51 arch: 1000_800_500_100"
path_to_session = os.path.join(settings.path_to_general_out_folder, iden_session)
path_to_meta_folder = os.path.join(path_to_session, "meta")
path_to_main_test = os.path.join(path_to_session, "post_train")
path_to_particular_test = os.path.join(path_to_main_test, test_name)
create_directories([path_to_main_test, path_to_particular_test])

score_file = open(path_to_particular_test + "/patient_score_per_region.log", "w")
labels_file = open(path_to_particular_test + "/patient_labels_per_region.log", "w") # Currently unused


list_regions = []
if regions_used == "all":
    list_regions = range(1, 117, 1) # 117 regions en total
elif regions_used == "most important":
    list_regions = settings.list_regions_evaluated

for region_selected in list_regions:

    region_voxels_index = mri_atlas.load_atlas_mri()[region_selected]
    region_voxels_values = dict_norad['stack'][:, region_voxels_index]
    region_voxels_values, max_denormalize = utils.normalize_array(region_voxels_values)
    region_voxels_label = dict_norad['labels']

    X_train = region_voxels_values
    Y_train = region_voxels_label

    suffix = 'region_' + str(region_selected)
    savefile = os.path.join(path_to_meta_folder, suffix + "-1500")
    metafile = os.path.join(path_to_meta_folder, suffix + "-1500.meta")
    print("Loading the file {}".format(metafile))

    tf.reset_default_graph()
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph(metafile)
    new_saver.restore(sess, savefile)

    # Hyperparameters and architecture is not used in loading setup
    v = VAE.VAE(architecture, HYPERPARAMS, meta_graph=savefile)

    print("Coding training data")
    code_train = v.encode(X_train)  # [mu, sigma]
    print("Coding test data")
    code_test = v.encode(X_train)  # [mu, sigma]

    # Fitting SVM
    print("Training SVM")
    clf = svm.SVC(decision_function_shape='ovr', kernel='linear')
    clf.fit(code_train[0], Y_train)    # Solo usando las media pq el codificador devuelve media y desviacion

    # Testing time
    print("Evaluating test samples")
    dec = clf.decision_function(code_train[0])

    print(len(dec))

    score_file.write("region_{0}".format(region_selected))

    for out in dec:
        score_file.write(",{}".format(out))
    score_file.write("\n")






# X_train, X_test, y_train, y_test = train_test_split(region_voxels_values, region_voxels_label,
#                                                   test_size=0.0, random_state=0)
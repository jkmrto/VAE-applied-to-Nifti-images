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
iden_session = "02_05_2017_21:09 arch: 1000_800_500_100_2"
#iden_session = "03_05_2017_08:12 arch: 1000_800_500_100"
test_name = "test_without_pre_records"
#regions_used = "all"
regions_used = "most important"
iter_to_meta_load = 1000

dict_norad = stack_NORAD.get_gm_stack()  # 'stack' 'voxel_index' 'labels'

path_to_session = os.path.join(settings.path_to_general_out_folder, iden_session)
path_to_meta_folder = os.path.join(path_to_session, "meta")
path_to_main_test = os.path.join(path_to_session, "post_train")
path_to_particular_test = os.path.join(path_to_main_test, test_name)
create_directories([path_to_main_test, path_to_particular_test])

score_file = open(path_to_particular_test + "/patient_score_per_region.log", "w")
labels_file = open(path_to_particular_test + "/patient_labels_per_region.log", "w") # Currently unused
per_region_accuracy_file = open(os.path.join(path_to_particular_test,
                                "per_region_accuracy.log"), "w")

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
    savefile = os.path.join(path_to_meta_folder, suffix + "-{}".format(iter_to_meta_load))
    metafile = os.path.join(path_to_meta_folder, suffix + "-{}.meta".format(iter_to_meta_load))
    print("Loading the file {}".format(metafile))

    tf.reset_default_graph()
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph(metafile)
    new_saver.restore(sess, savefile)

    # Hyperparameters and architecture is not used in loading setup
    v = VAE.VAE( meta_graph=savefile)

    print("Coding training data")
    code_train = v.encode(X_train)  # [mu, sigma]
 #   print("Coding test data")
 #   code_test = v.encode(X_train)  # [mu, sigma]

    # Fitting SVM
    print("Training SVM")
    clf = svm.SVC(decision_function_shape='ovr', kernel='linear')
    clf.fit(code_train[0], Y_train)    # Solo usando las media pq el codificador devuelve media y desviacion

    # Testing time
    print("Evaluating test samples")
    dec = clf.decision_function(code_train[0])

    # Post scritp to print accuracy per region
    dec_normalize = dec
    dec_normalize[dec_normalize < 0] = 0
    dec_normalize[dec_normalize > 0] = 1

    region_accuracy = metrics.accuracy_score(Y_train, dec_normalize)
    per_region_accuracy_file.write("region_{0},{1}\n".format(region_selected,
                                                           region_accuracy))
    per_region_accuracy_file.flush()

    score_file.write("region_{0}".format(region_selected))

    for out in dec:
        score_file.write(",{}".format(out))
    score_file.write("\n")

# X_train, X_test, y_train, y_test = train_test_split(region_voxels_values, region_voxels_label,
#                                                   test_size=0.0, random_state=0)
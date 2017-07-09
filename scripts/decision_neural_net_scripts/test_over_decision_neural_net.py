import os

import numpy as np
import tensorflow as tf

import settings
from lib.aux_functionalities.os_aux import create_directories
from lib.neural_net.decision_neural_net import DecisionNeuralNet
from lib.utils.evaluation_utils import evaluation_output

iter = 300000
iden_session = "03_05_2017_08:12 arch: 1000_800_500_100"
root_session_folder_name = "neural_net-2017_05_03-22:09-42_21_1"
test_name = "svm"

output_project_folder = "out"
main_test_folder_autoencoder_session = "post_train"
#test_name = "test_svm_important"

score_file_name = "patient_score_per_region.log"
cv_session_folder_name = "cv_data"


path_to_session_folder = os.path.join(settings.path_to_project,
                                      output_project_folder,
                                      iden_session,
                                      main_test_folder_autoencoder_session,
                                      test_name,
                                      root_session_folder_name)
path_to_cv = os.path.join(path_to_session_folder, cv_session_folder_name)
path_to_test_labels = os.path.join(path_to_cv, "y_test.csv")
path_to_test_data = os.path.join(path_to_cv, "X_test.csv") # Data already normalized
X_test = np.genfromtxt(path_to_test_data, delimiter=',')
y_test = np.genfromtxt(path_to_test_labels, delimiter=',')
y_test = np.reshape(y_test, (y_test.shape[0], 1))

path_to_meta_files = os.path.join(path_to_session_folder, "meta")
path_to_main_results_output = os.path.join(path_to_session_folder, "post_train")
path_to_own_results_output = os.path.join(path_to_main_results_output, "TestUsingTestData")
path_to_results_file = os.path.join(path_to_own_results_output, "results.csv")
path_to_roc_png = os.path.join(path_to_own_results_output, "curva_roc.png")
path_to_resume_file = os.path.join(path_to_own_results_output, "resume.txt")
create_directories([path_to_main_results_output, path_to_own_results_output])

tf.reset_default_graph()
sess = tf.Session()
metafile = os.path.join(path_to_meta_files, "-{}.meta".format(iter))
savefile = os.path.join(path_to_meta_files, "-{}".format(iter))
new_saver = tf.train.import_meta_graph(metafile)
new_saver.restore(sess, savefile)

v = DecisionNeuralNet(root_path=path_to_session_folder,
                      meta_graph=savefile)
# Taking the first element of the list returned
y_obtained = v.forward_propagation(X_test)[0]


evaluation_output(path_to_resume_file,
                  path_to_roc_png,
                  path_to_results_file,
                  y_obtained, y_test)
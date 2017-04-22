import os
import settings
import numpy as np
from lib.mri.stack_NORAD import load_patients_labels
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from lib.mri.stack_NORAD import load_patients_labels
from scripts.svm_score_aux_functions import load_svm_output_score
from lib.aux_functionalities.os_aux import create_directories


def make_binary(data):
    data[data <= 0] = 0
    data[data > 0] = 1

    return data

id_autoencoder_session = "01_04_2017_00:00 arch: 1000_800_500_100"
output_project_folder = "out"
main_test_folder_autoencoder_session = "post_train"
svm_test_name = "first_test"
score_file_name = "patient_score_per_region.log"
path_to_svm_test = os.path.join(settings.path_to_project,
                                output_project_folder,
                                id_autoencoder_session,
                                main_test_folder_autoencoder_session,
                                svm_test_name)
path_score_file = os.path.join(path_to_svm_test, score_file_name)

majority_vote_session_folder_name = "majority_vote"
path_to_majority_vote_session = os.path.join(path_to_svm_test,
                                             majority_vote_session_folder_name)
create_directories( [path_to_majority_vote_session])
path_to_roc_png = os.path.join(path_to_majority_vote_session,
                               "curva_roc.png")

data = load_svm_output_score(path_score_file)['raw']

#data = make_binary(data)
means = data.mean(axis=1)
#means_binary = means.round()
patients_labels = load_patients_labels()  # 417x1

[fpr, tpr, thresholds] = roc_curve(patients_labels, means)

plt.plot(fpr, tpr, linestyle='--')
plt.savefig(path_to_roc_png)


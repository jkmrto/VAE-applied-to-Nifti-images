import os
import settings
import numpy as np
from lib.mri.stack_NORAD import load_patients_labels
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from lib.mri.stack_NORAD import load_patients_labels
from scripts.svm_score_aux_functions import load_svm_output_score
from lib.aux_functionalities.os_aux import create_directories
from lib.aux_functionalities.functions import print_dictionary


def make_binary(data):
    data[data <= 0] = 0
    data[data > 0] = 1

    return data

idi_session = "03_05_2017_08:12 arch: 1000_800_500_100"
output_project_folder = "out"
main_test_folder_autoencoder_session = "post_train"
svm_folder_name = "svm"
score_file_name = "patient_score_per_region.log"
majority_vote_session_folder_name = "majority_vote"

path_to_svm_test = os.path.join(settings.path_to_project,
                                output_project_folder,
                                idi_session,
                                main_test_folder_autoencoder_session,
                                svm_folder_name)
path_score_file = os.path.join(path_to_svm_test, score_file_name)


path_to_majority_vote_session = os.path.join(path_to_svm_test,
                                             majority_vote_session_folder_name)
create_directories( [path_to_majority_vote_session])
path_to_roc_png = os.path.join(path_to_majority_vote_session, "curva_roc.png")
path_to_resume_file = os.path.join(path_to_majority_vote_session, "resume.txt")

data = load_svm_output_score(path_score_file)['raw']

#data = make_binary(data)
means = data.mean(axis=1)
#means_binary = means.round()
patients_labels = load_patients_labels()  # 417x1

precision = metrics.average_precision_score(patients_labels, means)
auc = metrics.roc_auc_score(patients_labels, means)
output_dic = {"precision": precision,
              "area under the curve": auc}
print_dictionary(path_to_resume_file, output_dic)

[fpr, tpr, thresholds] = roc_curve(patients_labels, means)

plt.figure()
plt.plot(fpr, tpr, linestyle='--')
plt.savefig(path_to_roc_png)


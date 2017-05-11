import os
import settings
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from lib.mri.stack_NORAD import load_patients_labels
from lib.svm_utils import load_svm_output_score
from lib.aux_functionalities.os_aux import create_directories
from lib.aux_functionalities.functions import print_dictionary
from lib.evaluation_utils import evaluation_output


def make_binary(data):
    data[data <= 0] = 0
    data[data > 0] = 1

    return data

TEST_OUTPUT_FOLDER = "test_out"
TRAIN_OUTPUT_FOLDER = "train_out"


idi_session = "05_05_2017_08:19 arch: 1000_800_500_100"
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

X_train = load_svm_output_score(path_file_train_score)['raw']
X_test = load_svm_output_score(path_file_test_score)['raw']
Y_train, Y_test = get_label_per_patient(path_to_cv_folder)

path_to_train_out = os.path.join(path_to_post_train, TRAIN_OUTPUT_FOLDER)
path_to_test_out = os.path.join(path_to_post_train, TEST_OUTPUT_FOLDER)


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


evaluation_output(os.path.join(path_to_train_out, RESUME_FILE_NAME),
                  os.path.join(path_to_train_out, CURVE_ROC_FILE_NAME),
                  os.path.join(path_to_train_out, FILE_TRUE_TO_GET_PER_PATIENT),
                  score_train, Y_train)

evaluation_output(os.path.join(path_to_test_out, RESUME_FILE_NAME),
                  os.path.join(path_to_test_out, CURVE_ROC_FILE_NAME),
                  os.path.join(path_to_test_out, FILE_TRUE_TO_GET_PER_PATIENT),
                  score_test, Y_test)

import os
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from numpy import linspace
from lib.utils import os_aux
from lib import file_reader
import settings


def plot_evaluation_parameters(list_parameters_dict, evaluation_method):

    list_rows = list_parameters_dict
    kernel_size_array = np.array([float(row['kernel_size']) for row in list_rows])
    f1_score_array = np.array([float(row['f1_score']) for row in list_rows])
    recall_score_array = np.array([float(row['recall_score']) for row in list_rows])
    accuracy_score_array = np.array([float(row['accuracy']) for row in list_rows])
    auc_score_array = np.array([float(row['area under the curve']) for row in list_rows])
    precision = np.array([float(row['precision']) for row in list_rows])

    string_ref = "{0} swap over {1}. {2} method".format(
        images_used, swap_variable, evaluation_method)
    plt.figure()
    plt.title(string_ref)
    plt.plot(kernel_size_array, f1_score_array, label="f1_score")
    plt.plot(kernel_size_array, recall_score_array, label="recall_score")
    plt.plot(kernel_size_array, accuracy_score_array, label="accuracy")
    plt.plot(kernel_size_array, auc_score_array, label="area under curve")
    plt.plot(kernel_size_array, precision, label="precision")
    plt.ylabel("% results")
    plt.xlabel("Kernel Size")
    plt.legend()
    plt.savefig(os.path.join(path_evaluation_images_folder, "{}.png".format(string_ref)))





start = 0.0
stop = 1.0
number_of_lines = 1000

images_used = "PET"
swap_variable = "kernel"

cm_subsection = linspace(start, stop, 9)
colors = [cm.jet(x) for x in cm_subsection]


#out_folder = "CVAE_session_swap_kernel_MRI"
out_folder = "CVAE_PET_session_swap_kernel_PET"
output_weighted_svm = "loop_output_weighted_svm.csv"
output_simple_majority_vote = "loop_output_simple_majority_vote.csv"
output_complex_majority_vote= "loop_output_complex_majority_vote.csv"

evaluation_images_folder_name = "evaluation_images"

# Paths references
path_to_folder = os.path.join(settings.path_to_general_out_folder, out_folder)
path_evaluation_images_folder = os.path.join(path_to_folder, evaluation_images_folder_name)
path_output_weighted_svm = os.path.join(path_to_folder, output_weighted_svm)
path_output_simple_majority_vote = os.path.join(path_to_folder, output_simple_majority_vote)
path_output_complex_majority_vote = os.path.join(path_to_folder, output_complex_majority_vote)

#
os_aux.create_directories([path_evaluation_images_folder])


list_SVM = file_reader.read_csv_as_list_of_dictionaries(
    path_output_weighted_svm)
list_SMV = file_reader.read_csv_as_list_of_dictionaries(
    path_output_simple_majority_vote)
list_CMV = file_reader.read_csv_as_list_of_dictionaries(
    path_output_complex_majority_vote)


plot_evaluation_parameters(list_parameters_dict=list_SVM,
                           evaluation_method="SVM")
plot_evaluation_parameters(list_parameters_dict=list_SMV,
                           evaluation_method="SMV")
plot_evaluation_parameters(list_parameters_dict=list_CMV,
                           evaluation_method="CMV")


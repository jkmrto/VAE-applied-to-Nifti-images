import os
import settings
import csv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from numpy import linspace
from lib.aux_functionalities.os_aux import create_directories as create
import copy

start = 0.0
stop = 1.0
number_of_lines = 1000
cm_subsection = linspace(start, stop, 9)
colors = [cm.jet(x) for x in cm_subsection]

out_folder = "CVAE_session_swap_kernel_MRI"
roc_logs_filename = "roc.logs"

path_to_folder = os.path.join(settings.path_to_general_out_folder,
                              out_folder)
create([path_to_folder])
path_roc_logs = os.path.join(path_to_folder, roc_logs_filename)

roc_log_file = open(path_roc_logs)
reader = csv.DictReader(roc_log_file, delimiter=";")

keys_used = [' evaluation', 'threshold ', ' false_positive_rate',
             ' test|train', ' true_positive_rate', 'kernel_size', ' fold']

# false positive rate -> X axis
# true positive rate -> Y axis
# kernel size, SWAP term
n_folds = 10
evaluation_type = ["SVM_weighted", "Complex_Majority_Vote",
                   "Simple_Majority_Vote"]
execution_type = ["test", "train"]
kernel_size = [2, 3, 4, 5, 6, 7, 8, 9, 10]
fold_selected = 5

list_rows = []
for row in reader:
    list_rows.append(row)

container_dict_by_evaluation_method = {
    "SVM_weighted": {},
    "Complex_Majority_Vote": {},
    "Simple_Majority_Vote": {},
}

intent = {' true_positive_rate',
          ' false_positive_rate',
          'threshold '}

folds_container = {}

for i in range(0, n_folds, 1):
    folds_container[i] = copy.deepcopy(intent)

kernel_container = {}

for kernel in kernel_size:
    kernel_container[kernel] = copy.deepcopy(folds_container)

for key in container_dict_by_evaluation_method.keys():
    container_dict_by_evaluation_method[key]["test"] = copy.deepcopy(
        kernel_container)
    container_dict_by_evaluation_method[key]["train"] = copy.deepcopy(
        kernel_container)

for row in list_rows:
    intent_dict = {
        'true_positive_rate': row[' true_positive_rate'],
        'false_positive_rate': row[' false_positive_rate'],
        'threshold': row['threshold ']
    }

    container_dict_by_evaluation_method[
        row[" evaluation"]][row[" test|train"]][int(row["kernel_size"])][
        int(row[" fold"])] = intent_dict

for eval_key, container_evaluation in container_dict_by_evaluation_method.items():
    print("hola")
    for exec_key, container_execution in container_evaluation.items():
        plt.figure()
        for kernel_key, container_kernel in container_execution.items():
            dict_by_fold = container_kernel[fold_selected]
            fpr = [float(str_number) for str_number in
                   dict_by_fold["false_positive_rate"].split(",")]
            tpr = [float(str_number) for str_number in
                dict_by_fold["true_positive_rate"].split(",")]
            plt.plot(np.array(fpr), np.array(tpr),
                     label="kernel_size:{}".format(kernel_key))

            img_idi = "{0},{1}".format(eval_key,exec_key)

        plt.title(img_idi)
        plt.legend()
        plt.savefig(os.path.join(path_to_folder,
                                     "{}.png".format(img_idi)))

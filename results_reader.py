import os
import settings
import csv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from numpy import linspace


start = 0.0
stop = 1.0
number_of_lines= 1000
cm_subsection = linspace(start, stop, 9)
colors = [cm.jet(x) for x in cm_subsection ]


out_folder = "CVAE_session_swap_kernel_MRI"
roc_logs_filename = "roc.logs"

path_to_folder = os.path.join(settings.path_to_general_out_folder,
                               out_folder)

path_roc_logs = os.path.join(path_to_folder, roc_logs_filename)

roc_log_file = open(path_roc_logs)
reader = csv.DictReader(roc_log_file, delimiter=";")

keys_used = [' evaluation', 'threshold ', ' false_positive_rate',
             ' test|train', ' true_positive_rate', 'kernel_size', ' fold']

# false positive rate -> X axis
# true positive rate -> Y axis
# kernel size, SWAP term

evaluation_type = ["SVM_weighted", "Complex_Majority_Vote", "Simple_Majority_Vote"]
execution_type = ["test", "train"]
kernel_size = [2,3,4,5,6,7,8,9,10]
fold_selected = 0

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
          'threshold '
          ""}

for key in container_dict_by_evaluation_method.keys():
    container_dict_by_evaluation_method[key] = "test"
    container_dict_by_evaluation_method[key] = "train"

for evaluation in evaluation_type:
    for execution in execution_type:
        for kernel in kernel_size:
            selectec_rows = []
            for row in list_rows:
                if row[" evaluation"] == evaluation:
                    if row[" test|train"] == execution:
                        if row[" fold"] == str(fold_selected):
                            plt.figure(int(row["kernel_size"]))
                            if row["kernel_size"] == str(kernel):
                                print("selected")
                                fpr = \
                                    [float(str_number) for str_number in
                                     row[" false_positive_rate"].split(",")]
                                print(fpr)
                                tpr = \
                                    [float(str_number) for str_number in
                                     row[" true_positive_rate"].split(",")]
                                plt.plot(np.array(fpr), np.array(tpr))

                            img_idi = "{0},{1},{2}".format(
                                    row[" evaluation"],
                                    row[" test|train"],
                                    row[" fold"])
                            plt.title(img_idi)

                            plt.savefig(os.path.join(path_to_folder,
                                        "{}.png".format(img_idi)))

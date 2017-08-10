import csv
import os

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from numpy import linspace

import settings

start = 0.0
stop = 1.0
number_of_lines = 1000
cm_subsection = linspace(start, stop, 9)
colors = [cm.jet(x) for x in cm_subsection]

out_folder = "CVAE_session_swap_kernel_MRI"
output_weighted_svm = "loop_output_weighted_svm.csv"

path_to_folder = os.path.join(settings.path_to_general_out_folder, out_folder)

path_output_weighted_svm = os.path.join(path_to_folder, output_weighted_svm)
output_weighted_svm_file = open(path_output_weighted_svm)

reader = csv.DictReader(output_weighted_svm_file)

list_rows = []
for row in reader:
    list_rows.append(row)

kernel_size_array = np.array([float(row['kernel_size']) for row in list_rows])
f1_score_array = np.array([float(row['f1_score']) for row in list_rows])
recall_score_array = np.array([float(row['recall_score']) for row in list_rows])
accuracy_score_array = np.array([float(row['accuracy']) for row in list_rows])
auc_score_array = np.array([float(row['area under the curve']) for row in list_rows])
precision = np.array([float(row['precision']) for row in list_rows])

plt.figure()
plt.title("MRI swap over kernel")
plt.plot(kernel_size_array, f1_score_array, label="f1_score")
plt.plot(kernel_size_array, recall_score_array, label="recall_score")
plt.plot(kernel_size_array, accuracy_score_array, label="accuracy")
plt.plot(kernel_size_array, auc_score_array, label="area under curve")
plt.plot(kernel_size_array, precision, label="precision")
plt.ylabel("% results")
plt.xlabel("Kernel Size")
plt.legend()
plt.savefig("intento.png")

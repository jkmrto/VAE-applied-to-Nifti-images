from matplotlib import pyplot as plt
import os
import numpy as np
from matplotlib import cm
from numpy import linspace


def generate_color_palette():
    start = 0.0
    stop = 1.0
    number_of_lines = 1000
    cm_subsection = linspace(start, stop, 9)
    colors = [cm.jet(x) for x in cm_subsection]


def plot_evaluation_parameters(list_parameters_dict, string_ref,
                               path_evaluation_images_folder,
                               swap_type, xlabel = "none"):

    list_rows = list_parameters_dict
    kernel_size_array = np.array([float(row[swap_type]) for row in list_rows])
    f1_score_array = np.array([float(row['f1_score']) for row in list_rows])
    recall_score_array = np.array([float(row['recall_score']) for row in list_rows])
    accuracy_score_array = np.array([float(row['accuracy']) for row in list_rows])
    auc_score_array = np.array([float(row['area under the curve']) for row in list_rows])
    precision = np.array([float(row['precision']) for row in list_rows])

    plt.figure()
    plt.title(string_ref)
    plt.plot(kernel_size_array, f1_score_array, label="f1_score")
    plt.plot(kernel_size_array, recall_score_array, label="recall_score")
    plt.plot(kernel_size_array, accuracy_score_array, label="accuracy")
    plt.plot(kernel_size_array, auc_score_array, label="area under curve")
    plt.plot(kernel_size_array, precision, label="precision")
    plt.ylabel("% results")
    plt.xlabel(xlabel)
    plt.legend()
    plt.savefig(os.path.join(path_evaluation_images_folder, "{}.png".format(string_ref)))
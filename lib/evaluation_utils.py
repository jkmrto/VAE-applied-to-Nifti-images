from lib.aux_functionalities.functions import print_dictionary
from sklearn import metrics
from matplotlib import pyplot as plt
import numpy as np


def evaluation_output(path_to_resume_file, path_to_roc_png,
                      path_to_results_file, y_obtained,  y_test):

    results = np.concatenate((y_test, y_obtained))

    precision = metrics.average_precision_score(y_test, y_obtained)
    auc = metrics.roc_auc_score(y_test, y_obtained)
    output_dic = {"precision": precision,

                  "area under the curve": auc}
    print_dictionary(path_to_resume_file, output_dic)
    [fpr, tpr, thresholds] = metrics.roc_curve(y_test, y_obtained)
    np.savetxt(path_to_results_file, results, delimiter=',')

    plt.figure()
    plt.plot(fpr, tpr, linestyle='--')
    plt.savefig(path_to_roc_png)
from lib.aux_functionalities.functions import print_dictionary
from sklearn import metrics
from matplotlib import pyplot as plt
import numpy as np


def evaluation_output(path_to_resume_file, path_to_roc_png,
                      path_to_results_file, y_obtained, y_test,
                      thresholds_establised=None):
    results = np.concatenate(([y_test], [y_obtained]))
    np.savetxt(path_to_results_file, results, delimiter=',')

    [fpr, tpr, thresholds_roc] = metrics.roc_curve(y_test, y_obtained)
    plot_roc_curve(fpr, tpr, path_to_roc_png)

    if thresholds_establised == None:
        threshold = get_thresholds_from_roc_curve(fpr, tpr, thresholds_roc)
    else:
        threshold = thresholds_establised

    scores_labeled = assign_binary_labels_based_on_threshold(
        y_obtained, threshold)

    accuracy = metrics.accuracy_score(y_test, scores_labeled)
    f1_score = metrics.f1_score(y_test, scores_labeled)
    recall_score = metrics.recall_score(y_test, scores_labeled)

    precision = metrics.average_precision_score(y_test, y_obtained)
    auc = metrics.roc_auc_score(y_test, y_obtained)
    output_dic = {"precision": precision,
                  "area under the curve": auc,
                  "accuracy": accuracy,
                  "f1_score": f1_score,
                  "recall_score": recall_score}

    print_dictionary(path_to_resume_file, output_dic)

    return threshold


def simple_evaluation_output(y_obtained, y_test, bool_test=False,
                             thresholds_establised=None):

    print(np.hstack((y_obtained, y_test)))

    y_obtained = np.row_stack(y_obtained)
    y_test = np.row_stack(y_test)

    [fpr, tpr, thresholds_roc] = metrics.roc_curve(y_test, y_obtained)

    if thresholds_establised == None:
        threshold = get_thresholds_from_roc_curve(fpr, tpr, thresholds_roc)
    else:
        threshold = thresholds_establised

    scores_labeled = assign_binary_labels_based_on_threshold(
        y_obtained, threshold)
    scores_labeled = np.row_stack(scores_labeled)

    print(np.hstack((y_obtained, y_test, scores_labeled)))

    accuracy = metrics.accuracy_score(y_test, scores_labeled)
    f1_score = metrics.f1_score(y_test, scores_labeled)
    recall_score = metrics.recall_score(y_test, scores_labeled)

    precision = metrics.average_precision_score(y_test, y_obtained)
    auc = metrics.roc_auc_score(y_test, y_obtained)
    output_dic = {"precision": precision,
                  "area under the curve": auc,
                  "accuracy": accuracy,
                  "f1_score": f1_score,
                  "recall_score": recall_score}

    return threshold, output_dic


def assign_binary_labels_based_on_threshold(scores, threshold):
    scores[scores <= threshold] = 0
    scores[scores > threshold] = 1

    return scores


def plot_roc_curve(fpr, tpr, path_to_roc_png):
    plt.figure()
    plt.plot(fpr, tpr, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curve ROC")
    plt.savefig(path_to_roc_png)


def get_thresholds_from_roc_curve(fpr, tpr, thresholds):
    distance_to_optimum = np.sqrt(np.power(fpr, 2) + np.power((1 - tpr), 2))
    pos_optimum = distance_to_optimum.argmin()
    thresholds_optimum = thresholds[pos_optimum]

    return thresholds_optimum

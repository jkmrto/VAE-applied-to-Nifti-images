import copy

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics

from lib.aux_functionalities.functions import print_dictionary
from lib.utils import svm_utils


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


def simple_evaluation_output(y_obtained, y_test,
                             thresholds_establised=None, bool_test=False):
    y_obtained = np.row_stack(y_obtained)
    y_test = np.row_stack(y_test)

    if bool_test:
        print(np.hstack((y_obtained, y_test)))

    [fpr, tpr, thresholds_roc] = metrics.roc_curve(y_test, y_obtained)

    if thresholds_establised == None:
        threshold = get_thresholds_from_roc_curve(fpr, tpr, thresholds_roc)
    else:
        threshold = thresholds_establised

    scores_labeled = assign_binary_labels_based_on_threshold(
        y_obtained, threshold)
    scores_labeled = np.row_stack(scores_labeled)

    if bool_test:
        print("y_obtained, y_test, scores_labeled")
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
    aux_scores = copy.deepcopy(scores)

    aux_scores[aux_scores <= threshold] = 0
    aux_scores[aux_scores > threshold] = 1

    return aux_scores


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


def simple_majority_vote(train_score_matrix, test_score_matrix, Y_train, Y_test,
                         bool_test=False):
    """
    :param train_score_matriz: type: np.array[] => [patients x nºregions]
    :param test_score_matriz:
    :return:
    """

    threshold = 0

    # simple majority vote
    train_labels_obatained = assign_binary_labels_based_on_threshold(
        train_score_matrix, threshold)

    test_labels_obatained = assign_binary_labels_based_on_threshold(
        test_score_matrix, threshold)

    # Means over each over, this is evalutating the activation per patient
    means_activation_train = np.row_stack(train_labels_obatained.mean(axis=1))
    means_activation_test = np.row_stack(test_labels_obatained.mean(axis=1))

    threshold = 0
    _, output_dic_train = simple_evaluation_output(means_activation_train,
                                                   Y_train,
                                                   threshold,
                                                   bool_test=bool_test)
    _, output_dic_test = simple_evaluation_output(means_activation_test, Y_test,
                                                  threshold,
                                                  bool_test=bool_test)

    return output_dic_train, output_dic_test


def get_average_over_metrics(list_dicts):
    """

    :param list_dicts:
    :return:
    """
    out = {}

    for key in list_dicts[0].keys():
        out[key] = np.array([float(dic[key]) for dic in list_dicts]).mean()

    return out


def complex_majority_vote_evaluation(data, bool_test=False):

    test_score_matriz = data["test"]["data"]
    Y_test = data["test"]["label"]
    train_score_matriz = data["train"]["data"]
    Y_train = data["train"]["label"]

    # COMPLEX MAJORITY VOTE
    complex_means_train = np.row_stack(train_score_matriz.mean(axis=1))
    complex_means_test = np.row_stack(test_score_matriz.mean(axis=1))

    if bool_test:
        print("TEST OVER FINAL RESULTS")
        test_train_score = np.hstack(
            (np.row_stack(complex_means_train), np.row_stack(Y_train)))
        test_test_score = np.hstack(
            (np.row_stack(complex_means_test), np.row_stack(Y_test)))
        print(test_train_score)
        print(test_test_score)

    threshold = 0
    _, complex_output_dic_train = simple_evaluation_output(complex_means_train,
                                                           Y_train, threshold,
                                                           bool_test=bool_test)
    _, complex_output_dic_test = simple_evaluation_output(complex_means_test,
                                                          Y_test, threshold,
                                                          bool_test=bool_test)

    print("Complex Majority Vote Test: " + str(complex_output_dic_test))
    print("Complex Majority Vote Train: " + str(complex_output_dic_train))

    return complex_output_dic_test, complex_output_dic_train


def weighted_svm_decision_evaluation(data, list_regions, bool_test=False):
    """

    :param data: Dicitionary["test"|"train"]["label"|"data"]
    :param list_regions: List 
    :param bool_test:
    :return:
    """

    test_score_matriz = data["test"]["data"]
    Y_test = data["test"]["label"]
    train_score_matriz = data["train"]["data"]
    Y_train = data["train"]["label"]

    if bool_test:
        print("SVM over weigthed regions shapes of data")
        print("train_data" + str(train_score_matriz.shape))
        print("test_data " + str(test_score_matriz.shape))
        print("train labels" + str(Y_train.flatten().shape))

    scores_train, scores_test, svm_coef = \
        svm_utils.fit_svm_and_get_decision_for_requiered_data_and_coefs_associated(
            train_score_matriz, Y_train.flatten(),
            test_score_matriz)

    # SVM weighted REGIONS RESULTS EVALUATION RESULTS
    threshold = 0
    _, weighted_output_dic_train = simple_evaluation_output(scores_train,
                                                            Y_train, 0,
                                                            bool_test=bool_test)
    _, weighted_output_dic_test = simple_evaluation_output(scores_test,
                                                           Y_test, 0,
                                                           bool_test=bool_test)

    aux_dic_regions_weight_coefs = {}
    [aux_dic_regions_weight_coefs.update({str(region): coef}) for region, coef
     in zip(list_regions, svm_coef)]

    print("Weighted SVM Vote Test results: " + str(weighted_output_dic_test))
    print("Weighted SVM  Vote Train results: " + str(weighted_output_dic_train))
    print("Weighted SVM  Coefs Gotten: " + str(aux_dic_regions_weight_coefs))

    return weighted_output_dic_test, weighted_output_dic_train, \
           aux_dic_regions_weight_coefs
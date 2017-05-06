from sklearn import svm
from sklearn import metrics
from lib.aux_functionalities import functions
from matplotlib import pyplot as plt
import os
import numpy
import copy


def fit_svm_and_get_decision_for_requiered_data(X_train, Y_train, X_test,
                                                decision_function_shape="ovr",
                                                kernel="linear"):

    clf = svm.SVC(decision_function_shape=decision_function_shape, kernel=kernel)
    clf.fit(X_train, Y_train)

    # Testing time
    scores_test = clf.decision_function(X_test)
    scores_train = clf.decision_function(X_train)

    return scores_train, scores_test


def per_region_evaluation(score, true_label, per_region_accuracy_file,
                          region_selected):
    dec_label = functions.assign_binary_labels_based_on_threshold(
        copy.copy(score), 0)

    region_accuracy = metrics.accuracy_score(true_label, dec_label)
    per_region_accuracy_file.write("region_{0},{1}\n".format(region_selected,
                                                             region_accuracy))
    per_region_accuracy_file.flush()


def log_scores(score, score_file, region_selected):
    score_file.write("region_{0}".format(region_selected))

    for out in score:
        score_file.write(",{}".format(out))
    score_file.write("\n")
    score_file.flush()


def load_svm_output_score(score_file, plot_hist=False):
    #    labels_file = open(path_to_particular_test + "patient_labels_per_region.log", "w")
    # print(score_file)
    my_data = numpy.genfromtxt(score_file, delimiter=',')
    # print(my_data)
    my_data = my_data[:, 1:]  # Deleting the first column of garbage
    dic = {}
    dic['min'] = min(my_data.flatten())
    dic['max'] = max(my_data.flatten())
    dic['range'] = dic['max'] - dic['min']
    dic['data_normalize'] = (my_data - dic['min']) / dic['range']
    dic['data_normalize'] = dic['data_normalize'].transpose()
    dic['raw'] = my_data.transpose()

    if plot_hist:
        plt.hist(dic['data_normalize'].flatten())
        plt.show()

    return dic
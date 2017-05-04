from sklearn import svm
from sklearn import metrics
from lib.aux_functionalities import functions
import os
import numpy
import copy

train_index_file = "train_index_to_stack.csv"
test_index_file = "test_index_to_stack.csv"


def fit_svm_and_get_decision_for_requiered_data(X_train, Y_train, X_test,
                                                decision_function_shape="ovr",
                                                kernel="linear"):

    clf = svm.SVC(decision_function_shape=decision_function_shape, kernel=kernel)
    clf.fit(X_train, Y_train)

    # Testing time
    scores_test = clf.decision_function(X_test)
    scores_train = clf.decision_function(X_train)

    return scores_train, scores_test


def get_train_and_test_index_from_files(path):
    train_index_path = os.path.join(path, train_index_file)
    test_index_path = os.path.join(path, test_index_file)

    train_index = numpy.genfromtxt(train_index_path).astype(int).tolist()
    test_index = numpy.genfromtxt(test_index_path).astype(int).tolist()

    return train_index, test_index


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
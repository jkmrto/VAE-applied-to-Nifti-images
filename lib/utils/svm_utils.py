from sklearn import svm
from matplotlib import pyplot as plt
import numpy as np


def fit_svm_and_get_decision_for_requiered_data(X_train, Y_train, X_test,
                                                decision_function_shape="None",
                                                kernel="linear",
                                                minimum_training_svm_error=0.001):

    clf = svm.SVC(decision_function_shape=decision_function_shape,
                  kernel=kernel)
    clf.fit(X_train, Y_train)

    # Testing time
    scores_test = clf.decision_function(X_test)
    scores_train = clf.decision_function(X_train)

    return scores_train, scores_test


def fit_svm_and_get_decision_for_requiered_data_and_coefs_associated(X_train,
    Y_train, X_test, decision_function_shape="ovr", kernel="linear"):

    clf = svm.SVC(decision_function_shape=decision_function_shape,
                  kernel=kernel)
    clf.fit(X_train, Y_train)

    # Testing time
    scores_test = clf.decision_function(X_test)
    scores_train = clf.decision_function(X_train)

    return scores_train, scores_test, clf.coef_.tolist()[0]



#def per_region_evaluation(score, true_label, per_region_accuracy_file,
#                          region_selected):
#    dec_label = evaluation.assign_binary_labels_based_on_threshold(
#        copy.copy(score), 0)

#    region_accuracy = metrics.accuracy_score(true_label, dec_label)
#    per_region_accuracy_file.write("region_{0},{1}\n".format(region_selected,
 #                                                            region_accuracy))
#    per_region_accuracy_file.flush()


def log_scores(score, score_file, region_selected):
    score_file.write("region_{0}".format(region_selected))

    for out in score:
        score_file.write(",{}".format(out))
    score_file.write("\n")
    score_file.flush()


def load_svm_output_score(score_file, plot_hist=False):
    #    labels_file = open(path_to_particular_test + "patient_labels_per_region.log", "w")
    # print(score_file)
    my_data = np.genfromtxt(score_file, delimiter=',')
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


def svm_mri_over_vae_output(vae_output, Y_train, Y_test, list_regions, bool_test=False,
                            minimum_training_svm_error=0.001):

    n_train_patient = Y_train.shape[0]
    n_test_patient = Y_test.shape[0]

    train_score_matriz = np.zeros((n_train_patient, len(list_regions)))
    test_score_matriz = np.zeros((n_test_patient, len(list_regions)))

    i = 0

    for region_selected in list_regions:

        print("SVM step")
        print("region {} selected".format(region_selected))
        train_output_wm = vae_output['wm'][region_selected]['train_output']
        test_output_wm = vae_output['wm'][region_selected]['test_output']

        train_output_gm = vae_output['gm'][region_selected]['train_output']
        test_output_gm = vae_output['gm'][region_selected]['test_output']

        train_means_gm = train_output_wm["mean"]
        test_means_gm = test_output_wm["mean"]

        train_means_wm = train_output_gm["mean"]
        test_means_wm = test_output_gm["mean"]

        wm_and_gm_train_data = np.concatenate((train_means_gm, train_means_wm),
                                              axis=1)
        wm_and_gm_test_data = np.concatenate((test_means_gm, test_means_wm),
                                             axis=1)
        if bool_test:
            print("\nShape wm+gm train data post encoder")
            print("Train shape: " +  str(wm_and_gm_train_data.shape))
            print("Test shape: " + str(wm_and_gm_test_data.shape))

        train_score, test_score = fit_svm_and_get_decision_for_requiered_data(
            wm_and_gm_train_data, Y_train, wm_and_gm_test_data,
            minimum_training_svm_error=minimum_training_svm_error)

        # [patient x regions] SVM results
        train_score_matriz[:, i] = train_score
        test_score_matriz[:, i] = test_score

        if bool_test:
            print("TEST SVM SCORE REGION {}".format(region_selected))
            print(train_score.shape)
            print(Y_train.shape)
            test_train_score = np.hstack(
                (np.row_stack(train_score), np.row_stack(Y_train)))
            test_test_score = np.hstack(
                (np.row_stack(test_score), np.row_stack(Y_test)))
            print(test_train_score)
            print(test_test_score)

        i += 1

    return train_score_matriz, test_score_matriz


def svm_pet_over_vae_output(vae_output, Y_train, Y_test, list_regions,
                            bool_test=False, minimum_training_svm_error=0.001):

    n_train_patient = vae_output[list_regions[0]]['train_output']["mean"].shape[0]
    n_test_patient = vae_output[list_regions[0]]['test_output']["mean"].shape[0]

    train_score_matriz = np.zeros((n_train_patient, len(list_regions)))
    test_score_matriz = np.zeros((n_test_patient, len(list_regions)))

    i = 0

    for region_selected in list_regions:

        print("SVM step")
        print("region {} selected".format(region_selected))
        train_output = vae_output[region_selected]['train_output']
        test_output = vae_output[region_selected]['test_output']

        train_means = train_output["mean"]
        test_means = test_output["mean"]

        if bool_test:
            print("\nShape wm+gm train data post encoder")
            print("Train shape: " + str(train_means.shape))
            print("Test shape: " + str(test_means.shape))

        train_score, test_score = fit_svm_and_get_decision_for_requiered_data(
            train_means, Y_train, test_means,
            minimum_training_svm_error=minimum_training_svm_error)

        # [regions x patients] SVM results
        train_score_matriz[:, i] = train_score
        test_score_matriz[:, i] = test_score

        if bool_test:
            print("TEST SVM SCORE REGION {}".format(region_selected))
            print(train_score.shape)
            print(Y_train.shape)
            test_train_score = np.hstack(
                (np.row_stack(train_score), np.row_stack(Y_train)))
            test_test_score = np.hstack(
                (np.row_stack(test_score), np.row_stack(Y_test)))
            print(test_train_score)
            print(test_test_score)

        i += 1

    return train_score_matriz, test_score_matriz
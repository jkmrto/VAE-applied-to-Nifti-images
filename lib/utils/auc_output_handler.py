import numpy as np


def array_to_str_csv_list(array):
    """

    :param array: ty[np.ndarray | list]
    :return:
    """
    if isinstance(array, np.ndarray):
        array = array.tolist()

    out = ",".join([str(value) for value in array])
    print(out)
    return out


def stringfy_auc_information(swap_over, k_fold_index, evaluation, roc_dic):
    """
    swap_variable;n_fold;evaluation_type;test|train;tpr;fpr,thr
    :param swap_over:
    :param k_fold_index:
    :return:
    """
    string_test = "{0};{1};{2};test;{3};{4};{5}".format(
        str(swap_over), str(k_fold_index), str(evaluation),
        array_to_str_csv_list(roc_dic["test"]["fpr"]),
        array_to_str_csv_list(roc_dic["test"]["tpr"]),
        array_to_str_csv_list(roc_dic["test"]["thresholds"]))

    string_train = "{0};{1};{2};train;{3};{4};{5}".format(
        str(swap_over), str(k_fold_index), str(evaluation),
        array_to_str_csv_list(roc_dic["train"]["fpr"]),
        array_to_str_csv_list(roc_dic["train"]["tpr"]),
        array_to_str_csv_list(roc_dic["train"]["thresholds"]))

    return string_test, string_train


def test():


    fpr = np.array([1, 2, 3, 4])
    tpr = [1, 2, 3, 4]
    thr = [0.2, 0.4, 0.8]

    aud_dic = {"test": {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thr},
        "train": {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thr}}

    test_string, train_string = stringfy_auc_information(
        swap_over=5,
        k_fold_index=1,
        evaluation="Simple Majority Vote",
        roc_dic=roc_dic)

    print(test_string)
    print(train_string)

#test()
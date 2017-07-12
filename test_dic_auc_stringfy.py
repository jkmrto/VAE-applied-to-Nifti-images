import numpy as np
k_fold_index = 5
swap_over = "kernel_size"

fpr = np.array([1,2,3,4])
tpr = [1,2,3,4]
thr = [0.2, 0.4, 0.8]

aud_dic = {"test":{
    "fpr":fpr,
    "tpr": tpr,
    "thresholds": thr}}




string_test = "{0};{1};{2};test;{2};{3};{4};{5}".format(
    swap_over, k_fold_index, "simple_majority_vote",
    array_to_str_csv_list(aud_dic["test"]["fpr"]),
    array_to_str_csv_list(aud_dic["test"]["tpr"]),
    array_to_str_csv_list(aud_dic["test"]["thresholds"]))

#string_train
print(array_to_str_csv_list(np.array([3,4,5,6,7,8,6,5,5,4])))
print(string_test)
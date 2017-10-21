import json
import numpy as np

# Structure.

#dic_container_evaluations = {
#    "SVM": {},
#    "SMV": {},
#    "CMV": {},

#kernel_list = [2, 3, 4, 5]
#for swap_variable_index in kernel_list:

#dic_container_evaluations["SVM"][swap_variable_index] = {}
#dic_container_evaluations["SMV"][swap_variable_index] = {}
#dic_container_evaluations["CMV"][swap_variable_index] = {}
#["SVM"|"CMV"|""][kernel_value][fold]

evaluation_methods = ["SVM", "CMV", "SMV"]


class JSONEncoder(json.JSONEncoder):
    """
    Class which will use in order to enconde the ObjectId (Mongodb ID)
    and serialize to JSON
    """

    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()

        return json.JSONEncoder.default(self, o)


def validate_samples_to_map(samples_to_map):

    if isinstance(samples_to_map, list):
        samples_to_map = np.array(samples_to_map)
    samples_to_map = np.reshape(samples_to_map, [samples_to_map.size, 1])

    return samples_to_map


def evaluation_container_to_log_file(path_file_test_out, path_file_full_out,
                                     evaluation_container,
                                     k_fold_container, swap_variable_list,
                                     n_samples):
    """

    :param path_to_file:
    :param evaluation_container:
    :param k_fold_dict: dict[kfold_index] -> dict["train"|"test"] -> indexes_samples
    :return:
    """
    # path_file_test_out
    # path_file_full_out

    test_simplified_container = {}
    full_container = {}

    for method in evaluation_methods:
        test_simplified_container[method] = {}
        full_container[method] = {}
        for swap_variable in swap_variable_list:
            full_container[method][swap_variable] = {}
            test_temp_array = np.zeros([n_samples, 1])
            # getting fold_indexes from the container
            k_fold_dict = k_fold_container[swap_variable]

            for k_fold_index in k_fold_dict.keys():

                full_temp_array = np.zeros([n_samples, 1])

                test_indexes = k_fold_dict[k_fold_index]["test"]
                train_indexes = k_fold_dict[k_fold_index]["train"]

                test_samples_to_map = validate_samples_to_map(
                    evaluation_container[method][swap_variable][k_fold_index]["test"])

                train_samples_to_map = validate_samples_to_map(
                    evaluation_container[method][swap_variable][k_fold_index]["train"])

                test_temp_array[test_indexes] = test_samples_to_map
                full_temp_array[test_indexes] = test_samples_to_map
                full_temp_array[train_indexes] = train_samples_to_map

                full_temp_dict = {
                    "threshold": evaluation_container[method][swap_variable][k_fold_index]["threshold"],
                    "values": full_temp_array,
                    "values_train": train_samples_to_map,
                    "values_test":  test_samples_to_map,
                    "indexes_train": train_indexes,
                    "indexes_test": test_indexes,
                }
                full_container[method][swap_variable][k_fold_index] = \
                    full_temp_dict

            test_simplified_container[method][swap_variable] = test_temp_array

    file_test = open(path_file_test_out, "w")
    json.dump(test_simplified_container, file_test, cls=JSONEncoder)
    file_test.close()

    file_full = open(path_file_full_out, "w")
    json.dump(full_container, file_full, cls=JSONEncoder)
    file_full.close()


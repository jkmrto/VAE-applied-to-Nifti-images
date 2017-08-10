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

evaluation_methods = ["SVM","CMV","SMV"]

class JSONEncoder(json.JSONEncoder):
    """
    Class which will use in order to enconde the ObjectId (Mongodb ID)
    and serialize to JSON
    """

    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()

        return json.JSONEncoder.default(self, o)

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


def evaluation_container_to_log_file(path_to_file, evaluation_container,
                                     k_fold_container, swap_variable_list,
                                     n_samples):
    """

    :param path_to_file:
    :param evaluation_container:
    :param k_fold_dict: dict[kfold_index] -> dict["train"|"test"] -> indexes_samples
    :return:
    """
    simplified_container = {}
    for method in evaluation_methods:
        simplified_container[method] = {}
        for swap_variable in swap_variable_list:
            temp_array = np.zeros([n_samples, 1])
            k_fold_dict = k_fold_container[swap_variable]

            for k_fold_index in k_fold_dict.keys():
                test_fold_index = k_fold_dict[k_fold_index]["test"]
                samples_to_map = evaluation_container[method][swap_variable][k_fold_index]["test"]
                if isinstance(samples_to_map,list):
                    samples_to_map = np.array(samples_to_map)
                samples_to_map = np.reshape(samples_to_map, [samples_to_map.size, 1])
                temp_array[test_fold_index] = samples_to_map

            simplified_container[method][swap_variable] = temp_array

    file = open(path_to_file, "w")
    json.dump(simplified_container, file, cls=JSONEncoder)
    file.close()

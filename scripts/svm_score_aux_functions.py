import matplotlib.pyplot as plt
from numpy import genfromtxt


def load_svm_output_score(score_file, plot_hist=False):
    #    labels_file = open(path_to_particular_test + "patient_labels_per_region.log", "w")
    # print(score_file)
    my_data = genfromtxt(score_file, delimiter=',')
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
from settings import LOG_DIR_GRADIENT_DESCEND_ERROR
import matplotlib.pyplot as plt
from lib.aux_functionalities import os_aux
import settings
from settings import list_regions_evaluated
from math import ceil

# Grouping 4 plots per figure
# Estimating the number of figures:
# plots_per_figure = 4
# number_regions_evaluated = len(list_regions_evaluated)
# number_figure_required = ceil(number_regions_evaluated / plots_per_figure)

# for number_figure in range(1, number_figure_required + 1, 1):
#    plt.figure(1)
#    plt.subplot(211)
#    plt.plot(t, s1)
#    plt.subplot(212)
#    plt.plot(t, 2 * s1)

path_to_dir_to_store_graph = settings.path_to_project + "/png/" + "/desc_grad_graph/"
os_aux.create_directories([path_to_dir_to_store_graph])

list_regions_evaluated = [3, 4, 7, 8, 9, 10, 23, 32, 61, 62 , 68]
figure_index = 0
for region in list_regions_evaluated:

    figure_index += 1
    plt.figure(figure_index)

    path_to_file = LOG_DIR_GRADIENT_DESCEND_ERROR + "region_{}_.log".format(region)
    file = open(path_to_file, 'r')
    iter = []
    error = []
    for line in file:
        [iter_aux, error_aux] = file.readline().split(",")
        iter.append(int(iter_aux))
        error.append(float(error_aux))

    plt.plot(error)

    plt.title('Region {}'.format(region))

    plt.savefig(path_to_dir_to_store_graph + 'grad_desc_region_{}.png'.format(region), dpi=200)


from settings import LOG_DIR_GRADIENT_DESCEND_ERROR
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


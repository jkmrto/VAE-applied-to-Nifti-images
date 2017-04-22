from settings import LOG_DIR_GRADIENT_DESCEND_ERROR
import matplotlib.pyplot as plt
from settings import list_regions_evaluated
from math import ceil

# Grouping 4 plots per figure
# Estimating the number of figures:
#plots_per_figure = 4
#number_regions_evaluated = len(list_regions_evaluated)
#number_figure_required = ceil(number_regions_evaluated / plots_per_figure)

#for number_figure in range(1, number_figure_required + 1, 1):
#    plt.figure(1)
#    plt.subplot(211)
#    plt.plot(t, s1)
#    plt.subplot(212)
#    plt.plot(t, 2 * s1)


list_regions_evaluated = [3,4,7,8,9,10,23]
fig = 0
figure_index = 0
plot_index = 0
for region in list_regions_evaluated:

    if plot_index % 4 == 0:
        figure_index += 1
        fig = plt.figure(figure_index)
        ax = []

        for index in range(1,5,1):
          ax.append(fig.add_subplot(2, 2, index))

    ax_selected = ax.pop(0)
    plot_index += 1

    path_to_file = LOG_DIR_GRADIENT_DESCEND_ERROR + "region_{}_.log".format(region)
    file = open(path_to_file, 'r')
    iter = []
    error = []
    for line in file:
        [iter_aux, error_aux] = file.readline().split(",")
        iter.append(int(iter_aux))
        error.append(float(error_aux))

    plt.tight_layout()
    ax_selected.plot(error)
    ax_selected.set_title('Region {}'.format(region))


    plt.savefig('prueba_1_{}.png'.format(figure_index), dpi=200)

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches


def plot_3dscatter_plot_2groups(samples, samples_labels, path_image, tittle=""):
    fig = plt.figure()
    ax = Axes3D(fig, elev=-150, azim=110)

    colors = ("blue", "red")
    groups = ("NOR", "AD")
    print(samples_labels == 0)
    print(samples)

    index_to_selected_images_0 = samples_labels == 0
    index_to_selected_images_0 = index_to_selected_images_0.flatten().tolist()

    index_to_selected_images_1 = samples_labels == 1
    index_to_selected_images_1 = index_to_selected_images_1.flatten().tolist()

    samples_0 = samples[index_to_selected_images_0, :]
    samples_1 = samples[index_to_selected_images_1, :]

    scatter1_proxy = ax.scatter(samples_0[:, 0], samples_0[:, 1],
                                samples_0[:, 2], c=colors[0], alpha=1)
    scatter_2_proxy = ax.scatter(samples_1[:, 0], samples_1[:, 1],
                                 samples_1[:, 2], c=colors[1], alpha=1)

    plt.title(tittle)
    ax.legend([scatter1_proxy, scatter_2_proxy], ['NOR', 'AD'],
              numpoints=1)
    plt.savefig(path_image, format="png")


def plot_2dscatter_plot_2groups(samples, samples_labels, path_image, tittle=""):
    index_to_selected_images_0 = samples_labels == 0
    index_to_selected_images_0 = index_to_selected_images_0.flatten().tolist()

    index_to_selected_images_1 = samples_labels == 1
    index_to_selected_images_1 = index_to_selected_images_1.flatten().tolist()

    samples_0 = samples[index_to_selected_images_0, :]
    samples_1 = samples[index_to_selected_images_1, :]

    colors = ("blue", "red")
    groups = ("NOR", "AD")

    plt.figure()
    plt.title(tittle)
    lo = plt.scatter(samples_0[:, 0], samples_0[:, 1], marker='o',
                     color=colors[0])
    ll = plt.scatter(samples_1[:, 0], samples_1[:, 1], marker='o',
                     color=colors[1])
    plt.legend([lo, ll], ["NOR", "AD"], loc='upper left', fontsize=8)

    plt.savefig(path_image, format="png")


def test3d():
    y = np.array([0, 0, 1, 1])

    x = np.array([[1, 1, 1],
                  [2, 2, 2],
                  [3, 3, 3],
                  [4, 4, 4]])

    plot_3dscatter_plot_2groups(
        samples=x,
        samples_labels=y,
        path_image="test.png")

#test3d()

def test2d():
    y = np.array([0, 0, 1, 1])

    x = np.array([[1, 1],
                  [2, 2],
                  [3, 3],
                  [4, 4]])

    plot_2dscatter_plot_2groups(
        samples=x,
        samples_labels=y,
        path_image="test.png")

test2d()

import numpy as np
from matplotlib import pyplot as plt


def reconstruct_3d_image(image_flatten, voxels_index, imgsize):

    print(imgsize)
    mri_image = np.zeros(imgsize[0] * imgsize[1] * imgsize[2])
    mri_image[voxels_index] = image_flatten
    mri_image_3d = mri_image.reshape(121, 145, 121)

    return mri_image_3d


def get_mean_over_samples_images(dict_norad, label):
    "mean over rows that should be the number of samples"

    desired_patients_pos = dict_norad['labels'] == label
    desired_patients_pos = desired_patients_pos.flatten()
    matriz_images = dict_norad['stack'][desired_patients_pos, :]
    mean_image = matriz_images.mean(axis=0)

    return mean_image


def plot_and_save_mri_section(image_mri_3d, index_section, png_name):
    plt.figure()
    plt.imshow(image_mri_3d[:,index_section,:], cmap="Greys")
    plt.savefig(png_name)
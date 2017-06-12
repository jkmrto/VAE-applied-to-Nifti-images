import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt


def reconstruct_3d_image(image_flatten, voxels_index, imgsize, reshape_order="C"):

    mri_image = np.zeros(imgsize)
    mri_image = mri_image.flatten()
    mri_image[voxels_index] = image_flatten
    mri_image_3d = np.reshape(mri_image, imgsize, reshape_order)

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


def plot_and_save_fig(flat_image, dict_norad, path_to_image, title="",
                      index_section_to_plot=77, path_to_3d_reconstruction=None,
                      reshape_order="C"):

    image_reconstructed_3d  = reconstruct_3d_image(flat_image,
            dict_norad['voxel_index'], dict_norad['imgsize'],
            reshape_order=reshape_order)

    plt.figure()
    plt.imshow(image_reconstructed_3d[:,index_section_to_plot,:], cmap="Greys")
    plt.title(title)
    plt.savefig(path_to_image)

    if path_to_3d_reconstruction is not None:
        img = nib.Nifti1Image(image_reconstructed_3d, np.eye(4))
        img.to_filename(path_to_3d_reconstruction)


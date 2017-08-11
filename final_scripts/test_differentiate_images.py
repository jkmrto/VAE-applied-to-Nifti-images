import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

from lib.data_loader import PET_stack_NORAD
from lib import reconstruct_helpers as recons
import numpy as np

stack_dict = PET_stack_NORAD.get_full_stack()
stack = stack_dict['stack']
patient_labels = stack_dict['labels']

label_selected = 0
index_to_selected_images = patient_labels == label_selected
index_to_selected_images = index_to_selected_images.flatten()
images_0 = stack[index_to_selected_images.tolist(), :]

label_selected = 1
index_to_selected_images = patient_labels == label_selected
index_to_selected_images = index_to_selected_images.flatten()
images_1 = stack[index_to_selected_images.tolist(), :]

print("images 0 shape {}".format(images_0.shape))
print("images 1 shape {}".format(images_1.shape))

means_0 = recons.get_mean_over_selected_samples(images=stack,
                                         label_selected=0,
                                         patient_labels=stack_dict['labels'])

means_1 = recons.get_mean_over_selected_samples(images=stack,
                                         label_selected=1,
                                         patient_labels=stack_dict['labels'])


dif_far_from_0 = np.abs(means_0 - images_1)
dif_far_from_1 = np.abs(means_1 - images_0)

pos_img_1 = np.argmax(dif_far_from_0.sum(axis=1))
pos_img_0 = np.argmax(dif_far_from_1.sum(axis=1))
print("AD {}".format(images_0.shape[0] + pos_img_1))
print("NOR {}".format(pos_img_0))

k_means3d_0 = recons.reconstruct_3d_image_from_flat_and_index(
    image_flatten=images_0[pos_img_0, :],
    voxels_index=stack_dict['voxel_index'],
    imgsize=stack_dict['imgsize'],
    reshape_kind="F")

k_means3d_1 = recons.reconstruct_3d_image_from_flat_and_index(
    image_flatten=images_1[pos_img_1, :],
    voxels_index=stack_dict['voxel_index'],
    imgsize=stack_dict['imgsize'],
    reshape_kind="F")


recons.plot_most_discriminative_section(
    img3d_1=k_means3d_0,
    img3d_2=k_means3d_1,
    path_to_save_image="kmeans_test_over_samples.png",
    cmap="jet")
from sklearn.cluster import KMeans
from lib.data_loader import PET_stack_NORAD
from lib import reconstruct_helpers as recons

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


print("sahpe images 0")
print(images_0.shape)
print(len(patient_labels) - sum(patient_labels)[0])

print("sahpe images 1")
print(images_1.shape)
print(sum(patient_labels))


kmeans_0 = KMeans(n_clusters=5, random_state=0).fit(images_0)
kmeans_1 = KMeans(n_clusters=5, random_state=0).fit(images_1)


print(kmeans_0.cluster_centers_[0].shape)
print(kmeans_1.cluster_centers_[0].shape)

means_0 = recons.get_mean_over_selected_samples(images=stack,
                                         label_selected=0,
                                         patient_labels=stack_dict['labels'])

means_1 = recons.get_mean_over_selected_samples(images=stack,
                                         label_selected=1,
                                         patient_labels=stack_dict['labels'])

print(sum(means_0-kmeans_0.cluster_centers_[0]))
print(sum(means_1-kmeans_1.cluster_centers_[0]))


k_means3d_0 = recons.reconstruct_3d_image_from_flat_and_index(
    image_flatten=kmeans_0.cluster_centers_[0],
    voxels_index=stack_dict['voxel_index'],
    imgsize=stack_dict['imgsize'],
    reshape_kind="F")

k_means3d_1 = recons.reconstruct_3d_image_from_flat_and_index(
    image_flatten=kmeans_1.cluster_centers_[0],
    voxels_index=stack_dict['voxel_index'],
    imgsize=stack_dict['imgsize'],
    reshape_kind="F")


recons.plot_most_discriminative_section(
    img3d_1=k_means3d_0,
    img3d_2=k_means3d_1,
    path_to_save_image="kmeans_test_over_samples.png",
    cmap="jet")




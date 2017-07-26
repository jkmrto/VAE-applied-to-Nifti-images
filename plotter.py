from sklearn.cluster import KMeans
from lib.data_loader import PET_stack_NORAD
from lib import reconstruct_helpers as recons

stack_dict = PET_stack_NORAD.get_full_stack()
stack = stack_dict['stack']
patient_labels = stack_dict['labels']
img_name= "prueba.png"

k_means3d_0 = recons.reconstruct_3d_image_from_flat_and_index(
    image_flatten=stack[22, :],
    voxels_index=stack_dict['voxel_index'],
    imgsize=stack_dict['imgsize'],
    reshape_kind="F")

k_means3d_1 = recons.reconstruct_3d_image_from_flat_and_index(
    image_flatten=stack[123pa, :],
    voxels_index=stack_dict['voxel_index'],
    imgsize=stack_dict['imgsize'],
    reshape_kind="F")


recons.plot_most_discriminative_section(
    img3d_1=k_means3d_0,
    img3d_2=k_means3d_1,
    path_to_save_image=img_name,
    cmap="jet")
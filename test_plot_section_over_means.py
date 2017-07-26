from lib.reconstruct_helpers import plot_most_discriminative_section
from lib.data_loader import PET_stack_NORAD
from lib.data_loader import pet_loader
from lib.data_loader import MRI_stack_NORAD
from lib.reconstruct_helpers import evaluate_cubes_difference_by_planes, \
    reconstruct_3d_image_from_flat_and_index
from lib import reconstruct_helpers as recons

images_used = "PET"
list_regions = "all"
print("Loading pet images")
stack = PET_stack_NORAD.get_full_stack()

mean_0 = recons.get_mean_over_selected_samples(
    images = stack["stack"],
    label_selected=stack["labels"],
    patient_labels=0)

mean_1 = recons.get_mean_over_selected_samples(
    images = stack["stack"],
    label_selected=stack["labels"],
    patient_labels=1)

patient_0_3d = reconstruct_3d_image_from_flat_and_index(
    image_flatten=mean_0,
    voxels_index=stack['voxel_index'],
    imgsize=stack['imgsize'],
    reshape_kind="F")

patient_1_3d = reconstruct_3d_image_from_flat_and_index(
    image_flatten=mean_1,
    voxels_index=stack['voxel_index'],
    imgsize=stack['imgsize'],
    reshape_kind="F")

plot_most_discriminative_section(img3d_1=patient_0_3d,
                                 img3d_2=patient_1_3d,
                                 path_to_save_image=
                                 "test_plot_section_output.png",
                                 cmap="jet")

plot_most_discriminative_section(img3d_1=patient_0_3d,
                                 img3d_2=patient_1_3d,
                                 path_to_save_image=
                                 "test_plot_section_output.png",
                                 cmap="jet")

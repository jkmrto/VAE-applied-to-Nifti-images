from lib.reconstruct_helpers import plot_most_discriminative_section
from lib.data_loader import PET_stack_NORAD
from lib.data_loader import MRI_stack_NORAD
from lib.reconstruct_helpers import evaluate_cubes_difference_by_planes, \
    reconstruct_3d_image_from_flat_and_index

print("Loading pet images")
#stack = PET_stack_NORAD.get_full_stack()
stack = MRI_stack_NORAD.get_gm_stack()
patient_0 = stack['stack'][0, :]
patient_1 = stack['stack'][-1, :]

patient_0_3d = reconstruct_3d_image_from_flat_and_index(
    image_flatten=patient_0,
    voxels_index=stack['voxel_index'],
    imgsize=stack['imgsize'],
    reshape_kind="F")

patient_1_3d = reconstruct_3d_image_from_flat_and_index(
    image_flatten=patient_1,
    voxels_index=stack['voxel_index'],
    imgsize=stack['imgsize'],
    reshape_kind="F")


plot_most_discriminative_section(img3d_1=patient_0_3d,
                                 img3d_2=patient_1_3d,
                                 path_to_save_image=
                                 "test_plot_section_output.png",
                                 cmap="Greys")

from lib.reconstruct_helpers import reconstruct_3d_image_from_flat_and_index
from lib.utils.output_utils import from_3d_image_to_nifti_file
import numpy as np
from lib.data_loader import pet_atlas
from lib.data_loader import PET_stack_NORAD
from lib.utils import utils3d

region = 3
sample = 1
print("Loading Pet Atlas")
atlas = pet_atlas.load_atlas()
reshape_kind = "F"

path_original_3dimg = "example"

print("Loading Pet stack")
stack_dict = PET_stack_NORAD.get_full_stack()

total_size = stack_dict['total_size']
stack = stack_dict["stack"]
imgsize = stack_dict['imgsize']
voxels_index = stack_dict['voxel_index']
map_region_voxels = atlas[region]  # index refered to nbground voxels
no_bg_region_voxels_index = voxels_index[map_region_voxels]

sample_voxels = stack[sample, :]
sample_region_voxels = sample_voxels[map_region_voxels]
sample_stack = np.vstack((sample_voxels, sample_voxels))

print("Reconstructing 3dimg")
img3d = reconstruct_3d_image_from_flat_and_index(
    image_flatten=sample_region_voxels,
    voxels_index=no_bg_region_voxels_index,
    imgsize=imgsize,
    reshape_kind="F")

img3d_segmented = utils3d.get_3dimage_segmented(img3d)

print("Saving")
from_3d_image_to_nifti_file(
    path_to_save=path_original_3dimg,
    image3d=img3d_segmented)

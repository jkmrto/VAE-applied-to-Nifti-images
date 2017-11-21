import sys
import os
from lib.data_loader import utils_mask3d
sys.path.append(os.path.dirname(os.getcwd()))

from lib.utils import output_utils
from lib.data_loader import mri_atlas
from lib.data_loader import pet_atlas
from lib.data_loader import PET_stack_NORAD
from lib.data_loader import MRI_stack_NORAD
from lib.utils.os_aux import create_directories

import settings

region = 75
#images = "MRI"
images = "PET"

path_folder3D =  os.path.join(settings.path_to_project, "folder3D")
path_folder_masks3d = os.path.join(path_folder3D, "masks3D")
path_mask = os.path.join(
    path_folder_masks3d, "{1}_region:{0}".format(region, images))
create_directories([path_folder3D, path_folder_masks3d])

atlas = None
reshape_kind = None
colour_kind = None
stack_dict = None

if images == "MRI":
    stack_dict = MRI_stack_NORAD.get_gm_stack()
    reshape_kind = "A"
    colour_kind = "Greys"
    atlas = mri_atlas.load_atlas_mri()

elif images == "PET":
    stack_dict = PET_stack_NORAD.get_full_stack()
    reshape_kind = "F"
    colour_kind = "jet"

total_size = stack_dict['total_size']
imgsize = stack_dict['imgsize']
voxels_index = stack_dict['voxel_index']
map_region_voxels = atlas[region]  # index refered to nbground voxels
no_bg_region_voxels_index = voxels_index[map_region_voxels]

mask3d = utils_mask3d.generate_region_3dmaskatlas(
        no_bg_region_voxels_index=no_bg_region_voxels_index,
        reshape_kind=reshape_kind,
        imgsize=imgsize,
        totalsize=total_size)

output_utils.from_3d_image_to_nifti_file(
        path_to_save=path_mask,
        image3d=mask3d)


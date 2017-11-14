import matplotlib.pyplot as plt
from lib.data_loader import PET_stack_NORAD
from lib.data_loader import MRI_stack_NORAD
from lib.reconstruct_helpers import reconstruct_3d_image_from_flat_and_index
from lib.utils import output_utils
import settings
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))


plt.interactive(False)

#images = "PET"
images = "WM"
#images = "GM"

print("Loading stack {}".format(images))
sample_selected = 50

stack = None
reshape_kind = None
colour_kind = None

path_folder_brain3d = os.path.join(
    settings.path_to_project, "folder3D", "brain3d",
    "brain3D_img:{0}_sample:{1}".format(images, sample_selected))

if images == "WM":
    stack = MRI_stack_NORAD. get_wm_stack()
    reshape_kind = "C"
    colour_kind = "Greys"
elif images == "GM":
    stack = MRI_stack_NORAD.get_gm_stack()
    reshape_kind = "C"
    colour_kind = "Greys"
elif images == "PET":
    stack = PET_stack_NORAD.get_full_stack()
    reshape_kind = "F"
    colour_kind = "jet"

img_selected = stack["stack"][sample_selected, :]
img_selected[img_selected > 1] = 1


img_3d = reconstruct_3d_image_from_flat_and_index(
            image_flatten=stack["stack"][sample_selected, :],
            voxels_index=stack['voxel_index'],
            imgsize=stack['imgsize'],
            reshape_kind=reshape_kind)

output_utils.from_3d_image_to_nifti_file(
    path_to_save=path_folder_brain3d,
    image3d=img_3d
)
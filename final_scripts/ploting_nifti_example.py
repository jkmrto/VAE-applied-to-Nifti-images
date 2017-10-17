import matplotlib.pyplot as plt
from lib.data_loader import PET_stack_NORAD
from lib.data_loader import MRI_stack_NORAD
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))


plt.interactive(False)

images = "PET"
print("Loading stack {}".format(images))

stack = None
colour_kind = "Greys"
reshape_kind = "C"

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

imgsize = stack['imgsize']
mri_image = np.zeros(imgsize[0]*imgsize[1]*imgsize[2])
images_index = stack['voxel_index'] # indices de cada pixel


image_select = stack["stack"][1, :]
image_select = np.reshape(image_select, [image_select.shape[0]])
image_select[image_select > 1] = 1

mri_image[images_index] = image_select  # Selecting one image

mri_image_3d = mri_image.reshape(imgsize, order=reshape_kind)

plt.imshow(np.rot90(mri_image_3d[:, 50, :]), cmap=colour_kind)
plt.colorbar()
plt.show()











import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import settings
import nibabel as nib


plt.interactive(False)

print("Loading stack_NORAD_GM")
f = sio.loadmat(settings.MRI_stack_path_GM)
images_stack = f['stack_NORAD_GM']

imgsize = f['imgsize'].astype('uint32')
mri_image = np.zeros(imgsize[0][0]*imgsize[0][1]*imgsize[0][2])
images_index = f['nobck_idx'] # indices de cada pixel

mri_image[images_index] = images_stack[1,:] # Selecting one image
mri_image_3d = mri_image.reshape(121,145,121)
plt.imshow(mri_image_3d[:,50,:], cmap="Greys")
plt.show()


img = nib.load(settings.mri_atlas_path)
img_data = img.get_data()
atlasdata = img_data.flatten()
bckvoxels = np.where(atlasdata != 0)
atlasdata = atlasdata[bckvoxels]
vals = np.unique(atlasdata)
reg_idx = {}


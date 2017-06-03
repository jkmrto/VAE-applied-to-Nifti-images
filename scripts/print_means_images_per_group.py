from lib.mri import stack_NORAD
from lib import regenerate_utils as utils
from matplotlib import pyplot as plt

import matplotlib

matplotlib.get_backend()
plt.interactive(False)

dict_norad = stack_NORAD.get_gm_stack()

# Truncate values over 1 to 1, and under 0 to 0
dict_norad['stack'][dict_norad['stack'] < 0 ] = 0
dict_norad['stack'][dict_norad['stack'] > 1 ] = 1

media_imagen_false = utils.get_mean_over_samples_images(dict_norad, 0)
media_imagen_pos = utils.get_mean_over_samples_images(dict_norad, 1)

media_3d_false = utils.reconstruct_3d_image(media_imagen_false,
                                      dict_norad['voxel_index'], dict_norad['imgsize'])

media_3d_pos = \
    utils.reconstruct_3d_image(media_imagen_pos, dict_norad['voxel_index'], dict_norad['imgsize'])

img_index = 40
sample_pos = utils.reconstruct_3d_image(dict_norad['stack'][-img_index,:], dict_norad['voxel_index'], dict_norad['imgsize'])
sample_neg = utils.reconstruct_3d_image(dict_norad['stack'][img_index,:], dict_norad['voxel_index'], dict_norad['imgsize'])

index = 77
plt.figure(1)
plt.imshow(media_3d_false[:,index,:], cmap="Greys")
plt.show(block=False)

plt.figure(2)
plt.imshow(media_3d_pos[:,index,:], cmap="Greys")
plt.show(block=False)

plt.figure(3)
plt.imshow(sample_pos[:,index,:], cmap="Greys")
plt.show(block=False)

plt.figure(4)
plt.imshow(sample_neg[:,index,:], cmap="Greys")
plt.show()
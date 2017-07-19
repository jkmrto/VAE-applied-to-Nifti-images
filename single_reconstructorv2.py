# Reconstructor over all images

import os
import settings
import tensorflow as tf
import numpy as np
from lib.data_loader import mri_atlas
from lib.aux_functionalities.os_aux import create_directories
from lib import utils
from lib.vae import VAE
from lib import session_helper as session
from lib import regenerate_utils
from lib import regenerate_utils
from matplotlib import pyplot as plt
from lib.data_loader import MRI_stack_NORAD

iden_session = "02_06_2017_23:20_arch:_1000_800_500_200"
test_name = "Encoding session"
regions_used = "all"
max_iter = 1500
latent_layer_dim = 200
bool_norm_truncate = True
bool_normalize_per_ground_images = False

# LOADING THE DATA
dict_norad = MRI_stack_NORAD.get_gm_stack()
patient_label = dict_norad['labels']
list_regions = session.select_regions_to_evaluate(regions_used)
atlas_mri = mri_atlas.load_atlas_mri()

# DIRECTORY TO THE NET SAVED
path_to_session = os.path.join(settings.path_to_general_out_folder,
                               iden_session)
path_to_meta_folder = os.path.join(path_to_session, "meta")
path_to_images_generated = os.path.join(path_to_session,
                                        "images_regenerated")

create_directories([path_to_images_generated])

if bool_norm_truncate:
    dict_norad['stack'][dict_norad['stack'] < 0] = 0
    dict_norad['stack'][dict_norad['stack'] > 1] = 1

img_index = 40
sample_pos = dict_norad['stack'][-img_index,:]
sample_neg = dict_norad['stack'][img_index,:]

images_reconstructed = np.zeros([dict_norad['stack'].shape[0], dict_norad['stack'].shape[1]])

for region_selected in list_regions:

    region_voxels_index = atlas_mri[region_selected]

    region_voxels_values = dict_norad['stack'][:, region_voxels_index]

    if bool_normalize_per_ground_images:
        _, max_denormalize = utils.normalize_array(region_voxels_values)

    print("Region {} selected".format(region_selected))

    suffix = 'region_' + str(region_selected)
    savefile = os.path.join(path_to_meta_folder,
                            suffix + "-{}".format(max_iter))
    metafile = os.path.join(path_to_meta_folder,
                            suffix + "-{}.meta".format(max_iter))
    print("Loading the file {}".format(metafile))

    tf.reset_default_graph()
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph(metafile)
    new_saver.restore(sess, savefile)

    v = VAE.VAE(meta_graph=savefile)

    images_reconstructed[:, region_voxels_index] = v.vae(region_voxels_values)

img_index=77
for index in range(0, images_reconstructed.shape[0], 1):

    image_reconstructed_3d  = \
        regenerate_utils.reconstruct_3d_image(images_reconstructed[index, :],
                                              dict_norad['voxel_index'], dict_norad['imgsize'])

    plt.figure()
    plt.imshow(image_reconstructed_3d[:,77,:], cmap="Greys")
    plt.title("image reconstructed")
    plt.savefig(path_to_images_generated + "/{}_patient.png".format(index))


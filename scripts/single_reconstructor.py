import os
import settings
import tensorflow as tf
import numpy as np
from lib.mri import mri_atlas
from lib.aux_functionalities.os_aux import create_directories
from lib import utils
from lib.vae import VAE
from lib import session_helper as session
from lib import regenerate_utils
from lib import regenerate_utils
from matplotlib import pyplot as plt
from lib.mri import stack_NORAD


def evaluate_cubes_difference_by_planes(cube1, cube2, bool_test=False):

   if bool_test:
        print(cube1.shape)
        print(cube2.shape)

   cube_dif = cube1-cube2
   cube_dif = cube_dif.__abs__()

   v1 = cube_dif.sum(axis=2).sum(axis=1)
   v2 = cube_dif.sum(axis=2).sum(axis=0)
   v3 = cube_dif.sum(axis=0).sum(axis=0)

   return np.argmax(v1), np.argmax(v2), np.argmax(v3)


# SESSION SETTINGS
#iden_session = "01_06_2017_18:35 arch: 1000_500_100"
#iden_session = "02_06_2017_18:24 arch: 1000_500_100"
# = "02_06_2017_19:01 arch: 1000_500_100"
iden_session = "02_06_2017_23:20 arch: 1000_800_500_200"
test_name = "Encoding session"
regions_used = "all"
max_iter = 1500
latent_layer_dim = 200
bool_norm_truncate = True
bool_normalize_per_ground_images = False

# LOADING THE DATA
dict_norad = stack_NORAD.get_gm_stack()
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

f_reconstructed = np.zeros((1, dict_norad['stack'].shape[1]))
p_reconstructed = np.zeros((1, dict_norad['stack'].shape[1]))

for region_selected in list_regions:

    region_voxels_index = atlas_mri[region_selected]

    region_voxels_values = dict_norad['stack'][:, region_voxels_index]

    if bool_normalize_per_ground_images:
        _, max_denormalize = utils.normalize_array(region_voxels_values)

    print("Region {} selected".format(region_selected))

    if bool_normalize_per_ground_images:
        f_reg_voxels_values = sample_neg[region_voxels_index]/max_denormalize
        p_reg_voxels_values = sample_pos[region_voxels_index]/max_denormalize
    else:
        f_reg_voxels_values = sample_neg[region_voxels_index]
        p_reg_voxels_values = sample_pos[region_voxels_index]

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

    print("Coding images")
    f_code_reg = v.encode(np.column_stack(f_reg_voxels_values))[0]  # [mu, sigma]
    p_code_reg = v.encode(np.column_stack(p_reg_voxels_values))[0] # selecting mu

    f_code_reg = np.reshape(f_code_reg, [1, latent_layer_dim])
    p_code_reg = np.reshape(p_code_reg, [1, latent_layer_dim])
    print(f_code_reg.shape)
    print(p_code_reg.shape)
    print(np.concatenate([np.row_stack(f_code_reg), np.row_stack(p_code_reg)], axis=0))
    code_reg = np.concatenate([f_code_reg, p_code_reg, f_code_reg-p_code_reg], axis=1)

    region_output = v.decode(zs=code_reg)
    print(region_output.shape)
    if bool_normalize_per_ground_images:
        f_reconstructed[0, region_voxels_index] = region_output[0,:] * max_denormalize
        p_reconstructed[0, region_voxels_index] = region_output[1,:] * max_denormalize
    else:
        f_reconstructed[0, region_voxels_index] = region_output[0, :]
        p_reconstructed[0, region_voxels_index] = region_output[1, :]

f_reconstructed_3d  = regenerate_utils.reconstruct_3d_image(f_reconstructed, dict_norad['voxel_index'], dict_norad['imgsize'])
p_reconstructed_3d =  regenerate_utils.reconstruct_3d_image(p_reconstructed, dict_norad['voxel_index'], dict_norad['imgsize'])
sample_neg_3d = regenerate_utils.reconstruct_3d_image(sample_neg, dict_norad['voxel_index'], dict_norad['imgsize'])
sample_pos_3d = regenerate_utils.reconstruct_3d_image(sample_pos, dict_norad['voxel_index'], dict_norad['imgsize'])


index_1, index_2, index_3 = evaluate_cubes_difference_by_planes(
    f_reconstructed_3d, p_reconstructed_3d, bool_test=False)

plt.figure(1)
plt.imshow(f_reconstructed_3d[index_1,:,:], cmap="Greys")
plt.title("false image reconstructed")
plt.show(block=False)

plt.figure(2)
plt.imshow(p_reconstructed_3d[index_1,:,:], cmap="Greys")
plt.title("positive image reconstructed")
plt.show(block=False)

plt.figure(3)
plt.imshow(f_reconstructed_3d[:,index_2,:], cmap="Greys")
plt.title("false image reconstructed")
plt.show(block=False)

plt.figure(4)
plt.imshow(p_reconstructed_3d[:,index_2,:], cmap="Greys")
plt.title("positive image reconstructed")
plt.show(block=False)

plt.figure(5)
plt.imshow(f_reconstructed_3d[:,:,index_3], cmap="Greys")
plt.title("false image reconstructed")
plt.show(block=False)

plt.figure(6)
plt.imshow(p_reconstructed_3d[:,:,index_3], cmap="Greys")
plt.title("positive image reconstructed")
plt.show(block=False)

img_index=77
plt.figure(7)
plt.imshow(f_reconstructed_3d[:,77,:], cmap="Greys")
plt.title("false image reconstructed")
plt.show(block=False)

plt.figure(8)
plt.imshow(p_reconstructed_3d[:,77,:], cmap="Greys")
plt.title("positive image reconstructed")
plt.show(block=True)
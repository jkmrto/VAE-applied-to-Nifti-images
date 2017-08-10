import os

import numpy as np
import tensorflow as tf

import settings
from lib import regenerate_utils
from lib import session_helper as session
from lib.data_loader import MRI_stack_NORAD
from lib.data_loader import mri_atlas
from lib.utils.os_aux import create_directories
from lib.vae import VAE

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

path_to_images_generated_own_folder = os.path.join(path_to_images_generated,
                                                   "reconstructor_extrapoling_over_2images")

create_directories([path_to_images_generated, path_to_images_generated_own_folder])

if bool_norm_truncate:
    dict_norad['stack'][dict_norad['stack'] < 0] = 0
    dict_norad['stack'][dict_norad['stack'] > 1] = 1

img_index = 40

ad_image = dict_norad['stack'][-img_index,:]
nor_image = dict_norad['stack'][img_index,:]

#nor++, nor+, nor, ad, ad+, ad++
images_reconstructed = np.zeros([8, dict_norad['stack'].shape[1]])

for region_selected in list_regions:

    region_voxels_index = atlas_mri[region_selected]

    region_nor_images = nor_image[region_voxels_index]
    region_ad_images = ad_image[region_voxels_index]

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

    print("encoding")
    nor_code_reg = v.encode(np.column_stack(region_nor_images))[0]  # [mu, sigma]
    ad_code_reg = v.encode(np.column_stack(region_ad_images))[0] # selecting mu

#    nor_code_reg = np.reshape(nor_code_reg, [1, latent_layer_dim])
#    ad_code_reg = np.reshape(ad_code_reg, [1, latent_layer_dim])

    nor_code_reg = np.reshape(nor_code_reg, [1, latent_layer_dim])
    ad_code_reg = np.reshape(ad_code_reg, [1, latent_layer_dim])

    nor_ad_dif_code = ad_code_reg - nor_code_reg

    code_reg = np.concatenate([
        nor_code_reg - 10 * nor_ad_dif_code,
        nor_code_reg - 5 * nor_ad_dif_code,
                               nor_code_reg - nor_ad_dif_code,
                               nor_code_reg, ad_code_reg,
                               ad_code_reg + nor_ad_dif_code,
                               ad_code_reg + 5 * nor_ad_dif_code,
                               ad_code_reg + 10 * nor_ad_dif_code], axis=0)

    print(code_reg.shape)
    region_output = v.decode(zs=code_reg)
    images_reconstructed[:, region_voxels_index]  = region_output


labels = ["nor10+",
    "nor+++++",
          "nor+",
          "nor",
          "ad",
          "ad+",
          "ad+++++",
          "ad10+"]

for index in range(0, images_reconstructed.shape[0], 1):

    regenerate_utils.plot_and_save_fig(
        images_reconstructed[index, :], dict_norad,
        path_to_image=os.path.join(path_to_images_generated_own_folder,
            "{}.png".format(labels[index])),
        title=labels[index], index_section_to_plot=77,
        path_to_3d_reconstruction = os.path.join(
            path_to_images_generated_own_folder,
            "{}.nii.gz".format(labels[index])))



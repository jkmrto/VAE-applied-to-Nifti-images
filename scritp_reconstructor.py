import os

import numpy as np
import tensorflow as tf

import settings
from lib import regenerate_utils
from lib import session_helper as session
from lib import utils
from lib.data_loader import MRI_stack_NORAD
from lib.data_loader import mri_atlas
from lib.utils.os_aux import create_directories
from lib.vae import VAE

iden_session = "bueno_05_05_2017_08:19 arch: 1000_800_500_100"
test_name = "Encoding session"
regions_used = "all"
max_iter = 1500
latent_layer_dim = 100
n_intervals = 10
norm_truncate = False

dict_norad = MRI_stack_NORAD.get_gm_stack()
patient_label = dict_norad['labels']
list_regions = session.select_regions_to_evaluate(regions_used)
atlas_mri = mri_atlas.load_atlas_mri()

# Directory to the net saved
path_to_session = os.path.join(settings.path_to_general_out_folder,
                               iden_session)
path_to_meta_folder = os.path.join(path_to_session, "meta")
path_to_images_generated = os.path.join(path_to_session,
                                        "images_regenerated")
create_directories([path_to_images_generated])

# Truncate values over 1 to 1, and under 0 to 0,
# because each voxels indicates the probability
# of having or not grey matter, so the inteval
# should be [0....1]

if norm_truncate:
    dict_norad['stack'][dict_norad['stack'] < 0 ] = 0
    dict_norad['stack'][dict_norad['stack'] > 1 ] = 1

mean_mri_image_false_patients = \
    regenerate_utils.get_mean_over_samples_images(dict_norad, 0)
mean_mri_image_positive_patient = \
    regenerate_utils.get_mean_over_samples_images(dict_norad, 1)

# init matriz to store results
#[nºregions *  dim_latent_layer]
output = np.zeros(((n_intervals + 1), dict_norad['stack'].shape[1]))

for region_selected in list_regions:

    print("Region {} selected".format(region_selected))

    region_voxels_index = atlas_mri[region_selected]
    f_reg_voxels_values = mean_mri_image_false_patients[region_voxels_index]
    p_reg_voxels_values = mean_mri_image_positive_patient[region_voxels_index]

    f_reg_voxels_values, max_denormalize_f = \
        utils.normalize_array(f_reg_voxels_values)

    p_reg_voxels_values, max_denormalize_p = \
        utils.normalize_array(p_reg_voxels_values)

    noramalize = max([max_denormalize_f, max_denormalize_p])

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
    difplus = (p_code_reg - f_code_reg)/n_intervals

    difplus = difplus.flatten()

    matriz_latent_codes = \
        np.array([f_code_reg.flatten() + incr * difplus for incr in
                  range(0, n_intervals + 1, 1)])

    region_output = v.decode(zs=matriz_latent_codes[:,:])

    output[:, region_voxels_index] = region_output * noramalize


for index in range(0, n_intervals + 1, 1):
    mri_imag_3d = regenerate_utils.reconstruct_3d_image(output[index,:],
        dict_norad['voxel_index'], dict_norad['imgsize'])

    path_to_image = os.path.join(path_to_images_generated, "{}.png".format(index))
    regenerate_utils.plot_and_save_mri_section(mri_imag_3d, 77,
                                               path_to_image)
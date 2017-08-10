import os

import numpy
import tensorflow as tf

import settings
from lib import session_helper as session
from lib import utils
from lib.data_loader import MRI_stack_NORAD
from lib.data_loader import mri_atlas
from lib.utils.os_aux import create_directories
from lib.vae import VAE

iden_session = "02_05_2017_21:09 arch: 1000_800_500_100_2"
test_name = "Encoding session"
regions_used = "most important"
max_iter = 1500

path_to_session = os.path.join(settings.path_to_general_out_folder,
                               iden_session)
path_to_meta_folder = os.path.join(path_to_session, "meta")
path_to_main_test = os.path.join(path_to_session, "post_train")
path_to_particular_test = os.path.join(path_to_main_test, test_name)
path_to_encoding_storage_folder = os.path.join(path_to_particular_test,
                                               "encoding_data")

create_directories([path_to_main_test, path_to_particular_test,
                    path_to_encoding_storage_folder])

dict_norad = MRI_stack_NORAD.get_gm_stack()  # 'stack' 'voxel_index' 'labels'
region_voxels_label = dict_norad['labels']

list_regions = session.select_regions_to_evaluate(regions_used)

for region_selected in list_regions:
    encoding_mean_file_saver = os.path.join(path_to_encoding_storage_folder,
                                            "region {}_means.txt".format(
                                                region_selected))
    encoding_desv_file_saver = os.path.join(path_to_encoding_storage_folder,
                                            "region {}_desv.txt".format(
                                                region_selected))

    region_voxels_index = mri_atlas.load_atlas_mri()[region_selected]
    region_voxels_values = dict_norad['stack'][:, region_voxels_index]
    region_voxels_values, max_denormalize = utils.normalize_array(
        region_voxels_values)

    X_train = region_voxels_values
    Y_train = region_voxels_label

    suffix = 'region_' + str(region_selected)
    savefile = os.path.join(path_to_meta_folder, suffix + "-{}".format(max_iter))
    metafile = os.path.join(path_to_meta_folder, suffix + "-{}.meta".format(max_iter))
    print("Loading the file {}".format(metafile))

    tf.reset_default_graph()
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph(metafile)
    new_saver.restore(sess, savefile)

    # Hyperparameters and architecture is not used in loading setup
    v = VAE.VAE(meta_graph=savefile)

    print("Coding training data")
    code_train = v.encode(X_train)  # [mu, sigma]

    numpy.savetxt(encoding_mean_file_saver, code_train[0], delimiter=',')
    numpy.savetxt(encoding_desv_file_saver, code_train[1], delimiter=',')

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from lib.utils.os_aux import create_directories
from lib.data_loader.pet_loader import load_pet_data_flat
from lib.data_loader import PET_stack_NORAD
from lib.data_loader import utils_mask3d
from lib.data_loader import pet_atlas
from datetime import datetime
from lib.vae import VAE
import tensorflow as tf
import settings


def get_pet_region_required_data(region):

    print("Loading Pet Atlas")
    atlas = pet_atlas.load_atlas()
    reshape_kind = "F"

    print("Loading Pet stack")
    stack_dict = PET_stack_NORAD.get_parameters()

    total_size = stack_dict['total_size']
    imgsize = stack_dict['imgsize']
    voxels_index = stack_dict['voxel_index']
    map_region_voxels = atlas[region]  # index refered to nbground voxels
    no_bg_region_voxels_index = voxels_index[map_region_voxels]

    print("Getting the Region Dimensions")
    mask3d = utils_mask3d.generate_region_3dmaskatlas(
        no_bg_region_voxels_index=no_bg_region_voxels_index,
        reshape_kind=reshape_kind,
        imgsize=imgsize,
        totalsize=total_size)

    minidx, maxidx = utils_mask3d.delim_3dmask(mask3d)

    dim_x = maxidx[0] - minidx[0] + 1
    dim_y = maxidx[1] - minidx[2] + 1
    dim_z = maxidx[1] - minidx[2] + 1
    brain_size  = imgsize
    region_size = [dim_x, dim_y, dim_z]
    return brain_size, region_size, no_bg_region_voxels_index


# VAE SETTINGS
# Net Configuration
hyperparams = {
    "batch_size": 64,
    "learning_rate": 1E-5,
    "dropout": 0.90,
    "lambda_l2_reg": 1E-5,
    "nonlinearity": tf.nn.elu,
    "squashing": tf.nn.sigmoid,
}

# Vae session cofiguration
session_conf = {
    "bool_normalized": True,
    "max_iter": 10000,
    "save_meta_bool": False,
    "show_error_iter": 100,
}

region = 3
after_input_architecture = [1000, 500, 100]  # no include hidden layer

# Path settings
own_datetime = datetime.now().strftime(r"%d_%m_%_Y_%H:%M")
session_prefix_name = "vae_reconstruction_evolution_region_{}".format(region)
session_name = session_prefix_name + "_" + own_datetime
path_to_session = os.path.join(settings.path_to_general_out_folder, session_name)
create_directories([path_to_session])

brain_size, region_size, region_voxels_location = \
    get_pet_region_required_data(region)
print(region_size)

dic_regions_to_flatten_voxels_pet, patient_labels, n_samples = \
    load_pet_data = load_pet_data_flat([region])

region_voxels_values = dic_regions_to_flatten_voxels_pet[region]
architecture = [region_voxels_values.shape[1]]
architecture.extend(after_input_architecture)

sgd_3Dimages = {
    "full_brain_size": brain_size,
    "sample": 1,
    "region_size": region_size,
    "voxels_location": region_voxels_location,
    "reshape_kind": "F",
}

tf.reset_default_graph()
v = VAE.VAE(architecture, hyperparams,
            path_to_session=path_to_session)

v.train(dic_regions_to_flatten_voxels_pet[region],
        max_iter=session_conf["max_iter"],
        suffix_files_generated="region_{}".format(region),
        iter_to_save=500,
        iters_to_show_error=session_conf['show_error_iter'],
        save_bool=session_conf["save_meta_bool"],
        sgd_3dimages=sgd_3Dimages)

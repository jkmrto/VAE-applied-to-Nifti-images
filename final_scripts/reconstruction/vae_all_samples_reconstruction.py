import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from lib import session_helper as session
from lib.utils.os_aux import create_directories
from lib.data_loader import pet_loader
from lib.data_loader import PET_stack_NORAD
from lib.data_loader import pet_atlas
from lib.vae import VAE
import numpy as np
import tensorflow as tf
import settings
from lib import reconstruct_helpers as recons
from lib.utils import output_utils as output
from lib.reconstruct_from_flat_utils import \
    reconstruct_from_flat_regions_to_full_3d_brain

explicit_iter_per_region = {}

# Configuration
logs = True
max_iters = 2000
images_used = "PET"

# Regions loader
regions_used = "all"
list_regions = session.select_regions_to_evaluate(regions_used)

# Directories Initialization
session_name = "vae_main_2000iters"
path_session = os.path.join(settings.path_to_general_out_folder, session_name)
path_meta = os.path.join(path_session, "meta")
path_images = os.path.join(path_session, "images")
path_3dsamples = os.path.join(path_images, "3d_samples_reconstruction_and_original")
path_section_compare = os.path.join(path_images,"section_samples_reconstructionVSoriginal")
create_directories([path_images, path_3dsamples, path_section_compare])

# Loading Data
stack_region_to_voxels, patient_labels, n_samples = \
    pet_loader.load_pet_data_flat(list_regions)

dic_params = PET_stack_NORAD.get_parameters()
totalsize = dic_params["total_size"]
imgsize = dic_params["imgsize"]

atlas = pet_atlas.load_atlas()

index_nobg_voxels = dic_params["voxel_index"]

list_regions = session.select_regions_to_evaluate(regions_used)
reconstruction_per_region = {}


reconstruction_per_region = {}
for region_selected in list_regions:
    print("Region {} selected".format(region_selected))

    iters = session.get_adequate_number_iterations(
        region_selected, explicit_iter_per_region, max_iters)

    index_nobg_voxels_region = index_nobg_voxels[atlas[region_selected]]

    # Loading Meta graph files
    meta_region_file = "region_{0}-{1}".format(region_selected, iters)
    path_meta_region = os.path.join(path_meta, meta_region_file)
    tf.reset_default_graph()
    vae = VAE.VAE(meta_graph=path_meta_region)

    print("Encoding")
    encoding_out = vae.encode(stack_region_to_voxels[region_selected])

    data_to_decode = encoding_out["mean"]

    decode_out = vae.decode(data_to_decode)

    reconstruction_per_region[region_selected] = decode_out

# reshape to 3d images
whole_reconstruction = reconstruct_from_flat_regions_to_full_3d_brain(
    reconstruction_per_region, images_used)
origin_image = reconstruct_from_flat_regions_to_full_3d_brain(
    stack_region_to_voxels, images_used)

for i in range(0, n_samples):
    path_3D_original_image = os.path.join(path_3dsamples, "sample:{}_original".format(i))
    path_3D_reconstruction_image = os.path.join(path_3dsamples, "sample:{}_reconstruction".format(i))
    path_section_compare_image = os.path.join(path_section_compare, "sample:{}".format(i))

    output.from_3d_image_to_nifti_file(
        path_3D_reconstruction_image, whole_reconstruction[i, :, :, :])
    output.from_3d_image_to_nifti_file(
        path_3D_original_image, origin_image[i, :, :, :])

    recons.plot_section_indicated(
        img3d_1=whole_reconstruction[i, :, :, :],
        img3d_2=origin_image[i, :, :, :],
        p1=settings.planos_hipocampo["p1"],
        p2=settings.planos_hipocampo["p2"],
        p3=settings.planos_hipocampo["p3"],
        path_to_save_image=path_section_compare_image,
        cmap=cmap)



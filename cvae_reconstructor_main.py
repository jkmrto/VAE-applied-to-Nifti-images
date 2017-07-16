import os

import numpy as np
import tensorflow as tf
import settings
from lib import session_helper as session
from lib.nifti_regions_loader import \
    load_pet_data_3d, load_mri_data_3d
from lib.vae import CVAE


def get_mean_3d_images_over_samples(region_to_3dimg_dict_pet):
    """

    :param region_to_3dimg_dict_pet: dict[region] -> 3d_image sh[n_samples, w, h, d]
    :return: dict[region]-> np.array 3d_mean_image sh[2, w, h, d]
                         -> array with the mean image negative pos 0, positive pos 1
    """
    region_to_class_to_3d_means_imgs = {}

    for region, cube_images in region_to_3dimg_dict_pet.items():
        class_to_3d_means_imgs = np.zeros([2, cube_images.shape[1],
                                          cube_images.shape[2], cube_images.shape[3]])

        index_to_selected_images = patient_labels == 0
        index_to_selected_images = index_to_selected_images.flatten()
        class_to_3d_means_imgs[0] = \
            cube_images[index_to_selected_images.tolist(), :, :, :].mean(axis=0)

        index_to_selected_images = patient_labels == 1
        index_to_selected_images = index_to_selected_images.flatten()
        class_to_3d_means_imgs[1] = \
            cube_images[index_to_selected_images, :, :, :].mean(axis=0)

        region_to_class_to_3d_means_imgs[region] = class_to_3d_means_imgs

    return region_to_class_to_3d_means_imgs



# Meta settings
#session_name = "test_saving_meta_PET_11_07_2017_15:15"
session_name = "test_saving_meta_PET_15_07_2017_21:34"
#images_used = "MRI"
images_used = "PET"

#vae_used = "dense_vae"
vae_used = "conv_vae"
iters = 100

regions_used = "all"
list_regions = session.select_regions_to_evaluate(regions_used)
path_session = os.path.join(settings.path_to_general_out_folder, session_name)
path_meta = os.path.join(path_session, "meta")
print(path_meta)


# Loading dataç
logs = False
n_samples=0
patient_labels = []
region_to_3dimg_dict_pet = None

if images_used == "PET":
    print("Loading Pet images")
    region_to_3dimg_dict_pet, patient_labels, n_samples = \
        load_pet_data_3d(list_regions)

elif images_used == "MRI":
    region_to_3dimg_dict_mri_gm, region_to_3dimg_dict_mri_wm,\
    patient_labels, n_samples = load_mri_data_3d(list_regions)

region_to_class_to_3d_means_images_pet = \
    get_mean_3d_images_over_samples(region_to_3dimg_dict_pet)


for region in list_regions:

    meta_region_file = "region_{0}-{1}".format(region, iters)
    path_meta_region = os.path.join(path_meta, meta_region_file)
    tf.reset_default_graph()

    # CVAE encoding
    hyperparams = {}
    hyperparams['image_shape'] = region_to_class_to_3d_means_images_pet[region].shape[1:]
    cvae = CVAE.CVAE(hyperparams=hyperparams, meta_path=path_meta_region)

    # encoding_images
    encoding_out = cvae.encode(region_to_class_to_3d_means_images_pet[region])

    if logs:
        print("Shape enconding_out mean {}".format(encoding_out["mean"].shape))

    images_3d_regenerated = cvae.decoder(latent_layer_input=encoding_out["mean"],
            original_images=region_to_class_to_3d_means_images_pet[region])

    if logs:
        print("images regenerated shape {}".format(images_3d_regenerated.shape))
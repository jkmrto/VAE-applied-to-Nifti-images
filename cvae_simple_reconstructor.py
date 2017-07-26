# PET test
# Testing reconstruction over only one sample

import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import tensorflow as tf
import settings
from lib import session_helper as session
from lib import reconstruct_helpers as recons
from lib.data_loader import utils_images3d
from lib.utils import output_utils as output
from lib.vae import CVAE

def get_adequate_number_iterations(region_selected, explicit_iter_per_region,
                                   predefined_iters):
    if region_selected in explicit_iter_per_region.keys():
        if explicit_iter_per_region[region_selected] < predefined_iters:
            max_train_iter = explicit_iter_per_region[region_selected]
        else:
            max_train_iter = predefined_iters
    else:
        max_train_iter = predefined_iters

    return max_train_iter


explicit_iter_per_region = {
    73: 300,
}

path_image = "simple_individual_reconstruction.png"
#AD 123
#NOR 22
logs = True
sample = 30
regions_used = "all"
session_name = "cvae_create_meta_nets_iter_500_26_07_2017_20:15"

images_used = "PET"

#vae_used = "dense_vae"

max_iters = 500

list_regions = session.select_regions_to_evaluate(regions_used)
path_session = os.path.join(settings.path_to_general_out_folder, session_name)
path_meta = os.path.join(path_session, "meta")
print(path_meta)

stack_region_to_3dimg, patient_labels, n_samples, cmap = \
    recons.load_desired_stacked_and_parameters(images_used, list_regions, )

selected_sample_3d_per_region = {}

origin_images_to_encode = {}

# selecting one sample repeated:
for region, cube_images in stack_region_to_3dimg.items():
    class_to_3d_means_imgs = np.zeros([2, cube_images.shape[1],
                                       cube_images.shape[2],
                                       cube_images.shape[3]])

    class_to_3d_means_imgs[0, :, :, :] = cube_images[sample, :, :, :]
    class_to_3d_means_imgs[1, :, :, :] = cube_images[sample, :, :, :]

    origin_images_to_encode[region] = class_to_3d_means_imgs

reconstruction_per_region = {}
for region in list_regions:

    iters = get_adequate_number_iterations(
        region_selected=region,
        explicit_iter_per_region = explicit_iter_per_region,
        predefined_iters = max_iters)

    print("region {} selected".format(region))
    meta_region_file = "region_{0}-{1}".format(region, iters)
    path_meta_region = os.path.join(path_meta, meta_region_file)
    tf.reset_default_graph()

    # CVAE encoding
    hyperparams = {}
    hyperparams['image_shape'] = origin_images_to_encode[region].shape[1:]
    cvae = CVAE.CVAE(hyperparams=hyperparams, meta_path=path_meta_region)

    # encoding_images
    print("Encoding")
    encoding_out = cvae.encode(origin_images_to_encode[region])

    data_to_decode = encoding_out["mean"]

    if logs:
        print("Shape enconding_out mean {}".format(data_to_decode.shape))

    print("Decoding")
    images_3d_regenerated = cvae.decoder(latent_layer_input=data_to_decode,
            original_images=origin_images_to_encode[region])

    reconstruction_per_region[region] = images_3d_regenerated
    if logs:
        print("images regenerated shape {}".format(images_3d_regenerated.shape))


print("Mapping Reconstructing images")
whole_reconstruction = \
    utils_images3d.map_region_segmented_over_full_image(reconstruction_per_region, images_used)
print("Mapping Reconstructing images ended")


print("Mapping Reconstructing images")
origin_image = \
    utils_images3d.map_region_segmented_over_full_image(origin_images_to_encode, images_used)

output.from_3d_image_to_nifti_file(path_to_save="example_neg",
                                   image3d=whole_reconstruction[0, :, :, :])

output.from_3d_image_to_nifti_file(path_to_save="example_pos",
                                   image3d=origin_image[0, :, :, :])

recons.plot_most_discriminative_section(
    img3d_1=whole_reconstruction[0, :, :, :],
    img3d_2=origin_image[0, :, :, :],
    path_to_save_image=path_image,
    cmap=cmap)

if logs:
    evaluate_difference_full_image = whole_reconstruction[0, :, :, :].flatten() \
                                     - origin_image[0, :, :, :].flatten()
    total_difference = sum(abs(evaluate_difference_full_image))
    print("Total difference between images reconstructed {0}".format(total_difference))
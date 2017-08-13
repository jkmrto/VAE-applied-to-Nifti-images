# PET test
# Testing reconstruction over only one sample
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import settings
from lib import session_helper as session
from lib import reconstruct_helpers as recons
from lib.data_loader import utils_images3d
from lib.utils import output_utils as output
from lib.vae import CVAE_supervised
from lib.utils.os_aux import create_directories

explicit_iter_per_region = {
    73: 300,
    74: 200,
}

extraploate_index = 5

session_name = "cvae_supevised_create_meta_nets_layer_500_iters_06_08_2017_14:58"
logs = True

# SAMPLES SELECTED
sample_NOR = 10
sample_AD = 120
samples_indexes = [sample_NOR, sample_AD]


regions_used = "all"
list_regions = session.select_regions_to_evaluate(regions_used)

images_used = "PET"
max_iters = 500

path_session = os.path.join(settings.path_to_general_out_folder, session_name)
path_meta = os.path.join(path_session, "meta")
path_images = os.path.join(path_session, "images")
path_reconstruction_images = os.path.join(path_images,
    "simple_reconstructor_with_extrapoling_{}".format(extraploate_index
                                                      ))
create_directories([path_reconstruction_images])


stack_region_to_3dimg, patient_labels, n_samples, cmap = \
    recons.load_desired_stacked_and_parameters(images_used, list_regions, )

# Selecting 3d images and labels
origin_images_to_encode = \
    utils_images3d.get_samples_in_stacked_region_to_3dsegmented_region(
    stack_region_to_3dimg=stack_region_to_3dimg,
    samples_indexes=samples_indexes
)
labels_selected = patient_labels[samples_indexes]

reconstruction_per_region = {}
for region in list_regions:

    iters = session.get_adequate_number_iterations(
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
    cvae = CVAE_supervised.CVAE(hyperparams=hyperparams, meta_path=path_meta_region)

    # encoding_images
    print("Encoding")
    encoding_out = cvae.encode(origin_images_to_encode[region])

    data_to_decode = encoding_out["mean"]

    # EXTRAPOLING OVER LATENT CODE
    latent_code_dif_rate = data_to_decode[1, :] - data_to_decode[0, :]
    data_to_decode[1, :] += latent_code_dif_rate * extraploate_index
    data_to_decode[0, :] += (-latent_code_dif_rate * extraploate_index)

    if logs:
        print("Shape enconding_out mean {}".format(data_to_decode.shape))

    print("Decoding")
    images_3d_regenerated = cvae.decoder(
        latent_layer_input=data_to_decode,
        original_images=origin_images_to_encode[region],
        labels=labels_selected
    )

    reconstruction_per_region[region] = images_3d_regenerated
    if logs:
        print("images regenerated shape {}".format(images_3d_regenerated.shape))


print("Mapping Reconstructed images")
whole_reconstruction = \
    utils_images3d.map_region_segmented_over_full_image(reconstruction_per_region,
                                                        images_used)
origin_image = \
    utils_images3d.map_region_segmented_over_full_image(origin_images_to_encode,
                                                        images_used)
recons_NOR = whole_reconstruction[0, :, :, :]
recons_AD = whole_reconstruction[1, :, :, :]

original_NOR = origin_image[0, :, :, :]
original_AD = origin_image[1, :, :, :]

output.from_3d_image_to_nifti_file(path_to_save="example_neg",
                                   image3d=whole_reconstruction[0, :, :, :])

output.from_3d_image_to_nifti_file(path_to_save="example_pos",
                                   image3d=origin_image[0, :, :, :])

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import matplotlib
matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
import settings
from lib import session_helper as session
from lib import reconstruct_helpers as recons
from lib.data_loader import utils_images3d
from lib.utils import output_utils as output
from lib.vae import CVAE
from lib.utils.os_aux import create_directories


explicit_iter_per_region = {
    73: 200,
    44: 800,
}

logs = True
max_iters = 2000
images_used = "PET"

# Region to use
regions_used = "all"
list_regions = session.select_regions_to_evaluate(regions_used)

# Directories Initialization
session_name = "main_cvae_net"
path_session = os.path.join(settings.path_to_general_out_folder, session_name)
path_meta = os.path.join(path_session, "meta")
path_images = os.path.join(path_session, "images")
path_3dsamples = os.path.join(path_images, "3d_samples_reconstruction_and_original")
path_section_compare = os.path.join(path_images,"section_samples_reconstructionVSoriginal")
create_directories([path_images, path_3dsamples, path_section_compare])

# Loadin Data
stack_region_to_3dimg, patient_labels, n_samples, cmap = \
    recons.load_desired_stacked_and_parameters(images_used, list_regions)

reconstruction_per_region = {}
for region in list_regions:

    iters = session.get_adequate_number_iterations(
        region, explicit_iter_per_region, max_iters)

    print("region {} selected".format(region))
    meta_region_file = "region_{0}-{1}".format(region, iters)
    path_meta_region = os.path.join(path_meta, meta_region_file)
    tf.reset_default_graph()

    # CVAE encoding
    hyperparams = {}
    hyperparams['image_shape'] = stack_region_to_3dimg[region].shape[1:]
    cvae = CVAE.CVAE(hyperparams=hyperparams, path_meta_graph=path_meta_region)
    cvae.generate_meta_net()

    # encoding_images
    print("Encoding")
    encoding_out = cvae.encode(stack_region_to_3dimg[region])

    data_to_decode = encoding_out["mean"]

    if logs:
        print("Shape enconding_out mean {}".format(data_to_decode.shape))

    print("Decoding")
    images_3d_regenerated = cvae.decoder(
        latent_layer_input=data_to_decode,
        original_images=stack_region_to_3dimg[region])

    reconstruction_per_region[region] = images_3d_regenerated
    if logs:
        print("images regenerated shape {}".format(images_3d_regenerated.shape))


print("Mapping Reconstructing images")
whole_reconstruction =\
        utils_images3d.map_region_segmented_over_full_image(
            reconstruction_per_region, images_used)
print("Mapping Reconstructing images ended")

print("Mapping Original images")
origin_image = \
        utils_images3d.map_region_segmented_over_full_image(
            stack_region_to_3dimg, images_used)
print("Mapping Original images ended")

for i in range(0, n_samples):
    path_3D_original = os.path.join(path_3dsamples, "sample:{}_original".format(i))
    path_3D_reconstruction = os.path.join(path_3dsamples, "sample:{}_reconstruction".format(i))
    path_section_compare = os.path.join(path_section_compare, "sample:{}".format(i))

    output.from_3d_image_to_nifti_file(
        path_3D_reconstruction, whole_reconstruction[i, :, :, :])
    output.from_3d_image_to_nifti_file(
        path_3D_original, origin_image[i, :, :, :])

    recons.plot_section_indicated(
        img3d_1=whole_reconstruction[i, :, :, :],
        img3d_2=origin_image[i, :, :, :],
        p1=settings.planos_hipocampo["p1"],
        p2=settings.planos_hipocampo["p2"],
        p3=settings.planos_hipocampo["p3"],
        path_to_save_image=path_section_compare,
        cmap=cmap)

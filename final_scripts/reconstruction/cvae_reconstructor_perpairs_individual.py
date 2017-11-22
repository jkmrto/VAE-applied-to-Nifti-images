import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import matplotlib

matplotlib.use('Agg')

import tensorflow as tf
import settings
from lib import session_helper as session
from lib import reconstruct_helpers as recons
from lib.data_loader import utils_images3d
from lib.utils import output_utils as output
from lib.vae import CVAE
from lib.utils import os_aux

# AD 123
# NOR 22

explicit_iter_per_region = {
    73: 200,
    44: 800,
}

patients_selected_per_class = {"NOR": 22, "AD": 123}
logs = True
regions_used = "all"
#session_name = settings.perclass_AD_session
session_name = "main_cvae_net"
folder_name_to_store_images_created = "Pair_reconstruction_comparaison"

max_iters = 2000
# images_used = "MRI_WM"
# images_used = "MRI_GM"
images_used = "PET"
list_regions = session.select_regions_to_evaluate(regions_used)

# Paths configurations
path_session = os.path.join(settings.path_to_general_out_folder, session_name)
path_meta = os.path.join(path_session, "meta")
path_images = os.path.join(path_session, "images")
path_where_store_images_generated = os.path.join(
    path_images, folder_name_to_store_images_created)

os_aux.create_directories([path_images, path_where_store_images_generated])

stack_region_to_3dimg, patient_labels, n_samples, cmap = \
    recons.load_desired_stacked_and_parameters(images_used, list_regions)

origin_images_to_encode = \
    recons.get_representatives_samples_over_region_per_patient_indexes(
        region_to_3d_images_dict=stack_region_to_3dimg,
        indexes_per_group=patients_selected_per_class)

reconstruction_per_region = {}
for region in list_regions:
    print("region {} selected".format(region))

    iters = session.get_adequate_number_iterations(
        region, explicit_iter_per_region, max_iters)

    meta_region_file = "region_{0}-{1}".format(region, iters)
    path_meta_region = os.path.join(path_meta, meta_region_file)
    tf.reset_default_graph()



    # CVAE encoding
    hyperparams = {}
    hyperparams['image_shape'] = origin_images_to_encode[region].shape[1:]
    cvae = CVAE.CVAE(hyperparams=hyperparams, path_meta_graph=path_meta_region)
    cvae.generate_meta_net()

    # encoding_images
    print("Encoding")
    encoding_out = cvae.encode(origin_images_to_encode[region])

    print("Decoding")
    images_3d_regenerated = cvae.decoder(
        latent_layer_input=encoding_out["mean"],
        original_images=origin_images_to_encode[region])

    reconstruction_per_region[region] = images_3d_regenerated
    if logs:
        print("images regenerated shape {}".format(images_3d_regenerated.shape))

print("Mapping Reconstructing images")
whole_reconstruction = \
    utils_images3d.map_region_segmented_over_full_image(
        reconstruction_per_region, images_used)
print("Mapping Reconstructing images ended")

print("Mapping Reconstructed images")
whole_reconstruction = \
    utils_images3d.map_region_segmented_over_full_image(
        reconstruction_per_region,
        images_used)
origin_image = \
    utils_images3d.map_region_segmented_over_full_image(
        origin_images_to_encode, images_used)

recons.plot_comparaision_images_ADvsNOR(whole_reconstruction, origin_image,
                                        path_where_store_images_generated, cmap)

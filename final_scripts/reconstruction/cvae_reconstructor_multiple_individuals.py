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
from lib.utils.os_aux import create_directories


patients_selected = [100, 123, 111, 88, 99, 101, 102]
number_samples_to_reconstruct = len(patients_selected)

logs = True
regions_used = "all"
folder_name_to_store_images_created = "reconstructed_multiple_individuals"
session_name = settings.perclass_AD_session
max_iters = 500

#images_used = "MRI_WM"
#images_used = "MRI_GM"
images_used = "PET"

list_regions = session.select_regions_to_evaluate(regions_used)

# Paths configurations
path_session = os.path.join(settings.path_to_general_out_folder, session_name)
path_meta = os.path.join(path_session, "meta")
path_images = os.path.join(path_session, "images")
path_where_store_images_generated = os.path.join(path_images, folder_name_to_store_images_created)
create_directories([path_images, path_where_store_images_generated])

stack_region_to_3dimg, patient_labels, n_samples, cmap = \
    recons.load_desired_stacked_and_parameters(images_used, list_regions, )

data_to_encode_per_region = \
    recons.get_3dsamples_indcated_by_indexes(
        region_to_3d_images_dict=stack_region_to_3dimg,
        indexes=patients_selected)


reconstruction_per_region = {}
for region in list_regions:

    iters = session.get_adequate_number_iterations(
        region_selected=region,
        explicit_iter_per_region = settings.explicit_iter_per_region,
        predefined_iters = max_iters)

    print("region {} selected".format(region))
    meta_region_file = "region_{0}-{1}".format(region, iters)
    path_meta_region = os.path.join(path_meta, meta_region_file)
    tf.reset_default_graph()

    # CVAE encoding
    hyperparams = {}
    hyperparams['image_shape'] = data_to_encode_per_region[region].shape[1:]
    cvae = CVAE.CVAE(hyperparams=hyperparams, meta_path=path_meta_region)

    # encoding_images
    print("Encoding")
    encoding_out = cvae.encode(data_to_encode_per_region[region])

    print("Decoding")
    images_3d_regenerated = cvae.decoder(latent_layer_input=encoding_out["mean"],
            original_images=data_to_encode_per_region[region])

    reconstruction_per_region[region] = images_3d_regenerated
    if logs:
        print("images regenerated shape {}".format(images_3d_regenerated.shape))


print("Mapping Reconstructing images")
whole_reconstruction = \
    utils_images3d.map_region_segmented_over_full_image(reconstruction_per_region, images_used)
print("Mapping Reconstructing images ended")

print("Mapping Original images")
origin_image = \
    utils_images3d.map_region_segmented_over_full_image(
        data_to_encode_per_region, images_used)
print("Mapping Original images ended")


for index in range(0,number_samples_to_reconstruct,1):

    idi = "{}".format(patients_selected[index])

    output.from_3d_image_to_nifti_file(path_to_save=os.path.join(
        path_where_store_images_generated, "Reconstructed {}".format(idi)),
        image3d=whole_reconstruction[index, :, :, :])

    output.from_3d_image_to_nifti_file(path_to_save=os.path.join(
        path_where_store_images_generated, "Original {}".format(idi)),
        image3d=origin_image[index, :, :, :])

    recons.plot_comparaison_images_ReconstructedvsOriginal(
        original_3dimg=origin_image[index,:,:,:],
        reconstruction_3dimg=whole_reconstruction[index, :, :, :],
        path_reconstruction_images=os.path.join(path_where_store_images_generated,
                                    "{}.png".format(idi)),
        cmap=cmap,
        title="Reconstructed AD vs Original AD. Sample {}".format(idi))
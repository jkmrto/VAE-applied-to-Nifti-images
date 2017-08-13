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
from lib.vae import CVAE

path_image = "reconstruction_individual_images.png"
#AD 123
#NOR 22
patients_selected = [22, 123, 23]
number_samples_to_reconstruct = len(patients_selected)
logs = True
regions_used = "all"
session_name = "test_saving_meta_PET_15_07_2017_21:34"

#images_used = "MRI_WM"
#images_used = "MRI_GM"
images_used = "PET"

#vae_used = "dense_vae"
vae_used = "conv_vae"
iters = 100



list_regions = session.select_regions_to_evaluate(regions_used)
path_session = os.path.join(settings.path_to_general_out_folder, session_name)
path_meta = os.path.join(path_session, "meta")
print(path_meta)

stack_region_to_3dimg, patient_labels, n_samples, cmap = \
    recons.load_desired_stacked_and_parameters(images_used, list_regions, )

data_to_encode_per_region = \
    recons.get_3dsamples_indcated_by_indexes(
        region_to_3d_images_dict=stack_region_to_3dimg,
        indexes=patients_selected)


reconstruction_per_region = {}
for region in list_regions:
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

for index in range(0,number_samples_to_reconstruct,1):
    output.from_3d_image_to_nifti_file(path_to_save="example_neg",
                                       image3d=whole_reconstruction[0, :, :, :])

    output.from_3d_image_to_nifti_file(path_to_save="example_pos",
                                       image3d=whole_reconstruction[1, :, :, :])


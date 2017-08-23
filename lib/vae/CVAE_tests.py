import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import matplotlib
matplotlib.use('Agg')
from lib.vae import CVAE
import settings
import lib.neural_net.kfrans_ops as ops
from lib import session_helper
from lib.data_loader.pet_loader import load_pet_regions_segmented
from lib.data_loader import pet_atlas
import region_plane_selector
from lib.data_loader import PET_stack_NORAD
from lib.utils import output_utils


def auto_execute_with_session_folders():
    print("Executing CVAE test")

    regions_used = "three"
    region_selected = 3
    list_regions = session_helper.select_regions_to_evaluate(regions_used)
    train_images = load_pet_regions_segmented(list_regions)[region_selected]

    pet_dict_stack = PET_stack_NORAD.get_parameters()
    atlas = pet_atlas.load_atlas()
    voxels_desired = atlas[region_selected]
    voxels_index = pet_dict_stack['voxel_index'] # no_bg_index to real position
    final_voxels_selected_index = voxels_index[voxels_desired]

    p1, p2, p3 = \
        region_plane_selector.get_maximum_activation_planes(
            voxels_index=final_voxels_selected_index,
            total_size=pet_dict_stack['total_size'],
            imgsize=pet_dict_stack['imgsize'],
            reshape_kind="C")

    hyperparams = {}
    hyperparams['latent_layer_dim'] = 100
    hyperparams['kernel_size'] = 5
    hyperparams['features_depth'] = [1, 16, 32]
    hyperparams['image_shape'] = train_images.shape[1:]
    hyperparams['activation_layer'] = ops.lrelu
    hyperparams['decay_rate'] = 0.002
    hyperparams['learning_rate'] = 0.001
    hyperparams['lambda_l2_regularization'] = 0.0001

    session_name = "test_over_cvae"
    path_to_session = \
        os.path.join(settings.path_to_general_out_folder, session_name)

    model = CVAE.CVAE(hyperparams=hyperparams,
                      test_bool=True,
                      meta_path=None,
                      path_to_session=path_to_session)

    model.train(X=train_images,
                n_iters=200,
                batchsize=32,
                suffix_files_generated="region_3",
                tempSGD_3dimages=True,
                iter_to_save=50,
                similarity_evaluation=True,
                dump_losses_log=True,
                save_bool=False,
                final_dump_comparison=True,
                final_dump_samples_to_compare=None,
                final_dump_planes_per_axis_to_show_in_compare=[p1, p2, p3])


auto_execute_with_session_folders()


def auto_execute_encoding_over_trained_net():
    regions_used = "three"
    region_selected = 3
    list_regions = session_helper.select_regions_to_evaluate(regions_used)
    train_images = load_pet_regions_segmented(list_regions)[region_selected]

    hyperparams = {}
    hyperparams['image_shape'] = train_images.shape[1:]

    session_name = "test_over_cvae"
    path_to_session = \
        os.path.join(settings.path_to_general_out_folder, session_name)
    path_to_meta_files = os.path.join(path_to_session, "meta", "region_3-500")

    cvae = CVAE.CVAE(hyperparams=hyperparams,
                     meta_path=path_to_meta_files)

    print("encoding")
    encoding = cvae.encode(train_images)  # [mu, sigma]

    return encoding


# encoding = auto_execute_encoder_over_trained_net()


def auto_execute_encoding_and_decoding_over_trained_net():
    regions_used = "three"
    region_selected = 3
    list_regions = session_helper.select_regions_to_evaluate(regions_used)
    train_images = load_pet_regions_segmented(list_regions)[region_selected]

    hyperparams = {}
    hyperparams['image_shape'] = train_images.shape[1:]

    session_name = "test_over_cvae"
    path_to_session = \
        os.path.join(settings.path_to_general_out_folder, session_name)
    path_to_meta_files = os.path.join(path_to_session, "meta", "region_3-50")
    path_to_images = os.path.join(path_to_session, "images")

    cvae = CVAE.CVAE(hyperparams=hyperparams,
                meta_path=path_to_meta_files)

    print("encoding")
    encoding_out = cvae.encode(train_images)  # [mean, stdev]
    z_in = encoding_out["mean"]
    print(type(z_in))
    print(z_in.shape)
    images_3d_regenerated = cvae.decoder(latent_layer_input=z_in,
                                         original_images=train_images)

    output_utils.from_3d_image_to_nifti_file(
        path_to_save=os.path.join(path_to_images, "example"),
        image3d=images_3d_regenerated[0, :, :, :])

    # auto_execute_with_session_folders()
    # auto_execute_encoding_and_decoding_over_trained_net()
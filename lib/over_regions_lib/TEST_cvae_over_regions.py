import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import matplotlib
matplotlib.use('Agg')
from lib.data_loader.pet_loader import load_pet_regions_segmented
from lib.utils import cv_utils
import lib.neural_net.kfrans_ops as ops
from lib import session_helper as session
from lib.data_loader import pet_atlas
import region_plane_selector
from lib.data_loader import PET_stack_NORAD
from lib.over_regions_lib.cvae_over_regions import \
    execute_saving_meta_graph_without_any_cv, execute_without_any_logs
import settings


def auto_execute_saving_meta_graph_without_any_cv(hyperparams=None,
                                                  session_conf=None):
    regions_used = "all"
    list_regions = session.select_regions_to_evaluate(regions_used)
    region_to_img_dict = load_pet_regions_segmented(list_regions,
                                                    bool_logs=False)

    # Autoenconder configuration
    if hyperparams is None:
        hyperparams = {'latent_layer_dim': 100,
                       'kernel_size': 5,
                       'activation_layer': ops.lrelu,
                       'features_depth': [1, 16, 32],
                        'decay_rate': 0.0002,
                        'learning_rate': 0.001,
                        'lambda_l2_regularization': 0.0001}

    if session_conf is None:
    # SESSION CONFIGURATION
        session_conf = {'bool_normalized': False,
                       'n_iters': 100,
                        "batch_size": 16,
                        "show_error_iter": 10}

    execute_saving_meta_graph_without_any_cv(
        region_cubes_dict=region_to_img_dict,
        hyperparams=hyperparams,
        session_conf=session_conf,
        list_regions=list_regions,
        path_to_root=settings.path_to_general_out_folder,
        session_prefix_name="test_saving_meta_PET")

#auto_execute_saving_meta_graph_without_any_cv()



def auto_execute_without_logs_over_one_region():
    regions_used = "three"
    list_regions = session.select_regions_to_evaluate(regions_used)
    region_to_img_dict = load_pet_regions_segmented(list_regions,
                                                    bool_logs=False)

    n_folds = 10
    n_samples = region_to_img_dict[1].shape[0]
    k_fold_dict = cv_utils.generate_k_folder_in_dict(n_samples=n_samples,
                                                     n_folds=n_folds)

    reg_to_group_to_images_dict = cv_utils.restructure_dictionary_based_on_cv_index_3dimages(
        dict_train_test_index=k_fold_dict[0],
        region_to_img_dict=region_to_img_dict)

    hyperparams = {'latent_layer_dim': 20,
                   'kernel_size': 5,
                   'activation_layer': ops.lrelu,
                   'features_depth': [1, 16, 32],
                   'decay_rate': 0.0002,
                   'learning_rate': 0.001,
                   'lambda_l2_regularization': 0.0001}

    session_conf = {'bool_normalized': False,
                    'n_iters': 50,
                    "batch_size": 16,
                    "show_error_iter": 10}

    results = execute_without_any_logs(
        region_train_cubes_dict=reg_to_group_to_images_dict['train'],
        hyperparams=hyperparams,
        session_conf=session_conf,
        path_to_root=None,
        list_regions=list_regions,
        region_test_cubes_dict=reg_to_group_to_images_dict['test'])

    return results

# out = auto_execute_without_logs_over_one_region()

def auto_execute_saving_meta_graph_without_any_cv(hyperparams=None,
                                                  session_conf=None):
    regions_used = "all"
    list_regions = session.select_regions_to_evaluate(regions_used)
    region_to_img_dict = load_pet_regions_segmented(list_regions,
                                                    bool_logs=False)

    # Autoenconder configuration
    if hyperparams is None:
        hyperparams = {'latent_layer_dim': 100,
                       'kernel_size': 5,
                       'activation_layer': ops.lrelu,
                       'features_depth': [1, 16, 32],
                        'decay_rate': 0.0002,
                        'learning_rate': 0.001,
                        'lambda_l2_regularization': 0.0001}

    if session_conf is None:
    # SESSION CONFIGURATION
        session_conf = {'bool_normalized': False,
                       'n_iters': 100,
                        "batch_size": 16,
                        "show_error_iter": 10}

    execute_saving_meta_graph_without_any_cv(
        region_cubes_dict=region_to_img_dict,
        hyperparams=hyperparams,
        session_conf=session_conf,
        list_regions=list_regions,
        path_to_root=settings.path_to_general_out_folder,
        session_prefix_name="test_saving_meta_PET")


def auto_execute_saving_meta_graph_and_3dtemp_images_and_final_dump(
        hyperparams=None, session_conf=None):

    regions_used = "all"
    list_regions = session.select_regions_to_evaluate(regions_used)
    region_to_img_dict = load_pet_regions_segmented(list_regions,
                                                    bool_logs=False)
    pet_dict_parameters = PET_stack_NORAD.get_parameters()
    atlas = pet_atlas.load_atlas()

    region_plane_selector.get_dict_region_to_maximum_activation_planes(
        list_regions=list_regions,
        atlas=atlas,
        stack_parameters=pet_dict_parameters,
    )

    final_dump_samples_to_compare = [10,20,30, 90, 100, 110]
    # Autoenconder configuration
    if hyperparams is None:
        hyperparams = {'latent_layer_dim': 100,
                       'kernel_size': 5,
                       'activation_layer': ops.lrelu,
                       'features_depth': [1, 16, 32],
                        'decay_rate': 0.0002,
                        'learning_rate': 0.001,
                        'lambda_l2_regularization': 0.0001}

    if session_conf is None:
    # SESSION CONFIGURATION
        session_conf = {'bool_normalized': False,
                        'n_iters': 100,
                        "batch_size": 16,
                        "show_error_iter": 10,
                        "final_dump_comparison":True,
                        "final_dump_samples_to_compare":
                            final_dump_samples_to_compare,
                        "final_dump_planes_per_axis_to_show_in_compare"=
    }

    execute_saving_meta_graph_without_any_cv(
        region_cubes_dict=region_to_img_dict,
        hyperparams=hyperparams,
        session_conf=session_conf,
        list_regions=list_regions,
        path_to_root=settings.path_to_general_out_folder,
        session_prefix_name="test_saving_meta_PET")

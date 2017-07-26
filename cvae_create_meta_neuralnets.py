import lib.neural_net.kfrans_ops as ops
import settings
from lib import session_helper
import os
from delete_pre_final_meta_data import delete_simple_session
from lib.data_loader.pet_loader import load_pet_regions_segmented
from lib.over_regions_lib.cvae_over_regions import \
    execute_saving_meta_graph_without_any_cv

regions_used = "all"
list_regions = session_helper.select_regions_to_evaluate(regions_used)
region_to_img_dict = load_pet_regions_segmented(list_regions,
                                                bool_logs=False)
explicit_iter_per_region = {
    73: 300,
}

hyperparams = {'latent_layer_dim': 100,
               'kernel_size': 5,
               'activation_layer': ops.lrelu,
               'features_depth': [1, 16, 32],
               'decay_rate': 0.005,
               'learning_rate': 0.001,
               'lambda_l2_regularization': 0.0001}

session_conf = {'bool_normalized': True,
                'n_iters': 500,
                "batch_size": 16,
                "show_error_iter": 10}

path_to_session = execute_saving_meta_graph_without_any_cv(
    region_cubes_dict=region_to_img_dict,
    hyperparams=hyperparams,
    session_conf=session_conf,
    list_regions=list_regions,
    path_to_root=settings.path_to_general_out_folder,
    session_prefix="cvae_create_meta_nets_iter_500",
    explicit_iter_per_region=explicit_iter_per_region)

# deleting temporal meta data generated
session_to_clean_meta_folder = os.path.join(path_to_session, "meta")

delete_simple_session(session_to_clean_meta_folder=session_to_clean_meta_folder)
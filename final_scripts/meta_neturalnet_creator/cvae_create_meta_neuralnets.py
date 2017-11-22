import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import settings

import lib.neural_net.kfrans_ops as ops
from lib import session_helper
from lib.data_loader.pet_loader import load_pet_regions_segmented
from lib.delete_pre_final_meta_data import delete_simple_session
from lib.over_regions_lib.cvae_over_regions import \
    execute_saving_meta_graph_without_any_cv

regions_used = "all"
last_valid_region_with_meta_generated = 30

list_regions = range(last_valid_region_with_meta_generated + 1, 117, 1)
region_to_img_dict = load_pet_regions_segmented(list_regions,
                                                bool_logs=False)
explicit_iter_per_region = {
    73: 300,
    44: 900,
}

max_iters = 2000

hyperparams = {'latent_layer_dim': 100,
               'kernel_size': [5, 5, 5],
               'activation_layer': ops.lrelu,
               'stride': 2,
               'features_depth': [1, 16, 32, 64],
               'decay_rate': 0.0025,
               'learning_rate': 0.001,
               'lambda_l2_regularization': 0.0001,
               'cvae_model': "3layers"}

session_conf = {'bool_normalized': False,
                'n_iters': max_iters,
                "batch_size": 64,
                "show_error_iter": 100}

path_to_session = execute_saving_meta_graph_without_any_cv(
    region_cubes_dict=region_to_img_dict,
    hyperparams=hyperparams,
    session_conf=session_conf,
    list_regions=list_regions,
    path_to_root=settings.path_to_general_out_folder,
    session_prefix_name="cvae_create_meta_nets_iter_1000-layer",
    explicit_iter_per_region=explicit_iter_per_region)

# deleting temporal meta data generated
session_to_clean_meta_folder = os.path.join(path_to_session, "meta")

delete_simple_session(session_to_clean_meta_folder=session_to_clean_meta_folder)
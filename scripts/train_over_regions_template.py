from scripts import train_over_regions_hub
from lib.data_loader import pet_atlas
from lib.data_loader import PET_stack_NORAD
import tensorflow as tf

# SESSION CONFIGURATION
HYPERPARAMS = {
    "batch_size": 16,
    "learning_rate": 5E-6,
    "dropout": 0.9,
    "lambda_l2_reg": 1E-5,
    "nonlinearity": tf.nn.elu,
    "squashing": tf.nn.sigmoid,
}

session_settings = {
    "regions_used": "all",
    "max_iter": 1500,
    "after_input_architecture" : [1000, 500, 100],
    "iter_to_save": 50,
    "iter_to_show_error":  10,
    "bool_normalized_per_region": False,
    "bool_norm_truncate": True,
    "bool_log_grad_desc_error": True,
}


dict_norad = PET_stack_NORAD.get_stack()  # 'stack' 'voxel_index' 'labels'
region_voxels_index_per_region = pet_atlas.load_atlas()


train_over_regions_hub.run_session(
    dict_norad=dict_norad,
    region_voxels_index_per_region=region_voxels_index_per_region,
    vae_hyperparams=HYPERPARAMS,
    session_settings=session_settings
)

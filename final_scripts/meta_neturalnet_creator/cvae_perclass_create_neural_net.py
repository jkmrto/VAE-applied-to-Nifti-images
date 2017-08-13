import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import lib.neural_net.kfrans_ops as ops
import settings
from lib import session_helper
from lib.data_loader.utils_images3d import get_stack_3dimages_filtered_by_label
from lib.data_loader.pet_loader import load_pet_regions_segmented
from lib.data_loader.PET_stack_NORAD import load_patients_labels
from lib.delete_pre_final_meta_data import delete_simple_session
from lib.over_regions_lib.cvae_over_regions import \
    execute_saving_meta_graph_without_any_cv


# Filtering imates to use base on AD or NOR images
regions_used = "all"
class_selected = "AD"
#class_selected = "AD"
list_regions = session_helper.select_regions_to_evaluate(regions_used)
region_to_img_dict = load_pet_regions_segmented(list_regions, bool_logs=False)
session_name = "cvae_perclass_{0}_create_meta_net_iter_{1}_latent_layer_{2}"

dic_class_to_label={
    "NOR": 0,
    "AD": 1,
}
explicit_iter_per_region = {}

filtered_stack = get_stack_3dimages_filtered_by_label(
    stack_region_to_3dimg=region_to_img_dict,
    samples_label=load_patients_labels(),
    label_selected=dic_class_to_label["AD"])

print("Dimensions stacked filtered: {}".format(
    filtered_stack[1].shape))

hyperparams = {'latent_layer_dim': 100,
               'kernel_size': 5,
               'activation_layer': ops.lrelu,
               'features_depth': [1, 16, 32],
               'decay_rate': 0.0025,
               'learning_rate': 0.001,
               'lambda_l2_regularization': 0.0001}

session_conf = {'bool_normalized': True,
                'n_iters': 500,
                "batch_size": 16,
                "show_error_iter": 10}


session_name = session_name.format(class_selected,
                                    hyperparams["latent_layer_dim"],
                                   session_conf["n_iters"])
print(session_name)

path_to_session = execute_saving_meta_graph_without_any_cv(
    region_cubes_dict=filtered_stack,
    hyperparams=hyperparams,
    session_conf=session_conf,
    list_regions=list_regions,
    path_to_root=settings.path_to_general_out_folder,
    session_prefix_name=session_name,
    explicit_iter_per_region=explicit_iter_per_region)

# deleting temporal meta data generated
session_to_clean_meta_folder = os.path.join(path_to_session, "meta")

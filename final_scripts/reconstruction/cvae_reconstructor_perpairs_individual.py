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

#AD 123
#NOR 22
patients_selected_per_class = {"NOR": 22, "AD": 123}
logs = True
regions_used = "all"
session_name = "test_saving_meta_PET_15_07_2017_21:34"

data_to_encode_per_region = \
    recons.get_representatives_samples_over_region_per_patient_indexes(
    region_to_3d_images_dict=region_to_3dimg_dict_pet,
    indexes_per_group=patients_selected_per_class)



plot_comparaision_images_ADvsNOR(original_NOR, original_AD, recons_NOR,
                                 recons_AD, path_reconstruction_images, cmap)
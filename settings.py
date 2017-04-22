import os
import numpy as np
from lib.aux_functionalities.os_aux import create_directories
from lib.mri.mri_atlas_settings import super_regions_atlas


def get_list_of_regions_evaluated(list_supper_region):

    supper_regions_evaluated = list_supper_region
    regions_evaluated = []
    for supper_region in supper_regions_evaluated:
        regions_evaluated = regions_evaluated + super_regions_atlas[supper_region]

    return regions_evaluated


list_supper_region_evaluated = ['frontal_lobe_val', 'parietal_lobe_val', 'occipital_lobe_val', 'temporal_lobe_val']

list_regions_evaluated = get_list_of_regions_evaluated(list_supper_region_evaluated)

path_to_project = os.path.dirname(os.path.abspath(__file__))
path_to_general_out_folder = os.path.join(path_to_project, "out")
create_directories([path_to_general_out_folder])

stack_path_GM = path_to_project + "/data/stack_NORAD_GM.mat"
stack_path_WM = path_to_project + "/data/stack_NORAD_WM.mat"
atlas_path = path_to_project + "/data/" + "ratlas116_MRI.nii"
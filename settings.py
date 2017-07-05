import os
from lib.aux_functionalities.os_aux import create_directories
from lib.data_loader.atlas_settings import super_regions_atlas


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

MRI_stack_path_GM = path_to_project + "/data/stack_NORAD_GM.mat"
MRI_stack_path_WM = path_to_project + "/data/stack_NORAD_WM.mat"
mri_atlas_path = path_to_project + "/data/" + "ratlas116_MRI.nii"

PET_stack_path = path_to_project + "/data/PET_stack_NORAD.mat"
pet_atlas_path = path_to_project + "/data/" + "ratlas116_PET.nii"

# AUTOENCODER SESSION FOLDER AND FILES ASSOCIATED
main_test_folder_autoencoder_session = "post_train"
main_cv_vae_session = "cv"


# SVM DEFAULT FOLDERS NAME AND FILES GENERATED
svm_file_name_per_region_accuracy = "per_reg_acc.log"
svm_file_name_scores_file_name = "scores.log"
svm_folder_name_test_out = "test_out"
svm_folder_name_train_out = "train_out"

# DECISION NEURAL NET

explicit_iter_per_region = {
    47: 160,
    44: 100,
    43: 130,
    73: 130,
    74: 130,
    75: 60,

}
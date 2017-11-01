import os

from lib.utils import os_aux
from lib import file_reader
from final_scripts.results_reader import reader_helper as helper
import settings

# CONFIGURATION
swap_type = "latent_layer"
#swap_type = "kernel"
images_used = "PET"
historical_folder_name = None


MRI_VAE_latent_layer_predefined_threshold = "/home/jkmrto/RemoteDisks/BiosipServer/new_folder/VAN-applied-to-Nifti-images/out/VAE_session_latent_layer_MRI/historical/threshold_predefined"
PET_VAE_latent_layer_predefined_threshold =  "/home/jkmrto/RemoteDisks/BiosipServer/temp_folder_3/VAN-applied-to-Nifti-images/out/VAE_session_{0}_{1}/historical/VAE_session_latent_layer_PET_with_defined_threshold"

out_folder = MRI_VAE_latent_layer_predefined_threshold
#out_folder = PET_VAE_latent_layer_predefined_threshold

output_weighted_svm = "loop_output_weighted_svm.csv"
output_simple_majority_vote = "loop_output_simple_majority_vote.csv"
output_complex_majority_vote = "loop_output_complex_majority_vote.csv"

evaluation_images_folder_name = "evaluation_images"

# Paths references
path_evaluation_images_folder = os.path.join(out_folder,
                                             evaluation_images_folder_name)
path_output_weighted_svm = os.path.join(out_folder, output_weighted_svm)
path_output_simple_majority_vote = os.path.join(out_folder,
                                                output_simple_majority_vote)
path_output_complex_majority_vote = os.path.join(out_folder,
                                                 output_complex_majority_vote)
#
os_aux.create_directories([path_evaluation_images_folder])

list_SVM = file_reader.read_csv_as_list_of_dictionaries(
    path_output_weighted_svm)
list_SMV = file_reader.read_csv_as_list_of_dictionaries(
    path_output_simple_majority_vote)
list_CMV = file_reader.read_csv_as_list_of_dictionaries(
    path_output_complex_majority_vote)

string_ref = helper.string_ref.format(
    images_used, helper.mapa_etiquetas[swap_type], "SVM")
helper.plot_evaluation_parameters(list_SVM, string_ref,
                                  path_evaluation_images_folder, swap_type)

string_ref = helper.string_ref.format(
    images_used, helper.mapa_etiquetas[swap_type], "SMV")
helper.plot_evaluation_parameters(list_SMV, string_ref,
                                  path_evaluation_images_folder, swap_type)

string_ref = helper.string_ref.format(
    images_used, helper.mapa_etiquetas[swap_type], "CMV")
helper.plot_evaluation_parameters(list_CMV, string_ref,
                                  path_evaluation_images_folder, swap_type)

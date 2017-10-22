import os

from lib.utils import os_aux
from lib import file_reader
from final_scripts.results_reader import reader_helper as helper
import settings

# CONFIGURATION
swap_type = "latent layer"
images_used = "PET"


out_folder = """/home/jkmrto/RemoteDisks/BiosipServer/
temp_folder_3/VAN-applied-to-Nifti-images/out/
VAE_session_{0}_{1}""".format(swap_type, images_used).strip("\n").replace("\n", '')


output_weighted_svm = "loop_output_weighted_svm.csv"
output_simple_majority_vote = "loop_output_simple_majority_vote.csv"
output_complex_majority_vote = "loop_output_complex_majority_vote.csv"

evaluation_images_folder_name = "evaluation_images"

# Paths references
path_to_folder = os.path.join(settings.path_to_general_out_folder, out_folder)
path_evaluation_images_folder = os.path.join(path_to_folder,
                                             evaluation_images_folder_name)
path_output_weighted_svm = os.path.join(path_to_folder, output_weighted_svm)
path_output_simple_majority_vote = os.path.join(path_to_folder,
                                                output_simple_majority_vote)
path_output_complex_majority_vote = os.path.join(path_to_folder,
                                                 output_complex_majority_vote)

#
os_aux.create_directories([path_evaluation_images_folder])

list_SVM = file_reader.read_csv_as_list_of_dictionaries(
    path_output_weighted_svm)
list_SMV = file_reader.read_csv_as_list_of_dictionaries(
    path_output_simple_majority_vote)
list_CMV = file_reader.read_csv_as_list_of_dictionaries(
    path_output_complex_majority_vote)

string_ref = "{0} swap over {1}. {2} method".format(
    images_used, swap_type, "SVM")

helper.plot_evaluation_parameters(list_SVM, string_ref,
                                  path_evaluation_images_folder, swap_type)

string_ref = "{0} swap over {1}. {2} method".format(
    images_used, swap_type, "SMV")
helper.plot_evaluation_parameters(list_SMV, string_ref,
                                  path_evaluation_images_folder, swap_type)

string_ref = "{0} swap over {1}. {2} method".format(
    images_used, swap_type, "CMV")
helper.plot_evaluation_parameters(list_CMV, string_ref,
                                  path_evaluation_images_folder, swap_type)

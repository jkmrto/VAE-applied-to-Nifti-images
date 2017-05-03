import os
import numpy
import settings
from lib.mri import stack_NORAD
from lib.aux_functionalities.os_aux import create_directories
from matplotlib import pyplot as plt


iden_session = "02_05_2017_21:09 arch: 1000_800_500_100_2"
test_name = "Encoding session"
regions_used = "most important"

path_to_session = os.path.join(settings.path_to_general_out_folder,
                               iden_session)
path_to_meta_folder = os.path.join(path_to_session, "meta")
path_to_main_test = os.path.join(path_to_session, "post_train")
path_to_particular_test = os.path.join(path_to_main_test, test_name)
path_to_encoding_storage_folder = os.path.join(path_to_particular_test,
                                               "encoding_data")
path_to_cluster_images = os.path.join(path_to_particular_test, "images")


create_directories([path_to_cluster_images])

dict_norad = stack_NORAD.get_gm_stack()  # 'stack' 'voxel_index' 'labels'
region_voxels_label = dict_norad['labels']

list_regions = []
if regions_used == "all":
    list_regions = range(1, 85, 1)  # 117 regions en total
elif regions_used == "most important":
    list_regions = settings.list_regions_evaluated

for region_selected in list_regions:
    encoding_mean_file_name = os.path.join(path_to_encoding_storage_folder,
                                           "region {}_desv.txt".format(
                                               region_selected))
    path_to_cluster_image = os.path.join(path_to_cluster_images, "region {}.png".
                                         format(region_selected))

    data_2 = numpy.genfromtxt(encoding_mean_file_name, delimiter=',')

    plt.figure()
    plt.scatter(data_2[:, 0], data_2[:, 1], c=region_voxels_label,
            cmap=plt.cm.viridis)
    plt.savefig(path_to_cluster_image, dpi=150)
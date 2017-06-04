import os
import settings
import tensorflow as tf
import numpy
from sklearn.decomposition import PCA
from lib.data_loader import mri_atlas
from lib.aux_functionalities.os_aux import create_directories
from matplotlib import pyplot as plt
from lib.data_loader import MRI_stack_NORAD
from lib import utils

test_name = "Encoding session"
regions_used = "all"

iden_session = "27_04_2017_21:54 arch: 1000_800_500_100_2"
path_to_session = os.path.join(settings.path_to_general_out_folder,
                               iden_session)
path_to_meta_folder = os.path.join(path_to_session, "meta")
path_to_main_test = os.path.join(path_to_session, "post_train")
path_to_particular_test = os.path.join(path_to_main_test, test_name)
path_to_encoding_storage_folder = os.path.join(path_to_particular_test,
                                               "encoding_data")
path_to_pca_over_encoding = os.path.join(path_to_encoding_storage_folder,
                                         "pca_2_dim_per_region")
path_to_images = os.path.join(path_to_pca_over_encoding, "images")
path_to_output = os.path.join(path_to_pca_over_encoding, "output")

create_directories([path_to_pca_over_encoding, path_to_images, path_to_output])

# We load the labels in order to assign the colour in the graph
dict_norad = MRI_stack_NORAD.get_gm_stack()  # 'stack' 'voxel_index' 'labels'
region_voxels_label = dict_norad['labels']

list_regions = []
if regions_used == "all":
    list_regions = range(1, 85, 1)  # 117 regions en total
elif regions_used == "most important":
    list_regions = settings.list_regions_evaluated

for region_selected in list_regions:
    encoding_mean_file_name = os.path.join(path_to_encoding_storage_folder,
                                           "region {}_means.txt".format(
                                               region_selected))
    path_to_cluster_image = os.path.join(path_to_images, "region {}.png".
                                         format(region_selected))
    path_to_pca_results_file = os.path.join(path_to_output, "region {}.txt".
                                            format(region_selected))
    # [417x100]
    data_100 = numpy.genfromtxt(encoding_mean_file_name, delimiter=',')

    pca = PCA(n_components=2)
    pca.fit(data_100)
    data_2 = pca.transform(data_100)

    numpy.savetxt(path_to_pca_results_file, data_2, delimiter=',')

    plt.figure()
    plt.scatter(data_2[:, 0], data_2[:, 1], c=region_voxels_label,
                cmap=plt.cm.viridis)
    plt.savefig(path_to_cluster_image, dpi=150)

# PET test
# Testing reconstruction over only one sample
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import matplotlib
from sklearn.decomposition import PCA
matplotlib.use('Agg')
import os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import settings
from lib import session_helper as session
from lib import reconstruct_helpers as recons
from lib.vae import CVAE
from lib import compare_helper as compare
import lib.scatter_plots_helper as scatter

explicit_iter_per_region = {
}
path_to_out_folder = os.path.join(settings.path_to_general_out_folder,
                                  "explore_latent_code")
path_image_diff_over_latent_layer_per_class_per_region = os.path.join(
    path_to_out_folder, "per_region_diff_per_class.png")
path_image_diff_over_latent_layer_over_NOR_per_region = os.path.join(
    path_to_out_folder, "per_region_diffs_over_nor_samples.png")
path_image_diff_over_latent_layer_over_AD_per_region = os.path.join(
    path_to_out_folder, "per_region_diffs_over_add_samples.png")
path_image_diff_over_latent_layer_all_region_per_sample = os.path.join(
    path_to_out_folder,
    "diff_over_samples_in_latent_layer_all_regions_concatenated.png")

regions_used = "all"
session_name = "test_saving_meta_PET_15_07_2017_21:34"

images_used = "PET"

#vae_used = "dense_vae"
max_iters = 100

list_regions = session.select_regions_to_evaluate(regions_used)
path_session = os.path.join(settings.path_to_general_out_folder, session_name)
path_meta = os.path.join(path_session, "meta")
print(path_meta)

stack_region_to_3dimg, patient_labels, n_samples, cmap = \
    recons.load_desired_stacked_and_parameters(images_used, list_regions, )

# stack_region_to_3dimg -> dict[region]  to 3d img
results_per_region_per_classes = {}
encoding_output_per_region = {}
diff_mean_over_AD_samples_per_region = {}
diff_mean_over_NOR_samples_per_region = {}
# sh[n_samples, latent_layer_dim x n_regions]
region_results_concatenated = np.zeros([n_samples, 0])
for region in list_regions:

    # Get specified iters per region, in order to avoid
    # not converging isssues
    iters = session.get_adequate_number_iterations(
        region_selected=region,
        explicit_iter_per_region = explicit_iter_per_region,
        predefined_iters = max_iters)

    print("region {} selected".format(region))
    meta_region_file = "region_{0}-{1}".format(region, iters)
    path_meta_region = os.path.join(path_meta, meta_region_file)
    tf.reset_default_graph()

    # CVAE encoding
    hyperparams = {}
    hyperparams['image_shape'] = stack_region_to_3dimg[region].shape[1:]
    cvae = CVAE.CVAE(hyperparams=hyperparams, meta_path=path_meta_region)

    # encoding_images
    print("Encoding")
    encoding_out = cvae.encode(stack_region_to_3dimg[region])
    encoding_means = encoding_out["mean"] #Selecting means as reference

    # regions concatenated matrix, compare samples over latent layer
    region_results_concatenated = np.concatenate(
        [region_results_concatenated, encoding_means], axis=1)


    # To calculate means per class, compare clases over latent layer
    means_over_latent_layer_per_class_and_region = \
        recons.get_means_by_label_over_flat_samples(
        data_samples=encoding_means,
        patient_labels=patient_labels)

    diff_encoding = compare.evaluate_diff_flat(
        means_over_latent_layer_per_class_and_region[0, :],
        means_over_latent_layer_per_class_and_region[1, :])

    results_per_region_per_classes[region] = diff_encoding

    # To calculate global difference per region

    NOR_samples = compare.get_samples_per_label(
        samples_matrix=encoding_means,
        labels=patient_labels,
        label_selected=0)
    AD_samples = compare.get_samples_per_label(
        samples_matrix=encoding_means,
        labels=patient_labels,
        label_selected=1)

    diff_mean_over_NOR_samples_per_region[region]=\
        compare.get_mean_difference_over_samples(NOR_samples)
    diff_mean_over_AD_samples_per_region[region]=\
        compare.get_mean_difference_over_samples(AD_samples)

    encoding_means_reduced = PCA(n_components=2).fit_transform(encoding_means)
    path_to_img = os.path.join(path_to_out_folder,
                               "2d_scatter_region_{}.png".format(region))
    scatter.plot_2dscatter_plot_2groups(samples=encoding_means_reduced,
                                        samples_labels=np.array(patient_labels),
                                        path_image=path_to_img,
                                        tittle = "region {}".format(region))

    # Scatter Analisis
    encoding_means_reduced = PCA(n_components=3).fit_transform(encoding_means)
    path_to_img = os.path.join(path_to_out_folder,
                               "3d_scatter_region_{}.png".format(region))
    scatter.plot_3dscatter_plot_2groups(samples=encoding_means_reduced,
                                        samples_labels=np.array(patient_labels),
                                        path_image=path_to_img,
                                        tittle="region {}".format(region))
    # Scatter Analisis

plt.figure()
plt.bar(list(results_per_region_per_classes.keys()),
        list(results_per_region_per_classes.values()))
plt.savefig(path_image_diff_over_latent_layer_per_class_per_region)

plt.figure()
plt.title("Cross Mean Difference over NOR samples")
plt.xlabel("Region")
plt.bar(list(diff_mean_over_NOR_samples_per_region.keys()),
        list(diff_mean_over_NOR_samples_per_region.values()))
plt.savefig(path_image_diff_over_latent_layer_over_NOR_per_region)

plt.figure()
plt.title("Cross Mean Difference over AD samples")
plt.xlabel("Region")
plt.bar(list(diff_mean_over_AD_samples_per_region.keys()),
        list(diff_mean_over_AD_samples_per_region.values()))
plt.savefig(path_image_diff_over_latent_layer_over_AD_per_region)


diff_over_samples_all_regions_imshow = \
    compare.get_comparision_over_matrix_samples(region_results_concatenated)

maximum = diff_over_samples_all_regions_imshow.max()
diff_over_samples_all_regions_imshow = \
    diff_over_samples_all_regions_imshow/maximum

plt.figure()
plt.imshow(diff_over_samples_all_regions_imshow, cmap="jet")
plt.title("Values normalized to {}".format(maximum))
plt.colorbar()
plt.savefig(path_image_diff_over_latent_layer_all_region_per_sample)
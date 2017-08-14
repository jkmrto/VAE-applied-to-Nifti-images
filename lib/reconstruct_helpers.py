import numpy as np
from lib.data_loader import pet_loader
from lib.data_loader import mri_loader
from matplotlib import pyplot as plt
import settings
import os


def load_desired_stacked_and_parameters(images_used, list_regions):
    """ Complete function for MRI"""
    n_samples = 0
    patient_labels = []
    stack_region_to_3dimg = None
    cmap = None

    if images_used == "PET":
        print("Loading Pet images")
        stack_region_to_3dimg, patient_labels, n_samples = \
            pet_loader.load_pet_data_3d(list_regions)
        cmap = "jet"

    elif images_used in ["MRI_GM", "MRI_WM"]:
        region_to_3dimg_dict_mri_gm, region_to_3dimg_dict_mri_wm, \
        patient_labels, n_samples = mri_loader.load_mri_data_3d(list_regions)
        if images_used == "MRI_GM":
            stack_region_to_3dimg = region_to_3dimg_dict_mri_gm
            cmap = "Greys"
        elif images_used == "MRI_WM":
            stack_region_to_3dimg = region_to_3dimg_dict_mri_wm
            cmap = "Greys"

    return stack_region_to_3dimg, patient_labels, n_samples, cmap


def get_data_to_encode_per_region(region_to_3dimg_dict_pet,
                                  where_to_mean_data,
                                  patient_labels,
                                  patients_selected_if_individual_treatment):
    """

    :param region_to_3dimg_dict_pet:  dict[region] -> 3d_img sh[n_samples,w,h,d]
    :param where_to_mean_data:
    :param patient_labels:
    :param patients_selected_if_individual_treatment: dict["NOR"|"AD"] -> index_patient_selected
    :return:
    """
    data_to_encode_per_region = None
    if where_to_mean_data == "before_encoding":
        data_to_encode_per_region = \
            get_mean_3d_images_over_samples_per_region(
                region_to_3dimg_dict_pet=region_to_3dimg_dict_pet,
                patient_labels=patient_labels)
    elif where_to_mean_data == "no_mean_individual_input":
        data_to_encode_per_region = \
            get_representatives_samples_over_region_per_patient_indexes(
                region_to_3d_images_dict=region_to_3dimg_dict_pet,
                indexes_per_group=patients_selected_if_individual_treatment)

    elif where_to_mean_data == "after_encoding":
        data_to_encode_per_region = region_to_3dimg_dict_pet

    return data_to_encode_per_region


def get_data_to_decode(where_to_mean_data, samples, patient_labels):
    data_to_decode = None
    if where_to_mean_data == "after_encode":
        data_to_decode = get_means_by_label_over_flat_samples(
            data_samples=samples,
            patient_labels=patient_labels)

    elif where_to_mean_data == "before_encoding":
        data_to_decode = samples

    elif where_to_mean_data == "no_mean_individual_input":
        data_to_decode = samples

    return data_to_decode


def get_representatives_samples_over_region_per_patient_indexes(
        region_to_3d_images_dict, indexes_per_group):
    region_to_class_to_3d_means_imgs = {}

    for region, cube_images in region_to_3d_images_dict.items():
        class_to_3d_means_imgs = np.zeros([2, cube_images.shape[1],
                                           cube_images.shape[2],
                                           cube_images.shape[3]])

        class_to_3d_means_imgs[0, :, :, :] = \
            cube_images[indexes_per_group["NOR"], :, :, :]

        class_to_3d_means_imgs[1, :, :, :] = \
            cube_images[indexes_per_group["AD"], :, :, :]

        region_to_class_to_3d_means_imgs[region] = class_to_3d_means_imgs

    return region_to_class_to_3d_means_imgs


def get_3dsamples_indcated_by_indexes(
        region_to_3d_images_dict, indexes):
    number_examples_to_extract = len(indexes)
    region_to_class_to_3d_means_imgs = {}

    for region, cube_images in region_to_3d_images_dict.items():
        class_to_3d_means_imgs = np.zeros([number_examples_to_extract,
                                           cube_images.shape[1],
                                           cube_images.shape[2],
                                           cube_images.shape[3]])

        class_to_3d_means_imgs[:, :, :, :] = cube_images[indexes, :, :, :]

        region_to_class_to_3d_means_imgs[region] = class_to_3d_means_imgs

    return region_to_class_to_3d_means_imgs


def get_mean_3d_images_over_samples_per_region(region_to_3dimg_dict_pet,
                                               patient_labels):
    """

    :param region_to_3dimg_dict_pet: dict[region] -> 3d_image sh[n_samples, w, h, d]
    :return: dict[region]-> np.array 3d_mean_image sh[2, w, h, d]
                         -> array with the mean image negative pos 0, positive pos 1
    """
    region_to_class_to_3d_means_imgs = {}

    for region, cube_images in region_to_3dimg_dict_pet.items():
        class_to_3d_means_imgs = np.zeros([2, cube_images.shape[1],
                                           cube_images.shape[2],
                                           cube_images.shape[3]])

        index_to_selected_images = patient_labels == 0
        index_to_selected_images = index_to_selected_images.flatten()
        class_to_3d_means_imgs[0] = \
            cube_images[index_to_selected_images.tolist(), :, :, :].mean(axis=0)

        index_to_selected_images = patient_labels == 1
        index_to_selected_images = index_to_selected_images.flatten()
        class_to_3d_means_imgs[1] = \
            cube_images[index_to_selected_images, :, :, :].mean(axis=0)

        region_to_class_to_3d_means_imgs[region] = class_to_3d_means_imgs

    return region_to_class_to_3d_means_imgs


def get_mean_over_flat_samples_per_region(dict_region_to_img, patient_labels):
    """
    """
    region_to_class_to_3d_means_imgs = {}

    for region, images in dict_region_to_img.items():
        region_to_class_to_3d_means_imgs[region] = \
            get_means_by_label_over_flat_samples(images, patient_labels)

    return region_to_class_to_3d_means_imgs


def get_means_by_label_over_flat_samples(data_samples, patient_labels):
    mean_images = np.zeros([2, data_samples.shape[1]])

    mean_images[0, :] = get_mean_over_selected_samples(data_samples, 0,
                                                       patient_labels)
    mean_images[1, :] = get_mean_over_selected_samples(data_samples, 1,
                                                       patient_labels)

    return mean_images


def get_mean_over_selected_samples(images, label_selected, patient_labels):
    index_to_selected_images = patient_labels == label_selected
    index_to_selected_images = index_to_selected_images.flatten()
    mean_over_images_selected = images[index_to_selected_images.tolist(),
                                :].mean(axis=0)
    return mean_over_images_selected


def evaluate_cubes_difference_by_planes(cube1, cube2, bool_test=False):
    if bool_test:
        print(cube1.shape)
        print(cube2.shape)

    cube_dif = cube1 - cube2
    cube_dif = cube_dif.__abs__()

    if bool_test:
        print("Diferncia dentre los cubos: {}".format(cube_dif.sum()))

    v1 = cube_dif.sum(axis=2).sum(axis=1)
    v2 = cube_dif.sum(axis=2).sum(axis=0)
    v3 = cube_dif.sum(axis=0).sum(axis=0)

    return np.argmax(v1), np.argmax(v2), np.argmax(v3)


def reconstruct_3d_image_from_flat_and_index(
        image_flatten, voxels_index, imgsize, reshape_kind):
    mri_image = np.zeros(imgsize)
    mri_image = mri_image.flatten()

    try:
        mri_image[voxels_index] = np.reshape(image_flatten,
                                             [image_flatten.shape[0], 1])
    except:
        mri_image[voxels_index] = image_flatten

    mri_image_3d = np.reshape(mri_image, imgsize, reshape_kind)

    return mri_image_3d


def plot_most_discriminative_section(img3d_1, img3d_2,
                                     path_to_save_image, cmap):
    dim1, dim2, dim3 = \
        evaluate_cubes_difference_by_planes(cube1=img3d_1,
                                            cube2=img3d_2,
                                            bool_test=True)

    plot_section_indicated(img3d_1, img3d_2, dim1, dim2,
                           dim3, path_to_save_image, cmap)


def plot_section_indicated(img3d_1, img3d_2, p1, p2, p3, path_to_save_image,
                           cmap, tittle=""):
    fig = plt.figure()
    fig.suptitle(tittle, fontsize=14)
    plt.subplot(321)
    plt.imshow(np.rot90(img3d_1[p1, :, :]), cmap=cmap)
    plt.subplot(323)
    plt.imshow(np.rot90(img3d_1[:, p2, :]), cmap=cmap)
    plt.subplot(325)
    plt.imshow(img3d_1[:, :, p3], cmap=cmap)
    plt.subplot(322)
    plt.imshow(np.rot90(img3d_2[p1, :, :]), cmap=cmap)
    plt.subplot(324)
    plt.imshow(np.rot90(img3d_2[:, p2, :]), cmap=cmap)
    plt.subplot(326)
    plt.imshow(img3d_2[:, :, p3], cmap=cmap)
    plt.savefig(filename=path_to_save_image, format="png")


def plot_individual_sample_by_planes_indicated(img3d, p1, p2, p3,
                                               path_to_save_image,
                                               cmap, tittle=""):
    fig = plt.figure()
    fig.suptitle(tittle, fontsize=14)
    plt.subplot(311)
    plt.imshow(np.rot90(img3d[p1, :, :]), cmap=cmap)
    plt.subplot(312)
    plt.imshow(np.rot90(img3d[:, p2, :]), cmap=cmap)
    plt.subplot(313)
    plt.imshow(img3d[:, :, p3], cmap=cmap)
    plt.savefig(filename=path_to_save_image, format="png")


def plot_comparaision_images_ADvsNOR(whole_reconstruction, origin_image,
                                     path_reconstruction_images, cmap):
    """

    :param whole_reconstruction:
    :param origin_image:
    :param path_reconstruction_images:
    :param cmap:
    :return:
    """

    recons_NOR = whole_reconstruction[0, :, :, :]
    recons_AD = whole_reconstruction[1, :, :, :]

    original_NOR = origin_image[0, :, :, :]
    original_AD = origin_image[1, :, :, :]

    # Original AD vs Original NOR
    plot_section_indicated(
        img3d_1=original_NOR,
        img3d_2=original_AD,
        p1=settings.planos_hipocampo["p1"],
        p2=settings.planos_hipocampo["p2"],
        p3=settings.planos_hipocampo["p3"],
        path_to_save_image=os.path.join(path_reconstruction_images,
                                        "Original_NORvsOriginal_AD.png"),
        cmap=cmap,
        tittle="Original NOR vs original AD")

    # Reconstructed AD vs Reconstructred Nor
    plot_section_indicated(
        img3d_1=recons_NOR,
        img3d_2=recons_AD,
        p1=settings.planos_hipocampo["p1"],
        p2=settings.planos_hipocampo["p2"],
        p3=settings.planos_hipocampo["p3"],
        path_to_save_image=os.path.join(path_reconstruction_images,
                                        "Reconstructed_NORvsReconstructed_AD.png"),
        cmap=cmap,
        tittle="Reconstructed NOR vs Reconstructed AD")

    # Reconstructed AD vs Original AD
    plot_section_indicated(
        img3d_1=recons_NOR,
        img3d_2=original_NOR,
        p1=settings.planos_hipocampo["p1"],
        p2=settings.planos_hipocampo["p2"],
        p3=settings.planos_hipocampo["p3"],
        path_to_save_image=os.path.join(path_reconstruction_images,
                                        "Reconstructed_NORvsOriginal_NOR.png"),
        cmap=cmap,
        tittle="Reconstructed NOR vs Original NOR")

    # Reconstructed NOR vs Original NOR
    plot_section_indicated(
        img3d_1=recons_AD,
        img3d_2=original_AD,
        p1=settings.planos_hipocampo["p1"],
        p2=settings.planos_hipocampo["p2"],
        p3=settings.planos_hipocampo["p3"],
        path_to_save_image=os.path.join(path_reconstruction_images,
                                        "Reconstructed_ADvsOriginal_AD.png"),
        cmap=cmap,
        tittle="Reconstructed AD vs Original AD")


def plot_comparaison_images_ReconstructedvsOriginal(
        original_3dimg, reconstruction_3dimg, path_reconstruction_images,
        cmap, title):

    # Reconstructed NOR vs Original NOR
    plot_section_indicated(
        img3d_1=original_3dimg,
        img3d_2=reconstruction_3dimg,
        p1=settings.planos_hipocampo["p1"],
        p2=settings.planos_hipocampo["p2"],
        p3=settings.planos_hipocampo["p3"],
        path_to_save_image=os.path.join(path_reconstruction_images,
                                        "Reconstructed_ADvsOriginal_AD.png"),
        cmap=cmap,
        tittle=title)

    # recons.plot_most_discriminative_section(
    #    img3d_1=whole_reconstruction[0, :, :, :],
    #    img3d_2=whole_reconstruction[1, :, :, :],
    #    path_to_save_image=path_image,
    #    cmap=cmap)

    # if logs:
    #    evaluate_difference_full_image = whole_reconstruction[0, :, :, :].flatten() \
    #                                     - whole_reconstruction[1, :, :, :].flatten()
    #    total_difference = sum(abs(evaluate_difference_full_image))
    #    print("Total difference between images reconstructed {0}".format(total_difference))

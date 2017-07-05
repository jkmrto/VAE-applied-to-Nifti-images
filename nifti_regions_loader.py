import numpy as np  # Se genera la mascara a partir del atlas
from lib.data_loader import PET_stack_NORAD
from lib.data_loader import pet_atlas
import nibabel as nib
from lib.data_loader import MRI_stack_NORAD
from lib.data_loader import PET_stack_NORAD
import os
from lib import output_utils
from lib.data_loader import mri_atlas
from lib.data_loader import pet_atlas
from lib import session_helper as session
import settings


def recortar_region(stack_dict, region, atlas, thval=0):
    """
    stack_dict = {'stack', 'imgsize', 'total_size', 'voxel_index'}
    """
    total_size = stack_dict['total_size']
    imgsize = stack_dict['imgsize']
    stack = stack_dict['stack']
    voxels_index = stack_dict['voxel_index']

    # atlas template
    mask_atlas = np.zeros(imgsize)
    mask_atlas = np.reshape(mask_atlas, [total_size], "F")
    map_region_voxels = atlas[region]
    real_region_voxels = voxels_index[map_region_voxels]
    mask_atlas[real_region_voxels] = 1

    mask_atlas = np.reshape(mask_atlas, imgsize, "F")

    eq = [[2, 1], [2, 0], [0, 0]]
    ndim = len(mask_atlas.shape)
    minidx = np.zeros(ndim, dtype=np.int)
    maxidx = np.zeros(ndim, dtype=np.int)

    for ax in range(ndim):
        filtered_lst = [idx for idx, y in enumerate(
            mask_atlas.any(axis=eq[ax][0]).any(axis=eq[ax][1])) if y > thval]
        minidx[ax] = min(filtered_lst)
        maxidx[ax] = max(filtered_lst)

    stack_masked = np.zeros((stack.shape[0], abs(minidx[0] - maxidx[0]),
                             abs(minidx[1] - maxidx[1]),
                             abs(minidx[2] - maxidx[2])))

    for patient in range(stack_masked.shape[0]):
        second_image = np.zeros(total_size)

        image = np.zeros(total_size)  # template

        voxels_patient_region_selected = stack[patient, map_region_voxels]
        print(voxels_patient_region_selected.shape)
        voxels_patient_region_selected = voxels_patient_region_selected.reshape(voxels_patient_region_selected.size, 1)
        image[real_region_voxels.tolist()] = voxels_patient_region_selected

        image = np.reshape(image, imgsize, "F")

        out_image = image[minidx[0]:maxidx[0],
                    minidx[1]:maxidx[1], minidx[2]:maxidx[2]]

        stack_masked[patient, :, :, :] = out_image
    return stack_masked


def load_pet_regions_segmented(list_regions, folder_to_store_3d_images=None,
                               bool_logs=True, out_csv_region_dimensions=None):
    dic_regions_segmented = {}
    list_dics_regions_dimensions = []

    patient_output = 10

    if bool_logs:
        print("Loading Pet Stack")
    dict_norad_pet = PET_stack_NORAD.get_full_stack()  # 'stack' 'voxel_index' 'labels'

    if bool_logs:
        print(" Stack Loaded")
    atlas = pet_atlas.load_atlas()

    for region in list_regions:

        if bool_logs:
            print("Segmenting Region {}".format(region))
        region_segmented = recortar_region(stack_dict=dict_norad_pet,
                                           region=region,
                                           atlas=atlas)
        dic_regions_segmented[region] = region_segmented

        # Debugging code
        if bool_logs:
            print("Region {} Segmented".format(region))

        if folder_to_store_3d_images is not None:
            img = nib.Nifti1Image(region_segmented[patient_output, :, :, :], np.eye(4))
            img.to_filename(os.path.join(folder_to_store_3d_images,
                                         "region_{}.nii".format(region)))

        if out_csv_region_dimensions is not None:
            dic_region_dimension = {
                "region": region,
                "n_voxels": atlas[region],
                "width_3d": region_segmented.shape[1],
                "height_3d": region_segmented.shape[2],
                "depth_3d": region_segmented.shape[3],
            }

            list_dics_regions_dimensions.append(dic_region_dimension)

    if out_csv_region_dimensions is not None:
        output_utils.print_dictionary_with_header(out_csv_region_dimensions, list_dics_regions_dimensions)

    return dic_regions_segmented


load_pet_regions_segmented([2, 3, 4, 5, 6, 7, 8],
                           folder_to_store_3d_images=None,
                           bool_logs=True,
                           out_csv_region_dimensions="regions_dimensions.csv")


def load_mri_regions_segmented(list_regions, folder_to_store_3d_images=None,
                               bool_logs=True):
    """

    :param list_regions:
    :param folder_to_store_3d_images:
    :param bool_logs:
    :return:
    """
    if folder_to_store_3d_images is not None:
        patient_to_output = 10

    dic_mri_gm_regions_segmented = {}
    dic_mri_wm_regions_segmented = {}

    if bool_logs:
        print("Loading MRI Stacks")

    dict_norad_gm = MRI_stack_NORAD.get_gm_stack()
    if bool_logs:
        print("GM MRI stack loaded")
    dict_norad_wm = MRI_stack_NORAD.get_wm_stack()
    if bool_logs:
        print("WM MRI stack loaded")

    if bool_logs:
        print(" Stack Loaded")

    atlas = pet_atlas.load_atlas()

    for region in list_regions:
        if bool_logs:
            print("Segmenting Region {}".format(region))
            print("Segmenting MRI GM".format(region))

        region_segmented_mri_gm = recortar_region(stack_dict=dict_norad_gm,
                                                  region=region,
                                                  atlas=atlas)
        if bool_logs:
            print("Segmenting MRI WM".format(region))

        region_segmented_mri_wm = recortar_region(stack_dict=dict_norad_wm,
                                                  region=region,
                                                  atlas=atlas)

        dic_mri_gm_regions_segmented[region] = region_segmented_mri_gm
        dic_mri_wm_regions_segmented[region] = region_segmented_mri_wm

        # Debugging code
        if bool_logs:
            print("Region {} Segmented".format(region))

        if folder_to_store_3d_images is not None:
            img = nib.Nifti1Image(region_segmented_mri_gm[patient_to_output, :, :, :], np.eye(4))
            img.to_filename(os.path.join(folder_to_store_3d_images,
                                         "mri_gm_region_{}.nii".format(region)))

            img = nib.Nifti1Image(region_segmented_mri_wm[patient_to_output, :, :, :], np.eye(4))
            img.to_filename(os.path.join(folder_to_store_3d_images,
                                         "mri_wm_region_{}.nii".format(region)))

    return dic_mri_gm_regions_segmented, dic_mri_wm_regions_segmented


# load_mri_regions_segmented([2,3,4,5,6,7,8], "test")


def load_mri_regions_flatten(list_regions):
    """

    :param list_regions:
    :return: dic_wm[region], dic_gm[region] || region -> voxels_flatten
    """
    dict_norad_gm = MRI_stack_NORAD.get_gm_stack()['stack']
    dict_norad_wm = MRI_stack_NORAD.get_wm_stack()['stack']

    atlas = mri_atlas.load_atlas_mri()

    dic_regions_to_flatten_voxels_mri_gm = {}
    dic_regions_to_flatten_voxels_mri_wm = {}

    for region in list_regions:
        dic_regions_to_flatten_voxels_mri_gm[region] = \
            dict_norad_gm[:, atlas[region]]

        dic_regions_to_flatten_voxels_mri_wm[region] = \
            dict_norad_wm[:, atlas[region]]

    return dic_regions_to_flatten_voxels_mri_gm, \
           dic_regions_to_flatten_voxels_mri_wm


# out_gm, out_wm = load_mri_regions_flatten([2, 3, 4, 5, 6])


def load_pet_regions_flatten(list_regions):
    """

    :param list_regions:
    :return: dic_pet[region] ->
                voxels_flatten Sh[nsamples x n_voxels_region]
    """

    dict_norad_pet = PET_stack_NORAD.get_full_stack()['stack']

    atlas = pet_atlas.load_atlas()

    dic_regions_to_flatten_voxels_pet = {}

    for region in list_regions:
        dic_regions_to_flatten_voxels_pet[region] = \
            dict_norad_pet[:, atlas[region]]

    return dic_regions_to_flatten_voxels_pet


# out = load_pet_regions_flatten([2, 3, 4, 5, 6])


def test(nifti_region_to_save, path_where_store_out="pet_regions_segmented"):
    """
    It save "region to save" nifti image in the folder "path_where_store_out"
    for all the patient
    :param region_to_save:
    :param path_where_store_out:
    :return:
    """
    regions_used = "three"
    list_regions = session.select_regions_to_evaluate(regions_used)
    dic_regions_segmented = load_pet_regions_segmented(list_regions)

    region_container_3d = dic_regions_segmented[nifti_region_to_save]  # [patients x heigh, width, depth]

    for patient in range(0, region_container_3d.shape[0], 1):
        img = nib.Nifti1Image(region_container_3d[patient, :, :, :], np.eye(4))
        img.to_filename(os.path.join(path_where_store_out,
                                     "region_{0},patient_{1}.nii".format(regions_used, patient)))


def over_test():
    path = os.path.join(settings.path_to_project, "pet_regions_segmented")
    test(region_to_save=3, path_where_store_out=path)

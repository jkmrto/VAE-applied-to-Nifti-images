import os
import nibabel as nib
import numpy as np  # Se genera la mascara a partir del atlas
from lib.data_loader import MRI_stack_NORAD
from lib.data_loader import mri_atlas
from lib.data_loader.utils_images3d import recortar_region


def load_mri_regions_segmented3d(list_regions, folder_to_store_3d_images=None,
                                 bool_logs=True):
    """
    THis functions return both stack, GM and WM stack, segmented 3d per region.
    :param list_regions:
    :param folder_to_store_3d_images:
    :param bool_logs:
    :return: 2 x [dic[region] -> ty[np.array] sh[n_samples, w, h, d]]
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

    atlas = mri_atlas.load_atlas_mri()

    for region in list_regions:
        if bool_logs:
            print("Segmenting Region {}".format(region))
            print("Segmenting MRI GM".format(region))

        region_segmented_mri_gm = recortar_region(stack_dict=dict_norad_gm,
                                                  region=region,
                                                  atlas=atlas,
                                                  reshape_kind="C"
                                                  )
        if bool_logs:
            print("Segmenting MRI WM".format(region))

        region_segmented_mri_wm = recortar_region(stack_dict=dict_norad_wm,
                                                  region=region,
                                                  atlas=atlas,
                                                  reshape_kind="C")

        dic_mri_gm_regions_segmented[region] = region_segmented_mri_gm
        dic_mri_wm_regions_segmented[region] = region_segmented_mri_wm

        # Debugging code
        if bool_logs:
            print("Region {} Segmented".format(region))

        if folder_to_store_3d_images is not None:
            img = nib.Nifti1Image(
                region_segmented_mri_gm[patient_to_output, :, :, :], np.eye(4))
            img.to_filename(os.path.join(folder_to_store_3d_images,
                                         "mri_gm_region_{}.nii".format(region)))

            img = nib.Nifti1Image(
                region_segmented_mri_wm[patient_to_output, :, :, :], np.eye(4))
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




def load_mri_data_flat(list_regions):
    # Loading the stack of images
    dic_regions_to_flatten_voxels_mri_gm, dic_regions_to_flatten_voxels_mri_wm = \
        load_mri_regions_flatten(list_regions)
    patient_labels = MRI_stack_NORAD.load_patients_labels()
    n_samples = dic_regions_to_flatten_voxels_mri_gm[list_regions[0]].shape[0]

    return dic_regions_to_flatten_voxels_mri_gm, dic_regions_to_flatten_voxels_mri_wm, \
           patient_labels, n_samples


def load_mri_data_3d(list_regions):
    # Loading the stack of images
    dic_regions_to_3dimg_mri_gm, dic_regions_to_3dimg_mri_wm = \
        load_mri_regions_segmented3d(list_regions)
    patient_labels = MRI_stack_NORAD.load_patients_labels()
    n_samples = len(patient_labels)

    return dic_regions_to_3dimg_mri_gm, dic_regions_to_3dimg_mri_wm, \
           patient_labels, n_samples

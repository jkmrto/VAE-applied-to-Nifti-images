import os

import nibabel as nib
import numpy as np

import settings
from lib import session_helper as session
from lib.data_loader import MRI_stack_NORAD
from lib.data_loader import PET_stack_NORAD
from lib.data_loader import mri_atlas
from lib.data_loader import pet_atlas
from lib.data_loader.pet_loader import \
    load_pet_regions_segmented
from lib.data_loader.utils_general import \
    load_parameters_and_atlas_by_images_used
from lib.data_loader.utils_mask3d import \
    get_whole_region_mask_and_region_segmented_mask
from lib.utils.os_aux import create_directories


def test_over_mask_over_regions_segmented_and_whole_extractor(region, images_used):
    """
    This functions evaluates if the number of voxels activated in the whole 3d
    mask [whole_mask_flatten] is the same that the number of voxels activated in
    :param region:
    :param images_used:
    :return:
    """
    atlas = None
    dict_parameters = None
    reshape_kind = None

    if images_used == "MRI":
        atlas = mri_atlas.load_atlas_mri()
        dict_parameters = MRI_stack_NORAD.get_parameters()
        reshape_kind = "C"

    elif images_used == "PET":
        atlas = pet_atlas.load_atlas()
        dict_parameters = PET_stack_NORAD.get_parameters()
        reshape_kind = "F"

    whole_mask_flatten, mask_segmented_flatten = \
        get_whole_region_mask_and_region_segmented_mask(
        atlas=atlas,
        dict_parameters=dict_parameters,
        region=region,
        reshape_kind=reshape_kind)

    print("Number voxels activaed in whole MRI: {0}\n"
         "length whole image: {1} \n"
          "Number voxles activaed in region segmented 3d: {2}\n"
          "length region segmented {3}".format(
        sum(whole_mask_flatten), len(whole_mask_flatten),
        sum(mask_segmented_flatten), len(mask_segmented_flatten)))

test_over_mask_over_regions_segmented_and_whole_extractor(region=3, images_used="MRI")

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

    region_container_3d = dic_regions_segmented[
        nifti_region_to_save]  # [patients x heigh, width, depth]

    for patient in range(0, region_container_3d.shape[0], 1):
        img = nib.Nifti1Image(region_container_3d[patient, :, :, :], np.eye(4))
        img.to_filename(os.path.join(path_where_store_out,
                                     "region_{0},patient_{1}.nii".format(
                                         regions_used, patient)))


def over_test():
    path = os.path.join(settings.path_to_project, "pet_regions_segmented")
    test(nifti_region_to_save=3, path_where_store_out=path)

#over_test()


def test_over_region_segmentation_and_reconstruction():
    images_used = "PET"
    test_out_folder = "test_segmentation_and_reconstruction"
    regions_used = "all"
    list_regions = session.select_regions_to_evaluate(regions_used)
    patient_selected = 0
    create_directories([test_out_folder])

    atlas, dict_parameters, reshape_kind = \
        load_parameters_and_atlas_by_images_used(images_used)

    dic_regions_segmented = load_pet_regions_segmented(list_regions=list_regions)
    image3d_reconstructed_flatten = np.zeros(dict_parameters['total_size'])  # template

    for region in list_regions:
        print(region)
        region_3d_segmented = \
            dic_regions_segmented[region][patient_selected, :, :, :]

        region_3d_segmented_totalsize = np.array(region_3d_segmented.shape).prod()
        region_3d_segmented_flatten = np.reshape(region_3d_segmented,
            [region_3d_segmented_totalsize], reshape_kind)

        whole_mask_flatten, mask_segmented_flatten=\
            get_whole_region_mask_and_region_segmented_mask(
            region=region,
            dict_parameters=dict_parameters,
            atlas=atlas,
            reshape_kind=reshape_kind)

        print(len(mask_segmented_flatten))
        print(len(region_3d_segmented_flatten))

        segmented_voxels_selected = region_3d_segmented_flatten[mask_segmented_flatten==1]

        image3d_reconstructed_flatten[whole_mask_flatten==1] = segmented_voxels_selected

    image3d_reconstructed = np.reshape(image3d_reconstructed_flatten,
        dict_parameters['imgsize'],reshape_kind)

    img = nib.Nifti1Image(image3d_reconstructed[:, :, :], np.eye(4))
    img.to_filename(os.path.join(test_out_folder,
                                 "region_{0},patient_{1}.nii".format(
                                     regions_used, patient_selected)))

#test_over_region_segmentation_and_reconstruction()

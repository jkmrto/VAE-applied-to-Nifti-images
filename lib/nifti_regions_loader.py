import os

import nibabel as nib
import numpy as np  # Se genera la mascara a partir del atlas

import settings
from lib import session_helper as session
from lib.data_loader import MRI_stack_NORAD
from lib.data_loader import PET_stack_NORAD
from lib.data_loader import mri_atlas
from lib.data_loader import pet_atlas
from lib.utils import output_utils
from lib.aux_functionalities.functions import create_directories

"The MRI segmentation 3d functions has not been tested yet"


def generate_region_3dmaskatlas_given_no_background_region_index(
        no_bg_region_voxels_index, reshape_kind, imgsize, totalsize):
    mask_atlas = np.zeros(imgsize)
    mask_atlas = np.reshape(mask_atlas, [totalsize], reshape_kind)
    mask_atlas[no_bg_region_voxels_index] = 1
    mask_atlas = np.reshape(mask_atlas, imgsize, reshape_kind)

    return mask_atlas


def delim_3dmask(mask_atlas, thval=0.5):
    """

    :param mask_atlas:
    :return:
    """
    eq = [[2, 1], [2, 0], [0, 0]]
    ndim = len(mask_atlas.shape)
    minidx = np.zeros(ndim, dtype=np.int)
    maxidx = np.zeros(ndim, dtype=np.int)

    for ax in range(ndim):
        filtered_lst = [idx for idx, y in enumerate(
            mask_atlas.any(axis=eq[ax][0]).any(axis=eq[ax][1])) if y > thval]
        minidx[ax] = min(filtered_lst)
        maxidx[ax] = max(filtered_lst)

    return minidx, maxidx


def get_whole_region_mask_and_region_segmented_mask(region, dict_parameters,
                                                    atlas, reshape_kind):
    """
    This functions returns two flat arrays. Each array only contents [0|1]
    indicating that voxel belong to the indicated region.
    :param region:
    :param dict_parameters:
    :param atlas:
    :param reshape_kind:
    :return:
    whole_mask_flatten: ty[np.darray] sh[MRI|PET 3d shape]
    mask_segmented_flatten: ty[np.darray] sh[region 3d segmented shape]
    """
    voxels_index = dict_parameters['voxel_index']
    imgsize = dict_parameters['imgsize']
    total_size = dict_parameters['total_size']

    # index to region voxels nobg
    no_bg_region_voxels_index = voxels_index[atlas[region]]

    mask_atlas = generate_region_3dmaskatlas_given_no_background_region_index(
        no_bg_region_voxels_index=no_bg_region_voxels_index,
        reshape_kind = reshape_kind,
        imgsize=imgsize,
        totalsize=total_size)

    whole_mask_flatten = np.reshape(mask_atlas, [total_size],reshape_kind )

    minidx, maxidx = delim_3dmask(mask_atlas, thval=0.5)

    # 3D Mask atlas segmeted
    atlas_segmented = mask_atlas[minidx[0]:maxidx[0]+1, minidx[1]:maxidx[1]+1,
                            minidx[2]:maxidx[2]+1]

    mask_segmented_flatten = np.reshape(atlas_segmented,
        [np.array(atlas_segmented.shape).prod()], reshape_kind)

    return whole_mask_flatten, mask_segmented_flatten


def test_over_mask_over_regions_segmented_and_whole_extractor(region, images_used):

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

#test_over_mask_over_regions_segmented_and_whole_extractor(region=1, images_used="PET")


def reconstruct_mri_image(list_regions, type_image="PET"):
    dict_parameters = PET_stack_NORAD.get_parameters()
    atlas = pet_atlas.load_atlas()


    for region in list_regions:
        out1, out2 = get_whole_region_mask_and_region_segmented_mask(
            atlas=atlas,
            dict_parameters=dict_parameters,
            region=3)


def get_region_indexes_in_3d_figure(atlas, region, stack_dict,
                                    reshape_kind="F"):
    """

    :param atlas: dic[region] -> index to voxels belonged to that region sh[len]
    :param region: region_index
    :param stack_dict: dict['total_size'|'img_size'|'voxel_index']
     stack_dict['voxel_index'] -> no background voxels index
    :param reshape_kind:
    :return:
    """
    total_size = stack_dict['total_size']
    imgsize = stack_dict['imgsize']
    voxels_index = stack_dict['voxel_index']

    # atlas template

    map_region_voxels = atlas[region]  # index refered to nbground voxels
    no_bg_region_voxels_index = voxels_index[map_region_voxels]

    mask_atlas = generate_region_3dmaskatlas_given_no_background_region_index(
        no_bg_region_voxels_index=no_bg_region_voxels_index,
        reshape_kind = reshape_kind,
        imgsize=imgsize,
        totalsize=total_size)

    return mask_atlas, no_bg_region_voxels_index


def recortar_region(stack_dict, region, atlas, thval=0, reshape_kind="F"):
    """
    stack_dict = {'stack', 'imgsize', 'total_size', 'voxel_index'}
    """
    total_size = stack_dict['total_size']
    imgsize = stack_dict['imgsize']
    stack = stack_dict['stack']
    voxels_index = stack_dict['voxel_index']
    map_region_voxels = atlas[region]

    mask_atlas, no_bg_region_voxels_index = \
        get_region_indexes_in_3d_figure(atlas, region, stack_dict,
                                        reshape_kind="F")

    minidx, maxidx = delim_3dmask(mask_atlas)

    stack_masked = np.zeros((stack.shape[0], abs(minidx[0] - maxidx[0])+1,
                             abs(minidx[1] - maxidx[1])+1,
                             abs(minidx[2] - maxidx[2])+1))

    for patient in range(stack_masked.shape[0]):

        image = np.zeros(total_size)  # template
        voxels_patient_region_selected = stack[patient, map_region_voxels]

        try:
            image[
                no_bg_region_voxels_index.tolist()] = voxels_patient_region_selected
        except:
            voxels_patient_region_selected = voxels_patient_region_selected.reshape(
                voxels_patient_region_selected.size,
                1)
            image[
                no_bg_region_voxels_index.tolist()] = voxels_patient_region_selected

        image = np.reshape(image, imgsize, reshape_kind)

        out_image = image[minidx[0]:maxidx[0]+1,
                    minidx[1]:maxidx[1]+1, minidx[2]:maxidx[2]+1]

        stack_masked[patient, :, :, :] = out_image
    return stack_masked


def load_pet_regions_segmented(list_regions, folder_to_store_3d_images=None,
                               bool_logs=True, out_csv_region_dimensions=None):
    """

    :param list_regions:
    :param folder_to_store_3d_images:
    :param bool_logs:
    :param out_csv_region_dimensions:
    :return: dic[region] -> ty[np.array] sh[n_samples, w, h, d]
    """
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
            img = nib.Nifti1Image(region_segmented[patient_output, :, :, :],
                                  np.eye(4))
            img.to_filename(os.path.join(folder_to_store_3d_images,
                                         "region_{}.nii".format(region)))

        if out_csv_region_dimensions is not None:
            dic_region_dimension = {
                "region": region,
                "n_voxels": atlas[region].size,
                "width_3d": region_segmented.shape[1],
                "height_3d": region_segmented.shape[2],
                "depth_3d": region_segmented.shape[3],
            }

            list_dics_regions_dimensions.append(dic_region_dimension)

    if out_csv_region_dimensions is not None:
        output_utils.print_dictionary_with_header(out_csv_region_dimensions,
                                                  list_dics_regions_dimensions)

    return dic_regions_segmented


# region_sample = load_pet_regions_segmented(list_regions=[1])[1][1,:,:,:]
# print(region_sample.shape)

# load_pet_regions_segmented(list_regions=session_helper.select_regions_to_evaluate("all"),
#                           folder_to_store_3d_images=None,
#                           bool_logs=True,
#                           out_csv_region_dimensions="regions_dimensions.csv")

def load_mri_regions_segmented3d(list_regions, folder_to_store_3d_images=None,
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

def load_pet_data_flat(list_regions):
    dic_regions_to_flatten_voxels_pet = load_pet_regions_flatten(list_regions)
    patient_labels = PET_stack_NORAD.load_patients_labels()
    n_samples = dic_regions_to_flatten_voxels_pet[list_regions[0]].shape[0]

    return dic_regions_to_flatten_voxels_pet, patient_labels, n_samples


# [out1, out2, out3] = load_pet_data([3,4,5,6,7,7])


def load_mri_data_flat(list_regions):
    # Loading the stack of images
    dic_regions_to_flatten_voxels_mri_gm, dic_regions_to_flatten_voxels_mri_wm = \
        load_mri_regions_flatten(list_regions)
    patient_labels = MRI_stack_NORAD.load_patients_labels()
    n_samples = dic_regions_to_flatten_voxels_mri_gm[list_regions[0]].shape[0]

    return dic_regions_to_flatten_voxels_mri_gm, dic_regions_to_flatten_voxels_mri_wm, \
           patient_labels, n_samples


def load_pet_data_3d(list_regions):
    """

    :param list_regions:
    :return:
    region_to_3d_img_dict_pet: dict[region] -> 3d_image sh[n_samples, w, h, d]
    patient_labels: ty[np.array] sh[n_samples] ]0,1[
    n_samples: value sh[1]
    """
    region_to_3dimg_dict_pet = load_pet_regions_segmented(list_regions,
                                                          bool_logs=False)
    patient_labels = PET_stack_NORAD.load_patients_labels()
    n_samples = len(patient_labels)

    return region_to_3dimg_dict_pet, patient_labels, n_samples


def load_mri_data_3d(list_regions):
    # Loading the stack of images
    dic_regions_to_3dimg_mri_gm, dic_regions_to_3dimg_mri_wm = \
        load_mri_regions_segmented3d(list_regions)
    patient_labels = MRI_stack_NORAD.load_patients_labels()
    n_samples = len(patient_labels)

    return dic_regions_to_3dimg_mri_gm, dic_regions_to_3dimg_mri_wm, \
           patient_labels, n_samples


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
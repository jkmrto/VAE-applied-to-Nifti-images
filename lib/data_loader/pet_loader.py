from lib.utils import output_utils
from lib.data_loader import PET_stack_NORAD
from lib.data_loader import pet_atlas
from lib.data_loader.utils_images3d import recortar_region
import os
import numpy as np
import nibabel as nib


def load_pet_regions_segmented(list_regions, folder_to_store_3d_images=None,
                               bool_logs=True, out_csv_region_dimensions=None):
    """
    This functions generates a dictionary indexed by region, which content per
    each region in the minimum 3d cube which contents all the voxels of that region
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
                                           reshape_kind="F",
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


def load_pet_regions_flatten(list_regions):
    """

    :param list_regions:
    :return: dic_pet[region] -> voxels_flatten Sh[nsamples, n_voxels_region]
    """

    dict_norad_pet = PET_stack_NORAD.get_full_stack()['stack']

    atlas = pet_atlas.load_atlas()

    dic_regions_to_flatten_voxels_pet = {}

    for region in list_regions:
        dic_regions_to_flatten_voxels_pet[region] = \
            dict_norad_pet[:, atlas[region]]

    return dic_regions_to_flatten_voxels_pet


# out = load_pet_regions_flatten([2, 3, 4, 5, 6])

def load_pet_data_flat(list_regions, bool_logs=False):
    dic_regions_to_flatten_voxels_pet = load_pet_regions_flatten(list_regions)
    patient_labels = PET_stack_NORAD.load_patients_labels()
    n_samples = dic_regions_to_flatten_voxels_pet[list_regions[0]].shape[0]

    return dic_regions_to_flatten_voxels_pet, patient_labels, n_samples


# [out1, out2, out3] = load_pet_data([3,4,5,6,7,7])


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

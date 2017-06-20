import numpy as np  # Se genera la mascara a partir del atlas
from lib.data_loader import PET_stack_NORAD
from lib.data_loader import pet_atlas
import nibabel as nib
import os
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

    for i in range(stack_masked.shape[0]):
        second_image = np.zeros(total_size)

        image = np.zeros(total_size) #template

        image[real_region_voxels.tolist()] = stack[i, map_region_voxels]

        image = np.reshape(image, imgsize, "F")

        out_image = image[minidx[0]:maxidx[0],
                            minidx[1]:maxidx[1], minidx[2]:maxidx[2]]

        stack_masked[i, :, :, :] = out_image
    return stack_masked


def load_regions_segmented(list_regions, folder_to_store_3d_images=None):

    dic_regions_segmented = {}

    print("Loading Pet Stack")
    dict_norad_pet = PET_stack_NORAD.get_full_stack()  # 'stack' 'voxel_index' 'labels'
    print(" Stack Loaded")
    patient = 10
    atlas = pet_atlas.load_atlas()

    for region in list_regions:
        print("Segmenting Region {}".format(region))
        region_segmented = recortar_region(stack_dict=dict_norad_pet,
                                       region=region,
                                       atlas=atlas)
        print("Region {} Segmented".format(region))

        if folder_to_store_3d_images is not None:
            img = nib.Nifti1Image(region_segmented[patient,:,:,:], np.eye(4))
            img.to_filename(os.path.join(folder_to_store_3d_images,
                                         "region_{}.nii".format(region)))

        dic_regions_segmented[region] = region_segmented
    return dic_regions_segmented


def test(region_to_save, path_where_store_out="pet_regions_segmented"):
    """
    It save "region to save" nifti image in the folder "path_where_store_out"
    of all the patient
    :param region_to_save:
    :param path_where_store_out:
    :return:
    """
    regions_used = "three"
    list_regions = session.select_regions_to_evaluate(regions_used)
    dic_regions_segmented = load_regions_segmented(list_regions)

    region_container_3d = dic_regions_segmented[region_to_save] #[patients x heigh, width, depth]

    for patient in range(0, region_container_3d.shape[0], 1):
        img = nib.Nifti1Image(region_container_3d[patient, :, :, :], np.eye(4))
        img.to_filename(os.path.join(path_where_store_out,
            "region_{0},patient_{1}.nii".format(regions_used, patient)))

def over_test():
    path = os.path.join(settings.path_to_project, "pet_regions_segmented")
    test(region_to_save=3, path_where_store_out=path)
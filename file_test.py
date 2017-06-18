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
        STACK = stack[:, map_region_voxels]
        print(STACK.shape)
        print(second_image.shape)
        print(real_region_voxels.shape)
        image[real_region_voxels.tolist()] = STACK

        image = np.reshape(image, imgsize, "F")

        stack_masked[i, :, :, :] = image[minidx[0]:maxidx[0],
                                   minidx[1]:maxidx[1], minidx[2]:maxidx[2]]

    return stack_masked


regions_used = "all"
list_regions = session.select_regions_to_evaluate(regions_used)
print("Loading Pet Stack")
dict_norad_pet = PET_stack_NORAD.get_full_stack()  # 'stack' 'voxel_index' 'labels'
print(" Stack Loaded")

path = os.path.join(settings.path_to_project, "pet_regions_segmented")
patient = 10
atlas = pet_atlas.load_atlas()

for region in list_regions:
    print("Segmenting Region {}".format(region))
    region_segmented = recortar_region(stack_dict=dict_norad_pet,
                                       region=region,
                                       atlas=atlas)
    print("Region {} Segmented".format(region))
    img = nib.Nifti1Image(region_segmented[patient,:,:,:], np.eye(4))
    img.to_filename(os.path.join(path, "region_{}.nii".format(region)))


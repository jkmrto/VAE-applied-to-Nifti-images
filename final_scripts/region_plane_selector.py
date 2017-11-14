from lib.data_loader import pet_atlas
from lib.data_loader import PET_stack_NORAD
from lib.data_loader import pet_loader
from lib import reconstruct_helpers as recons
from lib.data_loader import utils_mask3d as utils3d
import numpy as np


def get_maximum_activation_planes_over_3dmask(mask3d, logs=False):

    v1 = mask3d.sum(axis=2).sum(axis=1)
    v2 = mask3d.sum(axis=2).sum(axis=0)
    v3 = mask3d.sum(axis=0).sum(axis=0)

    m1, m2, m3 = np.max(v1), np.max(v2), np.max(v3)
    p1, p2, p3 = np.argmax(v1), np.argmax(v2), np.argmax(v3)

    if logs:
        print("Section selected per axis postion: {0}, {1}, {2}".format(m1, m2, m3))
        print("Section selected per axis maximum: {0}, {1}, {2}".format(p1, p2, p3))

    return p1, p2, p3


def get_3dmask_segmented(voxels_index, total_size, imgsize, reshape_kind):

    activaction_mask = list(voxels_index)

    mask_full_img = np.zeros([total_size])
    mask_full_img[activaction_mask] = 1
    mask_full_img_3d = np.reshape(mask_full_img, imgsize, reshape_kind)

    minidx, maxidx = utils3d.delim_3dmask(mask_full_img_3d, thval=0.5)

    # 3D Mask atlas segmeted
    mask3d_segmented = mask_full_img_3d[minidx[0]:maxidx[0] + 1,
                       minidx[1]:maxidx[1] + 1,
                       minidx[2]:maxidx[2] + 1]

    return mask3d_segmented


def get_maximum_activation_planes(voxels_index, total_size, imgsize, reshape_kind):

    mask3d_segmented = get_3dmask_segmented(voxels_index, total_size, imgsize, reshape_kind)
    p1, p2, p3 = get_maximum_activation_planes_over_3dmask(mask3d=mask3d_segmented)

    return p1, p2, p3


def get_middle_planes(img3d, logs=False):

    shape = np.array(img3d.shape)

    p1 = int(shape[0]/2)
    p2 = int(shape[1]/2)
    p3 = int(shape[2]/2)

    if logs:
        print("Shape 3dimg {}".format(str(shape)))
        print("Section selected per axis maximum: {0}, {1}, {2}".format(p1, p2, p3))

    return p1, p2, p3


def auto_test():

    sample_NOR = 10
    sample_AD = 120

    atlas = pet_atlas.load_atlas()
    region = 38
    voxels_desired = atlas[region]

    dict_region3d_stack = pet_loader.load_pet_regions_segmented(list_regions=[region])
    region_stack3d = dict_region3d_stack[region]

    pet_dict_stack = PET_stack_NORAD.get_parameters()
    voxels_index = pet_dict_stack['voxel_index'] # no_bg_index to real position

    final_voxels_selected_index = voxels_index[voxels_desired]

    p1, p2, p3 = get_maximum_activation_planes(
        voxels_index=final_voxels_selected_index,
        total_size=pet_dict_stack['total_size'],
        imgsize=pet_dict_stack['imgsize'],
        reshape_kind="F")

    recons.plot_section_indicated(
        img3d_1=region_stack3d[sample_NOR,:,:,:],
        img3d_2=region_stack3d[sample_AD,:,:,:],
        p1=p1, p2=p2, p3=p3,
        path_to_save_image="prueba.png",
        cmap="jet",
        tittle="NOR vs AD")

    p1, p2, p = get_middle_planes(img3d=region_stack3d[0,:,:,:])

    recons.plot_section_indicated(
        img3d_1=region_stack3d[sample_NOR,:,:,:],
        img3d_2=region_stack3d[sample_AD,:,:,:],
        p1=p1, p2=p2, p3=p3,
        path_to_save_image="prueba2.png",
        cmap="jet",
        tittle="NOR vs AD")


def get_dict_region_to_maximum_activation_planes(
        list_regions, atlas, stack_parameters):

    region_to_maximum_activation_planes = {}
    voxels_index = stack_parameters['voxel_index'] # no_bg_index to real position

    for region in list_regions:
        voxels_desired = atlas[region]
        final_voxels_selected_index = voxels_index[voxels_desired]

        p1, p2, p3 = get_maximum_activation_planes(
            voxels_index=final_voxels_selected_index,
            total_size=stack_parameters['total_size'],
            imgsize=stack_parameters['imgsize'],
            reshape_kind="F")

        region_to_maximum_activation_planes[region] = [p1, p2, p3]
#auto_test()


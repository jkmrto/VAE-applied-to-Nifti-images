import numpy as np


def generate_region_3dmaskatlas(
        no_bg_region_voxels_index, reshape_kind, imgsize, totalsize):
    """
    Generate 3d atlas of the given size.
    That atlas only will contain
    :param no_bg_region_voxels_index:
    :param reshape_kind:
    :param imgsize:
    :param totalsize:
    :return:
    """
    mask_atlas = np.zeros(imgsize)
    mask_atlas = np.reshape(mask_atlas, [totalsize], reshape_kind)
    mask_atlas[no_bg_region_voxels_index] = 1
    mask_atlas = np.reshape(mask_atlas, imgsize, reshape_kind)

    return mask_atlas


def delim_3dmask(mask_atlas, thval=0.5):
    """
    Given a 3d [0|1] mask, this function finds the delimitation over the mask,
    so all the activated voxels are contained in the segmented portion.
    :param mask_atlas: ty[np.array] sh[w,h,d]
    :return: minidx: sh[3] -> one index per axis
             maxidx: sh[3]-> one index per axis
    IMP, all the actived voxels are in:
    mask_atlas[minidx[0]:maxidx[0], minidx[1]:maxidx[1],minidx[2] - maxidx[2]]
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
    One array is the whole image masked for the region indicated
    One array is the segmented region [3d] masked for the voxels
        which are no background
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

    mask_atlas = generate_region_3dmaskatlas(
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
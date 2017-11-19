import numpy as np  # Se genera la mascara a partir del atlas
from lib.data_loader import utils_mask3d
from lib.data_loader.utils_general import load_parameters_and_atlas_by_images_used


def recortar_region(stack_dict, region, atlas, thval=0, reshape_kind="F"):
    """

    :param stack_dict:
    :param region:
    :param atlas:
    :param thval:
    :param reshape_kind: "C"|"F"  ("F" => MRI, "C" => PET)
    :return:
    """
    total_size = stack_dict['total_size']
    imgsize = stack_dict['imgsize']
    stack = stack_dict['stack']
    voxels_index = stack_dict['voxel_index']
    map_region_voxels = atlas[region]  # index refered to nbground voxels
    no_bg_region_voxels_index = voxels_index[map_region_voxels]

    mask_atlas = utils_mask3d.generate_region_3dmaskatlas(
        no_bg_region_voxels_index=no_bg_region_voxels_index,
        reshape_kind = reshape_kind,
        imgsize=imgsize,
        totalsize=total_size)

    minidx, maxidx = utils_mask3d.delim_3dmask(mask_atlas)

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
                voxels_patient_region_selected.size, 1)
            image[no_bg_region_voxels_index.tolist()] = \
                voxels_patient_region_selected

        image = np.reshape(image, imgsize, reshape_kind)

        out_image = image[minidx[0]:maxidx[0]+1,
                    minidx[1]:maxidx[1]+1, minidx[2]:maxidx[2]+1]

        stack_masked[patient, :, :, :] = out_image
    return stack_masked


def map_region_segmented_over_full_image(reconstruction_per_region, images_used):
    """
    Given a dictionary indexed by region and the kind of images reconstructed,
    this images is able to remap all the regions to their position in the
    whole image

    :param reconstruction_per_region: dict[region] -> ty[np.darray],
           sh[n_samples, w, h, d]
    :return: whole_reconstruction: ty[np.darray] sh[n_samples, h,w,d]

    """
    atlas, dict_parameters, reshape_kind = \
        load_parameters_and_atlas_by_images_used(images_used)

    list_regions = list(reconstruction_per_region.keys())
    number_3dimages_to_reconstruct = \
        reconstruction_per_region[list_regions[0]].shape[0]
    whole_3d_image_size = dict_parameters["imgsize"]

    whole_reconstruction = np.zeros([number_3dimages_to_reconstruct,
        whole_3d_image_size[0], whole_3d_image_size[1], whole_3d_image_size[2]])

    whole_reconstruction_length = np.array(whole_3d_image_size).prod()

    for region in list_regions:

        # Loading region masks
        whole_mask_flatten, mask_segmented_flatten=\
            utils_mask3d.get_whole_region_mask_and_region_segmented_mask(
            region=region,
            dict_parameters=dict_parameters,
            atlas=atlas,
            reshape_kind=reshape_kind)

        # Loop over samples
        for image_index in range(0,number_3dimages_to_reconstruct,1):
            region_3dimage_selected = \
                reconstruction_per_region[region][image_index, :, :, :]

            image_whole_reconstruction = whole_reconstruction[image_index, :,:,:]

            length_region_segmented_flatten = \
                np.array(region_3dimage_selected.shape).prod()

            # Reshape to flatten both arrays
            image_whole_reconstruction_flatten = np.reshape(image_whole_reconstruction,
                [whole_reconstruction_length], reshape_kind)

            region_3dimage_selected_flatten = np.reshape(region_3dimage_selected,
                [length_region_segmented_flatten], reshape_kind)

            segmented_voxels_selected = region_3dimage_selected_flatten[
                mask_segmented_flatten == 1]

            image_whole_reconstruction_flatten[
                whole_mask_flatten == 1] = segmented_voxels_selected

            whole_reconstruction[image_index, :, :, :] = \
                np.reshape(image_whole_reconstruction_flatten,
                    whole_3d_image_size, reshape_kind)

    return whole_reconstruction


def get_samples_in_stacked_region_to_3dsegmented_region(stack_region_to_3dimg,
                                                        samples_indexes):
    origin_images_to_encode = {}
    n_samples = len(samples_indexes)

    # selecting two samples required
    for region, cube_images in stack_region_to_3dimg.items():
        class_to_3d_means_imgs = np.zeros([n_samples, cube_images.shape[1],
                                       cube_images.shape[2],
                                       cube_images.shape[3]])

        for index in range(0, n_samples,1):
            class_to_3d_means_imgs[index, :, :, :] = \
                cube_images[samples_indexes[index], :, :, :]


        origin_images_to_encode[region] = class_to_3d_means_imgs

    return origin_images_to_encode


def get_stack_3dimages_filtered_by_label(stack_region_to_3dimg, samples_label,
                              label_selected):
    """

    :param stack_region_to_3dimg: stack[region] -> sh[n_samples, w_region,
                                                      h_region, d_region ]
    :param samples_label: ty[np.array] sh[n_samples] ct[0|1]
    :param label_selected: ty[Integer] 0|1
    :return: stack[region] => All samples will belong to the samples_label
                              indicated
    """

    stacker_filtered_region_to_3dimg = {}
    for region, images in stack_region_to_3dimg.items():
        index_to_selected_images = samples_label == label_selected
        index_to_selected_images = index_to_selected_images.flatten()
        stacker_filtered_region_to_3dimg[region] = \
            images[index_to_selected_images.tolist(), :]

    return stacker_filtered_region_to_3dimg


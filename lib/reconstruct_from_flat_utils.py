import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

from lib.data_loader import pet_loader
from lib.data_loader import PET_stack_NORAD
from lib.data_loader import pet_atlas
from lib import session_helper as session
from lib.utils import output_utils as output
from lib.utils.os_aux import create_directories
from lib import reconstruct_helpers as recons
import settings
import numpy as np
from scipy import misc


def reconstruct_from_flat_regions_to_full_3d_brain(
        dic_region_to_flat_voxels, images_used):

    if images_used == "PET":
        atlas = pet_atlas.load_atlas()
        dic_params = PET_stack_NORAD.get_parameters()
        reshape_kind = "F"
    else:
        raise("not supported kind of image {}".format(images_used))

    totalsize = dic_params["total_size"]
    imgsize = dic_params["imgsize"]
    index_nobg_voxels = dic_params["voxel_index"]

    list_regions = list(dic_region_to_flat_voxels.keys())
    n_samples = dic_region_to_flat_voxels[list_regions[0]].shape[0]

    whole_reconstruction_flat = np.zeros([n_samples, totalsize])
    for region in list_regions:
        region_output = dic_region_to_flat_voxels[region]
        index_nobg_voxels_regions = index_nobg_voxels[atlas[region]]
        whole_reconstruction_flat[:, list(index_nobg_voxels_regions)] = region_output

    # Defining size of the imag
    whole_reconstrution_size = [n_samples]
    whole_reconstrution_size.extend(imgsize)

    whole_reconstruction_3d = np.reshape(
        whole_reconstruction_flat, whole_reconstrution_size, reshape_kind)

    return whole_reconstruction_3d


def test_reconstruct_from_flat_regions_to_full_3d_brain():

    path_test_output = os.path.join(settings.path_to_general_out_folder,
                                    "temp_3d_pet_images_reconstructed")
    path_test_output_section = os.path.join(settings.path_to_general_out_folder,
        "temp_section_pet_images_reconstructed")
    print(path_test_output_section)
    create_directories([path_test_output, path_test_output_section])

    images_used = "PET"
    # Regions loader
    regions_used = "all"
    list_regions = session.select_regions_to_evaluate(regions_used)

    print("Loading Pet Data")
    stack_region_to_voxels, patient_labels, n_samples = \
        pet_loader.load_pet_data_flat(list_regions)

    print("Reorganizing data")
    whole_reconstruction_3d = reconstruct_from_flat_regions_to_full_3d_brain(
        stack_region_to_voxels, images_used)

    print("Saving images")
    for sample in range(0, n_samples):
        print("Saving sample {}".format(sample))
        path_3D_sample = os.path.join(path_test_output, "sample3d_{}".format(sample))
        output.from_3d_image_to_nifti_file(
            path_3D_sample, whole_reconstruction_3d[sample, :, :, :])

        path_section_sample = os.path.join(path_test_output_section,
                                           "sampleSection_{}".format(sample))
        recons.plot_section_indicated(
            tittle="Original vs Reconstructed. Sample {}".format(sample),
            img3d_1=whole_reconstruction_3d[sample, :, :, :],
            img3d_2=whole_reconstruction_3d[sample, :, :, :],
            p1=settings.planos_hipocampo["p1"],
            p2=settings.planos_hipocampo["p2"],
            p3=settings.planos_hipocampo["p3"],
            path_to_save_image=path_section_sample,
            cmap="jet")

test_reconstruct_from_flat_regions_to_full_3d_brain()

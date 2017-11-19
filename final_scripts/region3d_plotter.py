import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

from lib.data_loader.mri_loader import load_mri_regions_segmented3d
from lib.data_loader.pet_loader import load_pet_regions_segmented
from lib.session_helper import select_regions_to_evaluate
from lib.utils import output_utils
from lib.utils.os_aux import create_directories
import settings


images = "PET"
#images = "MRI"
list_regions = select_regions_to_evaluate("all")
sample_selected = 50


path_folder3D =  os.path.join(settings.path_to_project, "folder3D")
path_folder_region3d = os.path.join(path_folder3D, "region3D")
path_folder_images = os.path.join(path_folder_region3d,
    "brain3D_img:{0}_sample:{1}".format(images, sample_selected))

create_directories([path_folder3D, path_folder_region3d, path_folder_images])

pet_regions_segmented = None
mri_gm_regions_segmented = None
mri_wm_regions_segmented = None

if images == "PET":
    pet_regions_segmented = load_pet_regions_segmented(
        list_regions=list_regions,
        folder_to_store_3d_images=None,
        bool_logs=True,
        out_csv_region_dimensions=None)

    for region in list_regions:
        region_img_path = os.path.join(
            path_folder_images,"region:{}".format(region))
        output_utils.from_3d_image_to_nifti_file(
            path_to_save=region_img_path,
            image3d=pet_regions_segmented[region][sample_selected,:,:,:])

if images == "MRI":
    tuple_regions_segmented = load_mri_regions_segmented3d(
        list_regions=list_regions,
        folder_to_store_3d_images=None,
        bool_logs=True)

    [mri_gm_regions_segmented, mri_wm_regions_segmented] = \
        tuple_regions_segmented

    for region in list_regions:
        region_img_path = os.path.join(
            path_folder_images, "region:{}".format(region))

        output_utils.from_3d_image_to_nifti_file(
            path_to_save=region_img_path + "_wm",
            image3d=mri_wm_regions_segmented[region][sample_selected,:,:,:])

        output_utils.from_3d_image_to_nifti_file(
            path_to_save=region_img_path + "_gm",
            image3d=mri_gm_regions_segmented[region][sample_selected,:,:,:])

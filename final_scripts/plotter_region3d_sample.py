import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

from lib.data_loader.pet_loader import load_pet_regions_segmented
from lib.session_helper import select_regions_to_evaluate
from lib.utils import output_utils
from lib.utils.os_aux import create_directories
import settings


images = "PET"
#images = "MRI"


sample_selected = 1
region_selected = 3

#list_regions = select_regions_to_evaluate("all")
list_regions = [3]

path_folder3D =  os.path.join(settings.path_to_project, "folder3D")
path_folder_region3d = os.path.join(path_folder3D, "region3D_samples")
path_images = os.path.join(path_folder_region3d,
    "region:{0}_sample:{1}".format(images, sample_selected))

create_directories([path_folder3D, path_folder_region3d])

pet_regions_segmented = load_pet_regions_segmented(
    list_regions=list_regions,
    folder_to_store_3d_images=None,
    bool_logs=True,
    out_csv_region_dimensions=None)

output_utils.from_3d_image_to_nifti_file(
    path_to_save=path_images,
    image3d=pet_regions_segmented[region_selected][sample_selected, :, :, :])

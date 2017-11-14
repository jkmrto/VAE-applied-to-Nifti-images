from lib.data_loader.mri_loader import load_mri_regions_segmented3d
from lib.data_loader.pet_loader import load_pet_regions_segmented
from lib.session_helper import select_regions_to_evaluate
import settings
import os




images = "PET"
images = "MRI"
list_regions = select_regions_to_evaluate("all")
sample_selected = 50


path_folder_region3d = os.path.join(
    settings.path_to_project, "folder3D", "region3d",
    "brain3D_img:{0}_sample:{1}".format(images, sample_selected))


if images == "PET":
    load_pet_regions_segmented(
        list_regions=list_regions,
        folder_to_store_3d_images=None,
        bool_logs=True,
        out_csv_region_dimensions=None)

if images == "MRI":
    load_mri_regions_segmented3d(
    list_regions,
    folder_to_store_3d_images=None,
    bool_logs=True)

from lib.data_loader import MRI_stack_NORAD
from lib.data_loader import PET_stack_NORAD
from lib.data_loader import mri_atlas
from lib.data_loader import pet_atlas


def load_parameters_and_atlas_by_images_used(images_used):
    """
    This functions allows load the parameters from the stacks
    but leaving the heavy load in this scope (the voxels data per patient)
    so the python garbage collector can clean it
    :param images_used:
    :return:
    """
    atlas = None
    dict_parameters = None
    reshape_kind = None

    if images_used == "MRI":
        atlas = mri_atlas.load_atlas_mri()
        dict_parameters = MRI_stack_NORAD.get_parameters()
        reshape_kind = "C"
    elif images_used == "PET":
        atlas = pet_atlas.load_atlas()
        dict_parameters = PET_stack_NORAD.get_parameters()
        reshape_kind = "F"

    return atlas, dict_parameters, reshape_kind
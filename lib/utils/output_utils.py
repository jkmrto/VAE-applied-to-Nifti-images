import csv
import nibabel as nib
import numpy as np


def print_dictionary_with_header(file, list_of_dict):
    """

    :param file:
    :param list_of_dict:
    :return:
    """
    file = open(file, "w")

    writer = csv.DictWriter(file, delimiter=',',
        fieldnames=list(list_of_dict[0].keys()))
    writer.writeheader()
    for row in list_of_dict:
        writer.writerow(row)

    file.close()


def print_recursive_dict(dic, file=None, suffix=""):

    for key, item in dic.items():
        if isinstance(item, dict):
            next_suffix = suffix + "{},".format(key)
            print_recursive_dict(dic=item, file=file, suffix=next_suffix)
        else:
            if file is None:
                print(suffix + "{0}: {1}".format(key, item))
            else:
                file.write(suffix + "{0}: {1}\n".format(key, item))


def from_3d_image_to_nifti_file(path_to_save, image3d):
    img = nib.Nifti1Image(image3d, np.eye(4))
    img.to_filename("{}.nii".format(path_to_save))
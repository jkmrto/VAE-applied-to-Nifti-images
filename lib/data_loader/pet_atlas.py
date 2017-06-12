from lib.data_loader.atlas_settings import super_regions_atlas
import nibabel as nib
import settings
import numpy as np
import csv


def load_atlas():
    img = nib.load(settings.pet_atlas_path)
    img_data = img.get_data()
    atlasdata = img_data.flatten()
    bckvoxels = np.where(atlasdata != 0) #Sacamos los indices que no son 0
    atlasdata = atlasdata[bckvoxels] # Mapeamos aplicando los indices
    vals = np.unique(atlasdata)
    reg_idx = {}  # reg_idx contiene los indices de cada region
    for i in range(1, max(vals) + 1):
        reg_idx[i] = np.where(atlasdata == i)[0] # Saca los indices

    return reg_idx

#atlas = load_atlas()


def get_super_region_to_voxels():
    """
    This functions returns a dictionary.
    Each key of the dictionary is a region and his values are
    their voxels associated
    :return:
    """
    regions_dict = load_atlas()  # dictionary
    super_region_atlas_voxels = {}
    for super_region_key, regions_included in super_regions_atlas.items():
        super_region_atlas_voxels[super_region_key] = \
            np.concatenate(
                [voxels for region_index, voxels in regions_dict.items() if region_index in regions_included],
                axis=0)

    return super_region_atlas_voxels


def generate_super_regions_csv_template():
    regions_dict = get_super_region_to_voxels()

    with open('regions_neural_net_setting.template.csv', 'w+') as file:
        fieldnames = ['region', 'n_voxels', 'input_layer', 'first_layer', 'second_layer', 'latent_layer']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for region_name, voxels in regions_dict.items():
            aux_dic = {'region': str(region_name), 'n_voxels': str(len(voxels))}
            writer.writerow(aux_dic)


def generate_regions_csv_template():
    regions_dict = load_atlas()

    with open('regions_neural_net_setting.template.csv', 'w+') as file:
        fieldnames = ['region', 'n_voxels', 'input_layer', 'first_layer', 'second_layer', 'latent_layer']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for region_name, voxels in regions_dict.items():
            aux_dic = {'region': str(region_name), 'n_voxels': str(len(voxels))}
            writer.writerow(aux_dic)
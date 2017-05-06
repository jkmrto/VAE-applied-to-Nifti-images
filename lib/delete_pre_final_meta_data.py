import os
import settings


def get_list_files_suffix():
    data_suffix = 'data-00000-of-00001'
    index_suffix = 'index'
    meta_suffix = 'meta'
    return [data_suffix, index_suffix, meta_suffix]


def get_region_to_list_of_iters(files):
    """
    List of the files content in the meta folder
    :param files:
    :return: Dictionary wich maps from region_index(int)
    to the list of files (identified by the iter) content
    in the folder
    """
    region_to_iter_dict = {}

    for file in files:
        [region, iter] = file.split("-")
        region = region.split('_')[1]

        if region in region_to_iter_dict.keys():
            if int(iter) not in region_to_iter_dict[region]:
                region_to_iter_dict[region].append(int(iter))
        else:
            region_to_iter_dict[region] = [int(iter)]

    return region_to_iter_dict


def clear_meta_folder(iden_vae_session):
    # setting meta folder to clear

    path_to_meta_folder = os.path.join(settings.path_to_general_out_folder,
                                       iden_vae_session, "meta")

    suffixes = get_list_files_suffix()
    files_list = os.listdir(path_to_meta_folder)

    if 'checkpoint' in files_list:
        files_list.remove('checkpoint')
    files = [file.split(".")[0] for file in files_list]

    region_to_iter_dict = get_region_to_list_of_iters(files)

    # Loop over regions.
    for region, list_iters in region_to_iter_dict.items():
        list_iters.sort(reverse=True)
        region_prefix = 'region_{}'.format(region)

        for suffix in suffixes:
            for iter in list_iters[1:]:  # We leave only the meta of the higher iter
                file = "{0}-{1}.{2}".format(region_prefix, iter, suffix)

                os.remove(os.path.join(path_to_meta_folder, file))


iden_vae_session = "05_05_2017_08:19 arch: 1000_800_500_100"
clear_meta_folder(iden_vae_session)
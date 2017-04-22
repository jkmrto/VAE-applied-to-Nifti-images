import scipy.io as sio
import settings

# Script con la materia gris


def get_gm_stack():
    """
    This function returns a dictionary with these three values:
    1)
    :return:
    """
    f = sio.loadmat(settings.stack_path_GM)
    return {'labels': f['labels'], 'voxel_index': f['nobck_idx'], 'stack': f['stack_NORAD_GM'],
            'imgsize':f['imgsize'].astype('uint32'), 'n_patients': len(f['labels'])}


def load_patients_labels():
    dict_norad = get_gm_stack()  # 'stack' 'voxel_index' 'labels'
    return dict_norad['labels']
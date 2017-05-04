import scipy.io as sio
import settings
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
# Script con la materia gris


def get_gm_stack():
    """
    This function returns a dictionary with these three values:
    1)
    :return:
    """
    f = sio.loadmat(settings.stack_path_GM)
    return {'labels': f['labels'], 'voxel_index': f['nobck_idx'], 'stack': f['stack_NORAD_GM'],
            'imgsize':f['imgsize'].astype('uint32').tolist()[0], 'n_patients': len(f['labels'])}


def get_wm_stack():
    """
    This function returns a dictionary with these three values:
    1)
    :return:
    """
    f = sio.loadmat(settings.stack_path_WM)
    return {'labels': f['labels'], 'voxel_index': f['nobck_idx'], 'stack': f['stack_NORAD_WM'],
            'imgsize':f['imgsize'].astype('uint32').tolist()[0], 'n_patients': len(f['labels'])}


def load_patients_labels():
    dict_norad = get_gm_stack()  # 'stack' 'voxel_index' 'labels'
    return dict_norad['labels']


def test():
    data = get_gm_stack()
    sample = data['stack'][0,:].shape

    template = np.zeros(data['imgsize'], dtype=float)
    template = template.flatten()
    template[data['voxel_index']] = sample
    out = np.reshape(template, data['imgsize'])
    plt.imshow(out[:, 50, :], cmap='gray')

    img = nib.Nifti1Image(out, np.eye(4))
#test()
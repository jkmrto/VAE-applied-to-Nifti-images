import scipy.io as sio
import settings
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt


def get_stack():
    """
    This function returns a dictionary with these three values:
    1)
    :return:
    """
    f = sio.loadmat(settings.PET_stack_path)
    # f -> dict_keys(['bmask', 'normtype', 'tu', 'thr', 'labels_conv', 'labels', '__globals__',
    # 'nthr', 'maskind', 'atlas', 'stack_all_norm', 'CLASV', 'stack_PET', '__header__', '__version__',
    # 'clastring', 'patient'])

    images_size = [79, 95, 68]
    full_images = f['stack_all_norm'] # [138 x 510340]
    patient_labels = f['labels'] #[ 1x138]

    return {'labels': patient_labels,
            'stack': full_images,
            'imgsize':images_size,
            'n_patients': len(patient_labels)}


def get_stack_reduced():
    f = sio.loadmat(settings.PET_stack_path)
    # f -> dict_keys(['bmask', 'normtype', 'tu', 'thr', 'labels_conv', 'labels', '__globals__',
    # 'nthr', 'maskind', 'atlas', 'stack_all_norm', 'CLASV', 'stack_PET', '__header__', '__version__',
    # 'clastring', 'patient'])

    images_size = [79, 95, 68]
    reduced_images = f['stack_PET'] # [138 x 182562]
    reduce_index_to_full = f['maskind'] #[1 x 182562]
    patient_labels = f['labels'] #[ 1x138]

    return {'labels': patient_labels,
            'stack': reduced_images,
            'voxel_index': reduce_index_to_full,
            'imgsize':images_size,
            'n_patients': len(patient_labels)}

def load_patients_labels():
    dict_norad = get_stack()  # 'stack' 'voxel_index' 'labels'
    return dict_norad['labels']


def test():
    data = get_stack()
    sample = data['stack'][0,:].shape

    template = np.zeros(data['imgsize'], dtype=float)
    template = template.flatten()
    template[data['voxel_index']] = sample
    out = np.reshape(template, data['imgsize'])
    plt.imshow(out[:, 50, :], cmap='gray')

    img = nib.Nifti1Image(out, np.eye(4))
    img.to_filename('test4d.nii.gz')

#test()

stack = get_stack()
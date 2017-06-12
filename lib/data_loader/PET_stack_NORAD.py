import scipy.io as sio
import settings
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt


def get_full_stack():
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
    voxels_index = f['maskind']
    images = f['stack_all_norm'] # [138 x 510340]
    patient_labels = f['labels'] #[ 1x138]

    return {'labels': patient_labels,
            'stack': images,
    #        'voxel_index': voxels_index,
            'imgsize':images_size,
            'n_patients': len(patient_labels)}


def load_patients_labels():
    dict_norad = get_full_stack()  # 'stack' 'voxel_index' 'labels'
    return dict_norad['labels']


def test():
    data = get_stack()
    sample = data['stack'][50, :]

    template = np.zeros(data['imgsize'], dtype=float)
    template = template.flatten()

    template[data['voxel_index']] = sample
    out = np.reshape(template, [79, 95, 68], "F")
    plt.imshow(np.rot90(out[:, 30, :]), cmap='jet')
    plt.show(block=True)

    img = nib.Nifti1Image(out, np.eye(4))

    img.to_filename('test4d.nii.gz')

#test()

#stack = get_stack()
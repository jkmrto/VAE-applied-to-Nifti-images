from lib.data_loader.utils_mask3d import delim_3dmask
import numpy as np


def reshape_from_flat_to_3d(images_flat, dim3d, reshape_kind="C"):

    n_samples = images_flat.shape[0]
    dims = [n_samples, dim3d[0], dim3d[1], dim3d[2]]

    images3d = np.reshape(images_flat, dims, reshape_kind)

    return images3d


def reshape_from_3d_to_flat(images3d, total_size, reshape_kind="C"):

    n_samples = images3d.shape[0]
    dims = [n_samples, total_size]

    images_flat = np.reshape(images3d, dims, reshape_kind)

    return images_flat


def get_3dimage_segmented(img3d):
    minidx, maxidx = delim_3dmask(img3d, thval=0.01)
    img3d_segmented = img3d[minidx[0]:maxidx[0] + 1,
                      minidx[1]:maxidx[1] + 1,
                      minidx[2]:maxidx[2] + 1]
    return img3d_segmented

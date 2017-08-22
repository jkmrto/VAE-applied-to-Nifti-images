import numpy as np


def reshape_from_flat_to_3d(images_flat, dim3d, reshape_kind="F"):

    n_samples = images_flat.shape[0]
    dims = [n_samples, dim3d[0], dim3d[1], dim3d[2]]

    images3d = np.reshape(images_flat, dims, reshape_kind)

    return images3d


def reshape_from_3d_to_flat(images3d, total_size, reshape_kind="F"):

    n_samples = images3d.shape[0]
    dims = [n_samples, total_size]

    images_flat = np.reshape(images3d, dims, reshape_kind)

    return images_flat

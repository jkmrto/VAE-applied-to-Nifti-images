import numpy as np


def get_comparision_over_matrix_samples(matrix_samples):
    """
    :param matrix_samples: sh[n_samples, n_dims]
    :return: matrix_compare : sh[n_samples, n_samples].
    Each position [i,j] indicates the difference between the sample i and j,
    so it is a diagonal matrix
    """

    [n_samples, n_dims] = matrix_samples.shape
    matrix_compare = np.zeros([n_samples, n_samples])

    for i in range(0, n_samples, 1):
        for j in range(0, n_samples, 1):
            matrix_compare[i, j] = evaluate_diff_flat(array1=matrix_samples[i, :],
                                                      array2=matrix_samples[j, :])

    return matrix_compare


def evaluate_diff_flat(array1, array2):

    dist = np.linalg.norm(array1 - array2)

    return dist


def get_samples_per_label(samples_matrix, labels, label_selected):
    index_to_selected_images = labels == label_selected
    index_to_selected_images = index_to_selected_images.flatten()
    matrix_samples_selected = samples_matrix[index_to_selected_images.tolist(), :]

    return matrix_samples_selected


def get_mean_difference_over_samples(matrix_samples):

    matrix_differences = get_comparision_over_matrix_samples(matrix_samples)

    diff_mean_over_samples = \
        sum(sum(matrix_differences)) / matrix_differences.size

    return diff_mean_over_samples
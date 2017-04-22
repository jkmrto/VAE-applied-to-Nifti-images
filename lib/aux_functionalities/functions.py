import numpy as np


def get_batch_from_samples(X, Y, batch_size):
    index = np.random.choice(range(X.shape[0]), batch_size, replace=False)
    index = index.tolist()
    return X[index, :], Y[index]

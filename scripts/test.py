# def __old_generate_batch(self, X, Y):
import numpy as np


def get_batch_from_samples(X, Y, batch_size):
    print(np.random.choice(range(X.shape[0]), 2, replace=False))
    index = np.random.choice(range(X.shape[0]), batch_size, replace=False)
    index = index.tolist()
    return X[index, :], Y[index]

X = [[40,10], [30,10], [88,10], [55,10], [66,10], [11,10], [12,10], [45,10], [85,10], [20,10]]
Y = [1,2,3,4,5,6,7,8,9,10]
X = np.array(X)
Y = np.array(Y)
print(str(get_batch_from_samples(X, Y, 5)))

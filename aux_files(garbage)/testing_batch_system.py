import numpy as np

X = np.zeros((50,10), dtype=np.int)
l = np.arange(0, X.shape[0])
batch_size = 5


batch_idx = [l[i:i+batch_size] for i in range(0, len(l), batch_size)]
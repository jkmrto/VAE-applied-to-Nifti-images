from matplotlib import pyplot as plt
import numpy as np

learning_rate = 1e-03
print(learning_rate)
iter = np.array(range(1, 10000, 1))
print(iter)

out = learning_rate * np.exp(-iter/5000)

plt.figure(0)
plt.plot(out)
plt.show()
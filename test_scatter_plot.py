import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

y = [0,0,1,1]

x = np.array([[1,1,1],
             [2,2,2],
             [3,3,3],
            [4,4,4]])

fig = plt.figure()
ax = Axes3D(fig, elev=-150, azim=110)
colors = ("blue", "red")
groups = ("NOR","AD")

for data, color, group in zip(x, colors, groups):
    x, y, z = data
    ax.scatter(x, y, z, alpha=0.8, c=color,
               edgecolors='none', s=30, label=group)

plt.title('Matplot 3d scatter plot')
plt.legend(loc=2)
plt.show()

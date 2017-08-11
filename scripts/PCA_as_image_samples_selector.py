from lib.data_loader import PET_stack_NORAD
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot


print("Loading Pet images")
stack_dict = PET_stack_NORAD.get_full_stack()
stack = stack_dict['stack']
patient_labels = stack_dict['labels']

shape = stack.shape

out_matrix = np.zeros([shape[0], shape[0]])
for i in range(0, shape[0], 1):
    print("Evaluating difs over image {}".format(i))
    for j in range(0, shape[0], 1):
        out_matrix[i, j] = evaluate_dif(array1=stack[i, :],
                                        array2=stack[j, :])

file_out = "out_simple.csv"
fname = open(file_out, "wb")
np.savetxt(fname=fname,
              X=out_matrix,  delimiter=',',  fmt='%.2e',
              newline='\n', header='', footer='', comments='# ')


pyplot.figure()
pyplot.imshow(out_matrix, cmap="Greys")
pyplot.savefig("test_images_similitud")

n_components = 30
pca = PCA(n_components=n_components)
pca.fit(stack)
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))
pyplot.figure()
x = list(range(1, n_components+1, 1))
pyplot.plot(x, pca.explained_variance_ratio_.cumsum())
#pyplot.show()

out = pca.transform(X=stack)
print(out.shape)
shape = out.shape


out_matrix = np.zeros([shape[0], shape[0]])
for i in range(0, shape[0], 1):
    for j in range(0, shape[0], 1):
        out_matrix[i, j] = recons.evaluate_diff_flatr(array1=out[i, :],
                                        array2=out[j, :])

print(out_matrix)
file_out = "out.csv"
fname = open(file_out, "wb")
np.savetxt(fname=fname,
              X=out_matrix,  delimiter=',', fmt='%.2e',
              newline='\n', header='', footer='', comments='# ')
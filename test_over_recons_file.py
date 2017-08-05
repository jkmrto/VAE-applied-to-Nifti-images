from lib import reconstruct_helpers as recons
from lib import compare_helper as compare

import numpy as np

images = np.array([[-2,-2,-2,-2],
                   [-2,-2,-2,-2],
                   [-2,-2,-2,-2],
                   [-2,-2,-2,-2],
                   [3,3,3,3],
                   [3,3,3,3],
                   [3,3,3,3],
                   [3,3,3,3]])
patient_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
label_selected = 1

out_1 = recons.get_mean_over_selected_samples(images, 1, patient_labels)
out_2 = recons.get_mean_over_selected_samples(images, 0, patient_labels)

out_3 = compare.evaluate_diff_flat(out_1, out_2)

print(out_1)
print(out_2)
print(out_3)

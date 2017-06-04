import numpy as np
from lib.data_loader import MRI_stack_NORAD
from lib import regenerate_utils


def evaluate_cubes_diference_by_planes(cube1, cube2, bool_test=False):

   if bool_test:
        print(cube1.shape)
        print(cube2.shape)

   cube_dif = cube1-cube2
   cube_dif = cube_dif.__abs__()

   pos_max = 0
   max = 0
   v1 = cube_dif.sum(axis=2).sum(axis=1)
   v2 = cube_dif.sum(axis=2).sum(axis=0)
   v3 = cube_dif.sum(axis=0).sum(axis=0)


   return np.argmax(v1), np.argmax(v2), np.argmax(v3)


#   for index in range(0, cube_dif.shape[0], 1):
#       if cube_dif[index, : :].sum(axis=0).sum(axis=0) > max:
#           pos_max = index
#           max = cube_dif[index, : :].sum(axis=0).sum(axis=0)
           #print(max)
def test():
    dict_norad = MRI_stack_NORAD.get_gm_stack()  # 'stack' 'voxel_index' 'labels'
    img_index = 40
    sample_pos = dict_norad['stack'][-img_index,:]
    sample_neg = dict_norad['stack'][img_index,:]

    mri_imag_3d_pos = regenerate_utils.reconstruct_3d_image(sample_pos,
                                                    dict_norad['voxel_index'],
                                                    dict_norad['imgsize'])
    mri_imag_3d_neg = regenerate_utils.reconstruct_3d_image(sample_neg,
                                                    dict_norad['voxel_index'],
                                                    dict_norad['imgsize'])

    cube1 = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
    cube2 = cube1.transpose()
    evaluate_cubes_diference_by_planes(mri_imag_3d_pos, mri_imag_3d_neg)



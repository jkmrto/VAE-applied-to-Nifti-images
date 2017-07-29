from lib.data_loader import pet_atlas
from lib.data_loader import PET_stack_NORAD
from lib import reconstruct_helpers as recons
import numpy as np

logs = True
hipocampo_regions = [38, 39, 40]

# atlas load index referred to not background flatten voxels
atlas = pet_atlas.load_atlas()

all_voxels_desired = []
for region in hipocampo_regions:
    list_voxels_region = atlas[region]
    all_voxels_desired.extend(list_voxels_region)
    if logs:
        print("Number voxels desired {}".format(len(all_voxels_desired)))

pet_dict_stack = PET_stack_NORAD.get_full_stack()

reshape_kind = "F"
imgsize = pet_dict_stack['imgsize']
voxels_index = pet_dict_stack['voxel_index'] # no_bg_index to real position
total_size = pet_dict_stack['total_size']
stack = pet_dict_stack['stack']

activaction_mask = list(voxels_index[all_voxels_desired])

mask_full_img = np.zeros([total_size])
mask_full_img[activaction_mask] = 1

if logs:
    print("Number voxels activated after mapping over no_background_index {}".
          format(sum(mask_full_img)))


mask_full_img_3d = np.reshape(mask_full_img, imgsize, reshape_kind)

# activation planes.
v1 = mask_full_img_3d.sum(axis=2).sum(axis=1)
v2 = mask_full_img_3d.sum(axis=2).sum(axis=0)
v3 = mask_full_img_3d.sum(axis=0).sum(axis=0)

m1, m2, m3 = np.max(v1), np.max(v2), np.max(v3)
p1, p2, p3 = np.argmax(v1), np.argmax(v2), np.argmax(v3)

print("Section selected per axis postion: {0}, {1}, {2}".format(m1, m2, m3))
print("Section selected per axis maximum: {0}, {1}, {2}".format(p1, p2, p3))

nor_3d_sample = recons.reconstruct_3d_image_from_flat_and_index(
    image_flatten= stack[10,:],
    voxels_index=voxels_index,
    imgsize=imgsize,
    reshape_kind=reshape_kind)

ad_3d_sample = recons.reconstruct_3d_image_from_flat_and_index(
    image_flatten= stack[120,:],
    voxels_index=voxels_index,
    imgsize=imgsize,
    reshape_kind=reshape_kind)

recons.plot_section_indicated(
    img3d_1=nor_3d_sample,
    img3d_2=ad_3d_sample,
    p1=p1, p2=p2, p3=p3,
    path_to_save_image="sampe_hipocampo.png",
    cmap="jet")

print(images_stack.shape)
print(images_index.shape)


# print(f.keys())
# print("labels dimensiones" + str(f['labels'].shape))
# print("stack_NORAD_GM dimensiones" + str(f['stack_NORAD_GM'].shape))
# print("stack_NORAD_GM imsize" + str(f['imgsize']))
# noinspection PyPackageRequirements
# print("nobck dimensiones" + str(f['nobck_idx'].shape))

# plt.imshow(x_.reshape([dim, dim]), cmap="Greys")

# def load_stack(path_to_stack, stack_name):
# f = sio.loadmat(path_to_stack)










# Indices de cada region, es lo que tenemos para filtrar



def print_out_csv(regions_dict):
    with open('regions_neural_net_setting.template.csv', 'w+') as file:
        fieldnames = ['region', 'n_voxels', 'input_layer', 'first_layer',  'second_layer', 'latent_layer']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for region, value in regions_dict.items():
            aux_dic = {'region': str(region), 'n_voxels': str(len(value))}
            writer.writerow(aux_dic)


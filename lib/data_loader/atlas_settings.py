frontal_lobe_val = [3, 4, 7, 8, 23, 24, 9, 10, 27, 28, 31, 32]

parietal_lobe_val = [59, 60, 61, 62, 67, 68, 35, 36]

occipital_lobe_val = [49, 50, 51, 52, 53, 54]

temporal_lobe_val = [81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 55, 56, 37, 38, 39, 40]
cerebellum_val = [91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108]
vermis_val = [109, 110, 111, 112, 113, 114, 115, 116]

# lobes_mask_val=np.concatenate((frontal_lobe_val,parietal_lobe_val,occipital_lobe_val,temporal_lobe_val),axis=0)
# lobes_mask_val_cerebellum=np.concatenate((lobes_mask_val,cerebellum_val),axis=0)

super_regions_atlas = {'frontal_lobe_val': frontal_lobe_val,
                       'parietal_lobe_val': parietal_lobe_val,
                       'occipital_lobe_val': occipital_lobe_val,
                       'temporal_lobe_val': temporal_lobe_val,
                       'vermis_val': vermis_val,
                       'cerebellum_val': cerebellum_val,
                       }
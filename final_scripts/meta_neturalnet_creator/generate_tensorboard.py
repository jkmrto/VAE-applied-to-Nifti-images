import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from lib.vae import CVAE
from lib.vae import CVAE_helper
import tensorflow as tf
import settings
import lib.neural_net.kfrans_ops as ops
import numpy as np
from datetime import datetime
from lib.data_loader.pet_loader import load_pet_regions_segmented
from lib.utils.os_aux import create_directories

region_selected = 3
list_regions = [region_selected]
region_to_img_dict = load_pet_regions_segmented(list_regions, bool_logs=False)
train_cube_images = region_to_img_dict[region_selected]


path_to_session = os.path.join(settings.path_to_general_out_folder,
                                "tensorboard_example")
create_directories([path_to_session])

hyperparams = {'latent_layer_dim': 100,
               'kernel_size': [5, 5, 5],
               'activation_layer': ops.lrelu,
               'features_depth': [1, 64, 128],
               'decay_rate': 0.0025,
               'learning_rate': 0.001,
               'lambda_l2_regularization': 0.0001,
               "cvae_model": "2layers",
               'stride': 2
}

# Additional per region hyperparameters
hyperparams['image_shape'] = train_cube_images.shape[1:]

session_conf = {'bool_normalized': False,
                'n_iters': 5000,
                "batch_size": 16,
                "show_error_iter": 100}

tf.reset_default_graph()

cvae_model = CVAE_helper.select_model(hyperparams["cvae_model"])
model = cvae_model(hyperparams, path_to_session=path_to_session,
                  generate_tensorboard=True)
model.generate_meta_net()


#tensorboard --logdir=run1:/media/jkmrto/Multimedia/TFM/VAN-applied-to-Nifti-images/out/tensorboard_example/tb/ --port 6006

from lib.vae import CVAE_2layers
from lib.vae import CVAE_3layers
from lib.vae import CVAE_4layers
from lib.vae import CVAE_2layers_2DenseLayers


model_map = {
    "2layers": CVAE_2layers,
    "3layers": CVAE_3layers,
    "4layers": CVAE_4layers,
    "2layers_2dense": CVAE_2layers_2DenseLayers
}


def select_model(model):
    if model in list(model_map.keys()):
        return model_map[model]
    else:
        print("The model selected shold be one of {}".format(str(model_map)))
        raise Exception("Bad model assignation")
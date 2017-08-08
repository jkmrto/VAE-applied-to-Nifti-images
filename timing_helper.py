MRI_GM_NET_KEY = "MRI_GM_neuralnet"
MRI_WM__NETKEY = "MRI_WM_neuralnet"
PET_NET_KEY = "PET"


def get_averages_timing_dict_per_images_used(timing_dict, images_used):

    average_timing = {}
    if images_used == "MRI":
        average_timing = {
            "MRI_GM_neuralnet":
                sum(timing_dict["MRI_GM_neuralnet"]) / len(timing_dict["MRI_GM_neuralnet"]),
            "MRI_WM_neuralnet":
                sum(timing_dict["MRI_WM_neuralnet"]) / len(timing_dict["MRI_WM_neuralnet"]),
        }
    elif images_used == "PET":
        average_timing = {
            "PET":
                sum(timing_dict["PET"]) / len(timing_dict["PET"]),
        }

    return average_timing
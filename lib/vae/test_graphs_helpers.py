import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import matplotlib

matplotlib.use('Agg')
from lib import file_reader
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = [16, 9]
import settings
import matplotlib.ticker as mtick
from lib.utils.os_aux import create_directories
from lib import file_reader


def plot_mse_error_evolution(session_name, list_region):
    path_logs = os.path.join(settings.path_to_general_out_folder,
                 session_name, "logs", "losses_logs")
    path_images = os.path.join(settings.path_to_general_out_folder,
                 session_name, "images", "training_SGD_stats")

    create_directories([path_logs, path_images])

    for region in list_region:
        path_to_file = \
            os.path.join(settings.path_to_general_out_folder,
                     session_name, "logs","losses_logs",
                     "region_{}.txt".format(region))
        list_dicts = \
            file_reader.read_csv_as_list_of_dictionaries(path_to_file)

        region_prefix = "region_{0}".format(region)

        iters = []
        latent_layer_loss = []
        learning_rate = []
        generative_loss = []
        similarity_score = []
        MSE_error_over_samples = []

        for row in list_dicts:
            iters.append(int(row["iteration"]))
            latent_layer_loss.append(float(row["latent layer loss"]))
            learning_rate.append(float(row["learning rate"]))
            generative_loss.append(float(row["generative loss"]))
            similarity_score.append(float(row["similarity score"]))
            MSE_error_over_samples.append(float(row[" MSE error over samples"]))

        wrap_default_generate_image(
            idi= "generative_loss" ,
            tittle= "Generative Error",
            iters = iters,
            values= generative_loss,
            region_prefix=region_prefix,
            path_images=path_images
        )

        wrap_default_generate_image(
            idi= "latent_layer_loss" ,
            tittle= "Latent Layer Loss",
            iters = iters,
            values = latent_layer_loss,
            region_prefix=region_prefix,
            path_images=path_images)

        wrap_default_generate_image(
            idi= "similarity_score" ,
            tittle= "Similarity Score",
            iters = iters,
            values= similarity_score,
            region_prefix = region_prefix,
            path_images=path_images)

        wrap_default_generate_image(
            idi= "mse_error" ,
            tittle= "MSE error",
            iters = iters,
            values= MSE_error_over_samples,
            region_prefix=region_prefix,
            path_images=path_images
        )


        wrap_default_generate_image(
            idi= "learning_rate" ,
            tittle= "Learning Rate",
            iters = iters,
            values= learning_rate,
            region_prefix = region_prefix,
            path_images=path_images
            )


def generate_and_save_2xy_graphs_zoomed(title, xlabel, y_label,
        path_to_save, x_values, y_values):

    interval_xvalues_to_zoom = x_values[-1]/4

    zoomed_index = []
    for i in range(0,4,1):
        zoomed_index.append(round(i * interval_xvalues_to_zoom))

    # Figure Header
    fig = plt.figure()
    fig.suptitle(title, fontsize=14)
    # First subplot
    plt.subplot(221)
    plt.plot(x_values, y_values)
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%5.1e'))

    # Second subplot
    plt.subplot(222)
    post_iter_to_zoom = x_values.index(zoomed_index[1])
    plt.plot(x_values[post_iter_to_zoom:], y_values[post_iter_to_zoom:])
    plt.xlabel(xlabel, fontsize=14)

    plt.subplot(223)
    post_iter_to_zoom = x_values.index(zoomed_index[2])
    plt.plot(x_values[post_iter_to_zoom:], y_values[post_iter_to_zoom:])
    plt.xlabel(xlabel, fontsize=14)

    plt.subplot(224)
    post_iter_to_zoom = x_values.index(zoomed_index[3])
    plt.plot(x_values[post_iter_to_zoom:], y_values[post_iter_to_zoom:])
    plt.xlabel(xlabel, fontsize=14)
    plt.savefig(path_to_save)


def wrap_default_generate_image(idi, tittle, iters, values,
                                region_prefix, path_images):
    path_to_save = os.path.join(
        path_images, "{0}_{1}.png".format(region_prefix, idi))
    generate_and_save_2xy_graphs_zoomed(
        title="{} Over Iters".format(tittle),
        xlabel="Iters",
        y_label=tittle,
        path_to_save=path_to_save,
        x_values=iters,
        y_values=values)
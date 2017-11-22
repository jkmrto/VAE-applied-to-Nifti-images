import math
import os
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf

from final_scripts import region_plane_selector
from lib import reconstruct_helpers as recons
from lib.utils import output_utils
from lib.utils.functions import \
    get_batch_from_samples_unsupervised
from lib.utils.os_aux import create_directories
from lib.utils.utils3d import reshape_from_3d_to_flat
from lib.utils.utils3d import reshape_from_flat_to_3d

bool_save_meta = False


class CVAE():
    RESTORE_KEY = "restore"

    def __init__(self, hyperparams, test_bool=False,
                 path_to_session=None, generate_tensorboard=False,
                 path_meta_graph=None):

        assert "image_shape" in list(hyperparams), \
            "image_shape should be specified in hyperparams"
        self.image_shape = hyperparams['image_shape']
        self.total_size = np.array(self.image_shape).prod()

        self.session = tf.Session()
        self.hyperparams = hyperparams
        self.init_path_to_session = path_to_session
        self.generate_tensorboard = generate_tensorboard
        self.path_meta_graph = path_meta_graph

        self.dim_in_first_layer = None
        self.dim_out_first_layer = None
        self.dim_out_second_layer = None
        self.dim_out_third_layer = None
        self.dim_out_forth_layer = None
        self.input_dense_layer_dim = None

        if test_bool:
            print(hyperparams)

    def generate_meta_net(self):


        if self.path_meta_graph is None:

            # Initizalizing graph values
            self.n_z = self.hyperparams['latent_layer_dim']
            self.lambda_l2_reg = self.hyperparams['lambda_l2_regularization']
            self.learning_rate = self.hyperparams['learning_rate']
            self.features_depth = self.hyperparams['features_depth']
            self.kernel_size = self.hyperparams['kernel_size']
            self.activation_layer = self.hyperparams['activation_layer']
            self.decay_rate_value = float(self.hyperparams['decay_rate'])
            self.stride = self.hyperparams['stride']
            self.path_session_folder = self.init_path_to_session

            self.__build_graph()

            if self.path_session_folder is not None:
                self.__init_session_folders()

            self.__generate_tensorboard_files()

            handles = [self.in_flat_images, self.z_mean, self.z_stddev,
                       self.z_in_, self.regenerated_3d_images_]
            for handle in handles:
                tf.add_to_collection(CVAE.RESTORE_KEY, handle)

            self.session.run(tf.initialize_all_variables())
        else:
            self.load_meta_net()

    def load_meta_net(self):
        new_saver = tf.train.import_meta_graph(self.path_meta_graph + ".meta")
        new_saver.restore(self.session, self.path_meta_graph)

        # initializing attributes
        handles = self.session.graph.get_collection_ref(CVAE.RESTORE_KEY)
        print("laoding meta net")
        self.in_flat_images, self.z_mean, self.z_stddev, \
        self.z_in_, self.regenerated_3d_images_ = handles[0:5]

        # initialing variables
        self.n_z = self.z_in_.get_shape().as_list()[1]

    def __generate_tensorboard_files(self):
        print("Generating Tensorboard {}".format(self.generate_tensorboard))

        if self.generate_tensorboard:
            if self.path_session_folder is None:
                print("It is not possible to generate Tensorflow graph without a"
                      "path session specified")
            else:
                print("Generating Tensorboard")
                tb_path = os.path.join(self.path_session_folder, "tb")
                writer = tf.summary.FileWriter(
                    tb_path,graph=tf.get_default_graph())

    def __build_graph(self):
        """
        self.in_flat_images tensor--sh[n_samples x total_size]
        :return:
        """

        with tf.variable_scope("input"):
            # Placeholder location
            self.decay_rate = tf.placeholder_with_default(
                self.decay_rate_value, shape=[], name="decay_rate")

            self.in_flat_images = tf.placeholder(tf.float32,
                                                 [None, self.total_size],
                                                 "input_images")

            image_matrix = tf.reshape(self.in_flat_images,
                                      [-1, self.image_shape[0], self.image_shape[1],
                                       self.image_shape[2], 1])
            self.dim_in_first_layer = tf.shape(image_matrix)

        self.z_mean, self.z_stddev = self.recognition(image_matrix)

        with tf.variable_scope("reparametrization_trick"):
            samples = tf.random_normal(tf.shape(self.z_mean), 0, 1, dtype=tf.float32)
            guessed_z = self.z_mean + (self.z_stddev * samples)

        self.generated_images = self.generation(guessed_z, reuse_bool=False)

        generated_flat = tf.reshape(self.generated_images,
                                    tf.shape(self.in_flat_images))

        self.cost = self.__cost_calculation(generated_flat, self.z_mean,
                                            self.z_stddev)

        self.global_step = tf.Variable(0, trainable=False)

        self.temp_learning_rate = self.learning_rate * \
                                  tf.exp(- tf.multiply(
                                      tf.cast(self.global_step, tf.float32),
                                      self.decay_rate))

        self.optimizer = tf.train.AdamOptimizer(
            self.temp_learning_rate).minimize(
            self.cost, global_step=self.global_step)

        # ops to directly explore latent space
        # defaults to prior z ~ N(0, I)

        self.z_in_ = tf.placeholder(tf.float32,
                                    shape=[None, self.n_z], name="latent_in")
        generated_images_ = self.generation(self.z_in_, reuse_bool=True)

        self.regenerated_3d_images_ = \
            tf.reshape(generated_images_,
                       [-1, self.image_shape[0], self.image_shape[1],
                        self.image_shape[2]])

    def __init_session_folders(self):
        """
        This method will create inside the "out" folder a folder with the datetime
        of the execution of the neural net and with, with 3 folders inside it
        :return:
        """
        self.path_to_images = os.path.join(self.path_session_folder, "images")
        self.path_to_logs = os.path.join(self.path_session_folder, "logs")
        self.path_to_meta = os.path.join(self.path_session_folder, "meta")

        self.path_to_3dtemp_images = \
            os.path.join(self.path_to_images, "temp_3d_images")

        self.path_to_final_comparison_images = \
            os.path.join(self.path_to_images, "final_comparison")

        self.path_to_losses_log = \
            os.path.join(self.path_to_logs , "losses_logs")

        create_directories([self.path_session_folder, self.path_to_images,
                            self.path_to_logs, self.path_to_meta,
                            self.path_to_3dtemp_images,
                            self.path_to_final_comparison_images,
                            self.path_to_losses_log])

    def __cost_calculation(self, images_reconstructed, z_mean, z_stddev):
        bool_logs = True

        with tf.variable_scope("error_estimation"):
            with tf.variable_scope("generation_loss"):
                self.generation_loss = -tf.reduce_sum(
                    self.in_flat_images * tf.log(1e-8 + images_reconstructed) +
                    (1 - self.in_flat_images) * tf.log(1e-8 + 1 - images_reconstructed), 1)

            with tf.variable_scope("latent_layer_loss"):
                self.latent_loss = 0.5 * tf.reduce_sum(
                    tf.square(z_mean) + tf.square(z_stddev) - tf.log(
                        tf.square(z_stddev)) - 1, 1)

            cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
            #cost = self.generation_loss + self.latent_loss
            # self

            if bool_logs:
                print("shape generation_loss :{} ".format(
                    str(self.generation_loss.get_shape().as_list())))

                print(str("type latent_loss :{} ".format(
                    str(type(self.latent_loss)))))

            if self.lambda_l2_reg != 0:
                with tf.name_scope("l2_regularization"):
                    regularizers = [tf.nn.l2_loss(var) for var in
                                    self.session.graph.get_collection(
                                        "trainable_variables") if
                                    "weights" in var.name]

                    l2_reg = self.lambda_l2_reg * tf.add_n(regularizers)

                    cost += l2_reg

        return cost

    # encoder
    def recognition(self, input_images):
       pass

    def encode(self, input_images):
        """

        :param input_images: shape [n_samples]
        :return:
        """
        output_dic = {}

        input_images_flat = \
            self.__inspect_and_reshape_to_flat_input_images(input_images)

        feed_dict = {self.in_flat_images: input_images_flat}
        out_encode = \
            self.session.run([self.z_mean, self.z_stddev], feed_dict=feed_dict)

        output_dic["mean"] = out_encode[0]
        output_dic["stdev"] = out_encode[1]
        return output_dic

    # decoder
    def generation(self, z, reuse_bool):
        print("This functions needs to be created in the son class")
        raise Exception('Generation not defined in autoencoder')

    def __inspect_and_reshape_to_flat_input_images(self, input_images):
        """

        :param input_images: sh[n_samples, voxels_flat] ||
                             sh[n_samples, w, h, d]

        :return: np.array sh[n_samples, voxels_flat]
        """
        assert input_images.ndim in [2, 4], \
            "The shape of the input should be [n_samples, voxels_flat]," \
            " or  [n_samples, width, heigth, depth]"

        if input_images.ndim == 4:
            input_images_flat = np.reshape(input_images,
                                           [input_images.shape[0],
                                            self.total_size])
        else:
            input_images_flat = input_images

        return input_images_flat

    def decoder(self, latent_layer_input, original_images):
        """

        :param input_images: shape [n_samples]
        :return:
        """

        input_images = \
            self.__inspect_and_reshape_to_flat_input_images(original_images)

        feed_dict = {self.z_in_: latent_layer_input,
                     self.in_flat_images: input_images}

        return self.session.run(self.regenerated_3d_images_,
                                feed_dict=feed_dict)

    def __save(self, saver, suffix_file_saver_name):

        outfile = os.path.join(self.path_to_meta, suffix_file_saver_name)
        saver.save(self.session, outfile, global_step=self.global_step)

    def __generate_and_save_temp_3d_images(self, regen_batch, suffix):
        feed_dict = {self.in_flat_images: regen_batch}
        generated_test = self.session.run(
            self.generated_images[1, :],
            feed_dict=feed_dict)

        image_3d = np.reshape(generated_test, self.image_shape)
        image_3d = image_3d.astype(float)
        file_path = os.path.join(self.path_to_3dtemp_images, suffix)
        output_utils.from_3d_image_to_nifti_file(file_path, image_3d)

    def __full_reconstruction_error_evaluation(self, images_flat):

        n_samples = images_flat.shape[0]
        feed_dict = {self.in_flat_images: images_flat}
        bool_logs = False

        reconstructed_images = self.session.run(
            self.generated_images,
            feed_dict=feed_dict)

        images_3d_reconstructed = \
            reshape_from_flat_to_3d(reconstructed_images, self.image_shape)
        images_3d_original = reshape_from_flat_to_3d(images_flat,
                                                     self.image_shape)

        images_3d_original = images_3d_original.astype(float)
        images_3d_reconstructed = images_3d_reconstructed.astype(float)

        if bool_logs:
            print("Shape original images")
            print(images_3d_original.shape)
            print("Shape Modified images")
            print(images_3d_reconstructed.shape)

        diff_matrix = np.subtract(images_3d_original, images_3d_reconstructed)

        # similarity_evaluation
        total_diff = diff_matrix.sum()
        similarity_evaluation = abs(
            total_diff / np.array(images_flat.shape).prod())

        # MSE

        square_diff_matrix = np.power(diff_matrix, 2)
        mse_over_samples = square_diff_matrix.sum() / n_samples

        if bool_logs:
            print("Similarity {}%".format(similarity_evaluation))

        return similarity_evaluation, mse_over_samples

    @staticmethod
    def is_not_valid_lantent_and_reconstruction_loss(
            gen_loss, lat_loss, learning_rate, iter):

        is_not_valid = math.isnan(np.mean(gen_loss)) or \
                       math.isnan(np.mean(lat_loss)) or \
                       math.isinf(np.mean(lat_loss)) or \
                       math.isinf(np.mean(gen_loss))

        if is_not_valid:
            print("iter %d: genloss %f latloss %f learning_rate %f" % (
                iter, np.mean(gen_loss), np.mean(lat_loss), learning_rate))
            return True
        else:
            return False

    def __evaluate_and_restrict_output_if_session_folder_is_not_defined(
            self, tempSGD_3dimages, final_dump_comparison_images,
            dump_losses_log, similarity_evaluation):

        if self.path_session_folder is None:
            if tempSGD_3dimages:
                print("The session folder was not defined so 'tempSGD_3dimages'"
                      "will be set to automatically to False because the output"
                      "folder has not been defined")
                tempSGD_3dimages = False

            if final_dump_comparison_images:
                print(
                    "The session folder was not defined so 'final_dump_comparison_images'"
                    "will be set automatically to False because the output"
                    "folder has not been defined")
                final_dump_comparison_images = False

            if dump_losses_log:
                print("The session folder was not defined so 'dump_losses_log'"
                      "will be set automatically to False because the output"
                      "folder has not been defined")
                dump_losses_log = False

        return tempSGD_3dimages, final_dump_comparison_images, \
               dump_losses_log

    def __generate_losses_log_file(self, suffix, similarity_evaluation):

        path_to_file = \
            os.path.join(self.path_to_losses_log,
                         "{0}.txt".format(suffix))
        file = open(path_to_file, "w")

        if similarity_evaluation:
            file.write("{0},{1},{2},{3},{4}, {5}".format(
                "iteration", "generative loss", "latent layer loss",
                "learning rate", "similarity score", "MSE error over samples\n"))
        else:
            file.write("{0},{1},{2},{3}".format(
                "iteration", "generative loss", "latent layer loss",
                "learning rate\n"))

        return file

    def __compare_original_vs_reconstructed_samples(self, images_flat, suffix,
                                                    samples_to_compare,
                                                    planes_per_axis_to_show_in_compare):

        print("Final comparision between original and reconstructed images")

        feed_dict = {self.in_flat_images: images_flat}
        bool_logs = True

        reconstructed_images = self.session.run(
            self.generated_images,
            feed_dict=feed_dict)

        images_3d_reconstructed = \
            reshape_from_flat_to_3d(reconstructed_images, self.image_shape)
        images_3d_original = reshape_from_flat_to_3d(images_flat,
                                                     self.image_shape)

        images_3d_original = images_3d_original.astype(float)
        images_3d_reconstructed = images_3d_reconstructed.astype(float)

        if samples_to_compare is None:
            samples_to_compare = list(range(0, images_flat.shape[0], 1))

        if planes_per_axis_to_show_in_compare is None:
            p1, p2, p3 = region_plane_selector.get_middle_planes(
                images_3d_original[0, :, :, :])
        else:
            p1 = planes_per_axis_to_show_in_compare[0]
            p2 = planes_per_axis_to_show_in_compare[1]
            p3 = planes_per_axis_to_show_in_compare[2]

        for sample_index in samples_to_compare:
            img_path = os.path.join(
                self.path_to_final_comparison_images,
                "{0}_sample{1}.png".format(suffix, sample_index))

            recons.plot_section_indicated(
                img3d_1=images_3d_original[sample_index, :, :, :],
                img3d_2=images_3d_reconstructed[sample_index, :, :, :],
                p1=p1, p2=p2, p3=p3,
                path_to_save_image=img_path,
                cmap="jet",
                tittle="Original VS Reconstructres. {0} . Sample {1}.".format(
                    suffix, sample_index
                ))

    def __log_loss_data(self, iter_index, gen_loss, lat_loss, learning_rate,
                        images_flat, losses_log_file, similarity_evaluation):

        if similarity_evaluation is not None:
            # Generate %similarity in reconstruction
            similarity_score, mse_score = \
                self.__full_reconstruction_error_evaluation(images_flat=images_flat)

            print("iter {0}: genloss {1}, latloss {2}, "
                  "learning_rate {3}, Similarity Score: {4},"
                  "MSE {5}".format(
                iter_index, gen_loss, lat_loss,
                learning_rate, similarity_score, mse_score))

            if losses_log_file is not None:
                losses_log_file.write("{0},{1},{2},{3},{4},{5}\n".format(
                    iter_index, gen_loss, lat_loss,
                    learning_rate, similarity_score, mse_score))

        else:
            print("iter {0}: genloss {1}, latloss {2}, learning_rate {3}".format(
                    iter_index, gen_loss, lat_loss,learning_rate))

            if losses_log_file is not None:
                losses_log_file.write("{0},{1},{2},{3}\n".format(
                    iter_index, gen_loss, lat_loss, learning_rate))

    def train(self, X, n_iters=1000, batchsize=10, tempSGD_3dimages=False,
              iter_show_error=10, save_bool=False, suffix_files_generated=" ",
              iter_to_save=100, break_if_nan_error_value=True,
              similarity_evaluation=False,
              dump_losses_log=False,
              final_dump_comparison=False,
              final_dump_samples_to_compare=None,
              final_dump_planes_per_axis_to_show_in_compare=None):

        tempSGD_3dimages, final_dump_comparison_images, dump_losses_log = \
            self.__evaluate_and_restrict_output_if_session_folder_is_not_defined(
                tempSGD_3dimages, final_dump_comparison, dump_losses_log,
                similarity_evaluation)

        if dump_losses_log:
            losses_log_file = self.__generate_losses_log_file(
                suffix=suffix_files_generated,
                similarity_evaluation=similarity_evaluation)
        else:
            losses_log_file = None

        saver = None
        if save_bool:
            saver = tf.train.Saver(tf.global_variables())


        if tempSGD_3dimages:
            sample_image = X[0:2, :,:,:]
            sample_image_flat = reshape_from_3d_to_flat(sample_image, self.total_size)
            image_3d = sample_image.astype(float)
            file_path = os.path.join(self.path_to_3dtemp_images,
                                     "original_{}".format(suffix_files_generated))
            output_utils.from_3d_image_to_nifti_file(file_path, image_3d)

        # reshape from 3d to flat:
        X_flat = reshape_from_3d_to_flat(X, self.total_size)


        try:
            for iter in range(1, n_iters + 1, 1):

                batch_flat = get_batch_from_samples_unsupervised(
                    X_flat, batch_size=batchsize)

                feed_dict = {self.in_flat_images: batch_flat}
                _, gen_loss, lat_loss, global_step, learning_rate = \
                    self.session.run(
                        (self.optimizer, self.generation_loss, self.latent_loss,
                         self.global_step, self.temp_learning_rate),
                        feed_dict=feed_dict)

                if break_if_nan_error_value:
                    # Evaluate if the net is not converging and the error
                    # is a nan value, breaking the SGD loop
                    if self.is_not_valid_lantent_and_reconstruction_loss(
                            gen_loss, lat_loss, learning_rate, iter):
                        return -1

                if iter % iter_show_error == 0:
                    self.__log_loss_data(
                        iter_index=iter,
                        gen_loss=np.mean(gen_loss),
                        lat_loss=np.mean(lat_loss),
                        learning_rate=learning_rate,
                        images_flat=X_flat,
                        losses_log_file=losses_log_file,
                        similarity_evaluation=similarity_evaluation)

                    if tempSGD_3dimages:
                        self.__generate_and_save_temp_3d_images(
                            regen_batch=sample_image_flat,
                            suffix="{1}_iter_{0}".format(
                                iter, suffix_files_generated))

                if iter % iter_to_save == 0:
                    if save_bool:
                        self.__save(saver, suffix_files_generated)

            # End loop, End SGD
            # final dump data if the dump_comparaison option is activated
            if final_dump_comparison:
                self.__compare_original_vs_reconstructed_samples(
                    images_flat=X_flat,
                    suffix=suffix_files_generated,
                    samples_to_compare=final_dump_samples_to_compare,
                    planes_per_axis_to_show_in_compare=
                    final_dump_planes_per_axis_to_show_in_compare)

            return 0

        except(KeyboardInterrupt):
            print("iter %d: genloss %f latloss %f learning_rate %f" % (
                iter, np.mean(gen_loss), np.mean(lat_loss), learning_rate))
            now = datetime.now().isoformat()[11:]
            print("------- Training end: {} -------\n".format(now))
            sys.exit(0)


# little modification
import sys
import os
#sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import math
from datetime import datetime
from lib import reconstruct_helpers as recons
import numpy as np
import tensorflow as tf
import lib.neural_net.kfrans_ops as ops
from lib.utils import output_utils
from lib.utils.functions import \
    get_batch_from_samples_unsupervised
from lib.utils.os_aux import create_directories
import region_plane_selector
from lib.utils.utils_3dsamples import reshape_from_flat_to_3d
from lib.utils.utils_3dsamples import reshape_from_3d_to_flat

bool_save_meta = False


class CVAE():
    RESTORE_KEY = "restore"

    def __init__(self, hyperparams, test_bool=False, meta_path=None,
                 path_to_session=None):

        assert "image_shape" in list(hyperparams), \
            "image_shape should be specified in hyperparams"
        self.image_shape = hyperparams['image_shape']
        self.total_size = np.array(self.image_shape).prod()

        self.session = tf.Session()

        self.dim_in_first_layer = None
        self.dim_out_first_layer = None
        self.dim_out_second_layer = None
        self.input_dense_layer_dim = None

        if test_bool:
            print(hyperparams)

        if meta_path is None:

            # Initizalizing graph values
            self.n_z = hyperparams['latent_layer_dim']
            self.lambda_l2_reg = hyperparams['lambda_l2_regularization']
            self.learning_rate = hyperparams['learning_rate']
            self.features_depth = hyperparams['features_depth']
            self.kernel_size = hyperparams['kernel_size']
            self.activation_layer = hyperparams['activation_layer']
            self.decay_rate_value = float(hyperparams['decay_rate'])
            self.path_session_folder = path_to_session

            self.__build_graph()

            if self.path_session_folder is not None:
                self.__init_session_folders()

            handles = [self.in_flat_images, self.z_mean, self.z_stddev,
                       self.z_in_, self.regenerated_3d_images_]
            for handle in handles:
                tf.add_to_collection(CVAE.RESTORE_KEY, handle)

            self.session.run(tf.initialize_all_variables())
        else:
            new_saver = tf.train.import_meta_graph(meta_path + ".meta")
            new_saver.restore(self.session, meta_path)

            # initializing attributes
            handles = self.session.graph.get_collection_ref(CVAE.RESTORE_KEY)
            self.in_flat_images, self.z_mean, self.z_stddev, \
            self.z_in_, self.regenerated_3d_images_ = handles[0:5]

            # initialing variables
            self.n_z = self.z_in_.get_shape().as_list()[1]

    def __build_graph(self):
        """
        self.in_flat_images tensor--sh[n_samples x total_size]
        :return:
        """

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
        samples = tf.random_normal(tf.shape(self.z_mean), 0, 1,
                                   dtype=tf.float32)
        guessed_z = self.z_mean + (self.z_stddev * samples)

        self.generated_images = self.__generation(guessed_z, reuse_bool=False)
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
        generated_images_ = self.__generation(self.z_in_, reuse_bool=True)

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
        self.path_to_grad_desc_error = \
            os.path.join(self.path_to_logs, "DescGradError")

        self.path_to_3dtemp_images = \
            os.path.join(self.path_to_images, "temp_3d_images")

        self.path_to_final_comparison_images = \
            os.path.join(self.path_to_images, "final_comparison")

        create_directories([self.path_session_folder, self.path_to_images,
                            self.path_to_logs, self.path_to_meta,
                            self.path_to_3dtemp_images,
                            self.path_to_final_comparison_images])

    def __cost_calculation(self, images_reconstructed, z_mean, z_stddev):
        self.generation_loss = -tf.reduce_sum(
            self.in_flat_images * tf.log(1e-8 + images_reconstructed) + (
                1 - self.in_flat_images) * tf.log(
                1e-8 + 1 - images_reconstructed), 1)

        self.latent_loss = 0.5 * tf.reduce_sum(
            tf.square(z_mean) + tf.square(z_stddev) - tf.log(
                tf.square(z_stddev)) - 1, 1)
        cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
        # self

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
        with tf.variable_scope("recognition"):
            h1 = self.activation_layer(ops.conv3d(
                x=input_images,
                input_features=self.features_depth[0],
                output_features=self.features_depth[1],
                stride=2,
                kernel=self.kernel_size,
                name="first_layer"))  # 28x28x1 -> 14x14x16

            self.dim_out_first_layer = tf.shape(h1)
            h2 = self.activation_layer(ops.conv3d(
                x=h1,
                input_features=self.features_depth[1],
                output_features=self.features_depth[2],
                stride=2,
                kernel=self.kernel_size,
                name="second_layers"))  # 14x14x16 -> 7x7x32

            self.dim_out_second_layer = tf.shape(h2)

            self.input_dense_layer_dim = [
                -1, np.array(h2.get_shape().as_list()[1:]).prod()]
            h2_flat = tf.reshape(h2, self.input_dense_layer_dim)

            w_mean = ops.dense(h2_flat, self.input_dense_layer_dim[1], self.n_z,
                               "w_mean")
            w_stddev = ops.dense(h2_flat, self.input_dense_layer_dim[1],
                                 self.n_z,
                                 "w_stddev")

        return w_mean, w_stddev

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
    def __generation(self, z, reuse_bool):
        with tf.variable_scope("generation", reuse=reuse_bool):
            z_develop = ops.dense(z, self.n_z, self.input_dense_layer_dim[1],
                                  scope='z_matrix')

            z_matrix = self.activation_layer(
                tf.reshape(z_develop, self.dim_out_second_layer))

            h1 = self.activation_layer(ops.conv3d_transpose(
                x=z_matrix,
                output_shape=self.dim_out_first_layer,
                output_features=self.features_depth[1],
                input_features=self.features_depth[2],
                stride=2,
                kernel=self.kernel_size,
                name="g_h1"))

            h2 = ops.conv3d_transpose(x=h1,
                                      output_shape=self.dim_in_first_layer,
                                      output_features=self.features_depth[0],
                                      input_features=self.features_depth[1],
                                      stride=2,
                                      kernel=self.kernel_size,
                                      name="g_h2")
            h2 = tf.nn.sigmoid(h2)

        return h2

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

        feed_dict = {self.in_flat_images: images_flat}
        bool_logs = False

        reconstructed_images = self.session.run(
            self.generated_images,
            feed_dict=feed_dict)

        images_3d_reconstructed = \
            reshape_from_flat_to_3d(reconstructed_images, self.image_shape)
        images_3d_original = reshape_from_flat_to_3d(images_flat, self.image_shape)

        images_3d_original = images_3d_original.astype(float)
        images_3d_reconstructed = images_3d_reconstructed.astype(float)

        if bool_logs:
            print("Shape original images")
            print(images_3d_original.shape)
            print("Shape Modified images")
            print(images_3d_reconstructed.shape)

        diff_matrix = np.subtract(images_3d_original, images_3d_reconstructed)
        total_diff = diff_matrix.sum()
        mean_diff = abs(total_diff / np.array(images_flat.shape).prod()) * 2 * 100

        print("Similarity {}%".format(mean_diff))

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
            self,tempSGD_3dimages,final_dump_comparison_images):

        if self.path_session_folder:
            if tempSGD_3dimages:
                print("The session folder was not defined so 'tempSGD_3dimages'"
                      "will be set to automatically to False because the output"
                      "folder has not been defined")
                tempSGD_3dimages=False

            if final_dump_comparison_images:
                print("The session folder was not defined so 'final_dump_comparison_images'"
                      "will be set automatically to False because the output"
                      "folder has not been defined")
                final_dump_comparison_images=False

        return tempSGD_3dimages, final_dump_comparison_images

    def __compare_original_vs_reconstructed_samples(self, images_flat, suffix,
        samples_to_compare, planes_per_axis_to_show_in_compare):

        feed_dict = {self.in_flat_images: images_flat}
        bool_logs = True

        reconstructed_images = self.session.run(
            self.generated_images,
            feed_dict=feed_dict)

        images_3d_reconstructed = \
            reshape_from_flat_to_3d(reconstructed_images, self.image_shape)
        images_3d_original = reshape_from_flat_to_3d(images_flat, self.image_shape)

        images_3d_original = images_3d_original.astype(float)
        images_3d_reconstructed = images_3d_reconstructed.astype(float)

        if samples_to_compare is None:
            samples_to_compare = list(range(0,images_flat.shape[0], 1))

        if planes_per_axis_to_show_in_compare is None:
            p1,p2,p3 = region_plane_selector.get_middle_planes(
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

    def train(self, X, n_iters=1000, batchsize=10, tempSGD_3dimages=False,
              iter_show_error=10, save_bool=False, suffix_files_generated=" ",
              iter_to_save=100, break_if_nan_error_value=True,
              full_samples_evaluation=False,
              final_dump_comparison=False,
              final_dump_samples_to_compare=None,
              final_dump_planes_per_axis_to_show_in_compare=None):

        tempSGD_3dimages, final_dump_comparison_images = \
            self.__evaluate_and_restrict_output_if_session_folder_is_not_defined(
            tempSGD_3dimages, final_dump_comparison)

        saver = None
        if save_bool:
            saver = tf.train.Saver(tf.global_variables())

        #reshape from 3d to flat:
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
                    print("iter %d: genloss %f latloss %f learning_rate %f" % (
                        iter, np.mean(gen_loss), np.mean(lat_loss),
                        learning_rate))

                    if full_samples_evaluation:
                        all_images_flat = np.reshape(X, [X.shape[0], self.total_size])
                        # Generate %similarity in reconstruction
                        self.__full_reconstruction_error_evaluation(
                            images_flat=all_images_flat)

                    if tempSGD_3dimages:
                        self.__generate_and_save_temp_3d_images(
                            regen_batch=batch_flat[0:2, :],
                            suffix="{1}_iter_{0}".format(
                                iter, suffix_files_generated))

                if iter % iter_to_save == 0:
                    if save_bool:
                        self.__save(saver, suffix_files_generated)
            # End loop, End SGD


            #final dump data if the dump_comparaison option is activated
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


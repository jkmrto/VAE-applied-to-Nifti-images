import os
import sys
from datetime import datetime
import numpy as np
import tensorflow as tf
from numpy import inf
import lib.loss_function as loss
import settings
from lib.math_utils import sample_gaussian
from lib.neural_net.layers import Dense
from lib.aux_functionalities.os_aux import create_directories
from lib.utils import compose_all


class VAE():
    """Variational Autoencoder

    see: Kingma & Welling - Auto-Encoding Variational Bayes
    (http://arxiv.org/abs/1312.6114)
    """
    hyper_params = {
        "batch_size": 10,
        "learning_rate": 1E-3,
        "dropout": 1.,
        "lambda_l2_reg": 0.,
        "nonlinearity": tf.nn.elu,
        "squashing": tf.nn.sigmoid
    }

    RESTORE_KEY = "restore"

    def __init__(self, architecture, hyperparams, meta_graph=None, path_to_session=None):
        """(Re)build a symmetric VAE model with given:

            * architecture (list of nodes per encoder layer); e.g.
               [1000, 500, 250, 10] specifies a VAE with 1000-D inputs, 10-D latents,
               & end-to-end architecture [1000, 500, 250, 10, 250, 500, 1000]

            * hyperparameters (optional dictionary of updates to `DEFAULTS`)
        """

        self.architecture = architecture
        self.hyper_params.update(hyperparams)
        print("Hyperparamers indicated: " + str(VAE.hyper_params))

        self.session = tf.Session()

        if not meta_graph: # new model
            self.init_session_folders(path_to_session)
            assert len(self.architecture) > 2, \
                "Architecture must have more layers! (input, 1+ hidden, latent)"
            # build graph
            handles = self._build_graph()
            for handle in handles:
                tf.add_to_collection(VAE.RESTORE_KEY, handle)
            self.session.run(tf.global_variables_initializer())

        else: # restore saved model
            tf.train.import_meta_graph(meta_graph + ".meta").restore(self.session, meta_graph)
            handles = self.session.graph.get_collection_ref(VAE.RESTORE_KEY)

        print(handles)
        (self.x_in, self.dropout_, self.z_mean, self.z_log_sigma,
        self.x_reconstructed, self.z_, self.x_reconstructed_,
        self.cost, self.global_step, self.train_op) = handles[0:10]

    def init_session_folders(self, path_to_session):
        """
        This method will create inside the "out" folder a folder with the datetime
        of the execution of the neural net and with, with 3 folders inside it
        :return:
        """
        self.path_session_folder = path_to_session

        self.path_to_images = os.path.join(self.path_session_folder, "images")
        self.path_to_logs = os.path.join(self.path_session_folder, "logs")
        self.path_to_meta = os.path.join(self.path_session_folder, "meta")
        self.path_to_grad_desc_error = os.path.join(self.path_to_logs, "DescGradError")

        create_directories([self.path_session_folder, self.path_to_images,
                            self.path_to_logs, self.path_to_meta])

    @property
    def step(self):
        """Train step"""
        return self.global_step.eval(session=self.session)

    def __build_cost_estimate(self, x_reconstructed, x_in, z_mean, z_log_sigma):

        # reconstruction loss: mismatch b/w x & x_reconstructed
        # binary cross-entropy -- assumes x & p(x|z) are iid Bernoullis
        rec_loss = loss.crossEntropy(x_reconstructed, x_in)

        # Kullback-Leibler divergence: mismatch b/w approximate vs. imposed/true posterior
        kl_loss = loss.kullbackLeibler(z_mean, z_log_sigma)

        with tf.name_scope("l2_regularization"):
            regularizers = [tf.nn.l2_loss(var) for var in self.session.graph.get_collection(
                "trainable_variables") if "weights" in var.name]
            l2_reg = self.hyper_params['lambda_l2_reg'] * tf.add_n(regularizers)

        with tf.name_scope("cost"):
            # average over minibatch
            cost = tf.reduce_mean(rec_loss + kl_loss, name="vae_cost")
            cost += l2_reg

        return cost

    def _build_graph(self):
        x_in = tf.placeholder(tf.float32, shape=[None, self.architecture[0]], name="x")
        dropout = tf.placeholder_with_default(1., shape=[], name="dropout")

        # encoding / "recognition": q(z|x) ->  outer -> inner
        encoding = [Dense("encoding", hidden_size, dropout, self.hyper_params['nonlinearity'])
                    for hidden_size in reversed(self.architecture[1:-1])]
        h_encoded = compose_all(encoding)(x_in)

        # latent distribution parametetrized by hidden encoding
        # z ~ N(z_mean, np.exp(z_log_sigma)**2)
        z_mean = Dense("z_mean", self.architecture[-1], dropout)(h_encoded)
        z_log_sigma = Dense("z_log_sigma", self.architecture[-1], dropout)(h_encoded)

        # kingma & welling: only 1 draw necessary as long as minibatch large enough (>100)
        z = sample_gaussian(z_mean, z_log_sigma)

        # decoding / "generative": p(x|z)
        decoding = [Dense("decoding", hidden_size, dropout, self.hyper_params['nonlinearity'])
                    for hidden_size in self.architecture[1:-1]] # assumes symmetry

        # final reconstruction: restore original dims, squash outputs [0, 1]
        decoding.insert(0, Dense( # prepend as outermost function
            "x_decoding", self.architecture[0], dropout, self.hyper_params['squashing']))
        x_reconstructed = tf.identity(compose_all(decoding)(z), name="x_reconstructed")

        cost = self.__build_cost_estimate(x_reconstructed, x_in, z_mean, z_log_sigma)

        # optimization
        global_step = tf.Variable(0, trainable=False)
        with tf.name_scope("Adam_optimizer"):
            optimizer = tf.train.AdamOptimizer(self.hyper_params['learning_rate'])
            tvars = tf.trainable_variables()
            grads_and_vars = optimizer.compute_gradients(cost, tvars)
            clipped = [(tf.clip_by_value(grad, -5, 5), tvar) # gradient clipping
                    for grad, tvar in grads_and_vars]
            train_op = optimizer.apply_gradients(clipped, global_step=global_step,
                                                 name="minimize_cost")

        # ops to directly explore latent space
        # defaults to prior z ~ N(0, I)
        with tf.name_scope("latent_in"):
            z_ = tf.placeholder_with_default(tf.random_normal([1, self.architecture[-1]]),
                                            shape=[None, self.architecture[-1]],
                                            name="latent_in")
        x_reconstructed_ = compose_all(decoding)(z_)

        return (x_in, dropout, z_mean, z_log_sigma, x_reconstructed,
                z_, x_reconstructed_, cost, global_step, train_op)

    def encode(self, x):
        """Probabilistic encoder from inputs to latent distribution parameters;
        a.k.a. inference network q(z|x)
        """
        # np.array -> [float, float]
        feed_dict = {self.x_in: x}
        return self.session.run([self.z_mean, self.z_log_sigma], feed_dict=feed_dict)

    def decode(self, zs=None):
        """Generative decoder from latent space to reconstructions of input space;
        a.k.a. generative network p(x|z)
        """
        # (np.array | tf.Variable) -> np.array
        feed_dict = dict()
        if zs is not None:
            is_tensor = lambda x: hasattr(x, "eval")
            zs = (self.session.run(zs) if is_tensor(zs) else zs) # coerce to np.array
            feed_dict.update({self.z_: zs})
        # else, zs defaults to draw from conjugate prior z ~ N(0, I)
        return self.session.run(self.x_reconstructed_, feed_dict=feed_dict)

    def vae(self, x):
        """End-to-end autoencoder"""
        # np.array -> np.array
        return self.decode(sample_gaussian(*self.encode(x)))

    def save(self, saver, suffix_file_saver_name):

        outfile = os.path.join(self.path_to_meta, suffix_file_saver_name)
        saver.save(self.session, outfile, global_step=self.step)

    def generate_batch(self, n_samples):

        l = np.arange(0, n_samples)
        batch_idx = [l[i:i + self.hyper_params['batch_size']] for i in
                     range(0, len(l), self.hyper_params['batch_size'])]
        return batch_idx

    def training_end(self, saver, save_bool, err_train, i, suffix):

        print("final avg cost %1.5f" % (err_train / i))
        now = datetime.now().isoformat()[11:]
        print("------- Training end: {} -------\n".format(now))

        self.save(saver, suffix) if save_bool else None

    def train(self, X, max_iter=np.inf, save_bool=True, suffix_files_generated=" ",
              iter_to_save=1000, iters_to_show_error=100):

        """
        :param iters_to_show_error:
        :param X:
        :param max_iter:
        :param save_bool:
        :param iters_to_save:
        """
        saver = tf.train.Saver(tf.global_variables()) if save_bool else None
        err_train = 0

        path_to_file = os.path.join(self.path_to_grad_desc_error,
                                    suffix_files_generated + "log")
        gradient_descent_log = open(path_to_file, "w")

        try:
            now = datetime.now().isoformat()[11:]
            print("------- Training begin: {} -------\n".format(now))
            i = 0
            while True:  # Se ejecuta hasta condicion i>max_iter -> break

                # batch selector
                index = np.random.choice(range(X.shape[0]), self.hyper_params['batch_size'], replace=False)
                x = X[index.tolist(), :]

                feed_dict = {self.x_in: x, self.dropout_: self.hyper_params['dropout']}
                fetches = [self.x_reconstructed, self.cost, self.global_step, self.train_op]
                x_reconstructed, cost, i, _ = self.session.run(fetches, feed_dict)

                err_train += cost
                gradient_descent_log.write("{0},{1}\n".format(i, cost))

                if i % iters_to_show_error == 0:
                    print("round {} --> avg cost: ".format(i), err_train/iters_to_show_error)
                    err_train = 0  # Reinitialzing the counting error

                if i % iter_to_save == 0:
                    self.save(saver, suffix_files_generated)

                if i >= max_iter:
                    self.training_end(saver, save_bool, err_train/iters_to_show_error, i, suffix_files_generated)
                    break

        except(KeyboardInterrupt):
            print("final avg cost (@ step {} = epoch {}): {}".format(
                i, X.train.epochs_completed, err_train / i))
            now = datetime.now().isoformat()[11:]
            print("------- Training end: {} -------\n".format(now))
            sys.exit(0)





# restore saved model
       #     model_datetime, model_name = os.path.basename(meta_graph).split("_vae_")
       #     self.datetime = "{}_reloaded".format(model_datetime)
       #     *model_architecture, _ = re.split("_|-", model_name)
       #     self.architecture = [int(n) for n in model_architecture]
        #    meta_graph = os.path.abspath(meta_graph)

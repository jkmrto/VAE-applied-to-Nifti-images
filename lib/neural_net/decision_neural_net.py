import tensorflow as tf
from lib.neural_net.layers import Dense
from lib.utils import compose_all
import numpy as np
from datetime import datetime
from lib.aux_functionalities.functions import get_batch_from_samples
from lib.aux_functionalities.os_aux import create_directories
import sys
import os

RESTORE_KEY = 'key'


class DecisionNeuralNet():
    def __init__(self, architecture=None, hyperparams=None, meta_graph=None,
                 root_path=""):
        self.architecture = architecture
        self.hyperparams = hyperparams
        self.session = tf.Session()
        self.root_path = root_path

        print("Hyperparamers indicated: " + str(self.hyperparams))

        if not meta_graph:  # new model
            # assert len(self.architecture) > 1, \
            self.init_session_folders()
            handles = self.__build_graph()
            for handle in handles:
                tf.add_to_collection(RESTORE_KEY, handle)
            self.session.run(tf.global_variables_initializer())

        else:  # restore saved model
            tf.train.import_meta_graph(meta_graph + ".meta").restore(self.session, meta_graph)
            handles = self.session.graph.get_collection_ref(RESTORE_KEY)

        print(handles)
        self.x_in, self.y_true, self.y_obtained, self.dropout, \
        self.cost, self.global_step, self.train_op = handles[0:7]

        #     if save_graph_def:  # tensorboard
        #         self.logger = tf.summary.FileWriter(log_dir, self.session.graph)

    @property
    def step(self):
        """Train step"""
        return self.global_step.eval(session=self.session)

    def init_session_folders(self):

        self.path_to_images = os.path.join(self.root_path, "images")
        self.path_to_logs = os.path.join(self.root_path, "logs")
        self.path_to_meta = os.path.join(self.root_path, "meta")

        create_directories([self.root_path, self.path_to_images,
                            self.path_to_logs, self.path_to_meta])

    @staticmethod
    def __build_cost_estimate(x_true, x_obtained):
        print("build cost estimate")
        print(x_true.get_shape())
        print(x_obtained.get_shape())
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=x_true, logits=x_obtained))
        # return l1_loss(x_true, x_obtained)

    def __build_graph(self):

        x_in = tf.placeholder(tf.float32, shape=[None, self.architecture[0]], name="x")
        y_true = tf.placeholder(tf.float32, shape=[None, self.architecture[-1]], name="y")
        dropout = tf.placeholder_with_default(1., shape=[], name="dropout")

        # encoding / "recognition": q(z|x) ->  outer -> inner
        dense_layers = [Dense("net", hidden_size, dropout, self.hyperparams['nonlinearity'])
                        for hidden_size in reversed(self.architecture)]

        y_obtained = compose_all(dense_layers)(x_in)

        global_step = tf.Variable(0, trainable=False)

        cost = self.__build_cost_estimate(y_true, y_obtained)

        with tf.name_scope("Adam_optimizer"):
            optimizer = tf.train.AdamOptimizer(self.hyperparams['learning_rate'])
            tvars = tf.trainable_variables()
            grads_and_vars = optimizer.compute_gradients(cost, tvars)
            clipped = [(tf.clip_by_value(grad, -5, 5), tvar)  # gradient clipping
                       for grad, tvar in grads_and_vars]
            train_op = optimizer.apply_gradients(clipped, global_step=global_step,
                                                 name="minimize_cost")

        return x_in, y_true, y_obtained, dropout, cost, global_step, train_op

    def forward_propagation(self, x, y):
        """Probabilistic encoder from inputs to latent distribution parameters;
        a.k.a. inference network q(z|x)
        """
        # np.array -> [float, float]
        feed_dict = {self.x_in: x}
        return self.session.run([self.y_obtained], feed_dict=feed_dict)

    def __old_generate_batch(self, n_samples):

        l = np.arange(0, n_samples)
        batch_idx = [l[i:i + self.hyperparams['batch_size']] for i in
                     range(0, len(l), self.hyperparams['batch_size'])]
        return batch_idx

    def save(self, saver):

        outfile = self.path_to_meta + "/"
        saver.save(self.session, outfile, global_step=self.step)

    def training_end(self, saver, save_bool, last_avg_cost):

        print("final avg cost %1.5f" % (last_avg_cost))
        now = datetime.now().isoformat()[11:]
        print("------- Training end: {} -------\n".format(now))

        self.save(saver) if save_bool else None

    def train(self, X, Y, max_iter=1000, save_bool=True, iter_to_show_error=100,
              iter_to_save=1000, path_to_grad_error_log_file_name=""):

        saver = tf.train.Saver(tf.global_variables()) if save_bool else None
        err_train = 0

        if not (path_to_grad_error_log_file_name == ""):
            gradient_descent_log = open(path_to_grad_error_log_file_name, "w")

        try:
            now = datetime.now().isoformat()[11:]
            print("------- Training begin: {} -------\n".format(now))
            last_avg_cost = 0

            while True:  # Se ejecuta hasta condicion i>max_iter -> break

                x, y = get_batch_from_samples(X, Y, self.hyperparams['batch_size'])

                #  x = X[batch_idx[bc], :]  # Selecting the required batch
                # y = Y[batch_idx[bc]]  # Selecting the required batch
                # bc = 0 if (bc > len(batch_idx) - 2) else bc + 1

                feed_dict = {self.x_in: x, self.dropout: self.hyperparams['dropout'], self.y_true: y}
                fetches = [self.cost, self.global_step, self.train_op, self.y_obtained, self.y_true]
                cost, i, _, y_obtained, y_true = self.session.run(fetches, feed_dict)

                err_train += cost
                if not (path_to_grad_error_log_file_name == ""):
                    gradient_descent_log.write("{0},{1}\n".format(i, cost))

                if i % iter_to_show_error == 0:
                    last_avg_cost = err_train / iter_to_show_error
                    print("round {} --> avg cost: ".format(i), last_avg_cost)
                    err_train = 0  # Reinitialzing the counting error

                if i % iter_to_save == 0:
                    if save_bool:
                        self.save(saver)

                if i >= max_iter:
                    self.training_end(saver, save_bool, last_avg_cost)
                    if not (path_to_grad_error_log_file_name == ""):
                        gradient_descent_log.close()
                    break


        except(KeyboardInterrupt):
            print("final avg cost (@ step {} = epoch {}): {}".format(
                i, X.train.epochs_completed, err_train / i))
            now = datetime.now().isoformat()[11:]
            print("------- Training end: {} -------\n".format(now))
            sys.exit(0)


def __load_svm_score_file():
    dir = os.getcwd()
    file_name = '/Resultados_lineal_todas_regiones.log'
    svm_output_file = open(dir + file_name)

    svm_output_region_index = []
    svm_output = []
    for line in svm_output_file:
        print(line)
        region_index, tail = line.split(':')
        head, score = tail.split(' ')
        svm_output_region_index.append(region_index)
        svm_output.append(float(score.replace('\n', '')))

    return svm_output, svm_output_region_index


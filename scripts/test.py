# def __old_generate_batch(self, X, Y):
import tensorflow as tf

HYPERPARAMS = {
    "batch_size": 16,
    "learning_rate": 5E-6,
    "dropout": 0.9,
    "lambda_l2_reg": 1E-5,
    "nonlinearity": tf.nn.elu,
    "squashing": tf.nn.sigmoid,
}

session_descriptor = {}
session_descriptor.update(HYPERPARAMS)
print(str(session_descriptor))


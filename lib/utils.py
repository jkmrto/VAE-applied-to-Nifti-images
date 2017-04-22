import functools
from functional import compose, partial
import tensorflow as tf


def compose_all(*args):
    """Util for multiple function composition

    i.e. composed = composeAll([f, g, h])
         composed(x) == f(g(h(x)))
    """
    # adapted from https://docs.python.org/3.1/howto/functional.html
    return partial(functools.reduce, compose)(*args)


def print_(var, name: str, first_n=5, summarize=5):
    """Util for debugging, by printing values of tf.Variable `var` during training"""
    # (tf.Tensor, str, int, int) -> tf.Tensor
    return tf.Print(var, [var], "{}: ".format(name), first_n=first_n,
                    summarize=summarize)


def normalize_array(array):
    out = array / array.max()
    return out, array.max


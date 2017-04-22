from lib.mnist import mnist_plot
from settings import *

def all_plots(model, mnist):
    if model.architecture[-1] == 2: # only works for 2-D latent
        print("Plotting in latent space...")
        plot_all_in_latent(model, mnist)

        print("Exploring latent...")
        mnist_plot.exploreLatent(model, nx=20, ny=20, range_=(-4, 4), outdir=PLOTS_DIR)
        for n in (24, 30, 60, 100):
            mnist_plot.exploreLatent(model, nx=n, ny=n, ppf=True, outdir=PLOTS_DIR,
                                     name="explore_ppf{}".format(n))

    print("Interpolating...")
    interpolate_digits(model, mnist)

    print("Plotting end-to-end reconstructions...")
    plot_all_end_to_end(model, mnist)

    print("Morphing...")
    morph_numbers(model, mnist, ns=[9,8,7,6,5,4,3,2,1,0])

    print("Plotting 10 MNIST digits...")
    for i in range(10):
        mnist_plot.justMNIST(get_mnist(i, mnist), name=str(i), outdir=PLOTS_DIR)


def plot_all_in_latent(model, mnist):
    names = ("train", "validation", "test")
    datasets = (mnist.train, mnist.validation, mnist.test)
    for name, dataset in zip(names, datasets):
        mnist_plot.plotInLatent(model, dataset.images, dataset.labels, name=name,
                                outdir=PLOTS_DIR)


def interpolate_digits(model, mnist):
    imgs, labels = mnist.train.next_batch(100)
    idxs = np.random.randint(0, imgs.shape[0] - 1, 2)
    mus, _ = model.encode(np.vstack(imgs[i] for i in idxs))
    mnist_plot.interpolate(model, *mus, name="interpolate_{}->{}".format(
        *(labels[i] for i in idxs)), outdir=PLOTS_DIR)


def plot_all_end_to_end(model, mnist):
    names = ("train", "validation", "test")
    datasets = (mnist.train, mnist.validation, mnist.test)
    for name, dataset in zip(names, datasets):
        x, _ = dataset.next_batch(10)
        x_reconstructed = model.VAE(x)
        mnist_plot.plotSubset(model, x, x_reconstructed, n=10, name=name,
                              outdir=PLOTS_DIR)


def morph_numbers(model, mnist, ns=None, n_per_morph=10):
    if not ns:
        import random
        ns = random.sample(range(10), 10) # non-in-place shuffle

    xs = np.squeeze([get_mnist(n, mnist) for n in ns])
    mus, _ = model.encode(xs)
    mnist_plot.morph(model, mus, n_per_morph=n_per_morph, outdir=PLOTS_DIR,
                     name="morph_{}".format("".join(str(n) for n in ns)))


def get_mnist(n, mnist):
    """Returns 784-D numpy array for random MNIST digit `n`"""
    assert 0 <= n <= 9, "Must specify digit 0 - 9!"
    import random

    SIZE = 500
    imgs, labels = mnist.train.next_batch(SIZE)
    idxs = iter(random.sample(range(SIZE), SIZE)) # non-in-place shuffle

    for i in idxs:
        if labels[i] == n:
            return imgs[i] # first match


def load_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets("./data/mnist_data")
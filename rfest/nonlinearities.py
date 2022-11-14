import jax.numpy as jnp

__all__ = ['apply_nonlinearity', 'softplus', 'softmax', 'sigmoid', 'sigmoid_shifted',
           'relu', 'leaky_relu', 'selu', 'swish', 'elu', 'identity']

import numpy as np
from matplotlib import pyplot as plt


def apply_nonlinearity(x, kind):
    if kind == 'softplus':
        return softplus(x)

    elif kind == 'exponential':
        return jnp.exp(x)

    elif kind == 'softmax':
        return softmax(x)

    elif kind == 'sigmoid':
        return sigmoid(x)

    elif kind == 'sigmoid_shifted':
        return sigmoid_shifted(x)

    elif kind == 'tanh':
        return jnp.tanh(x)

    elif kind == 'relu':
        return relu(x)

    elif kind == 'leaky_relu':
        return leaky_relu(x)

    elif kind == 'selu':
        return selu(x)

    elif kind == 'swish':
        return swish(x)

    elif kind == 'elu':
        return elu(x)

    elif kind == 'none' or kind == 'identity':
        return x

    else:
        raise ValueError(f'Input filter nonlinearity `{kind}` is not supported.')


def plot_nonlinearity(kind, xmin=-5, xmax=5):
    x = np.linspace(xmin, xmax, 1001)
    y = apply_nonlinearity(x, kind)
    plt.figure()
    plt.plot(x, y)
    plt.show()


def softplus(x):
    return jnp.log(1. + jnp.exp(x))


def softmax(x):
    z = jnp.exp(x)
    return 100. * z / z.sum()


def sigmoid(x):
    return 1. / (1. + jnp.exp(-x))


def sigmoid_shifted(x):
    return 1. / (1. + jnp.exp(-x - 0.5))


def relu(x):
    return jnp.where(x > 0., x, 0.)


def leaky_relu(x):
    return jnp.where(x > 0., x, x * 0.01)


def selu(x):
    return 1.0507 * jnp.where(x > 0., x, 1.6733 * jnp.exp(x) - 1.6733)


def swish(x):
    return x / (1. + jnp.exp(-x))


def elu(x):
    return jnp.where(x > 0., x, jnp.exp(x) - 1.)


def identity(x):
    return x

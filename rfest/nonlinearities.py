import jax.numpy as jnp

__all__ = ['softplus', 'softmax', 'sigmoid', 'relu', 'leaky_relu', 'selu', 'swish', 'elu', 'identity']


def softplus(x):
    return jnp.log(1 + jnp.exp(x))


def softmax(x):
    z = jnp.exp(x)
    return z / z.sum()


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def relu(x):
    return jnp.where(x > 0., x, 0.)


def leaky_relu(x):
    return jnp.where(x > 0., x, x * 0.01)


def selu(x):
    return 1.0507 * jnp.where(x > 0., x, 1.6733 * jnp.exp(x) - 1.6733)


def swish(x):
    return x / (1 + jnp.exp(-x))


def elu(x):
    return jnp.where(x > 0, x, jnp.exp(x) - 1)


def identity(x):
    return x

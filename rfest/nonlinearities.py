import jax.numpy as np

def softplus(x):
    return np.log(1 + np.exp(x)) 

def softmax(x):
    z = np.exp(x)
    return z / z.sum()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.where(x > 0., x, 0.)

def leaky_relu(x):
    return np.where(x > 0., x, x * 0.01)

def selu(x):
    return 1.0507 * np.where(x > 0., x, 1.6733 * np.exp(x) - 1.6733)

def swish(x):
    return x / (1 + np.exp(-x))

def elu(x):
    return np.where(x > 0, x, np.exp(x)-1)
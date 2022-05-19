import jax.numpy as jnp


def loss_neglogli(y, r, dt=1.):
    r = jnp.where(r != jnp.inf, r, 0.)  # remove inf
    r = jnp.maximum(r, 1e-20)  # remove zero to avoid nan in log.

    term0 = - jnp.log(r / dt) @ y  # spike term from poisson log-likelihood
    term1 = jnp.sum(r)  # non-spike term
    neglogli = term0 + term1
    return neglogli


def loss_mse(y, r):
    mse = jnp.mean((y - r) ** 2)
    return mse


def loss_penalty(w, alpha, beta):
    l1 = jnp.linalg.norm(w, 1) if alpha != 0. else 0.
    l2 = jnp.linalg.norm(w, 2) if alpha != 1. else 0.
    penalty = beta * ((1. - alpha) * l2 + alpha * l1)
    return penalty

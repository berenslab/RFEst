![showcase](./misc/showcase.png)

RFEst v2 is a Python3 toolbox for neural receptive field estimation, featuring methods such as spline-based GLMs,
Empirical Bayes with various Gaussian priors, and a few matrix factorization methods.

## Supported Methods

**Spline-based GLMs** [1]

The new GLM module unified both vanilla and spline GLMs.

```python
from rfest import GLM

lnp = GLM(distr='poisson', output_nonlinearity='softplus')

# add training data
lnp.add_design_matrix(X_train, dims=[25, ], df=[8, ], smooth='cr', name='stimulus')  # use spline for stimulus filter
lnp.add_design_matrix(y_train, dims=[20, ], df=[8, ], smooth='cr', shift=1,
                      name='history')  # use spline for history filter

# add validation data
lnp.add_design_matrix(X_dev, name='stimulus')  # basis will automatically apply to dev set
lnp.add_design_matrix(y_dev, name='history')

# intialize model parameters
lnp.initialize(num_subunits=1, dt=dt, method='random', random_seed=2046)

# fit model
lnp.fit(y={'train': y_train, 'dev': y_dev},
        num_iters=1000, verbose=100, step_size=0.1, beta=0.01)
```

**Evidence Optimization**

* Ridge Regression
* Automatic Relevance Determination (ARD) [2]
* Automatic Smoothness Determination (ASD) [3]
* Automatic Locality Determination (ALD) [4]

```python
from rfest import ASD

asd = ASD(X, y, dims=[5, 20, 15])  # nT, nX, nY
p0 = [1., 1., 2., 2., 2.]  # sig, rho, 𝛿t, 𝛿y, 𝛿x
asd.fit(p0=p0, num_iters=300)
```

**Matrix Factorization**

A few matrix factorization methods have been implemented as a submodule (`MF`).

```python
from rfest.MF import KMeans, semiNMF
```

For more information, see [here](https://github.com/berenslab/RFEst/blob/master/rfest/MF/README.md).

## Installation

`rfest` is available on [`pypi`](https://pypi.org/project/rfest/):

```sh
pip install rfest
```

This will install `rfest` with CPU support. 

Alternative, you can clone this repo into a local directory and install via pip editable mode:

```sh
git clone https://github.com/berenslab/RFEst
pip install -e RFEst
```

If you want GPU support, follow the instructions on the [`JAX` github repository](https://github.com/google/jax) to install `JAX` with GPU support (**before** installing `rfest`). For example, for NVIDIA GPUs, run

```sh
pip install -U "jax[cuda12]"
```

## Dependencies

    numpy
    scipy
    sklearn
    matplotlib
    jax
    jaxlib

## Tutorial

Tutorial notebooks can be found
here: [https://github.com/huangziwei/notebooks_RFEst](https://github.com/huangziwei/notebooks_RFEst)

## Reference

[1] Huang, Z., Ran, Y., Oesterle, J., Euler, T., & Berens, P. (2021). Estimating smooth and sparse neural receptive
fields with a flexible spline basis. Neurons, Behavior, Data Analysis, and Theory, 5(3),
1–30. https://doi.org/10.51628/001c.27578

[2] MacKay, D. J. (1994). Bayesian nonlinear modeling for the prediction competition. ASHRAE transactions, 100(2),
1053-1062.

[3] Sahani, M., & Linden, J. F. (2003). Evidence optimization techniques for estimating stimulus-response functions. In
Advances in neural information processing systems (pp. 317-324).

[4] Park, M., & Pillow, J. W. (2011). Receptive field inference with localized priors. PLoS computational biology, 7(10)
, e1002219.


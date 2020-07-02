![showcase](./misc/showcase.png)

RFEst is a Python3 toolbox for neural receptive field estimation, featuring methods such as spline-based GLMs, Empirical Bayes with various Gaussian priors, and a few matrix factorization methods. 

## Supported Methods

**Spline-based GLMs** [1]

`splineLG`, `splineLNP` and `splineLNLN` use *natural cubic regression splines* to approximate spatio-temporal RFs. 

Given a stimulus design matrix (X) and the corresponding response (y), an optimized RF is calculated with respect to the dimension of the RF `dims=(nT, nY, nX)` :

```python
from rfest import splineLG

spl = splineLG(X, y, dims=(5, 20, 15), df=9, smooth='cr') 
spl.fit(num_iters=500, alpha=1, beta=0.025, verbal=100)
```

**Evidence Optimization**

* Ridge Regression 
* Automatic Relevance Determination (ARD) [2]
* Automatic Smoothness Determination (ASD) [3]
* Automatic Locality Determination (ALD) [4]

```python
from rfest import ASD

asd = ASD(X, y, dims=(5, 20, 15)) # nt, ny, nx
p0 = [1., 1., 2., 2., 2.] # sig, rho, 𝛿t, 𝛿y, 𝛿x
asd.fit(p0=p0, num_iters=300)
```

**Matrix Factorization**

A few matrix factorization methods have been implemented as a submodule (`MF`). 

```python
from rfest.MF import KMeans, semiNMF
```

For more information, see [here](https://github.com/berenslab/RFEst/blob/master/rfest/MF/README.md). 

## Installation

RFEst uses [JAX](https://github.com/google/jax) for automatic differentiation and JIT compilation to GPU/CPU, so you need to install JAX first. 

To install CPU-only version, simply clone this repo into local directory and then run:

```bash
git clone https://github.com/berenslab/RFEst
pip install -e RFEst
```

To enable GPU support on **Linux**, you need to consult the [JAX install guide](https://github.com/google/jax#pip-installation). For reference purpose, I copied the relevant steps here, but please always check the JAX README page for up-to-date information.

```bash
# install jaxlib
PYTHON_VERSION=cp37  # alternatives: cp36, cp37, cp38
CUDA_VERSION=cuda100  # alternatives: cuda100, cuda101, cuda102, cuda110
PLATFORM=manylinux2010_x86_64  # alternatives: manylinux2010_x86_64
BASE_URL='https://storage.googleapis.com/jax-releases'
pip install --upgrade $BASE_URL/$CUDA_VERSION/jaxlib-0.1.50-$PYTHON_VERSION-none-$PLATFORM.whl

pip install --upgrade jax  # install jax
```

### A Note For Windows Users

JAX has no native Windows support yet, but can be installed on **CPU via the Windows Subsystem for Linux** (Windows 10 only, and make sure that the pip version is the latest). 

## Dependencies

    numpy
    scipy
    sklearn
    matplotlib
    jax
    jaxlib

## Data

The accompanying data for example notebooks are from:

Ran, Y., Huang, Z., et al. (2020). Type-specific dendritic integration in mouse retinal ganglion cells. Nat Commun 11, 2101. https://doi.org/10.1038/s41467-020-15867-9
 
## Reference

[1] Huang, Z., &  Berens, P. (2020). Efficient high dimensional receptive field inference using a flexible spline basis. Cosyne Abstracts 2020, Denver, CO. [[abstract]](./misc/Huang_et_al_Cosyne_abstract.pdf) [[poster]](./misc/Huang_et_al_Cosyne_poster.pdf)

[2] MacKay, D. J. (1994). Bayesian nonlinear modeling for the prediction competition. ASHRAE transactions, 100(2), 1053-1062.

[3] Sahani, M., & Linden, J. F. (2003). Evidence optimization techniques for estimating stimulus-response functions. In Advances in neural information processing systems (pp. 317-324).

[4] Park, M., & Pillow, J. W. (2011). Receptive field inference with localized priors. PLoS computational biology, 7(10), e1002219.

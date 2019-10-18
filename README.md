# RFEst

A Python 3 tool for neural receptive field (RF) estimation.

## Installation

RFEst uses [JAX](https://github.com/google/jax) for automatic differentiation and JIT compilation to GPU/CPU, so you need to install JAX first. 

### For Linux and MacOS users

To install CPU-only version, simply clone this repo into local directory and then run `pip install -e`, JAX and other dependencies will be installed automatically:

    git clone https://github.com/berenslab/RFEst
    pip install -e RFEst

To enable GPU support on **Linux**, you need to consult the [JAX installation guide](https://github.com/google/jax#pip-installation). For reference purpose, I copied the relevant steps here, but please always check the JAX README page for up-to-date information.

    # install jaxlib
    PYTHON_VERSION=cp37  # alternatives: cp27, cp35, cp36, cp37
    CUDA_VERSION=cuda92  # alternatives: cuda90, cuda92, cuda100, cuda101
    PLATFORM=linux_x86_64  # alternatives: linux_x86_64
    BASE_URL='https://storage.googleapis.com/jax-releases'
    pip install --upgrade $BASE_URL/$CUDA_VERSION/jaxlib-0.1.30-$PYTHON_VERSION-none-$PLATFORM.whl

    pip install --upgrade jax  # install jax
    
### For Windows Users

JAX doen't support Windows yet. However, if you are running Windows 10, you can install JAX within the [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10) (but only the relatively outdated versions of JAX and jaxlib, sadly).

    git clone https://github.com/berenslab/RFEst
    pip install -r RFEst/requirements_win.txt
    pip install -e RFEst
    
## Supported Methods

* Ridge Regression
* Automatic Relevance Determination (ARD) [1]
* Automatic Smoothness Determination (ASD) [2]
* Automatic Locality Determination (ALD) [3]

## Usage

Given a stimulus design matrix (X) and the corresponding response (y), a optimized RF is calculated with respect to the dimension of the RF `dims=(nT, nY, nX)` 

    ```python
    from rfest import ASD

    asd = ASD(X, y, dims=(5, 20, 15))
    asd.fit(initial_params=[1., 1., 2., 2., 2.], num_iters=300)
    ```

This package also comes with a simple linear gaussian data generator with three spatial filters ('gaussian', 'mexican_hat', 'gabor').

    from rfest import make_data

    ((X, y), (Xtest, Ytest), 
     w_true) = make_data(dims=(5, 20, 15), sigma=(1.5, 1.5),
                               n_samples=2000, nsevar=0.025, 
                               filter_type='gaussian', seed=2046)    

## Dependencies

    numpy
    scipy
    sklearn
    matplotlib
    jax
    jaxlib

## Reference

[1] Tipping, M. E. (2001). Sparse Bayesian learning and the relevance vector machine. Journal of machine learning research, 1(Jun), 211-244.

[2] Sahani, M., & Linden, J. F. (2003). Evidence optimization techniques for estimating stimulus-response functions. In Advances in neural information processing systems (pp. 317-324).

[3] Park, M., & Pillow, J. W. (2011). Receptive field inference with localized priors. PLoS computational biology, 7(10), e1002219.

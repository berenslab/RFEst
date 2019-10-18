# RFEst

A Python 3 tool for receptive field (RF) estimation using Empirical Bayes and automatic differentiation. 

## Installation

To install, clone this repo into local directory and then use `pip install -e`:

    git clone https://github.com/berenslab/RFEst
    pip install -e RFEst

## Supported Methods

* Ridge Regression
* Automatic Relevance Determination (ARD) [1]
* Automatic Smoothness Determination (ASD) [2]
* Automatic Locality Determination (ALD) [3]

## Usage

Given a stimulus design matrix (X) and the corresponding response (y), a optimized RF is calculated with respect to the dimension of the RF `dims=(nT, nY, nX)` 

    from rfest import ASD

    asd = ASD(X, y, dims=(5, 20, 15))
    asd.fit(initial_params=[1., 1., 2., 2., 2.], num_iters=300)

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
    jaxlib[^1]

## Reference

[1] Tipping, M. E. (2001). Sparse Bayesian learning and the relevance vector machine. Journal of machine learning research, 1(Jun), 211-244.

[2] Sahani, M., & Linden, J. F. (2003). Evidence optimization techniques for estimating stimulus-response functions. In Advances in neural information processing systems (pp. 317-324).

[3] Park, M., & Pillow, J. W. (2011). Receptive field inference with localized priors. PLoS computational biology, 7(10), e1002219.

## Note

[^1]: Jax doen't support Windows yet, but it might work on Windows Subsystem for Linux. Quoted from Jax's installation guide:

> We support installing or building jaxlib on Linux (Ubuntu 16.04 or later) and macOS (10.12 or later) platforms, but not yet Windows. We're not currently working on Windows support, but contributions are welcome (see #438). Some users have reported success with building a CPU-only jaxlib from source using the Windows Subsytem for Linux.

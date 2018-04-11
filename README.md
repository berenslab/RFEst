# RFEst

A Python 3 tool for receptive field (RF) estimation using Empirical Bayes and automatic differentiation. 

## Installation

To install, clone this repo into local directory and then use `pip install -e`:

    git clone https://github.com/berenslab/RFEst
    pip install -e RFEst

## Supported Methods

* Ridge Regression 
    * Ridge 
    * RidgeFixedPoint 
* Automatic Relevance Determination [1]
    * ARD 
* Automatic Smoothness Determination [2]
    * ASD
* Automatic Locality Determination [3]
    * ALDs 

**NOTED** In case of data with 3 dimensions, the current implementations does not optimized for temporal dimension due to limited amount of data in our own dataset. Instead, we lagged the response matrix, and treat each time-lagged response as a 2D mapping, and optimized the average loss of each time-lagged with a shared set of parameters.

## Usage

Given a stimulus matrix (X) and the corresponding response matrix (Y), a optimized RF is calculated with respect to the dimension of the RF rf_dims=(nX, nY, nT). The response matrix would be lagged by `nT` automatically if it presents. 

    from rfest import ASD

    asd = ASD(X, Y, rf_dims=(15, 20, 5))
    asd.fit(num_iters=300)

The optimized spatial and temporal RFs are stored in `self.sRF_opt` and `self.tRF_opt`.

This package also comes with a simple linear gaussian data generator with three spatial filters ('gaussian', 'mexican_hat', 'gabor').

    from rfest import make_data

    ((X, Y), (Xtest, Ytest), 
     w_true) = make_data(dims=(15, 20, 5), sigma=(1.5, 1.5),
                               n_samples=2000, nsevar=1, 
                               filter_type='mexican_hat', seed=2046)    

## Dependencies

    numpy
    scipy
    sklearn
    matplotlib
    autograd

## Reference

[1] Tipping, M. E. (2001). Sparse Bayesian learning and the relevance vector machine. Journal of machine learning research, 1(Jun), 211-244.

[2] Sahani, M., & Linden, J. F. (2003). Evidence optimization techniques for estimating stimulus-response functions. In Advances in neural information processing systems (pp. 317-324).

[3] Park, M., & Pillow, J. W. (2011). Receptive field inference with localized priors. PLoS computational biology, 7(10), e1002219.
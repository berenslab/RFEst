# Nonnegative Matrix Factorization (NMF) with spline

It has been shown that subunits from retinal ganglion cell receptive fields and other visual neurons in the cortex can be retrieved using clustering methods (e.g. semiNMF ([Liu, et al. 2017](https://www.nature.com/articles/s41467-017-00156-9)) and soft clustering ([Shah et al. 2020](https://elifesciences.org/articles/45743))). 

## Implemented methods

* k-means clustering
* NMF ([Lee & Seung, 2001](https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf))
* SemiNMF ([Ding et al, 2010](https://people.eecs.berkeley.edu/~jordan/papers/ding-li-jordan-pami.pdf))

and their corresponding spline-based versions, based on and modified from [Zdunek, et al, 2014](https://www.researchgate.net/profile/Rafal_Zdunek2/publication/274899525_B-Spline_Smoothing_of_Feature_Vectors_in_Nonnegative_Matrix_Factorization/links/553156010cf2f2a588ad4947/B-Spline-Smoothing-of-Feature-Vectors-in-Nonnegative-Matrix-Factorization.pdf).

## Usage

```python

nmf = NMF(V, k)
nmf.fit(num_iters=100, verbal=10)

```

NMF with spline-based left factor:

```python

nmf = NMF(V, k, build_L=True, dims_L=[5, 20, 15], df_L=7)
nmf.fit(num_iters=100, verbal=10)

```

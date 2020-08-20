# Matrix Factorization with spline

It has been shown that subunits from retinal ganglion cell receptive fields and other visual neurons in the cortex can be retrieved using clustering methods (e.g. semiNMF ([Liu, et al. 2017](https://www.nature.com/articles/s41467-017-00156-9)) and soft clustering ([Shah et al. 2020](https://elifesciences.org/articles/45743))). 

NMF can be easily combined with splines and produce smooth factors directly using multiplicative updates (see: [Zdunek, et al, 2014](https://www.researchgate.net/profile/Rafal_Zdunek2/publication/274899525_B-Spline_Smoothing_of_Feature_Vectors_in_Nonnegative_Matrix_Factorization/links/553156010cf2f2a588ad4947/B-Spline-Smoothing-of-Feature-Vectors-in-Nonnegative-Matrix-Factorization.pdf)), and k-means clustering can also be augmented with splines by simply replacing the original means with spline-approximated means. 

## Implemented methods

* k-means clustering
* NMF ([Lee & Seung, 2001](https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf))
* SemiNMF ([Ding et al, 2010](https://people.eecs.berkeley.edu/~jordan/papers/ding-li-jordan-pami.pdf))

and their corresponding spline-based versions.

## Usage

K-means clustering with spline-based cluster centroids:

```python

kms = KMenas(V, k, build_S=True, dims=[20, 20], df=11)
kms.fit(num_iters=100, verbose=10)

```

NMF with spline-based left factor:

```python

nmf = NMF(V, k, build_L=True, dims_L=[5, 20, 15], df_L=7)
nmf.fit(num_iters=100, verbose=10)

```
If spline-based right factor is prefered, than use flags `build_R`, `dims_R` and `df_R`. 

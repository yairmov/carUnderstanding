import numpy as np
from scipy.spatial.distance import cdist
from scipy import linalg
# import numba
from numba.decorators import autojit, jit

# @jit('f8[:,:](f8[:,:],f8[:,:],i8,f8)')
def LLC_encoding(B, X, knn=5, beta=3e-2):
  '''
  ========================================================================
  USAGE: coeff = LLC_coding(B,X,knn,beta)
  Approximated Locality-constraint Linear Coding

  Inputs
         B       -M x d codebook, M entries in a d-dim space
         X       -N x d matrix, N data points in a d-dim space
         knn     -number of nearest neighboring
         beta  -regulerization to improve condition

   Outputs
         coeff   -N x M matrix, each row is a code for corresponding X

  original MATLAB code: Jinjun Wang, march 19, 2010
  Python code: Yair Movshovitz-Attias, Marcg 4, 2014
  ========================================================================
  '''
  nframe = X.shape[0]
  nbase  = B.shape[0]

  # find k nearest neighbors
  D = cdist(X, B, 'sqeuclidean')
  sort_idx = D.argsort(axis=1)
  IDX = sort_idx[:, :knn]


  # llc approximation coding
  II = np.eye(knn, knn)
  coeff = np.zeros([nframe, nbase])

#   from pprint import pprint
  
  for ii in np.arange(nframe):
#   for ii in [1423]:
#     print ii
    idx = IDX[ii,:]
#     pprint(('B_idx', B[idx,:10]))
#     pprint(('X_ii', X[ii,:10]))
    z = B[idx,:] - X[ii,:]     # shift ith pt to origin
    C = z.dot(z.T)             # local covariance
    C = C + II*beta*np.trace(C)   # regularlization (K>D)
    w = linalg.lstsq(C, np.ones([knn, 1]))[0]
    w = w / w.sum()
    coeff[ii, idx] = w.T.reshape(w.size,)


  return coeff







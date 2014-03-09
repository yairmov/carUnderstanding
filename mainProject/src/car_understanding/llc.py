import numpy as np
from scipy.spatial.distance import cdist
from scipy import linalg


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

  from pprint import pprint
  np.savetxt("tmp.csv", B[:,:10], delimiter=",  ", fmt='%.3g')
  
#   for ii in np.arange(nframe):
  for ii in [146]:
    print ii
    idx = IDX[ii,:]
    z = B[idx,:] - X[ii,:]     # shift ith pt to origin
    pprint( ("B_idx", B[idx,:10]))
    pprint( ("X_ii", X[ii,:10]))
    C = z.dot(z.T)             # local covariance
    C = C + II*beta*np.trace(C)   # regularlization (K>D)
    w = linalg.lstsq(C, np.ones([knn, 1]))[0]
    print w, w.sum()
    w = w / w.sum()
    # print "w.shape: {}, coeff[ii, idx].shape: {}".format(w.T.reshape(w.size,).shape, coeff[ii, idx].shape)
    # return
    coeff[ii, idx] = w.T.reshape(w.size,)


  return coeff







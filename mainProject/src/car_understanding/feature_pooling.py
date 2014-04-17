'''
Created on Apr 17, 2014

@author: ymovshov
'''

import numpy as np


def contains(xmin, ymin, xmax, ymax, points):
  '''
  For each point in points checks if it is in the box.
  box can be any tuple like container.
  points should be a numpy array.
  box = (xmin, ymin, xmax, ymax)
  points = N-by-2, where each line is x, y
  
  returns a boolean numpy array of length N.
  '''
  
#   assert type(points) == np.ndarray, 'points should be numpy array'
  
  x = points[:,0]
  y = points[:,1]
  contains_x = (x >= xmin) * (x <= xmax)
  contains_y = (y >= ymin) * (y <= ymax)
  
  return np.logical_and(contains_x, contains_y)



class SpatialPooler(object):
  '''
  A class for spatial pooling of features.
  '''


  def __init__(self, pooling_box):
    '''
    pooling_box = (xmin, ymin, xmax, ymax) where each value is in [0,1]
    indicating the desired area in the image.
    E.g. (0, 0, 0.5, 1) will pool features from the left half of the image.  
    '''
    self.pooling_box = np.array(pooling_box)
    
  
  @staticmethod
  def to_pool(locations, features, pooling_box):
    M = np.max(locations, axis = 0)
    
    xmin = M[0] * pooling_box[0]
    ymin = M[1] * pooling_box[1]
    xmax = M[0] * pooling_box[2]
    ymax = M[1] * pooling_box[3]
  
    to_pool = contains(xmin, ymin, xmax, ymax, locations) 
    
    return features[to_pool, :]
  
  def features_to_pool(self, locations, features):
    '''
    locations - numpy array of size Nx2 where each row is the x,y location
    in the image.
    features - numpy array of size Nxd where each row is a d dimentional feature
    vector to be pooled.
    
    returns the subset of features that should be pooled. 
    '''
    
    assert type(locations) == np.ndarray, ('locations must be an ' + 
                                           'N-by-2 numpy.ndarray' +  
                                           'where each row is the x,y '+
                                           'location in the image')
    
    return self.to_pool(locations, features, self.pooling_box)
    
    
    
    
      
        
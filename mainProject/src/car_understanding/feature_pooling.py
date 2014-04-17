'''
Created on Apr 17, 2014

@author: ymovshov
'''

import numpy as np

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
    self.pooling_box = pooling_box
    
  
  @staticmethod
  def contains(box, points):
    '''
    For each point in points checks if it is in the box.
    box can be any tuple like container.
    points should be a numpy array.
    box = (xmin, ymin, xmax, ymax)
    points = N-by-2, where each line is x, y
    
    returns a boolean numpy array of length N.
    '''
    
    assert type(points) == np.ndarray, 'points should be numpy array'
    
    contains_x = np.logical_and(points[:,0] >= box[0], points[:,0] <= box[2])
    contains_y = np.logical_and(points[:,1] >= box[1], points[:,1] <= box[3])
    
    return np.logical_and(contains_x, contains_y)
    
    
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
    
    xM, yM = tuple(locations.max(axis = 0))
    xm, ym = tuple(locations.min(axis = 0))
    
    xmin = xm * self.pooling[0]
    ymin = ym * self.pooling[1]
    xmax = xM * self.pooling[2]
    ymax = yM * self.pooling[3]
    
    to_pool = self.contains((xmin, ymin, xmax, ymax), locations)
    
    return features[to_pool, :]
    
      
        
'''
Created on Mar 3, 2014

@author: ymovshov
'''

import cv2 as cv
from skimage import feature
import numpy as np

class LbpFeatureExtractor(object):
  '''
  Local Binary Pattern Histogram feature extractor class
  '''


  def __init__(self, config):
    '''
    Constructor
    '''
    self.config = config
    
  def extract_feature(self, img):
    if (type(img) == str):
      img = cv.imread(img)
    if img.ndim > 1:
      img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
      
    # Extract LBP
    P = self.config.LPB.P
    R = self.config.LPB.R
    
    lbp = feature.local_binary_pattern(img, P, R, method='default')
    return np.bincount(lbp.reshape([lbp.size,]).astype(np.int32))
      
    
'''
Created on Mar 3, 2014

@author: ymovshov
'''

import cv2 as cv
from skimage import feature

class HogFeatureExtractor(object):
  '''
  A class that implements HOG descriptor extraction from an image
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
      
    # extract HOG
    hog = feature.hog(img, 
                      orientations=self.config.HOG.orientations, 
                      normalise=True, 
                      pixels_per_cell=self.config.HOG.pixels_per_cell)
    return hog
          
        
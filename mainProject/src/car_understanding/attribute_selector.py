# encoding: utf-8
'''
car_understanding.attribute_selector -- A convinience class for selecting attributes 

@author:     Yair Movshovitz-Attias

@copyright:  2014 Yair Movshovitz-Attias. All rights reserved.

@contact:    yair@cs.cmu.edu
'''

from sklearn.externals.joblib import Parallel, delayed, Memory, dump, load
import sklearn as sk
import numpy as np
import os

import Bow
from docutils.languages.af import labels

class AttributeSelector:
  """A class that provides easy selection of images based on a n attribute."""
  
  def __init__(self, config, class_meta):
    """ Ctor.
    
    Args:
          config     - Config object created by configuration.get_config()
          class_meta - Pandas table defining the class metadata (see fgcomp_dataset_utils)
    """
    self.config  = config
    self.class_meta = class_meta.copy()
    
    
    self.process_dataset()
    
  def class_ids_for_attribute(self, attrib_name):
    '''
    Return all class ids that have the attribuet attrib_name
    '''
    class_ids = []
    attrib_name = str.lower(attrib_name)
    for ii in range(len(self.class_meta)):
      class_name = str.lower(class_meta['class_name'].iloc[ii])
      if has_attribute_by_name(class_name, attrib_name):
        class_ids.append(class_meta['class_index'].iloc[ii])
  
    return class_ids
  
  
  def has_list_attributes_by_index(self, class_id, attrib_names):
    '''
    Does class:class_index has the attribute list atrrib_names?
    '''
    return has_list_attributes_by_name(self.class_meta[class_id].class_name, 
                                 attrib_names)
    
  def has_attribute_by_index(self, class_id, attrib_name):
    '''
    Does class:class_index has the attribute atrrib_name?
    '''
    return has_attribute_by_name(self.class_meta[class_id].class_name, 
                                 attrib_name)
  
  # Static" functions
  # -----------------
  @staticmethod
  def has_attribute_by_name(class_name, attrib_name):
    '''
    Does class:class_name has the attribute atrrib_name?
    uses the class name as an indicator.
    i.e. the class "Acura RL Sedan 2012" has the attributes:
    [Acura, Sedan, 2012], but not the attribute Ford. 
    '''
    return str.find(class_name, attrib_name) != -1
  
  @staticmethod  
  def has_list_attributes_by_name(class_name, attrib_names):
    '''
    Does class:class_name has all attributes in attrib_names?
    '''
    has_attrib = [has_attribute_by_name(class_name, a_name) for
                  a_name in attrib_names]
    
    return np.array(has_attrib).all()
    
    
      
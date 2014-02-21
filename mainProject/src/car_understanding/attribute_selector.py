# encoding: utf-8
'''
car_understanding.attribute_selector -- A convinience class for selecting attributes 

@author:     Yair Movshovitz-Attias

@copyright:  2014 Yair Movshovitz-Attias. All rights reserved.

@contact:    yair@cs.cmu.edu
'''

import numpy as np
import pandas as pd

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
    
    
    
    
  def create_attrib_meta(self, attrib_names):
    classes = self.class_meta
    attrib_meta = pd.DataFrame(np.zeros([classes.shape[0], len(attrib_names)],
                                        dtype=int), 
                               columns = attrib_names,
                               index = classes.index)
    for class_index in attrib_meta.index:
      class_name = classes.class_name[class_index]
      for name in attrib_meta.columns:
        print "class_name: " + class_name
        print "name: " + name
        attrib_meta.ix[class_index, name] = \
        AttributeSelector.has_attribute_by_name(class_name, name)
      
    return attrib_meta
    
  def class_ids_for_attribute(self, attrib_name):
    '''
    Return all class ids that have the attribuet attrib_name
    '''
    class_ids = []
    attrib_name = str.lower(attrib_name)
    for ii in range(len(self.class_meta)):
      class_name = str.lower(self.class_meta['class_name'].iloc[ii])
      if AttributeSelector.has_attribute_by_name(class_name, attrib_name):
        class_ids.append(self.class_meta['class_index'].iloc[ii])
  
    return class_ids
  
  
  def has_list_attributes_by_index(self, class_id, attrib_names):
    '''
    Does class:class_index has the attribute list atrrib_names?
    '''
    return AttributeSelector.has_list_attributes_by_name(
                                      self.class_meta.class_name[class_id], 
                                      attrib_names)
    
  def has_attribute_by_index(self, class_index, attrib_name):
    '''
    Does class:class_index has the attribute atrrib_name?
    '''
    return AttributeSelector.has_attribute_by_name(
                                 self.class_meta.class_name[class_index], 
                                 attrib_name)
    
  def prune_attributes(self, class_index, attrib_names):
    '''
    From the list of attributes, return only the ones that this class has.
    '''
    has_attrib = [self.has_attribute_by_index(class_index, n) for
                n in attrib_names]

    new_attrib_names = np.array(attrib_names)
    return new_attrib_names[np.array(has_attrib)]

  
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
    return str.find(str.lower(class_name), str.lower(attrib_name)) != -1
  
  @staticmethod  
  def has_list_attributes_by_name(class_name, attrib_names):
    '''
    Does class:class_name has all attributes in attrib_names?
    '''
    has_attrib = [AttributeSelector.has_attribute_by_name(class_name, a_name) for
                  a_name in attrib_names]
    
    return np.array(has_attrib).all()
  
  




      
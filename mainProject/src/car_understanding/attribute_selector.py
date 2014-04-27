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
  
  def __init__(self, config, class_meta, attrib_meta):
    """ Ctor.
    
    Args:
          config     - Config object created by configuration.get_config()
          class_meta - Pandas table defining the class metadata (see fgcomp_dataset_utils)
    """
    self.config  = config
    self.class_meta = class_meta.copy()
    self.attrib_meta = attrib_meta.copy()
    self.attrib_matrix = \
      self.create_attrib_matrix(np.concatenate([np.unique(attrib_meta[x]) 
                                                for x in attrib_meta.columns]))
      #         self.create_attrib_matrix(['uk', 'italy', 'germany'])
    
    
    
  def create_attrib_matrix(self, attrib_names):
    classes = self.class_meta
    attrib_matrix = pd.DataFrame(np.zeros([classes.shape[0], len(attrib_names)],
                                        dtype=bool), 
                               columns = attrib_names,
                               index = classes.index)
    
    for name in attrib_names:
      print name
      attrib_matrix[name] = np.sum(self.attrib_meta == name, axis=1)  > 0
          
    return attrib_matrix
    
  def class_ids_for_attribute(self, attrib_name):
    '''
    Return all class ids that have the attribuet attrib_name
    '''
    attrib_mask = self.attrib_matrix[attrib_name]
  
    return np.array(self.attrib_matrix.index[attrib_mask])
  
  
  def has_list_attributes_by_index(self, class_index, attrib_names):
    '''
    Does class:class_index has the attribute list atrrib_names?
    '''
    return list(self.attrib_matrix.loc[class_index][attrib_names])
    
  def has_attribute_by_index(self, class_index, attrib_name):
    '''
    Does class:class_index has the attribute atrrib_name?
    '''
    return self.attrib_matrix.loc[class_index][attrib_name].iloc[0]
    
  def prune_attributes(self, class_index, attrib_names):
    '''
    From the list of attributes, return only the ones that this class has.
    '''
    has_attrib = self.has_list_attributes_by_index(class_index, attrib_names)

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
    raise Exception('Depricated: Should not be calling this!!!!!')
    return str.find(str.lower(class_name), str.lower(attrib_name)) != -1
  
  @staticmethod  
  def has_list_attributes_by_name(class_name, attrib_names):
    '''
    Does class:class_name has all attributes in attrib_names?
    '''
    raise Exception('Depricated: Should not be calling this!!!!!')
    has_attrib = [AttributeSelector.has_attribute_by_name(class_name, a_name) for
                  a_name in attrib_names]
    
    return np.array(has_attrib).all()
  
  




      
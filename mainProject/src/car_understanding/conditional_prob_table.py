'''
Created on Apr 8, 2014

@author: ymovshov
'''

import pandas as pd
import numpy as np

class CPT(object):
  '''
  A class implementing a container for a Conditional Probability Table. Useful for when the
  number of rows in the table is huge as it saves only seen values.
  Note that the class does not enforce that the values in it are actual
  probabilities, as it is sometimes convenient to build the CPT iterativly by
  counting.  
  '''


  def __init__(self, smooth_value=0, default_true_value=0.5, name=''):
    '''
    Constructor
    smooth_value: 'fake' counts to add to values. Used to give non zero probabilities
    to unseen values.
    '''
    self.index = set()
    columns=['True', 'False']
    self.smooth_value = smooth_value
    self.cpt = pd.DataFrame(columns=columns)
    self.is_normalized = False
    self.default_true_value = default_true_value
    
  
  def has_row(self, row_ind):
    row_ind = str(row_ind)
    return row_ind in self.index
    
  def create_row(self, row_ind):
    row_ind = str(row_ind)
    if not self.has_row(row_ind):
      self.cpt.loc[row_ind] = pd.Series(data=np.array([0,0]), 
                                        index=['True', 'False'])
      self.index.add(row_ind)
  
  def set_value(self, row_ind, column, value):
    row_ind = str(row_ind)
    if not self.has_row(row_ind):
      raise LookupError()
    
    self.cpt.ix[row_ind, column] = value
  
  def add_count(self, row_ind, column):
    if self.is_normalized:
      raise StandardError('CPT has been normalized. No more values can be added.')
    
    row_ind = str(row_ind)
    if not self.has_row(row_ind):
      raise LookupError()
    
    self.cpt.ix[row_ind, column] += 1
    
  def get_value(self, row_ind, column):
    row_ind = str(row_ind)
    if not self.has_row(row_ind):
      print '{{' + row_ind + '}}'
      raise LookupError()
      if self.is_normalized:
        return self.default_true_value if column else 1 - self.default_true_value
      else:
        return 0
    
    return self.cpt.ix[row_ind, column]
  
  def __str__(self):
    return self.cpt.__str__()
    
  def normalize_rows(self):
    '''
    Normalizes each row of the table to make it into a probability function. 
    '''
    # Add smoothing factor
    self.cpt += self.smooth_value
    
    # Normalize
    self.cpt = self.cpt.divide(self.cpt.sum(axis=1), axis='index')
    self.is_normalized = True
    
    
    
#   def __getitem__(self, row, col):
#     if row in self.index:
#       return self.cpt.ix[row, col]
    
        
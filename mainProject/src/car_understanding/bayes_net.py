# encoding: utf-8
'''
car_understanding.bayes_net -- 
Defines a bayes net PGM that fuses information from attribute classifiers to
predict the car class. 

@author:     Yair Movshovitz-Attias

@copyright:  2014 Yair Movshovitz-Attias. All rights reserved.

@contact:    yair@cs.cmu.edu
'''

# from bayesian.bbn import build_bbn
import numpy as np
import os
import itertools
from clint.textui import progress
import pandas as pd

import Bow
from attribute_selector import AttributeSelector


class BayesNet:
  """A Bayes net model."""
  
  def __init__(self, config, train_annos, class_meta, attrib_clfs, desc=""):
    """ Ctor.
    
    Args:
          config      - Config object created by configuration.get_config()
          train_annos - Pandas table defining the training data (see fgcomp_dataset_utils)
          class_meta  - (see fgcomp_dataset_utils)
          attrib_clfs - A list of AttributeClassifiers
          desc        - Longer string description of attribute (optional)
          
    """
    self.config       = config.copy()
    self.train_annos  = train_annos.copy()
    self.class_meta   = class_meta.copy()
    self.attrib_clfs  = attrib_clfs 
    self.desc         = desc
    self.clf_res      = None
    self.CPT          = {}
    self.clf_names    = [self.attrib_clfs[ii].name for 
                                  ii in range(len(self.attrib_clfs))]
    
  
  
  def create_attrib_res_on_images(self):
    '''
    Calculates the predicion of all attribute classifiers on training images.
    This table can be used to caclulate all the Conditional Probability tables
    for the Bayes Net.
    '''
    # Define some conviniece pointers 
    train_annos = self.train_annos
    config = self.config
    attrib_classifiers = self.attrib_clfs
    
    print "Load image Bow histograms from disk"
    features = np.empty(shape=[len(train_annos), config.SIFT.BoW.num_clusters])
    for ii in progress.bar(range(len(train_annos))):
      img_name = train_annos.iloc[ii]['basename']
      img_name = os.path.splitext(img_name)[0]
      hist_filename = os.path.join(config.SIFT.BoW.hist_dir,
                                   img_name) + '_hist.dat'
      hist = Bow.load(hist_filename) # hist[0] = values, hist[1] = bin edges
      features[ii, :] = hist[0]
  
    print "Apply attribute classifiers on images"
    res = {}
#     for attrib_clf in attrib_classifiers:
    for ii in progress.bar(range(len(attrib_classifiers))):
      attrib_clf = attrib_classifiers[ii]
      curr_res = attrib_clf.clf.decision_function(features)
#       curr_res = attrib_clf.clf.predict(features)
      res[attrib_clf.name] = curr_res.reshape(len(curr_res))
  
    res = pd.DataFrame(data=res, index=train_annos.index)
  
  
    res = pd.concat([res, train_annos.ix[:, ['class_index']]], axis=1)
    return res
  
  
  def cpt_for_attrib(self, attrib_name, attrib_selector):
    # Create all tuples of True/False classifier score
    rows = list(itertools.product(*[(1, 0) for 
                                    ii in range(len(names))]))
    
    clf_names = self.clf_names
    clf_res = self.clf_res
    
    attrib_class_ids = attrib_selector.class_ids_for_attribute(attrib_name)
    # intersect attrib_class_ids with clf_res.class_index
    attrib_class_ids = \
    [val for val in attrib_class_ids if val in list(clf_res.class_index)]
    
    clf_res_descrete = clf_res.copy()
    clf_res_descrete.ix[:, clf_names] = \
                  clf_res.ix[:, clf_names] > self.config.attribute.high_thresh
    
    cpt = pd.DataFrame(np.ones([len(rows), 2]), 
                       index=rows, columns=['True', 'False'])
    
    for ii, row in enumerate(rows):
      tmp = clf_res_descrete.copy()
      for jj, name in enumerate(clf_names):
        tmp = tmp[(tmp[name] == row[jj])]
      
      cpt.iloc[jj][0] += tmp.class_index.isin(attrib_class_ids).sum()
      cpt.iloc[jj][1] += tmp.shape[0] - cpt.iloc[jj][0]
        
    # normalize all the rows, to create a probability function
    cpt = cpt.divide(cpt.sum(axis=1), axis='index')
    return cpt
    
  
  
  def cpt_for_class(self, class_index, attribute_selector):    
    # figure out which attributes does this class have
    attrib_names = self.clf_names
    has_attrib = [attribute_selector.has_attribute_by_index(class_index, n) for
                n in attrib_names]
    
    attrib_names = np.array(attrib_names)
    attrib_names = attrib_names[np.array(has_attrib)]
    
    # Create all tuples of True/False for indicator variable
    rows = list(itertools.product(*[('1', '0') for 
                                    ii in range(len(attrib_names))]))
    
    cpt = pd.DataFrame(np.zeros([len(rows), 2]), 
                       index=rows, columns=['True', 'False'])
    
    # All rows except the one that has ALL the attributes should be zero.
    # The one in which all attribs are true, should be the proportion of this class
    # with respect to all classes that have all these attributes.
    cpt['False'] = 1
    
    num_classes_with_attrib = 0
    for k, class_index in self.class_meta.iterkv():
       if attribute_selector.has_list_attributes_by_index(class_index, 
                                                          attrib_names):
         num_classes_with_attrib += 1
         
    cpt.ix[[tuple(*ones(shape=[1, len(attrib_names)], 
                        dtype=str))], 'True'] = 1 / num_classes_with_attrib
    
    
    
      
    
  def init_CPT(self):
    '''
    Initialize the Conditional Probability Tables for all the nodes
    in the net.
    '''
    if self.clf_res == None:
      self.clf_res = self.create_attrib_res_on_images()
    
    attrib_selector = AttributeSelector(self.config, 
                                        self.class_meta)
    attrib_names = self.clf_names
    class_inds  = self.train_annos.class_index.unique()
    
    # P(class | attributes)
    #----------------------
    for class_index in class_inds:
      self.CPT['p({} | atrr)'.format(class_index)] = \
        self.cpt_for_class(class_index, attribute_selector)
    
    
    # P(attribute | res of attrib classifiers)
    #-----------------------------------------
    for attrib_name in names:
      self.CPT['p({}|theta)'.format(attrib_name)] = \
        self.cpt_for_attrib(attrib_name, attrib_selector)
    
    
    
    
    
    
    
    
    
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
import pymc as mc

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
    
    # sort by attrib name (keep the attributs sorted at all times!)
    inds = np.argsort(self.clf_names)
    self.clf_names = list(np.array(self.clf_names)[inds])
    self.attrib_clfs = list(np.array(self.attrib_clfs)[inds]) 
     
    
  
  
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
    clf_names = np.array(self.clf_names)
    clf_res = self.clf_res
    
    
    attrib_class_ids = attrib_selector.class_ids_for_attribute(attrib_name)
    # intersect attrib_class_ids with clf_res.class_index
    attrib_class_ids = \
    [val for val in attrib_class_ids if val in list(clf_res.class_index)]
    
    clf_res_descrete = clf_res.copy()
    clf_res_descrete.ix[:, clf_names] = \
                  clf_res.ix[:, clf_names] > self.config.attribute.high_thresh
    
    # Create all tuples of True/False classifier score
    rows = list(itertools.product(*[(1, 0) for 
                                    ii in range(len(clf_names))]))
    cpt = pd.DataFrame(np.ones([len(rows), 2]), 
                       index=rows, columns=['True', 'False'])
    
    for ii in range(clf_res_descrete.shape[0]):
      cc = clf_res_descrete.iloc[ii]
      print clf_names
      row = tuple(cc[clf_names])
      print row
      has_attrib = cc['class_index'] in attrib_class_ids
      cpt.ix[row, str(has_attrib)] += 1
    
    print cpt
    return
    
    
#     for ii, row in enumerate(rows):
#       tmp = clf_res_descrete.copy()
#       for jj, name in enumerate(clf_names):
#         tmp = tmp[(tmp[name] == row[jj])]
#       
#       cpt.iloc[ii][0] += tmp.class_index.isin(attrib_class_ids).sum()
#       cpt.iloc[ii][1] += tmp.shape[0] - cpt.iloc[ii][0]
        
    # normalize all the rows, to create a probability function
    cpt = cpt.divide(cpt.sum(axis=1), axis='index')
    return cpt
    
  
  
  def cpt_for_class(self, class_index, attrib_list, attribute_selector):    

    # Create all tuples of True/False for indicator variable
    rows = list(itertools.product(*[(1, 0) for 
                                    ii in range(len(attrib_list))]))
    
    cpt = pd.DataFrame(np.zeros([len(rows), 2]), 
                       index=rows, columns=['True', 'False'])
    
    # All rows except the one that has ALL the attributes should be zero.
    # The one in which all attribs are true, should be the proportion of this class
    # with respect to all classes that have all these attributes.
    cpt['False'] = 1
    
    num_classes_with_attrib = 0
    for k, class_index in self.class_meta.iterkv():
      if attribute_selector.has_list_attributes_by_index(class_index, 
                                                      attrib_list):
        num_classes_with_attrib += 1
         
    cpt.ix[[tuple(*np.ones(shape=[1, len(attrib_list)], 
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
    
    
    self.cpt_for_attrib('suv', attrib_selector)
    return
    
    # P(attribute | res of attrib classifiers)
    #-----------------------------------------
    for attrib_name in attrib_names:
      self.CPT['p({}|theta)'.format(attrib_name)] = \
        self.cpt_for_attrib(attrib_name, attrib_selector)
    
    # P(class | attributes)
    #----------------------
    for class_index in class_inds:
      # figure out which attributes does this class have
      attrib_list = attrib_selector.prune_attributes(class_index, attrib_names)
      
      class_key = 'p({} | {})'.format(class_index, 
                                   BayesNet.format_attribute_list(attrib_list))
      
      self.CPT[class_key] = self.cpt_for_class(class_index, 
                                               attrib_list, 
                                               attrib_selector)
    
    
        
  
  def predict_one(self, clf_res):
    # building model
    # first start with observed variables - the results of all the classifiers 
    # on the image
    clf_res_descrete = clf_res.copy()
    clf_res_descrete[self.clf_names] = \
        clf_res[self.clf_names] > self.config.attribute.high_thresh
    
    # the actual distrobution used is not important as these are 
    # *observed* variables. We set the value to be the result of the classifiers              
    theta = mc.DiscreteUniform('theta', 0, 1,
                               size=len(self.clf_names),
                               observed=True,
                               value=clf_res_descrete[self.clf_names])
    
    # The hidden layer. Each attriute is connected to all attribute classifiers
    # as its parents.
    attrib_names = self.clf_names
    attrib_bnet_nodes = []
    for attrib_name in attrib_names:
      identifier = 'p({}|theta)'.format(attrib_name)
      cpt = self.CPT[identifier]
      p_function = mc.Lambda(identifier, 
                             self.prob_function_builder_for_mid_layer(cpt, 
                                                                      theta))
      attrib_bnet_nodes.append(mc.Bernoull(attrib_name, p_function))
      
   
    class_inds = self.train_annos.class_index.unique() 
    for class_index in class_inds:
      attrib_list = attrib_selector.prune_attributes(class_index, attrib_names)
      class_key = 'p({} | {})'.format(class_index, 
                                   BayesNet.format_attribute_list(attrib_list))
      cpt = self.CPT[class_key]
      
      
  
  
  def prob_function_builder_for_class_layer(self, cpt, attrib_values):
    return lambda attrib_values=attrib_values: float(cpt.ix[[tuple(attrib_values)],
                                                        'True'])
  
  def prob_function_builder_for_mid_layer(self, cpt, theta):    
    return lambda theta=theta: float(cpt.ix[[tuple(theta.value)],'True'])
       
  
  
  @staticmethod      
  def format_attribute_list(attrib_list):
    '''
    Returns a string representation of the list of attributes (comma seperated).
    The attributes are sorted such that the representation is dependant only on
    the contents of attrib_list, not the order of it.
    '''
    l = list(np.sort(attrib_list))
    return ','.join(l)
    
    
    
    
    
    
    
    
    
    
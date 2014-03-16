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
# from sklearn.externals.joblib import Parallel, delayed
import sys

import Bow as Bow
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
    self.class_inds   = self.class_meta.class_index.unique()
    
    self.attrib_clfs  = attrib_clfs
    for ii in range(len(self.attrib_clfs)):
      attrib_clfs[ii].name = str.lower(attrib_clfs[ii].name)
       
    self.desc         = desc
    self.clf_res      = None
    self.CPT          = {}
    self.clf_names    = [self.attrib_clfs[ii].name for 
                                  ii in range(len(self.attrib_clfs))]
    
    # sort by attrib name (keep the attributs sorted at all times!)
    inds = np.argsort(self.clf_names)
    self.clf_names = list(np.array(self.clf_names)[inds])
    self.attrib_clfs = list(np.array(self.attrib_clfs)[inds]) 
    
    self.attrib_selector = AttributeSelector(self.config, 
                                        self.class_meta)
     
    
  
  
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
      features[ii, :] = hist
  
    print "Apply attribute classifiers on images"
    res = {}
#     for attrib_clf in attrib_classifiers:
    for ii in progress.bar(range(len(attrib_classifiers))):
      attrib_clf = attrib_classifiers[ii]
      curr_res = attrib_clf.decision_function(features,
                                              use_prob=config.attribute.use_prob)
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
    cpt = pd.DataFrame(np.ones([len(rows), 2], dtype=np.float64), 
                       index=rows, columns=['True', 'False'])
    
    
    for ii in range(clf_res_descrete.shape[0]):
      cc = clf_res_descrete.iloc[ii]
      row = tuple(cc[clf_names])
      has_attrib = cc['class_index'] in attrib_class_ids
      cpt.ix[row, str(has_attrib)] += 1
    
    
    # normalize all the rows, to create a probability function
    cpt = cpt.divide(cpt.sum(axis=1), axis='index')
    print "CPT for attrib: {}".format(attrib_name)
    print "----------------------------"
    print cpt
    return cpt
    
  
  
  def cpt_for_class(self, class_index, attrib_list, attribute_selector):    

    # Create all tuples of True/False for indicator variable
    rows = list(itertools.product(*[(1, 0) for 
                                    ii in range(len(attrib_list))]))
    
    min_prob = 1e-2
    cpt = pd.DataFrame(min_prob * np.ones([len(rows), 2], dtype=np.float64), 
                       index=rows, columns=['True', 'False'])
    
    # All rows except the one that has ALL the attributes should have p(true)=min_prob.
    # The one in which all attribs are true, should be the proportion of this class
    # with respect to all classes that have all these attributes.
    cpt['False'] = 1-min_prob
    
    num_classes_with_attrib = 0
    for cind in self.class_meta.index:
      if attribute_selector.has_list_attributes_by_index(cind, 
                                                      attrib_list):
        num_classes_with_attrib += 1
         
         
    print "attrib_list: {}".format(attrib_list)
    print "num_classes_with_attribs: {}".format(num_classes_with_attrib)
    p = 1.0 / num_classes_with_attrib
    cpt.ix[[tuple(*np.ones(shape=[1, len(attrib_list)], 
                        dtype=int))], 'True'] = p
    cpt.ix[[tuple(*np.ones(shape=[1, len(attrib_list)], 
                        dtype=int))], 'False'] = 1 - p
                        
    print "CPT for class: {}".format(self.class_meta.class_name[class_index])
    print "---------------------------------"
    print cpt                        
    return cpt
    
      
      
    
  def init_CPT(self):
    '''
    Initialize the Conditional Probability Tables for all the nodes
    in the net.
    '''
    if self.clf_res == None:
      self.clf_res = self.create_attrib_res_on_images()
    
    attrib_selector = self.attrib_selector
    
    attrib_names = self.clf_names
    print attrib_names
    class_inds  = self.class_inds
    
    
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
    
    print self.class_meta
        
  
  '''
  fake: will use the ground truth attribute values for the middle 
  layer, to check what is the best we can hope for.
  '''
  def predict(self, use_gt=False):
    class_inds = self.class_inds
    class_prob = pd.DataFrame(np.zeros([self.clf_res.shape[0], 
                                        len(class_inds)]),
                              index=self.train_annos.index, 
                              columns=class_inds)
    
    attrib_names = self.clf_names
    attrib_prob = pd.DataFrame(np.zeros([self.clf_res.shape[0], 
                                         len(attrib_names)]),
                              index=self.train_annos.index, 
                              columns=attrib_names)
        
    clf_res_descrete = self.clf_res.copy()
    clf_res_descrete[self.clf_names] = \
        self.clf_res[self.clf_names] > self.config.attribute.high_thresh
        
    # using ground truth    
    if use_gt:
      attrib_meta = self.attrib_selector.create_attrib_meta(attrib_names)
        
    # Create cache for results - we only have 2^num_attrib options.
    class_prob_cache = {}
    attrrib_prob_cache = {}
    
    for ii in range(clf_res_descrete.shape[0]):
      print "=================={}========================".format(ii)
      if use_gt:
        desc = attrib_meta.loc[clf_res_descrete.iloc[ii]['class_index']]
        key = np.array(desc) 
      else:
        desc = clf_res_descrete.iloc[ii]
        key = np.array(desc[attrib_names])
      print "key: {}".format(key)
      key = key.tostring()
      if (not class_prob_cache.has_key(key)):
        print "Never got this key before, computing...."
        (class_prob_ii, attrib_prob_ii) = self.predict_one(desc, use_gt)
        class_prob_cache[key] = class_prob_ii
        attrrib_prob_cache[key] = attrib_prob_ii
      
      class_prob.iloc[ii] = class_prob_cache[key]
      attrib_prob.iloc[ii] = attrrib_prob_cache[key]
      
    return (class_prob, attrib_prob)
      
  def predict_one(self, clf_res_descrete, use_gt=False):
    # building model
    # first start with observed variables - the results of all the classifiers 
    # on the image
        
    # the actual distrobution used is not important as these are 
    # *observed* variables. We set the value to be the result of the classifiers              
    theta = mc.DiscreteUniform('theta', 0, 1,
                               size=len(self.clf_names),
                               observed=True,
                               value=clf_res_descrete[self.clf_names])
    
    
    
    # The hidden layer. Each attriute is connected to all attribute classifiers
    # as its parents. If we are using the ground truth, then we set this layer to
    # also be observed.
    attrib_names = self.clf_names
    attrib_bnet_nodes = {}
    if use_gt:
      for attrib_name in attrib_names:
        print "predict one: ", attrib_name
        rv = mc.DiscreteUniform(attrib_name, 
                                observed=True, 
                                value=clf_res_descrete[attrib_name])
        attrib_bnet_nodes[attrib_name] = rv
    else:
      for attrib_name in attrib_names:
        identifier = 'p({}|theta)'.format(attrib_name)
        cpt = self.CPT[identifier]
        p_function = mc.Lambda(identifier, 
                               self.prob_function_builder_for_mid_layer(cpt, 
                                                                        theta))
        attrib_bnet_nodes[attrib_name] = mc.Bernoulli(attrib_name, p_function)
      
     
    # The top layer Each class is connected to the attributes it has 
    class_inds = self.class_inds
    attrib_selector = self.attrib_selector
    
    class_bnet_nodes = {}
    for class_index in class_inds:
      attrib_name_list = attrib_selector.prune_attributes(class_index, attrib_names)
      class_key = 'p({} | {})'.format(class_index, 
                                   BayesNet.format_attribute_list(attrib_name_list))
      cpt = self.CPT[class_key]
      curr_attribs = [attrib_bnet_nodes[name] for name in attrib_name_list]
      
      p_function = mc.Lambda(class_key, 
                             self.prob_function_builder_for_class_layer(cpt, 
                                                                      curr_attribs))
      class_bnet_nodes[class_index] = mc.Bernoulli(str(class_index), p_function)
      
    nodes = attrib_bnet_nodes.values()
    nodes.extend(class_bnet_nodes.values())
    nodes.extend(theta)
    model = mc.Model(nodes)
    mc.graph.dag(model).write_pdf('tmp.pdf')
    MAP = mc.MAP(model)
    MAP.fit() # first do MAP estimation
    mcmc = mc.MCMC(model)
    mcmc.sample(10000, 3000)
    print()

##     use    
#     mcmc.summary()
##     or:
#     for node in attrib_bnet_nodes.values():
#       node.summary()       
#     for node in class_bnet_nodes.values():
#       node.summary()

    attrib_probs = pd.Series(np.zeros([len(attrib_names),]), index=attrib_names)
    for attrib_name in attrib_names:
      samples = mcmc.trace(str(attrib_name))[:]
      attrib_probs[attrib_name] = samples.mean()
    
#     print "attrib_probs:"
#     print attrib_probs   
    
    class_probs = pd.Series(np.zeros([len(class_inds),]), index=class_inds)
    for class_index in class_inds:
      samples = mcmc.trace(str(class_index))[:]
      class_probs[class_index] = samples.mean()
    
#     print "class_probs:"
#     print class_probs 
    
    return (class_probs, attrib_probs)
  
  
  def prob_function_builder_for_class_layer(self, cpt, attribs):
    return lambda attribs=attribs: np.float(cpt.ix[[tuple([int(a) for a in attribs])],
                                                        'True'])
  
  def prob_function_builder_for_mid_layer(self, cpt, theta):    
    return lambda theta=theta: np.float(cpt.ix[[tuple(theta)],'True'])
       
  
  
  @staticmethod      
  def format_attribute_list(attrib_list):
    '''
    Returns a string representation of the list of attributes (comma seperated).
    The attributes are sorted such that the representation is dependant only on
    the contents of attrib_list, not the order of it.
    '''
    l = list(np.sort(attrib_list))
    return ','.join(l)
    
    
    
    
    
    
    
    
    
    
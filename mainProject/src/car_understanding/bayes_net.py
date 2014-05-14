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
from util import ProgressBar
import pandas as pd
import pymc as mc
from sklearn.externals.joblib import Parallel, delayed
from sklearn import cross_validation
from bayesian.bbn import build_bbn
import copy
from sklearn.preprocessing import normalize

import Bow as Bow
from attribute_selector import AttributeSelector
from conditional_prob_table import CPT
import util
from numpy import zeros_like

global_CPT = None

class BayesNet2():
  """A Bayes net model ("arrows going down")"""
  
  
  def __init__(self, config, train_annos, class_meta, attrib_clfs, attrib_meta,
               multi_class_clf= None, desc="", use_gt=False):
    """ Ctor.
    
    Args:
          config      - Config object created by configuration.get_config()
          train_annos - Pandas table defining the training data (see fgcomp_dataset_utils)
          class_meta  - (see fgcomp_dataset_utils)
          attrib_clfs - A list of AttributeClassifiers
          desc        - Longer string description of attribute (optional)
          For debugging - use ground truth labels instead of attribute classifier scores
          
    """
    self.config       = config.copy()
    self.train_annos  = train_annos.copy()
    self.class_meta   = class_meta.copy()
    self.class_inds   = self.class_meta.class_index.unique()

    self.desc         = desc
    self.CPT          = {}
    self.g            = None # Will hold the bnet graph

    # sort by attrib name (keep the attributs sorted at all times!)
    #-----
    self.attrib_clfs  = attrib_clfs 
    self.attrib_names    = [self.attrib_clfs[ii].name for 
                                  ii in range(len(self.attrib_clfs))]
    inds = np.argsort(self.attrib_names)
    self.attrib_names = list(np.array(self.attrib_names)[inds])
    self.attrib_clfs = list(np.array(self.attrib_clfs)[inds])
    #----- 
    
    self.attrib_selector = AttributeSelector(self.config, 
                                        self.class_meta,
                                        attrib_meta)
    
    self.use_gt = use_gt
    self.multi_class_clf = multi_class_clf
    
    self.is_init = False
    
    
  def init(self):
    self.init_CPT()
    self.build_bnet()
    
  def init_CPT(self):
    self.is_init = True
    
    print 'init class nodes'
    print '----------------'
    self.init_class_nodes_CPT()
    
    print '\n\ninit attrib nodes'
    print '----------------'
    self.init_attrib_nodes_CPT()
    
    print '\n\ninit attrib clf nodes'
    print '----------------'
    self.init_attrib_clf_nodes_CPT()
    
    print '\n\ninit multiclass clf nodes'
    print '----------------'
    self.init_multi_class_clf_nodes_CPT()
    

  def init_class_nodes_CPT(self):
    '''
    Learning a CPT for p(c) using prior on the labels.
    '''
    n_imgs = float(self.train_annos.shape[0])
    class_counts = np.array([sum(self.train_annos.class_index == ind) 
                    for ind in self.class_inds])
    class_priors = class_counts / n_imgs
    
    for ii in range(len(self.class_inds)):
      prior = pd.DataFrame(data={'True': [class_priors[ii]],
                            'False': [1-class_priors[ii]]},
                           columns=['True', 'False'])
      self.CPT['p({})'.format(self.class_inds[ii])] = prior
      
#       print 'p({})'.format(self.class_inds[ii])
#       print prior 
    
  
  def init_attrib_nodes_CPT(self):
    has_attrib_prob = 0.8
    no_attrib_prob = 0.001
    
    for a_name in self.attrib_names:
      classes_for_attrib = self.attrib_selector.class_ids_for_attribute(a_name)
      classes_for_attrib = np.sort(classes_for_attrib)
      l = classes_for_attrib.shape[0]
      class_list = ','.join([str(x) for x in classes_for_attrib])
      cpt = CPT(default_true_value=has_attrib_prob, 
                name='p({0}|{1})'.format(a_name, class_list))
      false_row_key = ['False' for x in range(l)]
      cpt.create_row(false_row_key)
      cpt.set_value(false_row_key, cpt.TRUE, no_attrib_prob)
      cpt.set_value(false_row_key, cpt.FALSE, 1-no_attrib_prob)
      cpt.is_normalized = True
      
      self.CPT['p({}|{})'.format(a_name, class_list)] = cpt
      
#       print 'p({}|{})'.format(a_name, class_list)
#       print cpt 


  def init_attrib_clf_nodes_CPT(self):
    attrib_clfs = self.attrib_clfs
    bins = [-np.inf, -1, -0.2, 0.2, 1, np.inf]
    c = np.zeros(shape=[2, 5], dtype=np.float32)
    
    for clf in attrib_clfs:
      y_true = clf.labels_train
      p_scores = clf.train_pred_scores[y_true]
      n_scores = clf.train_pred_scores[np.logical_not(y_true)]
      c[0,:], b = np.histogram(p_scores, bins)
      c[1,:], b = np.histogram(n_scores, bins)
      c[:] += 1
      normalize(c, axis=1, norm='l1', copy=False)
      
      cpt = pd.DataFrame(data=c,
                         index=['True', 'False'], 
                         columns=['nn', 'n', 'u', 'p', 'pp'],
                         dtype=np.float32)

      print 'p({0}_clf|{0})'.format(clf.name)
      print cpt

      self.CPT['p({0}_clf|{0})'.format(clf.name)] = cpt

#   def init_attrib_clf_nodes_CPT(self):
#     attrib_clfs = self.attrib_clfs
#     
#     for clf in attrib_clfs:
#       y_true = clf.labels_train
#       y_pred = clf.train_pred_labels
#       
#       n_postive = float(np.sum(y_true))
#       n_negative = y_true.shape[0] - n_postive
#       n_tp = np.sum(np.logical_and(y_pred, y_true))
#       n_fp = np.sum(np.logical_and(y_pred, np.logical_not(y_true)))
#       
#       p_clf_given_attrib =  n_tp / n_postive
#       p_clf_given_not_attrib =  n_fp / n_negative
#       
#       cpt = pd.DataFrame(index=['True', 'False'], columns=['True', 'False'])
#       cpt.index.name = 'Hidden attrib value'
#       cpt.loc['True'] = [p_clf_given_attrib, 1-p_clf_given_attrib]
#       cpt.loc['False'] = [p_clf_given_not_attrib, 1-p_clf_given_not_attrib]
#       
# #       print 'p({0}_clf|{0})'.format(clf.name)
# #       print cpt
#       
#       self.CPT['p({0}_clf|{0})'.format(clf.name)] = cpt


  def init_multi_class_clf_nodes_CPT(self):
    scores = self.multi_class_clf.train_pred_scores
    bins = [-np.inf, -1, -0.2, 0.2, 1, np.inf]
    c = np.zeros(shape=[2, 5], dtype=np.float32)
    for ii, class_id in enumerate(self.class_inds):
      y_true = self.multi_class_clf.labels_train == class_id
      p_scores = scores[y_true,ii]
      n_scores = scores[np.logical_not(y_true),ii]
      
      c[0,:], b = np.histogram(p_scores, bins)
      c[1,:], b = np.histogram(n_scores, bins)
      c[:] += 1
      normalize(c, axis=1, norm='l1', copy=False)
      
      cpt = pd.DataFrame(data=c,
                         index=['True', 'False'], 
                         columns=['nn', 'n', 'u', 'p', 'pp'],
                         dtype=np.float32)
      
      print 'p(m_clf_{0}|{0})'.format(class_id)
      print cpt
      self.CPT['p(m_clf_{0}|{0})'.format(class_id)] = cpt

#   def init_multi_class_clf_nodes_CPT(self):
#     
#     for class_id in self.class_inds:
#       y_true = self.multi_class_clf.labels_train == class_id
#       y_pred = self.multi_class_clf.train_pred_labels == class_id
#       
#       n_postive = float(np.sum(y_true))
#       n_negative = y_true.shape[0] - n_postive
#       n_tp = np.sum(np.logical_and(y_pred, y_true))
#       n_fp = np.sum(np.logical_and(y_pred, np.logical_not(y_true)))
#       
#       p_clf_given_attrib =  n_tp / n_postive
#       p_clf_given_not_attrib =  n_fp / n_negative
#       
#       cpt = pd.DataFrame(index=['True', 'False'], columns=['True', 'False'])
#       cpt.index.name = 'class_id'
#       cpt.loc['True'] = [p_clf_given_attrib, 1-p_clf_given_attrib]
#       cpt.loc['False'] = [p_clf_given_not_attrib, 1-p_clf_given_not_attrib]
#     
# #       print 'p(m_clf_{0}|{0})'.format(class_id)
# #       print cpt
#       
#       self.CPT['p(m_clf_{0}|{0})'.format(class_id)] = cpt
      
      
  
  def build_functions_for_nodes(self):
    functions = []
    domains = {}
     
    global global_CPT
    global_CPT = self.CPT
    
    f,d = self.build_functions_for_class_nodes()
    functions.extend(f)
    domains.update(d)
    
    f,d = self.build_functions_for_attrib_nodes()
    functions.extend(f)
    domains.update(d)
    
    f,d = self.build_functions_for_attrib_clf_nodes()
    functions.extend(f)
    domains.update(d)
    
    f,d = self.build_functions_for_multiclass_clf_nodes()
    functions.extend(f)
    domains.update(d)      

    return functions, domains
  
  
  def build_functions_for_class_nodes(self):
    functions = []
    domains = {}
    
    f_str = '''def f_c_{class_id}(c_{class_id}):
      cpt = global_CPT['p({class_id})']
      return cpt.iloc[0][c_{class_id}]
    '''
    for class_id in self.class_inds:
      f_name = 'f_c_{}'.format(class_id)
      curr_f = function_builder(f_str.format(class_id=class_id), f_name)
      curr_d = {'c_' + str(class_id): ['True', 'False']} 
      
      functions.append(curr_f)
      domains.update(curr_d)
      
    return functions, domains
  
  def build_functions_for_attrib_nodes(self):
    functions = []
    domains = {}
    
    # make template function using string
    f_str = '''def f_a_{a_name}(a_{a_name}, {class_list1}):
      global global_CPT
      cpt = global_CPT['p({a_name}|{class_list2})']
      return cpt.get_value(({class_list1}), a_{a_name})
    '''
    
    for a_name in self.attrib_names:
      classes_for_attrib = self.attrib_selector.class_ids_for_attribute(a_name)
      classes_for_attrib = np.sort(classes_for_attrib)
      class_list1 = ','.join(['c_' + str(x) for x in classes_for_attrib])
      class_list2 = ','.join([str(x) for x in classes_for_attrib])
      f_name = 'f_a_{}'.format(a_name)
      curr_f = function_builder(f_str.format(a_name=a_name, 
                                             class_list1=class_list1,
                                             class_list2=class_list2),
                                f_name)
      functions.append(curr_f)
      domains.update({'a_' + a_name: ['True', 'False']})
      
    return functions, domains
  
  def build_functions_for_attrib_clf_nodes(self):
    functions = []
    domains = {}
    
    # make template function using string
    f_str = '''def f_clf_{a_name}(clf_{a_name}, a_{a_name}):
      global global_CPT
      cpt = global_CPT['p({a_name}_clf|{a_name})']
      return cpt.loc[a_{a_name}][clf_{a_name}]
    '''
    for a_name in self.attrib_names:
      f_name = 'f_clf_{}'.format(a_name)
      curr_f = function_builder(f_str.format(a_name=a_name), 
                                f_name)
      functions.append(curr_f)
      domains.update({'clf_' + a_name: ['nn', 'n', 'u', 'p', 'pp']})
    
    return functions, domains

  def build_functions_for_multiclass_clf_nodes(self):
    functions = []
    domains = {}
    
    # make template function using string
    f_str = '''def f_m_{class_id}(m_{class_id}, c_{class_id}):
      global global_CPT
      cpt = global_CPT['p(m_clf_{class_id}|{class_id})']
      return cpt.loc[c_{class_id}][m_{class_id}]
    '''
    for class_id in self.class_inds:
      f_name = 'f_m_{}'.format(class_id)
      curr_f = function_builder(f_str.format(class_id=class_id), 
                              f_name)
      functions.append(curr_f)
      domains.update({'m_' + str(class_id): ['nn', 'n', 'u', 'p', 'pp']})
    
    return functions, domains
      
      
  def build_bnet(self):
    functions, domains = self.build_functions_for_nodes()
    self.g = build_bbn(*functions, domains=domains)
    
  def q(self, *args):
    self.g.q(args)
    
    
  def create_attrib_res_on_images(self, data_annos, features=None):
    '''
    Calculates the predicion of all attribute classifiers on training images.
    This table can be used to caclulate all the Conditional Probability tables
    for the Bayes Net.
    '''
    if self.use_gt:
      return None, None
    
    # Define some conviniece pointers 
    config = self.config
    attrib_classifiers = self.attrib_clfs
    
    if features is None:
      print "create_attrib_res_on_images: Load image Bow histograms from disk"
      features = Bow.load_bow(data_annos, config)
  
    print "Apply attribute classifiers on images"
    res = {}
    res_descrete = {}
    pbar = ProgressBar(len(attrib_classifiers))
    for ii in range(len(attrib_classifiers)):
      attrib_clf = attrib_classifiers[ii]
      curr_res = attrib_clf.decision_function(features,
                                              use_prob=config.attribute.use_prob)
      curr_res_d = attrib_clf.predict(features)
      
      res[attrib_clf.name] = curr_res.reshape(len(curr_res))
      res_descrete[attrib_clf.name] = curr_res_d.reshape(len(curr_res_d))
      pbar.animate(ii)
    print ''
  
    res = pd.DataFrame(data=res, index=data_annos.index)
    res_descrete = pd.DataFrame(data=res_descrete, index=data_annos.index)
  
  
    res = pd.concat([res, data_annos.ix[:, ['class_index']]], axis=1)
    res_descrete = pd.concat([res_descrete, data_annos.ix[:, ['class_index']]], axis=1)
    
    
    return res, res_descrete
    
  def predict(self, test_annos):
    n_imgs = test_annos.shape[0]
    use_gt = self.use_gt
    class_inds = self.class_inds
    attrib_names = self.attrib_names
    
    print "Load image Bow histograms from disk"
    features = Bow.load_bow(test_annos, self.config)
    
    
    if not use_gt:
      clf_res, clf_res_discrete = self.create_attrib_res_on_images(test_annos,
                                                                   features)
        
    # using ground truth    
    if use_gt:
      attrib_meta = self.attrib_selector.create_attrib_meta(attrib_names)
        
        
    # apply multi class classifier on test annos
    
    # (if boolean)
#     m = self.multi_class_clf.predict(features=features)
#     m_clf_values = pd.DataFrame(data=np.zeros([m.shape[0], len(class_inds)]), 
#                            index=test_annos.index, 
#                            columns=class_inds, dtype=bool)
#     for ii in range(m.shape[0]):
#           m_clf_values.iloc[ii][m[ii]] = True

    # (if using score values)
    m = self.multi_class_clf.decision_function(features=features)
    mm = np.zeros_like(m, dtype=str)

    mm[m <= -1] = 'nn'
    mm[np.logical_and(m > -1, m  <= -0.2)] = 'n'
    mm[np.logical_and(m > -0.2, m  <= 0.2)] = 'u'
    mm[np.logical_and(m > 0.2, m  <= 1)] = 'p'
    mm[m > 1] = 'pp'
    m_clf_values = pd.DataFrame(data=mm, index=test_annos.index, 
                                columns=class_inds, dtype=str)
    
    c = np.array(clf_res[attrib_names])
    tmp = clf_res['class_index']
    cc = np.zeros_like(c, dtype=str)
    cc[c <= -1] = 'nn'
    cc[np.logical_and(c > -1, c  <= -0.2)] = 'n'
    cc[np.logical_and(c > -0.2, c  <= 0.2)] = 'u'
    cc[np.logical_and(c > 0.2, c  <= 1)] = 'p'
    cc[c > 1] = 'pp'
    clf_res = pd.DataFrame(data=cc, index=test_annos.index, 
                                columns=attrib_names, dtype=str)
    clf_res['class_index'] = tmp
    
    

    class_prob = pd.DataFrame(np.zeros([n_imgs, 
                                        len(class_inds)]),
                              index=test_annos.index, 
                              columns=class_inds)
    
    attrib_prob = pd.DataFrame(np.zeros([n_imgs, 
                                         len(attrib_names)]),
                              index=test_annos.index, 
                              columns=attrib_names)
    
    pbar = ProgressBar(n_imgs)
    print 'Prdicting class probabilities on images'
    for ii in range(n_imgs):
#       print "=================={}/{}========================".format(ii+1, n_imgs)
#       print "Image: {}, class_id: {}, class_name: {}".format(test_annos.iloc[ii]['basename'],
#                                                             test_annos.iloc[ii]['class_index'], 
#                                                             test_annos.iloc[ii]['class_name'])
      if use_gt:
        attrib_res = attrib_meta.loc[test_annos.iloc[ii]['class_index']]
      else:
#         attrib_res = clf_res_discrete.iloc[ii]
        attrib_res = clf_res.iloc[ii]
      
      m_clf_values_one = m_clf_values.iloc[ii]
      (class_prob_ii, attrib_prob_ii) = self.predict_one(attrib_res, m_clf_values_one)
      class_prob.iloc[ii] = class_prob_ii
      attrib_prob.iloc[ii] = attrib_prob_ii
      pbar.animate(ii)
      
    print ' '
    return (class_prob, attrib_prob)
  
  
  def predict_one(self, clf_res_discrete, m_clf_values):
    use_gt = self.use_gt
    
    if use_gt:
      raise NotImplementedError()
    
    # build dictionary with all query parameters:
    # parameters are the observed values for the classifiers
    q_params = {}
    # attrib clfs
    for a_name in self.attrib_names:
      q_params.update({'clf_'+a_name: str(clf_res_discrete.loc[a_name])})
    # multi class clf
    for c_id in self.class_inds:
      q_params.update({'m_'+str(c_id): str(m_clf_values.loc[c_id])})
      
    
    
    # Run inference with query parameters
    marginals = self.g.query(**q_params)
    
    # Build class probs
    class_probs = pd.Series(index=self.class_inds, dtype=np.float64)
    for c_id in self.class_inds:
      class_probs.loc[c_id] = marginals[('c_' + str(c_id), 'True')]
    
    # Build attribute probs
    attrib_probs = pd.Series(index=self.attrib_names, dtype=np.float64)
    for a_name in self.attrib_names:
      attrib_probs.loc[a_name] = marginals[('a_' + a_name, 'True')]

    return class_probs, attrib_probs
  
  def export_to_BIF(self, filename):
    from lxml import etree
    # Define preamble
    preamble = '''\
<?xml version="1.0"?>
<!-- DTD for the XMLBIF 0.3 format -->
<!DOCTYPE BIF [
        <!ELEMENT BIF ( NETWORK )*>
              <!ATTLIST BIF VERSION CDATA #REQUIRED>
        <!ELEMENT NETWORK ( NAME, ( PROPERTY | VARIABLE | DEFINITION )* )>
        <!ELEMENT NAME (#PCDATA)>
        <!ELEMENT VARIABLE ( NAME, ( OUTCOME |  PROPERTY )* ) >
              <!ATTLIST VARIABLE TYPE (nature|decision|utility) "nature">
        <!ELEMENT OUTCOME (#PCDATA)>
        <!ELEMENT DEFINITION ( FOR | GIVEN | TABLE | PROPERTY )* >
        <!ELEMENT FOR (#PCDATA)>
        <!ELEMENT GIVEN (#PCDATA)>
        <!ELEMENT TABLE (#PCDATA)>
        <!ELEMENT PROPERTY (#PCDATA)>
]>


<BIF VERSION="0.3">
</BIF>'''
    root = etree.XML(preamble)
    print 'here'
    tree = etree.ElementTree(root)
    etree.SubElement(root, 'NETWORK')
    network = root[0]
    etree.SubElement(network, 'NAME')
    network[0].text = self.desc
    
    node_str = '''
    <VARIABLE TYPE="nature">
    <NAME>{name}</NAME>
    <OUTCOME>{Value1}</OUTCOME>
    <OUTCOME>{Value2}</OUTCOME>
    <PROPERTY>position = (0,0)</PROPERTY>
    </VARIABLE>'''
    
    def c_node_maker(name, values):
      node = etree.Element('VARIABLE', TYPE="nature")
      NAME = etree.Element('NAME')
      NAME.text = name
      node.append(NAME)
      for v in values:
        outcome = etree.Element('OUTCOME')
        outcome.text = v
        node.append(outcome)
      p = etree.Element('PROPERTY')
      p.text = 'position = (0,0)'
      node.append(p)
    
    # make XML object for class nodes
    for c_ind in self.class_inds:
      n = c_node_maker('c_'+str(c_ind), ['True', 'False'])
      network.append(n)
      
    print etree.tostring(tree)
    
    
def function_builder(f_str, f_name):
#   print f_str
  exec f_str in globals(), locals()
  return locals()[f_name]



  

#------------------------------------------------------------------------------
#----------------------------OLD-----------------------------------------------
#------------------------------------------------------------------------------

def cpt_for_attrib(attrib_name, attrib_selector, 
                     clf_names, clf_res_discrete):
    
    attrib_class_ids = attrib_selector.class_ids_for_attribute(attrib_name)
    # intersect attrib_class_ids with clf_res.class_index
    attrib_class_ids = \
    [val for val in attrib_class_ids if val in list(clf_res_discrete.class_index)]
    
    
    cpt = CPT(smooth_value=1, name='{}_attribute_cpt'.format(attrib_name))
    
#     pbar = ProgressBar(clf_res_discrete.shape[0])
    for ii in range(clf_res_discrete.shape[0]):
#       pbar.animate(ii)
      cc = clf_res_discrete.iloc[ii]
      row = tuple(cc[clf_names])
      has_attrib = cc['class_index'] in attrib_class_ids
      if not cpt.has_row(row):
          cpt.create_row(row)
      cpt.add_count(row, str(has_attrib))
#     print('')
    
    # normalize all the rows, to create a probability function
    cpt.normalize_rows()
#     print "CPT for attrib: {}".format(attrib_name)
#     print "----------------------------"
#     print cpt
    return cpt

class BayesNet:
  """A Bayes net model."""
  
  def __init__(self, config, train_annos, class_meta, attrib_clfs, attrib_meta,
               multi_class_clf = None, desc="", use_gt=False):
    """ Ctor.
    
    Args:
          config      - Config object created by configuration.get_config()
          train_annos - Pandas table defining the training data (see fgcomp_dataset_utils)
          class_meta  - (see fgcomp_dataset_utils)
          attrib_clfs - A list of AttributeClassifiers
          desc        - Longer string description of attribute (optional)
          For debugging - use ground truth labels instead of attribute classifier scores
          
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
    self.clf_res_discrete = None
    self.CPT          = {}
    self.clf_names    = [self.attrib_clfs[ii].name for 
                                  ii in range(len(self.attrib_clfs))]
    
    # sort by attrib name (keep the attributs sorted at all times!)
    inds = np.argsort(self.clf_names)
    self.clf_names = list(np.array(self.clf_names)[inds])
    self.attrib_clfs = list(np.array(self.attrib_clfs)[inds]) 
    
    self.attrib_selector = AttributeSelector(self.config, 
                                        self.class_meta,
                                        attrib_meta)
    
    self.use_gt = use_gt
    self.multi_class_clf = multi_class_clf
     
    
  
  def create_attrib_res_on_images(self, data_annos, features=None):
    '''
    Calculates the predicion of all attribute classifiers on training images.
    This table can be used to caclulate all the Conditional Probability tables
    for the Bayes Net.
    '''
    if self.use_gt:
      return None, None
    
    
    # Define some conviniece pointers 
    config = self.config
    attrib_classifiers = self.attrib_clfs
    
    if features is None:
      print "Load image Bow histograms from disk"
      features = Bow.load_bow(data_annos, config)
  
    print "Apply attribute classifiers on images"
    res = {}
    res_descrete = {}
    pbar = ProgressBar(len(attrib_classifiers))
    for ii in range(len(attrib_classifiers)):
      attrib_clf = attrib_classifiers[ii]
      curr_res = attrib_clf.decision_function(features,
                                              use_prob=config.attribute.use_prob)
      curr_res_d = attrib_clf.predict(features)
      
      res[attrib_clf.name] = curr_res.reshape(len(curr_res))
      res_descrete[attrib_clf.name] = curr_res_d.reshape(len(curr_res_d))
      pbar.animate(ii)
    print ''
  
    res = pd.DataFrame(data=res, index=data_annos.index)
    res_descrete = pd.DataFrame(data=res_descrete, index=data_annos.index)
  
  
    res = pd.concat([res, data_annos.ix[:, ['class_index']]], axis=1)
    res_descrete = pd.concat([res_descrete, data_annos.ix[:, ['class_index']]], axis=1)
    
    
    return res, res_descrete
  
  
  
    
  
  
  def cpt_for_class(self, class_index, attrib_list, attribute_selector):    

    # Create all tuples of True/False for indicator variable
#     rows = list(itertools.product(*[(1, 0) for 
#                                     ii in range(len(attrib_list))]))
    
    min_prob = 1e-2
#     cpt = pd.DataFrame(min_prob * np.ones([len(rows), 2], dtype=np.float64), 
#                        index=rows, columns=['True', 'False'])
    
    cpt = CPT(name='class_cpt', default_true_value=min_prob)

    # All rows except the one that has ALL the attributes should have p(true)=min_prob.
    # The one in which all attribs are true, should be the proportion of this class
    # with respect to all classes that have all these attributes.
#     cpt['False'] = 1-min_prob
    

    
    num_classes_with_attrib = 0
    for cind in self.class_meta.index:
      if np.all(attribute_selector.has_list_attributes_by_index(cind, 
                                                      attrib_list)):
        num_classes_with_attrib += 1
         
         
#     print "attrib_list: {}".format(attrib_list)
#     print "num_classes_with_attribs: {}".format(num_classes_with_attrib)
    p = 1.0 / num_classes_with_attrib
    row = tuple(True for ii in range(len(attrib_list)))
    cpt.create_row(row)
    cpt.set_value(row, 'True', p)
    cpt.set_value(row, 'False', 1-p)
    cpt.normalize_rows()
#     cpt.ix[[tuple(*np.ones(shape=[1, len(attrib_list)], 
#                         dtype=int))], 'True'] = p
#     cpt.ix[[tuple(*np.ones(shape=[1, len(attrib_list)], 
#                         dtype=int))], 'False'] = 1 - p
                        
#     print "CPT for class: {}".format(self.class_meta.class_name[class_index])
#     print "---------------------------------"
#     print cpt
    return cpt
    
      
      
    
  def init_CPT(self):
    '''
    Initialize the Conditional Probability Tables for all the nodes
    in the net.
    '''
    
    attrib_selector = self.attrib_selector
    
    attrib_names = self.clf_names
    print '\n-------' + str(attrib_names) + '\n--------'
    class_inds  = self.class_inds
    
    
    
    # P(attribute | res of attrib classifiers)
    #-----------------------------------------
    print('Building CPT for attributes')
    if not self.use_gt: # if using ground truth we don't need to calculate this
      
      # read tables from attrib classifiers
      for ii, attrib_name in enumerate(attrib_names):
        curr_cpt = CPT(smooth_value=0, name='{}_attribute_cpt'.format(attrib_name))
        row_ind = tuple([True])
        curr_cpt.create_row(row_ind)
        curr_cpt.set_value(row_ind, 'True', self.attrib_clfs[ii].stats.loc['True', 'True'])
        curr_cpt.set_value(row_ind, 'False', self.attrib_clfs[ii].stats.loc['True', 'False'])
        row_ind = tuple([False])
        curr_cpt.create_row(row_ind)
        curr_cpt.set_value(row_ind, 'True', self.attrib_clfs[ii].stats.loc['False', 'True'])
        curr_cpt.set_value(row_ind, 'False', self.attrib_clfs[ii].stats.loc['False', 'False'])
        
        self.CPT['p({}|theta)'.format(attrib_name)] = curr_cpt
        print attrib_name
        print self.CPT['p({}|theta)'.format(attrib_name)]
        
#       clf_res, clf_res_discrete = \
#         self.create_attrib_res_on_images(self.train_annos)      
#       n_attribs = len(attrib_names)
#       cpts = Parallel(n_jobs=self.config.n_cores, 
#                       verbose=self.config.logging.verbose)(
#                       delayed(cpt_for_attrib)(attrib_names[ii], 
#                                                    attrib_selector,
#                                                    np.array([attrib_names[ii]]),
#                                                    clf_res_discrete)
#                       for ii in range(n_attribs))                       
#       for ii, attrib_name in enumerate(attrib_names):
#         self.CPT['p({}|theta)'.format(attrib_name)] = cpts[ii]
      


        
    # P(class | attributes)
    #----------------------
    print('Building CPT classes')
#     pbar = ProgressBar(len(class_inds))
    for ii, class_index in enumerate(class_inds):
      # figure out which attributes does this class have
      attrib_list = attrib_selector.prune_attributes(class_index, attrib_names)
      
      class_key = 'p({} | {})'.format(class_index, 
                                   BayesNet.format_attribute_list(attrib_list))
      
      self.CPT[class_key] = self.cpt_for_class(class_index, 
                                               attrib_list, 
                                               attrib_selector)
#       pbar.animate(ii)
#     print ''
    
#     print self.class_meta
        
  
  '''
  use_gt: will use the ground truth attribute values for the middle 
  layer, to check what is the best we can hope for.
  '''
  def predict(self, test_annos):
    if len(test_annos.shape) == 1:
      n_imgs = 1
    else:
      n_imgs = test_annos.shape[0]
    use_gt = self.use_gt
    class_inds = self.class_inds
    class_prob = pd.DataFrame(np.zeros([n_imgs, 
                                        len(class_inds)]),
                              index=test_annos.index, 
                              columns=class_inds)
    
    attrib_names = self.clf_names
    attrib_prob = pd.DataFrame(np.zeros([n_imgs, 
                                         len(attrib_names)]),
                              index=test_annos.index, 
                              columns=attrib_names)
    
    
    print "Load image Bow histograms from disk"
    features = Bow.load_bow(test_annos, self.config)
    
    if not use_gt:
      clf_res, clf_res_discrete = self.create_attrib_res_on_images(test_annos,
                                                                   features)
        
    # using ground truth    
    if use_gt:
      attrib_meta = self.attrib_selector.create_attrib_meta(attrib_names)
        
        
    # apply multi class classifier on test annos
    m_proba = self.multi_class_clf.predict_proba(features)
    m_proba = pd.DataFrame(data=m_proba, 
                           index=test_annos.index, 
                           columns=class_inds)
        
    
    # Create cache for results:
    # we don't want to waste time on options we have seen before.
    class_prob_cache = {}
    attrrib_prob_cache = {}
    
    for ii in range(n_imgs):
      print "=================={}/{}========================".format(ii+1, n_imgs)
      print "Image: {}, class_id: {}, class_name: {}".format(test_annos.iloc[ii]['basename'],
                                                            test_annos.iloc[ii]['class_index'], 
                                                            test_annos.iloc[ii]['class_name'])
      if use_gt:
        discr = attrib_meta.loc[test_annos.iloc[ii]['class_index']]
        key = np.array(discr) 
      else:
        discr = clf_res_discrete.iloc[ii]
        key = np.array(discr[attrib_names])
      
      m_proba_one = m_proba.iloc[ii]
      key = np.concatenate([key, np.array(m_proba_one)])
      print "key: {}".format(key)
      key = key.tostring()
      if (not class_prob_cache.has_key(key)):
        print "Never got this key before, computing...."
        (class_prob_ii, attrib_prob_ii) = self.predict_one(discr, m_proba_one)
        class_prob_cache[key] = class_prob_ii
        attrrib_prob_cache[key] = attrib_prob_ii
      
      class_prob.iloc[ii] = class_prob_cache[key]
      attrib_prob.iloc[ii] = attrrib_prob_cache[key]
      
    return (class_prob, attrib_prob)
  
  def predict_one(self, clf_res_discrete, m_proba_one, method='pymc'):
    if method == 'pymc':
      return self.predict_one_pymc(clf_res_discrete, m_proba_one)
    else:
      return self.predict_one_ebayes(clf_res_discrete, m_proba_one)
    
  def predict_one_ebayes(self, clf_res_discrete):
    from bayesian.bbn import build_bbn
    
#     #------- Dynamically rename a function ---------
#     def bind_function(name, func):
#       func.__name__ = name
#       return func
#     #------- Dynamically rename a function ---------

    
    
    # building model
    # first start with observed variables - the results of all the classifiers 
    # on the image
    clf_nodes = {}
    for clf_name in self.clf_names:
      print 'jhf'
    
    return    
  
  def predict_one_pymc(self, clf_res_discrete, multi_clf_probs):
    use_gt = self.use_gt
    class_inds = self.class_inds
    # building model
    # first start with observed variables - the results of all the classifiers 
    # on the image
        
    # the actual distrobution used is not important as these are 
    # *observed* variables. We set the value to be the result of the classifiers              
    theta = mc.DiscreteUniform('theta', 0, 1,
                               size=len(self.clf_names),
                               observed=True,
                               value=clf_res_discrete[self.clf_names])
    
    
#     M = mc.DiscreteUniform('M', 0, 1,
#                                size=len(class_inds),
#                                observed=True,
#                                value=multi_clf_probs)
    
    
    # The hidden layer. Each attriute is connected to all attribute classifiers
    # as its parents. If we are using the ground truth, then we set this layer to
    # also be observed.
    attrib_names = self.clf_names
    attrib_bnet_nodes = {}
    if use_gt:
      for attrib_name in attrib_names:
        rv = mc.DiscreteUniform(attrib_name, 
                                observed=True, 
                                value=clf_res_discrete[attrib_name])
        attrib_bnet_nodes[attrib_name] = rv
    else:
      for ii, attrib_name in enumerate(attrib_names):
        identifier = 'p({}|theta)'.format(attrib_name)
        cpt = self.CPT[identifier]
        p_function = mc.Lambda(identifier, 
                               self.prob_function_builder_for_mid_layer(cpt, 
                                                                        theta,
                                                                        ii))
        attrib_bnet_nodes[attrib_name] = mc.Bernoulli(attrib_name, p_function)
      
     
    # The top layer Each class is connected to the attributes it has  and to the
    # multi class classifier
    attrib_selector = self.attrib_selector
    
    M = []
    class_bnet_nodes = {}
    for class_index in class_inds:
      attrib_name_list = attrib_selector.prune_attributes(class_index, attrib_names)
      class_key = 'p({} | {})'.format(class_index, 
                                   BayesNet.format_attribute_list(attrib_name_list))
      cpt = self.CPT[class_key]
      curr_attribs = [attrib_bnet_nodes[name] for name in attrib_name_list]
      
      location = np.where(multi_clf_probs.index == class_index)[0][0]
      m = mc.DiscreteUniform('M_{}'.format(class_index),
                                       observed=True,value=np.array(multi_clf_probs)[location])
      M.append(m)
      p_function = mc.Lambda(class_key, 
                             self.prob_function_builder_for_class_layer(cpt, 
                                                                      curr_attribs,
                                                                      m))
      class_bnet_nodes[class_index] = mc.Bernoulli(str(class_index), p_function)
      
    nodes = attrib_bnet_nodes.values()
    nodes.extend(class_bnet_nodes.values())
    nodes.extend(theta)
    nodes.extend(M)
    model = mc.Model(nodes)
    mc.graph.dag(model).write_pdf('tmp.pdf')
#     import sys;sys.exit(0)
    
#     if not self.use_gt:
#       MAP = mc.MAP(model)
#       MAP.fit() # first do MAP estimation
      
    mcmc = mc.MCMC(model)
#     mcmc.sample(10000, 3000)
    mcmc.sample(2000) 
    print()
    
#     from pymc.Matplot import plot as mcplot
#     mcplot(mcmc.trace("bmw"), common_scale=False)
#     from matplotlib.pyplot import savefig; savefig('mcmc.pdf')
#     import sys;sys.exit(0)

##     use    
#     mcmc.summary()
##     or:
#     for node in attrib_bnet_nodes.values():
#       node.summary()       
#     for node in class_bnet_nodes.values():
#       node.summary()

    attrib_probs = pd.Series(np.zeros([len(attrib_names),]), index=attrib_names)
    if not use_gt:
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
    
#     import sys;sys.exit(0)
    return (class_probs, attrib_probs)
  
  
  def prob_function_builder_for_class_layer(self, cpt, attribs, M):
#     return lambda attribs=attribs: np.float(cpt.ix[[tuple([int(a) for a in attribs])],
#                                                         'True'])
#     return lambda attribs=attribs: np.float(cpt.get_value(tuple([a for a in attribs]), 'True'))

    def func(attribs=attribs, M=M):
      attrib_prob = np.float(cpt.get_value(tuple([a for a in attribs]), 'True'))
      multi_prob = M
      from_attrib = attrib_prob > 1e-2
      from_multi = multi_prob >= 0.5
      
      if from_attrib and from_multi:
        return (multi_prob + attrib_prob)/2.0
      if from_attrib and (not from_multi):
        return attrib_prob
      if (not from_attrib) and from_multi:
        return multi_prob
      return attrib_prob
      
  
    return func                         
#     return lambda attribs=attribs, M=M: max(np.float(cpt.get_value(tuple([a for a in attribs]), 'True')),
#                                             M)
  
  def prob_function_builder_for_mid_layer(self, cpt, theta, ii):    
#     return lambda theta=theta: np.float(cpt.ix[[tuple(theta)],'True'])
#     return lambda theta=theta: np.float(cpt.get_value([bool(v) for v in tuple(theta)],'True'))
    return lambda theta=theta: np.float(cpt.get_value([theta[ii]],'True'))

    
  
  @staticmethod      
  def format_attribute_list(attrib_list):
    '''
    Returns a string representation of the list of attributes (comma seperated).
    The attributes are sorted such that the representation is dependant only on
    the contents of attrib_list, not the order of it.
    '''
    l = list(np.sort(attrib_list))
    return ','.join(l)
    
    
    
    
    
    
    
    
    
    
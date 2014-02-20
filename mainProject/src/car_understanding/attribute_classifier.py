# encoding: utf-8
'''
car_understanding.attribute_classifier -- a single attribute classifier

@author:     Yair Movshovitz-Attias

@copyright:  2014 Yair Movshovitz-Attias. All rights reserved.

@contact:    yair@cs.cmu.edu
'''

from sklearn.externals.joblib import dump, load
import sklearn as sk
import numpy as np
import os

import Bow

class AttributeClassifier:
  """A module for classifying attributes."""
  
  def __init__(self, config, dataset, pos_inds, name, desc=""):
    """ Ctor.
    
    Args:
          config   - Config object created by configuration.get_config()
          dataset  - Pandas table defining the dataset (see fgcomp_dataset_utils)
          pos_inds - a list or numpy array of image indices to use as positive
          examples for training the classifier).
          name     - Short string name for object.
          desc     - Longer string description of attribute (optional)
          
    """
    self.config       = config
    self.name         = name
    self.pos_img_inds = pos_inds
    self.dataset      = dataset.copy()
    self.desc         = desc
    self.clf          = sk.svm.SVC(kernel='linear', 
                                   class_weight='auto',
#                                    C=0.005,
                                   probability=True)
#     self.memory   = Memory(cachedir=config.SIFT.BoW.hist_dir.format(name), 
#                            verbose=0)
    
    
    # Creating memoiztion for functions
#     self.calc_raw_feature = self.memory.cache(self.calc_raw_feature) 
    
  
  def create_feature_matrix(self, features=None):
    # Load histograms from disk into a matrix
    if features == None:
      # Preallocate feature matrix
      features = np.empty(shape=[len(self.dataset), 
                                 self.config.SIFT.BoW.num_clusters])
      for ii in range(len(self.dataset)):
        img_name = self.dataset.iloc[ii]['basename']
        img_name = os.path.splitext(img_name)[0]
        hist_filename = os.path.join(self.config.SIFT.BoW.hist_dir, 
                                     img_name) + '_hist.dat'
        hist = Bow.load(hist_filename) # hist[0] = values, hist[1] = bin edges
        features[ii, :] = hist[0]
  
  
    # preprocess features
    features = sk.preprocessing.scale(features)
  
    # create pos/neg labels
    labels = self.dataset.index.isin(self.pos_img_inds)
    
    num_pos = sum(labels)
    num_neg = labels.shape[0] - num_pos
    
    print "num_pos: {}, num_neg: {}".format(num_pos, num_neg)
    assert num_neg >= num_pos, "num_neg >= num_pos"
    
    
    # make pos/neg sets of equal size
    pos_inds = labels.nonzero()[0]
    neg_inds = np.logical_not(labels).nonzero()[0]
    neg_inds = np.random.permutation(neg_inds)[:num_pos]
     
    features = features[np.concatenate([pos_inds, neg_inds]), :]
    labels  = np.concatenate([np.ones(shape=pos_inds.shape),
                             np.zeros(shape=neg_inds.shape)])
    
    dump([features, labels], 'features_all.tmp')
     
     
     
    num_pos = sum(labels)
    num_neg = labels.shape[0] - num_pos
    print "equalized sets: num_pos: {}, num_neg: {}".format(num_pos, num_neg)
    assert num_neg == num_pos, "num_neg == num_pos"
    assert features.shape[0] == num_pos + num_neg, \
    "features.shape[0] == num_pos + num_neg"
     
    dump([features, labels], 'features_eq.tmp')
  
    return (features, labels)
    
  
  def fit(self, features, labels):
    self.clf.fit(features, labels)
    
    
  def cross_validate(self, features, labels):
    self.my_print("Num pos examples: {}".format(np.sum(labels)))
    cv = 5
    score_method = 'precision'
#     score_method = 'accuracy'
#     cv = sk.cross_validation.StratifiedKFold(labels, 3)
    scores = sk.cross_validation.cross_val_score(self.clf, 
                                                 features, 
                                                 labels,
                                                 n_jobs=min(cv, 12), 
                                                 verbose=1,
                                                 cv=cv,
                                                 scoring=score_method)
    
    # Report results
    self.my_print("scores: {}".format(scores))
    self.my_print(score_method + ": %0.2f (+/- %0.2f)" % 
                  (scores.mean(), scores.std() * 2))
    rand_pred = np.random.choice([True, False], size=labels.shape)
    self.my_print("Random (precision): {}".format(
                               sk.metrics.precision_score(labels, rand_pred)))
    
  def my_print(self, str):
    print "AttributeCLassifier(" + self.name + "):" + str
    
  def run_training_pipeline(self, cv=False):
    """ The full sequence of operations that trains an attribute classifier"""
    
    self.my_print("Loading feature-word histograms from disk, and creating " + 
                  "matrix for attribute classification.")
    (features, labels) = self.create_feature_matrix()
    return
    
    
    if cv:
      self.my_print("Training classifier [Cross Validation]")
      self.cross_validate(features, labels)
#     else:
    self.my_print("Training classifier")
    self.fit(features, labels)
    
    
  def decision_function(self, features):
    if self.clf.probability:
      return self.clf.predict_proba(features)[:,0]
    return self.clf.decision_function(features)
    
  
    
  # Static" functions
  # -----------------
  @staticmethod
  def contains(box, point):
    '''
    box = (xmin, xmax, ymin, ymax)
    point = (x, y)
    '''
    return (box[0] <= point[0] and box[1] >= point[0] and
            box[2] <= point[1] and box[3] >= point[1])
    
  @staticmethod  
  def save(attrib_classifier, filename):
    dump(attrib_classifier, filename)
      
  @staticmethod    
  def load(filename):
    return load(filename)
    
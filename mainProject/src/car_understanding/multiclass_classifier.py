'''
Created on May 1, 2014

@author: ymovshov
'''

from sklearn.svm import SVC, LinearSVC
from sklearn import cross_validation
from sklearn.externals.joblib import dump, load
from sklearn.preprocessing import StandardScaler
import numpy as np

import Bow as Bow

class MultiClassClassifier(object):
  '''
  A wrapper module around a multi class classifier. Gathers statistics of
  accuracy while training.  
  '''


  def __init__(self, train_annos, class_meta, config):
    '''
    Constructor.
    
    Args:
      train_annos - pandas Dataframe with annotations of training images.
      class_meta - pandas Dataframe with class_id to class name mappings.
      config - config object created by configuration.get_config()
    '''
  
    self.train_annos = train_annos.copy()
    self.class_meta = class_meta.copy()
    self.config = config
    
    self.clf = LinearSVC(class_weight='auto', loss='l2', C=1)
    self.scaler = StandardScaler()
    
    self.class_inds = np.array(class_meta.class_index)
    self.n_folds = 4 # used for gathering stats
    
    self.labels_train = np.array(self.train_annos.class_index)
    # predicted labels for training data. each instance is predicted when its
    # fold was used as a validation set
    self.train_pred_labels = np.zeros_like(self.labels_train)
    self.train_pred_scores = np.zeros(shape=[self.train_pred_labels.shape[0], 
                                             len(self.class_inds)],
                                       dtype=np.float64)
    
  
  def fit(self):
    
    # load features form disk
    features = Bow.load_bow(self.train_annos, self.config)
#     features = self.scaler.fit_transform(features)
    
    labels = self.labels_train
    
    print('Training with these {} classes:'.format(len(np.unique(labels))))
    print(labels)
    
    skf = cross_validation.StratifiedKFold(self.labels_train, n_folds=self.n_folds)
    
    ii = 0
    for train_index, test_index in skf:
      ii += 1
      print('Training using fold {} of {}'.format(ii, self.n_folds))
      self.clf.fit(features[train_index,:], labels[train_index])
      self.train_pred_labels[test_index] = \
        self.clf.predict(features[test_index,:])
      self.train_pred_scores[test_index,:] = \
        self.clf.decision_function(features[test_index,:])
        
    # after all stats are gathered, retrain with all data for best perfromance.
    print('Training with all data')
    self.clf.fit(features, labels)
    
  def predict(self, test_annos=None, features=None):
    assert (not (test_annos is None)) or (not (features is None)), 'test_annos or features need to be not None' 
    
    if features is None:
      features = Bow.load_bow(test_annos, self.config)
      
#     features = self.scaler.transform(features)
    
    return self.clf.predict(features)
  
  def decision_function(self, test_annos=None, features=None):
    assert (not (test_annos is None)) or (not (features is None)), 'test_annos or features need to be not None' 
    
    if features is None:
      features = Bow.load_bow(test_annos, self.config)
      
#     features = self.scaler.transform(features)
    
    return self.clf.decision_function(features)
  
  
  # Static" functions
  # -----------------
  @staticmethod  
  def save(clf, filename):
    dump(clf, filename, compress=3)
    
  @staticmethod    
  def load(filename):
    return load(filename)
    
    
    
    
    
  
      
        
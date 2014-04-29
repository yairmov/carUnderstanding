# encoding: utf-8
'''
car_understanding.attribute_classifier -- a single attribute classifier

@author:     Yair Movshovitz-Attias

@copyright:  2014 Yair Movshovitz-Attias. All rights reserved.

@contact:    yair@cs.cmu.edu
'''

import sklearn as sk
from sklearn.externals.joblib import dump, load
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble.forest import RandomForestClassifier
import numpy as np
import os
import pandas as pd

import Bow as Bow
import util


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
    self.probability  = config.attribute.use_prob
    self.n_cores      = config.n_cores
    self.thresh       = 0
#     self.clf          = SVC(kernel='rbf', 
#                            class_weight='auto',
#                            C=1, gamma=1e-3,
#                            probability=True)

    self.clf          = LinearSVC(class_weight='auto', loss='l2', C=1e-3)
#     self.clf          = RandomForestClassifier(n_estimators=100, max_depth=4,
#                                                min_samples_split=1,
#                                                min_samples_leaf=1,
#                                                oob_score=True,
#                                                n_jobs=self.n_cores)
#     self.clf          = GradientBoostingClassifier(n_estimators=1000, 
#                                                    learning_rate=1.0, 
#                                                    max_depth=1)
    self.Scaler       = StandardScaler()
      
  
  def create_feature_matrix(self, features=None):
    # Load histograms from disk into a matrix
    if features == None:
      features = Bow.load_bow(self.dataset, self.config)
  
  
    # create pos/neg labels
    labels = self.dataset.index.isin(self.pos_img_inds)
    
    
    # preprocess features
    features = self.Scaler.fit_transform(features)
    
    num_pos = sum(labels)
    num_neg = labels.shape[0] - num_pos
    
    print "num_pos: {}, num_neg: {}".format(num_pos, num_neg)
    assert num_neg >= num_pos, "num_neg >= num_pos"
    
#     dump([features, labels], 'features_all.tmp')
    
    
    # make pos/neg sets of equal size
  #     pos_inds = labels.nonzero()[0]
  #     neg_inds = np.logical_not(labels).nonzero()[0]
  #     neg_inds = np.random.permutation(neg_inds)[:num_pos]
  #         
  #     features = features[np.concatenate([pos_inds, neg_inds]), :]
  #     labels  = np.concatenate([np.ones(shape=pos_inds.shape, dtype=bool),
  #                              np.zeros(shape=neg_inds.shape, dtype=bool)])
     
     
    num_pos = sum(labels)
    num_neg = labels.shape[0] - num_pos
    print "equalized sets: num_pos: {}, num_neg: {}".format(num_pos, num_neg)
#     assert num_neg == num_pos, "num_neg == num_pos"
    assert features.shape[0] == num_pos + num_neg, \
    "features.shape[0] == num_pos + num_neg"
     
     
#     string_labels = np.empty(shape=labels.shape, dtype=np.object)
#     trueval = self.name
#     falseval = 'NOT-' + self.name
#     string_labels[labels] = trueval
#     string_labels[np.logical_not(labels)] = falseval
#     labels = string_labels
    
#     dump([features, labels], 'features_eq.tmp') 
  
    return (features, labels)
    
  
  def fit(self, features, labels, grid_search=False):
    '''
    Fits the classifier. 
    Can use gridsearch for finding best parameters.
    '''
    if grid_search:
      self.clf = self.grid_search(features, labels)
      return
    
    
    # fit on k folds to find best theshold and confidence
    n_folds = 4
    self.my_print('Running {} folds to find best threshold and confidence of classifier.'.format(n_folds))
    eer = [] 
    skf = sk.cross_validation.StratifiedKFold(labels, 
                                              n_folds=n_folds)
    clfs = []
    stats = pd.DataFrame(index = ['True', 'False'], columns=['True', 'False'], dtype=np.float32)
    stats[:] = 0
    for train_index, test_index in skf:
      self.clf.fit(features[train_index,:], labels[train_index])
      clfs.append(np.copy(self.clf))
      responses = self.clf.decision_function(features[test_index,:])
      curr_eer = util.find_equal_err_rate(labels[test_index], responses)
      eer.append(curr_eer)
      pred = responses > curr_eer
      l = labels[test_index]
      stats.loc['True', 'True'] = stats.loc['True', 'True'] + np.sum(np.logical_and(pred, l))
      stats.loc['True', 'False'] = stats.loc['True', 'False'] + np.sum(np.logical_and(pred, np.logical_not(l)))
      stats.loc['False','True'] = stats.loc['False', 'True'] + np.sum(np.logical_and(np.logical_not(pred), l))
      stats.loc['False', 'False'] = stats.loc['False', 'False'] + np.sum(np.logical_and(np.logical_not(pred), np.logical_not(l)))
      print stats
      
    self.my_print("Equal Error Rates: {}".format(eer))
    self.thresh = np.array(eer).mean()
    self.my_print('selected: {}'.format(self.thresh))
     
    sk.preprocessing.normalize(stats ,axis=1,norm='l1')
    self.my_print(''.format(stats))
      
  
  
  
  
  
  
  
  def grid_search(self, features, labels):
    raise Exception('Check this code, it is old.')
  
    # Set the parameters by cross-validation
    tuned_parameters_SVC = [
#                           {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                            'C': [1, 10, 100, 1000], 
# #                            'class_weight': ['auto']
#                            },
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000],
                          'class_weight': ['auto']
                         }
                        ]
    
    tuned_parameters_LinearSVC = [{'penalty': ['l2', 'l1'],
                         'C': [1e-4, 1e-3, 1e-2, 1e-1, 1],
                         'class_weight': ['auto']}]
    
    tuned_parameters_GradientBoosting = [{'n_estimators': [100, 1000],
                         'learning_rate': [1, 0.1, 0.01],
                         'max_depth': [1, 2, 3]}]

    tuned_parameters_RandomForest = [{'n_estimators': [100, 200, 1000],
                                      'max_depth': [1, 10, 20],
                                      'min_samples_split': [1, 2, 5]}]

    
    print("# Tuning hyper-parameters")
    print('')
  
#       clf = GridSearchCV(SVC(C=1), tuned_parameters_SVC, cv=5, scoring='precision',
#                           n_jobs=self.n_cores,
#                           verbose=3)
#       clf = GridSearchCV(GradientBoostingClassifier(), tuned_parameters_GradientBoosting, cv=5, scoring='precision',
#                           n_jobs=self.n_cores,
#                           verbose=1)

#       clf = GridSearchCV(RandomForestClassifier(), tuned_parameters_RandomForest, 
#                          cv=5, scoring='precision',
#                           n_jobs=self.n_cores,
#                           verbose=3)
    clf = GridSearchCV(LinearSVC(C=1, dual=False), 
                       tuned_parameters_LinearSVC, cv=5, scoring='precision',
                        n_jobs=self.n_cores,
                        verbose=3)
    clf.fit(features, labels)
  
    print("Best parameters set found on development set:")
    print()
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print()
    
    self.clf = clf.best_estimator_
    self.fit(features, labels)
    
    
  def my_print(self, s):
    print "AttributeCLassifier(" + self.name + "):" + s
    
  def run_training_pipeline(self, features=None, grid_search=False):
    """ The full sequence of operations that trains an attribute classifier"""
    
    
    self.my_print("Loading feature-word histograms from disk, and creating " + 
                  "matrix for attribute classification.")
    (features, labels) = self.create_feature_matrix(features)
    
    self.my_print("Training classifier")
    self.fit(features, labels, grid_search=grid_search)
    
    
  def predict(self, features):
    return self.decision_function(features) > self.thresh
#     return self.clf.predict(self.Scaler.transform(features))
  
  def decision_function(self, features, use_prob=False):
    f = self.Scaler.transform(features)
    import sklearn
    if (use_prob and self.probability) or type(self.clf) == sklearn.ensemble.forest.RandomForestClassifier:
      return self.clf.predict_proba(f)[:,1]
    return self.clf.decision_function(f)
    
  
    
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
    dump(attrib_classifier, filename, compress=3)
      
  @staticmethod    
  def load(filename):
    return load(filename)
    
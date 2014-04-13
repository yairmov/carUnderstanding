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
from sklearn.ensemble.forest import RandomForestClassifier
import numpy as np
import os

import Bow as Bow


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
    
#     self.clf          = Pipeline([('Scaler', Scaler()), 
#                                   ('Classifier', SVC(kernel='linear',
#                                                      class_weight='auto',
#                                                      probability=True))])
    
  
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
        hist = Bow.load(hist_filename)
        if type(hist) == tuple:
          hist = hist[0]
        features[ii, :] = hist
  
  
    # create pos/neg labels
    labels = self.dataset.index.isin(self.pos_img_inds)
    
    
    # preprocess features
#     features = sk.preprocessing.scale(features)
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
    if not grid_search:
      self.clf.fit(features, labels)
    else:
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
#       clf = GridSearchCV(LinearSVC(C=1, dual=False), tuned_parameters_LinearSVC, cv=5, scoring='precision',
#                           n_jobs=self.n_cores,
#                           verbose=3)
#       clf = GridSearchCV(GradientBoostingClassifier(), tuned_parameters_GradientBoosting, cv=5, scoring='precision',
#                           n_jobs=self.n_cores,
#                           verbose=1)

      clf = GridSearchCV(RandomForestClassifier(), tuned_parameters_RandomForest, 
                         cv=5, scoring='precision',
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
    
    
  def cross_validate(self, features, labels):
    self.my_print("Num pos examples: {}".format(np.sum(labels)))
    cv = 5
    score_method = 'precision'
#     score_method = 'accuracy'
#     cv = sk.cross_validation.StratifiedKFold(labels, 3)
    scores = sk.cross_validation.cross_val_score(self.clf, 
                                                 features, 
                                                 labels,
                                                 n_jobs=min(cv, self.n_cores), 
                                                 verbose=1,
                                                 cv=cv,
                                                 scoring=score_method)
    
    # Report results
    self.my_print("scores: {}".format(scores))
    self.my_print(score_method + ": %0.2f (+/- %0.2f)" % 
                  (scores.mean(), scores.std() * 2))
    rand_pred = np.random.choice([True, False], size=labels.shape)
    self.my_print("Random (precision): {}".format(
                               metrics.precision_score(labels, rand_pred)))
    
  def my_print(self, s):
    print "AttributeCLassifier(" + self.name + "):" + s
    
  def run_training_pipeline(self, cv=False, grid_search=False):
    """ The full sequence of operations that trains an attribute classifier"""
    
    self.my_print("Loading feature-word histograms from disk, and creating " + 
                  "matrix for attribute classification.")
    (features, labels) = self.create_feature_matrix()
    
    if cv:
      self.my_print("Training classifier [Cross Validation]")
      self.cross_validate(features, labels)
#     else:
    self.my_print("Training classifier")
    self.fit(features, labels, grid_search=grid_search)
    
    
  def predict(self, features):
    return self.clf.predict(features)
  
  def decision_function(self, features, use_prob=True):
    features = self.Scaler.transform(features)
    import sklearn
    if (use_prob and self.probability) or type(self.clf) == sklearn.ensemble.forest.RandomForestClassifier:
      return self.clf.predict_proba(features)[:,1]
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
    dump(attrib_classifier, filename, compress=3)
      
  @staticmethod    
  def load(filename):
    return load(filename)
    
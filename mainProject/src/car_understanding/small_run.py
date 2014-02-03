'''
Created on Jan 16, 2014

@author: ymovshov
'''

import os as os
# import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn import preprocessing
from sklearn.externals.joblib import Parallel, delayed
import cv2 as cv

from configuration import get_config
import fgcomp_dataset_utils as fgu
from dense_SIFT import dense_SIFT, save_to_disk, load_from_disk
import Bow
from boto import config

def preprocess():
  config = get_config()
  (train_annos, class_meta, domain_meta) = fgu.get_all_metadata(config)

  # Filter the class meta and train annotations according to the small use
  # case definitions
  class_meta = class_meta[class_meta['domain_index'] == config.dataset.domains[0]]
  train_annos = train_annos[
                            train_annos.class_index.isin(
                            config.dataset.class_ids.pos +
                            config.dataset.class_ids.neg)
                            ]
  return ({'train_annos': train_annos,
             'class_meta': class_meta,
             'domain_meta': domain_meta},
          config)




def calc_dense_SIFT_one_img(annotation, config):
  rel_path = annotation['rel_path']
  img_file = os.path.join(config.dataset.main_path, rel_path)
  img = cv.imread(img_file)
  (kp, desc) = dense_SIFT(img, grid_spacing=config.SIFT.grid_spacing)
  save_to_disk(kp, desc, img_file, config.SIFT.raw_dir)


def calc_dense_SIFT_on_dataset(dataset, config):
  '''
  Just calles calc_dense_SIFT_one_img on all images in dataset using a
  parallel wrapper.
  '''
  train_annos = dataset['train_annos']

  Parallel(n_jobs=-1, verbose=2)(
                 delayed(calc_dense_SIFT_one_img)(train_annos.iloc[ii], config)
                 for ii in range(len(train_annos)))

def contains(box, point):
  '''
  box = (xmin, xmax, ymin, ymax)
  point = (x, y)
  '''
  return (box[0] <= point[0] and box[1] >= point[0] and
          box[2] <= point[1] and box[3] >= point[1])



def load_SIFT_from_a_file(curr_anno, config):
  curr_file = os.path.splitext(curr_anno['basename'])[0] + '.dat'
  (kp, desc) = load_from_disk(os.path.join(config.SIFT.raw_dir, curr_file))

  # Only keep points that are inside the bounding box
  box = (curr_anno['xmin'], curr_anno['xmax'],
         curr_anno['ymin'], curr_anno['ymax'])

  inds = np.zeros(shape=[len(kp),], dtype=bool)
  for jj in range(len(kp)):
    inds[jj] = contains(box, kp[jj].pt)

  desc = desc[inds, :]
  kp = np.asarray(kp)[inds].tolist()

  # Random selection of a subset of the keypojnts/descriptors
  inds  = np.random.permutation(desc.shape[0])
  desc = desc[inds, :]
  desc = desc[:config.SIFT.BoW.max_desc_per_img, :]
#   kp    = [kp[i] for i in inds]
#   kp    = kp[:config.SIFT.BoW.max_desc_per_img]

  return desc

def load_SIFT_from_files(dataset, config):
  train_annos = dataset['train_annos']

  nfiles = len(train_annos)
  print 'Loading dense SIFT for %d training images ' % nfiles
  features = Parallel(n_jobs=-1, verbose=2)(
                 delayed(load_SIFT_from_a_file)(train_annos.iloc[ii], config)
                 for ii in range(nfiles))

  # convert to numy arry
  features = np.concatenate(features)
  return features




def create_feature_matrix(dataset, config):
  train_annos = dataset['train_annos']

  # Preallocate feature matrix
  features = np.empty(shape=[len(train_annos), config.SIFT.BoW.num_clusters])

  # Load histograms from disk into a matrix
  print 'Loading histograms from disk'
  for ii in range(len(train_annos)):
    img_name = train_annos.iloc[ii]['basename']
    img_name = os.path.splitext(img_name)[0]
    hist_filename = os.path.join(config.SIFT.BoW.hist_dir, img_name) + '_hist.dat'
    hist = Bow.load(hist_filename) # hist[0] = values, hist[1] = bin edges
    features[ii, :] = hist[0]


  # preprocess features
  features = preprocessing.scale(features)

  # create pos/neg labels
  print 'Creating pos/neg labels'
  labels = train_annos.class_index.isin(
                    config.dataset.class_ids.pos).values

  return (features, labels)

def evaluate(features, labels):
  from sklearn.cross_validation import train_test_split
  from sklearn.grid_search import GridSearchCV
  from sklearn.metrics import classification_report

#   # Split the train_annos in two parts
#   X_train, X_test, y_train, y_test = train_test_split(
#     features, labels, test_size=0.3, random_state=0)
#
#
#   # Set the parameters by cross-validation
#   tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                        'C': [1, 10, 100, 1000]},
#                       {'kernel': ['linear'], 'C': [0.005, 1, 10, 100, 1000]}]
#
#   scores = ['precision', 'recall']
#
#   for score in scores:
#       print("# Tuning hyper-parameters for %s" % score)
#       print()
#
#       clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5, scoring=score)
#       clf.fit(X_train, y_train)
#
#       print("Best parameters set found on development set:")
#       print()
#       print(clf.best_estimator_)
#       print()
#       print("Grid scores on development set:")
#       print()
#       for params, mean_score, scores in clf.grid_scores_:
#           print("%0.3f (+/-%0.03f) for %r"
#                 % (mean_score, scores.std() / 2, params))
#       print()
#
#       print("Detailed classification report:")
#       print()
#       print("The model is trained on the full development set.")
#       print("The scores are computed on the full evaluation set.")
#       print()
#       y_true, y_pred = y_test, clf.predict(X_test)
#       print(classification_report(y_true, y_pred))
#       print()


########

  # SVM params
  print 'setting classifier parameters'
#   clf = svm.SVC(kernel='linear', C=0.0005)
  clf = svm.SVC(kernel='rbf', C=10, gamma=0.0001)
#   from sklearn.ensemble import AdaBoostClassifier
#   clf = AdaBoostClassifier(svm.SVC(kernel='linear', C=0.005),
#                            algorithm="SAMME",
#                          n_estimators=10)
  scores = cross_validation.cross_val_score(clf, features, labels, cv=5)

  # Report results
  print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

if __name__ == '__main__':
    (dataset, config) = preprocess()

    #  RUN dense SIFT on alll images
    print "Saving Dense SIFT to disk"
    calc_dense_SIFT_on_dataset(dataset, config)

    # Create BoW model
    features = load_SIFT_from_files(dataset, config)
    print "Loaded %d SIFT features from disk" % features.shape[0]
    print "K-Means CLustering"
    Bow.create_BoW_model(features, config.SIFT.BoW.model_file)

    # Assign cluster labels to all images
    print "Assigning to histograms"
    Bow.create_word_histograms_on_dataset(dataset['train_annos'], config)

    # Extract final features
    print "Building training matrix for classifier"
    (features, labels) = create_feature_matrix(dataset, config)

    # Evaluate
    print "Cross Validation on classifier"
    evaluate(features, labels)



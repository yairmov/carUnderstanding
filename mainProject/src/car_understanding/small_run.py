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
import cv2 as cv

from configuration import get_config
import fgcomp_dataset_utils as fgu
from dense_SIFT import dense_SIFT, save_to_disk, load_from_disk
import Bow

def preprocess():
  config = get_config()

  # Adding pos/neg class definitions to the config
  config.dataset.class_ids.neg = [188, 190, 196, 207, 213] # not SUV
  config.dataset.class_ids.pos = [184, 220, 231, 235, 303] # SUV

  (dataset, class_meta, domain_meta) = fgu.get_all_metadata(config)

  # Filter the class meta and train annotations according to the small use
  # case definitions
  class_meta = class_meta[class_meta['domain_index'] == config.dataset.domains[0]]
  dataset = dataset[
                            dataset.class_index.isin(
                            config.dataset.class_ids.pos +
                            config.dataset.class_ids.neg)
                            ]
  return ({'dataset': dataset,
             'class_meta': class_meta,
             'domain_meta': domain_meta},
          config)




def calc_dense_SIFT_on_dataset(dataset, config):
  dataset = dataset['dataset']
  for row_tuple in dataset.iterrows():
    # row_tuple[0]=index row_tuple[1]=data
    row = row_tuple[1]
    rel_path = row['rel_path']
    img_file = os.path.join(config.dataset.main_path, rel_path)

    # Read image and resize such that bounding box is of specific size
    print 'Resize img such that BB is of width = ' + str(config.bb_width)
    img = cv.imread(img_file)
    img = set_width_to_normalize_bb(img, row['xmin'], 
                                    row['xmax'], config.bb_width)

    print 'Extracting dense-SIFT from image:', img_file, '...',
    (kp, desc) = dense_SIFT(img, grid_spacing=config.SIFT.grid_spacing)
    print 'Done.'
    print 'Saving dense-SIFT to folder: "', config.SIFT.raw_dir, '" ...',
    save_to_disk(kp, desc, img_file, config.SIFT.raw_dir)
    print 'Done.'


def load_SIFT_from_files(dir_path):
  config = get_config()
  files = os.listdir(dir_path)

  # Create a pandas dataFrame to hold all the features. Will enter more details later
#   features = pd.DataFrame()
  features = np.empty(shape=[0,128])

  for ii in range(len(files)):
    curr_file = files[ii]
    print 'loading data from {} ({}/{})'.format(curr_file, ii+1, len(files))
    (kp, desc) = load_from_disk(os.path.join(dir_path, curr_file))

    # Random selection of a subset of the keypojnts/descriptors
    inds  = np.random.permutation(desc.shape[0])
    desc = desc[inds, :]
    desc = desc[:config.SIFT.BoW.max_desc_per_img, :]
#     kp    = [kp[i] for i in inds]
#     kp    = kp[:config.SIFT.BoW.max_desc_per_img]

    # Add descriptors/Keypoints to features
    features = np.concatenate([features, desc])

  return features




def create_feature_matrix(dataset, config):
  dataset = dataset['dataset']

  # Preallocate feature matrix
  features = np.empty(shape=[len(dataset), config.SIFT.BoW.num_clusters])

  # Load histograms from disk into a matrix
  print 'Loading histograms from disk'
  for ii in range(len(dataset)):
    img_name = dataset.iloc[ii]['basename']
    img_name = os.path.splitext(img_name)[0]
    hist_filename = os.path.join(config.SIFT.BoW.hist_dir, img_name) + '_hist.dat'
    hist = Bow.load(hist_filename) # hist[0] = values, hist[1] = bin edges
    features[ii, :] = hist[0]


  # preprocess features
  features = preprocessing.scale(features)

  # create pos/neg labels
  print 'Creating pos/neg labels'
  labels = dataset.class_index.isin(
                    config.dataset.class_ids.pos).values

  return (features, labels)

def evaluate(features, labels):
  from sklearn.cross_validation import train_test_split
  from sklearn.grid_search import GridSearchCV
  from sklearn.metrics import classification_report

  # Split the dataset in two equal parts
  X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=0)


  # Set the parameters by cross-validation
  tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                       'C': [1, 10, 100, 1000]},
                      {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

  scores = ['precision', 'recall']

  for score in scores:
      print("# Tuning hyper-parameters for %s" % score)
      print()

      clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5, scoring=score)
      clf.fit(X_train, y_train)

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

      print("Detailed classification report:")
      print()
      print("The model is trained on the full development set.")
      print("The scores are computed on the full evaluation set.")
      print()
      y_true, y_pred = y_test, clf.predict(X_test)
      print(classification_report(y_true, y_pred))
      print()


########

  # SVM params
#   print 'setting classifier parameters'
#   clf = svm.SVC(kernel='linear', C=4)
#   from sklearn.ensemble import AdaBoostClassifier
#   clf = AdaBoostClassifier(svm.SVC(kernel='linear', C=4),
#                            algorithm="SAMME",
#                          n_estimators=200)
#   scores = cross_validation.cross_val_score(clf, features, labels, cv=5)

  # Report results
#   print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

if __name__ == '__main__':
    (dataset, config) = preprocess()

    #  RUN dense SIFT on alll images
#     calc_dense_SIFT_on_dataset(dataset, config)

    # Create BoW model
#     features = load_SIFT_from_files(config.SIFT.raw_dir)
#     Bow.create_BoW_model(features, config.SIFT.BoW.model_file)

    # Assign cluster labels to all images
#     Bow.create_word_histograms_on_dir(config)

    # Extract final features
    (features, labels) = create_feature_matrix(dataset, config)

    # Evaluate
    evaluate(features, labels)



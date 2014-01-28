'''
functions for BoW related functionality

Created on Jan 19, 2014

@author: ymovshov
'''


import os as os
import pandas as pd
import numpy as np
# import sklearn as sk
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.externals import joblib
from sklearn import preprocessing

# Import my code
from configuration import get_config
# import fgcomp_dataset_utils as fgu
import dense_SIFT


# Cluster features to create the 'words'
def cluster_to_words(features, config=None):
  if config is None:
    config = get_config()

    # Create clustering estimator

    # KMEANS
#     estimator = KMeans(init='k-means++',
#                           n_clusters=config.SIFT.BoW.num_clusters,
#                           n_init=3*config.SIFT.BoW.num_clusters, verbose=True)


    # Mini batch KMEANS
    batch_size = int(np.round(float(config.SIFT.BoW.num_clusters) / 10))
    estimator = MiniBatchKMeans(init='k-means++',
                            n_clusters=config.SIFT.BoW.num_clusters,
                            batch_size=batch_size,
                            max_no_improvement=10,
                            verbose=True)


    # DBSCAN
#     estimator = DBSCAN(eps=0.3, min_samples=10)

    # normalize SIFT features
    features = normalize_features(features)

    # Cluster features
    print "Clustering features into {} clusters".format(estimator.n_clusters)
    estimator.fit(features)

    return estimator


def save(model, filename):
  (dirname, name) = os.path.split(filename)
  if not os.path.isdir(dirname):
    os.makedirs(dirname)

  joblib.dump(model, filename)

def load(filename):
  return joblib.load(filename)


def create_BoW_model(features, model_file):
  bow_model = cluster_to_words(features)
  save(bow_model, model_file)


# Assign each features vector in features (row) to a cluster center from
# bow_model, and return count histogram
def word_histogram(features, bow_model, config=None):
  if config is None:
    config = get_config()

  # normalize SIFT features
  features = normalize_features(features)

  word_ids = bow_model.predict(features)
  return np.histogram(word_ids, bins=config.SIFT.BoW.num_clusters,
                      range=[0, config.SIFT.BoW.num_clusters], density=True)


def normalize_features(features):
  return preprocessing.normalize(features, norm='l1')


def create_word_histograms_on_dir(config=None):
  if config is None:
    config = get_config()

  bow_model = load(config.SIFT.BoW.model_file)

  dir_path = config.SIFT.raw_dir
  sift_files = os.listdir(dir_path)
  for ii in range(len(sift_files)):
    curr_sift_file = sift_files[ii]
    print 'loading data from {} ({}/{})'.format(curr_sift_file, ii+1, len(sift_files))
    (kp, desc) = dense_SIFT.load_from_disk(os.path.join(dir_path, curr_sift_file))

    hist = word_histogram(desc, bow_model)

    # save histogram
    (name, ext) = os.path.splitext(os.path.split(curr_sift_file)[1])
    hist_file_name = os.path.join(config.SIFT.BoW.hist_dir, name + '_hist.dat')
    print 'Saving histogram to {} ({}/{})'.format(hist_file_name, ii+1, len(sift_files))
    save(hist, hist_file_name)


if __name__ == "__main__":
  features = load_SIFT_from_files('/usr0/home/ymovshov/Documents/Research/Code/car_understanding/SIFT/raw')
  bow_model = cluster_to_words(features)
  from sklearn import metrics
  labels = bow_model.labels_
  print metrics.silhouette_score(features, labels, metric='euclidean')






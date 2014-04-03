'''
functions for BoW related functionality

Created on Jan 19, 2014

@author: ymovshov
'''


import os as os
import pandas as pd
import numpy as np
# import sklearn as sk
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.externals.joblib import Parallel, delayed

# Import my code
# from configuration import get_config
# import fgcomp_dataset_utils as fgu
import dense_SIFT
from util import ProgressBar


# Cluster features to create the 'words'
def cluster_to_words(features, config):

    # Create clustering estimator

    # KMEANS
#     estimator = KMeans(init='k-means++',
#                           n_clusters=config.SIFT.BoW.num_clusters,
#                           n_init=10, verbose=True, n_jobs=-2, tol=1e-3)


    # Mini batch KMEANS
    batch_size = config.SIFT.BoW.num_clusters * 10
    estimator = MiniBatchKMeans(init='k-means++',
                            n_clusters=config.SIFT.BoW.num_clusters,
                            batch_size=batch_size,
                            tol=0.001,
                            init_size=10*config.SIFT.BoW.requested_n_clusters,
                            n_init = 10,
                            verbose=True)


    # normalize SIFT features
    # features = normalize_features(features)

    # Cluster features
    print "Clustering features into {} clusters".format(estimator.n_clusters)
    estimator.fit(features)

    # Drop duplicate clusters (usually empty clusters)
    clusters = pd.DataFrame(data=estimator.cluster_centers_)
    clusters.drop_duplicates(inplace=True)
    estimator.cluster_centers_ = np.array(clusters)
    estimator.n_clusters = clusters.shape[0]

    # Update config to show new number of clusters
    config.SIFT.BoW.num_clusters = estimator.n_clusters

    return estimator


def save(model, filename):
  (dirname, name) = os.path.split(filename)
  if not os.path.isdir(dirname):
    os.makedirs(dirname)

  joblib.dump(model, filename)

def load(filename):
  return joblib.load(filename)


def create_BoW_model(features, config):
  bow_model = cluster_to_words(features, config)
  return bow_model


# Assign each features vector in features (row) to a cluster center from
# bow_model, and return count histogram
def word_histogram(features, bow_model, config):
  from llc import LLC_encoding

  # normalize SIFT features
  # features = normalize_features(features)


  # using LLC encoding
  if config.SIFT.LLC.use:
    codebook = bow_model.cluster_centers_
    encoding = LLC_encoding(codebook, features, config.SIFT.LLC.knn, config.SIFT.LLC.beta)

    # use max pooling for LLC
    hist = encoding.max(axis=0)


  else:

    word_ids = bow_model.predict(features)
    hist = np.histogram(word_ids, bins=config.SIFT.BoW.num_clusters,
                        range=[0, config.SIFT.BoW.num_clusters], density=True)
    hist = hist[0]

  return hist



def normalize_features(features):
  return preprocessing.normalize(features, norm='l1')


def create_word_histogram_on_file(raw_feature_file, bow_model, config):
#   logfile = os.path.join('tmp', os.path.basename(raw_feature_file))
#   with open(logfile, 'w') as f:
#     f.write('Start\n')
  (kp, desc) = dense_SIFT.load_from_disk(raw_feature_file)
  hist = word_histogram(desc, bow_model, config)
  (name, ext) = os.path.splitext(os.path.split(raw_feature_file)[1])
  hist_filename = os.path.join(config.SIFT.BoW.hist_dir, name + '_hist.dat')
  save(hist, hist_filename)
#   with open(logfile, 'w') as f:
#     f.write('Done\n')

def create_word_histograms_on_dataset(train_annos, config):
  bow_model = load(config.SIFT.BoW.model_file)

  dir_path = config.SIFT.raw_dir
  n_files = train_annos.shape[0]

  if not os.path.isdir(config.SIFT.BoW.hist_dir):
    os.makedirs(config.SIFT.BoW.hist_dir)

  Parallel(n_jobs=-1, verbose=config.logging.verbose)(
                 delayed(create_word_histogram_on_file)(
                 os.path.join(dir_path,
                              os.path.splitext(train_annos.iloc[ii]['basename'])[0] + '.dat'),
                 bow_model,
                 config)
                 for ii in range(n_files))

#   for ii in range(n_files):
#   for ii in [13714]:
#     print ii
#     create_word_histogram_on_file(os.path.join(dir_path,
#                                 os.path.splitext(train_annos.loc[ii]['basename'])[0] + '.dat'),
#                                 bow_model,
#                                 config)


def load_bow(data_annos, config):
  features = np.empty(shape=[len(data_annos), config.SIFT.BoW.num_clusters])
  progress = ProgressBar(len(data_annos))
  for ii in range(len(data_annos)):
    img_name = data_annos.iloc[ii]['basename']
    img_name = os.path.splitext(img_name)[0]
    hist_filename = os.path.join(config.SIFT.BoW.hist_dir,
                                 img_name) + '_hist.dat'
    hist = load(hist_filename)
    features[ii, :] = hist
    progress.animate(ii)

  return features

if __name__ == "__main__":
  print 'la'
  #   features = load_SIFT_from_files('/usr0/home/ymovshov/Documents/Research/Code/car_understanding/SIFT/raw')
  #   bow_model = cluster_to_words(features)
  #   from sklearn import metrics
  #   labels = bow_model.labels_
  #   print metrics.silhouette_score(features, labels, metric='euclidean')

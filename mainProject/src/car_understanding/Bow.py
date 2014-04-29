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
from path import path

# Import my code
import dense_SIFT
from util import ProgressBar
from dense_SIFT import load_from_disk
import configuration
from feature_pooling import SpatialPooler
from llc import LLC_encoding

def fit_model(train_annos, config):
  features = load_sift(train_annos, config)
  print "Loaded %d SIFT features from disk" % features.shape[0]
  print "K-Means CLustering"
  return cluster_to_words(features, config)
  


def contains(box, point):
  '''
  box = (xmin, xmax, ymin, ymax)
  point = (x, y)
  '''
  return (box[0] <= point[0] and box[1] >= point[0] and
          box[2] <= point[1] and box[3] >= point[1])



# def load_SIFT_from_a_file(curr_anno, config):
#   curr_file = os.path.splitext(curr_anno['basename'])[0] + '.dat'
#   (kp, desc) = load_from_disk(os.path.join(config.SIFT.raw_dir, curr_file),
#                               matlab_version=True)
# 
#   # Only keep points that are inside the bounding box
#   box = (curr_anno['xmin'], curr_anno['xmax'],
#          curr_anno['ymin'], curr_anno['ymax'])
# 
# #   inds = np.zeros(shape=[len(kp),], dtype=bool)
# #   for jj in range(len(kp)):
# #     inds[jj] = contains(box, kp[jj].pt)
#   
#   n_pts = int(kp.shape[0])
#   inds = np.zeros(shape=[n_pts,], dtype=bool)
#   for jj in range(n_pts):
#     inds[jj] = contains(box, kp[jj,:2])
# 
#   desc = desc[inds, :]
# #   kp = np.asarray(kp)[inds].tolist()
#   kp = kp[inds,:]
# 
#   # Random selection of a subset of the keypojnts/descriptors
#   # Select one patch size and get all descriptors from it
#   patch_size = np.random.choice(np.unique(kp[:,-1]), 1)[0]
#   inds = np.where(kp[:,-1] == patch_size)[0]
#   desc = desc[inds,:]
#   kp = kp[inds,:]
#   
#   
# #   inds  = np.random.permutation(desc.shape[0])
# #   desc = desc[inds, :]
# #   desc = desc[:config.SIFT.BoW.max_desc_per_img, :]
# # #   kp    = [kp[i] for i in inds]
# # #   kp    = kp[:config.SIFT.BoW.max_desc_per_img]
# 
#   return desc
# 
# def load_SIFT_from_files(train_annos, config):
# 
#   nfiles = len(train_annos)
#   print 'Loading dense SIFT for %d training images ' % nfiles
#   features = Parallel(n_jobs=config.n_cores, verbose=config.logging.verbose)(
#                  delayed(load_SIFT_from_a_file)(train_annos.iloc[ii], config)
#                  for ii in range(nfiles))
# 
# #   features = []
# #   pbar = ProgressBar(nfiles)
# #   for ii in range(nfiles):
# #     pbar.animate(ii)
# #     features.append(load_SIFT_from_a_file(train_annos.iloc[ii], config))
# 
#   # convert to numy arry
#   features = np.concatenate(features)
# 
#   # sample max_desc features
#   inds  = np.random.permutation(features.shape[0])
#   features = features[inds, :]
#   features = features[:config.SIFT.BoW.max_desc_total, :]
# 
#   return features  


def load_sift_from_file(sift_filename, max_num_desc=None):
  (kp, desc) = load_from_disk(sift_filename, matlab_version=True)
  
  
  # Randomly select a scale and get all the descriptors from it
  p_size = np.random.choice(np.unique(kp[:,-1]))
  inds = kp[:,-1] == p_size
  kp = kp[inds, :]
  desc = desc[inds, :]
  
  # Random selection of a subset of the descriptors
  inds  = np.random.permutation(desc.shape[0])
  desc = desc[inds, :]
  desc = desc[:max_num_desc, :]
  
  return desc.astype(np.float32)
  
  
def load_sift(data_annos, config):
  nfiles = len(data_annos)
  fnames = data_annos.basename.apply(lambda x: path(x).splitext()[0] + '.dat')
  fnames = list(fnames.apply(lambda x: path(config.SIFT.raw_dir).joinpath(x)))
  print 'Loading dense SIFT from %d images ' % nfiles
  features = Parallel(n_jobs=-1, verbose=config.logging.verbose)(
                 delayed(load_sift_from_file)(fnames[ii], config.SIFT.BoW.max_desc_per_img)
                 for ii in range(nfiles))

#   features = []
#   for ii in range(nfiles):
#     features.append(load_sift_from_file(fnames[ii], config.SIFT.BoW.max_desc_per_img))
  
  features = np.concatenate(features)
  # sample max_desc features
  inds  = np.random.permutation(features.shape[0])
  features = features[inds[:config.SIFT.BoW.max_desc_total], :]

  return features

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
                            n_clusters=config.SIFT.BoW.requested_num_clusters,
                            batch_size=batch_size,
                            tol=0.001,
                            init_size=10*config.SIFT.BoW.requested_num_clusters,
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
    configuration.update_config(config, 
                                'SIFT.BoW.num_clusters', 
                                estimator.n_clusters)
#     config.SIFT.BoW.num_clusters = estimator.n_clusters
    

    return estimator


def save(model, filename):
  (dirname, name) = os.path.split(filename)
  if not os.path.isdir(dirname):
    os.makedirs(dirname)

  joblib.dump(model, filename)

def load(filename):
  return joblib.load(filename)


# def create_BoW_model(features, config):
#   bow_model = cluster_to_words(features, config)
#   return bow_model


# Assign each features vector in features (row) to a cluster center from
# bow_model, and return count histogram.
# Can also apply LLC enocoding to use more than one cluster center. 
def word_histogram(features, locations, bow_model, config):

  # normalize SIFT features
  # features = normalize_features(features)


  # using LLC encoding
  if config.SIFT.LLC.use:
    codebook = bow_model.cluster_centers_
    encoding = LLC_encoding(codebook, features, config.SIFT.LLC.knn, config.SIFT.LLC.beta)

    # use max pooling for LLC
#     hist = encoding.max(axis=0)
    
    spatial_poolers = [SpatialPooler(x) for x in config.SIFT.pool_boxes] 

    pooled_enc = [sp.features_to_pool(locations, encoding).max(axis=0) 
             for sp in spatial_poolers]
        
    hist = np.concatenate(pooled_enc, axis=0)

  else:

    print('NO SPATIAL POOLING IMPLEMENTED FOR SIMPLE BAG OF WORDS!!!')
    word_ids = bow_model.predict(features)
    hist = np.histogram(word_ids, bins=config.SIFT.BoW.num_clusters,
                        range=[0, config.SIFT.BoW.num_clusters], density=True)
    hist = hist[0]

  return hist



def normalize_features(features):
  return preprocessing.normalize(features, norm='l1')


def create_word_histogram_on_file(raw_feature_file, bow_model, config):
  (name, ext) = os.path.splitext(os.path.split(raw_feature_file)[1])
  hist_filename = os.path.join(config.SIFT.BoW.hist_dir, name + '_hist.dat')
  
  (frames, desc) = dense_SIFT.load_from_disk(raw_feature_file,
                                         matlab_version=True)
  hist = word_histogram(desc, frames[:,:2], bow_model, config)
  save(hist, hist_filename)

def create_word_histograms_on_dataset(data_annos, config):
  bow_model = load(config.SIFT.BoW.model_file)

  dir_path = config.SIFT.raw_dir
  n_files = data_annos.shape[0]

#   if not os.path.isdir(config.SIFT.BoW.hist_dir):
#     os.makedirs(config.SIFT.BoW.hist_dir)

  Parallel(n_jobs=config.n_cores, verbose=config.logging.verbose)(
                 delayed(create_word_histogram_on_file)(
                 os.path.join(dir_path,
                              os.path.splitext(data_annos.iloc[ii]['basename'])[0] + '.dat'),
                 bow_model,
                 config)
                 for ii in range(n_files))

#   for ii in range(n_files):
#     print ii
#     create_word_histogram_on_file(os.path.join(dir_path,
#                                 os.path.splitext(data_annos.iloc[ii]['basename'])[0] + '.dat'),
#                                 bow_model,
#                                 config)


def load_bow(data_annos, config):
  features = np.empty(shape=[len(data_annos), 
                             config.SIFT.BoW.num_clusters * 
                             len(config.SIFT.pool_boxes)])
  progress = ProgressBar(len(data_annos))
  for ii in range(len(data_annos)):
    img_name = data_annos.iloc[ii]['basename']
    img_name = os.path.splitext(img_name)[0]
    hist_filename = os.path.join(config.SIFT.BoW.hist_dir,
                                 img_name) + '_hist.dat'
    hist = load(hist_filename)
    features[ii, :] = hist
    progress.animate(ii)
  print('')

  return features

if __name__ == "__main__":
  print 'la'
  #   features = load_SIFT_from_files('/usr0/home/ymovshov/Documents/Research/Code/car_understanding/SIFT/raw')
  #   bow_model = cluster_to_words(features)
  #   from sklearn import metrics
  #   labels = bow_model.labels_
  #   print metrics.silhouette_score(features, labels, metric='euclidean')

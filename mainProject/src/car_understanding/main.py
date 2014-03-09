'''
Created on Mar 3, 2014

@author: ymovshov
'''

import os
import cv2 as cv
from sklearn.externals.joblib import Parallel, delayed, dump, load
import numpy as np


import fgcomp_dataset_utils as fgu
from configuration import get_config
from dense_SIFT import dense_SIFT, save_to_disk, load_from_disk
from SIFT_feature_extractor import SiftFeatureExtractor


def load_SIFT_from_a_file(curr_anno, config):
  
  def contains(box, point):
    '''
    box = (xmin, xmax, ymin, ymax)
    point = (x, y)
    '''
    return (box[0] <= point[0] and box[1] >= point[0] and
            box[2] <= point[1] and box[3] >= point[1])
  
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
  features = Parallel(n_jobs=-1, verbose=config.logging.verbose)(
                 delayed(load_SIFT_from_a_file)(train_annos.iloc[ii], config)
                 for ii in range(nfiles))

  # convert to numy arry
  features = np.concatenate(features)

  # sample max_desc features
  inds  = np.random.permutation(features.shape[0])
  features = features[inds, :]
  features = features[:config.SIFT.BoW.max_desc_total, :]

  return features


def extract_features(dataset, config):
  # SIFT
  sift_extractor = SiftFeatureExtractor(config)
  sift_extractor.calc_dense_SIFT_on_dataset(dataset, config)
  

  

if __name__ == '__main__':
  # Load config
  makes = ['bmw', 'ford']
  types = ['sedan', 'SUV']
  args = makes + types
  config = get_config(args)
  (dataset, config) = fgu.get_all_metadata(config)
  
  # extract all features
  print "Saving raw features to disk"
  extract_features(dataset, config)
  
  # load features
  print "Loading raw features from disk"
  sift_features = load_SIFT_from_files(dataset, config)
  
  
  
  
  
  
  
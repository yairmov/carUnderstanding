'''
Created on Mar 3, 2014

@author: ymovshov
'''

import numpy as np
from path import path
from sklearn.externals.joblib import Parallel, delayed, load, dump

from configuration import get_config, update_config
import util
from matlab_dense_sift import dense_sift_matlab
import fgcomp_dataset_utils as fgu
from dense_SIFT import load_from_disk
import Bow


def preprocess_dataset(config):
  # Copy data set to cache
  util.copy_dataset('../../../fgcomp2013/release', 
                    config)
  
  # RUN THIS ONCE(it will crop and resize the images)
  if not config.dataset.is_cropped:
    # train
    util.crop_and_resize_dataset(config.dataset.train_annos_file_bk,
                                 config.dataset.train_annos_file,
                                 config.dataset.main_path,
                                 config.bb_area)
    #test
    util.crop_and_resize_dataset(config.dataset.train_annos_file_bk,
                                 config.dataset.train_annos_file,
                                 config.dataset.main_path,
                                 config.bb_area)
    update_config(config, 'dataset.is_cropped', True)




def calculate_dense_sift(data_annos, config):  
  dense_sift_matlab(data_annos, config)
  


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
  
def main():
  config = get_config()
  
  preprocess_dataset(config)
  
  (dataset, config) = fgu.get_all_metadata(config)
  
#   print('DENSE SIFT - train set')
#   calculate_dense_sift(dataset['train_annos'], config)
#   print('DENSE SIFT - test set')
#   calculate_dense_sift(dataset['test_annos'], config)
  
  # Create BoW model
  features = load_sift(dataset['train_annos'], config)
  dump(features, 'tmp.dat', compress=3)
  return
  print "Loaded %d SIFT features from disk" % features.shape[0]
  print "K-Means CLustering"
  bow_model = Bow.cluster_to_words(features, config)
  
  
    
if __name__ == '__main__':
  main()
  
    
    
  
  
  
  
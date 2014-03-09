'''
Created on Mar 3, 2014

@author: ymovshov
'''

import os
import cv2 as cv
from sklearn.externals.joblib import Parallel, delayed, dump, load


import fgcomp_dataset_utils as fgu
from configuration import get_config
from dense_SIFT import dense_SIFT, save_to_disk, load_from_disk


def calc_dense_SIFT_one_img(annotation, config):
  rel_path = annotation['rel_path']
  img_file = os.path.join(config.dataset.main_path, rel_path)

  # Replace extension to .dat and location in config.SIFT.raw_dir
  (name, ext) = os.path.splitext(os.path.split(img_file)[1])
  save_name = os.path.join(config.SIFT.raw_dir, name + '.dat')

  if os.path.exists(save_name):
    return

  img = cv.imread(img_file)
  (kp, desc) = dense_SIFT(img, grid_spacing=config.SIFT.grid_spacing)
  save_to_disk(kp, desc, save_name)


def calc_dense_SIFT_on_dataset(dataset, config):
  '''
  Just calls calc_dense_SIFT_one_img on all images in dataset using a
  parallel wrapper.
  '''
  train_annos = dataset['train_annos']

  Parallel(n_jobs=-1, verbose=config.logging.verbose)(
                 delayed(calc_dense_SIFT_one_img)(train_annos.iloc[ii], config)
                 for ii in range(len(train_annos)))


def extract_features(dataset, config):
  # SIFT
  calc_dense_SIFT_on_dataset(dataset, config)
  
  # LBP
  

if __name__ == '__main__':
  # Load config
  makes = ['bmw', 'ford']
  types = ['sedan', 'SUV']
  args = makes + types
  config = get_config(args)
  (dataset, config) = fgu.get_all_metadata(config)
  
  
  
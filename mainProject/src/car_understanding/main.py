'''
Created on Mar 3, 2014

@author: ymovshov
'''

import numpy as np
from path import path
from sklearn.externals.joblib import Parallel, delayed, load, dump
import pandas as pd

from configuration import get_config, update_config
import util
from matlab_dense_sift import dense_sift_matlab
import fgcomp_dataset_utils as fgu
from dense_SIFT import load_from_disk
import Bow


def preprocess_dataset(config):
  # Copy data set to cache
  util.copy_dataset(config)
  
  # RUN THIS ONCE(it will crop and resize the images)
  if not config.dataset.is_cropped:
    # train
    util.crop_and_resize_dataset(config.dataset.train_annos_file_bk,
                                 config.dataset.train_annos_file,
                                 config.dataset.main_path,
                                 config.bb_area,
                                 has_class=True)
    #test
    util.crop_and_resize_dataset(config.dataset.test_annos_file_bk,
                                 config.dataset.test_annos_file,
                                 config.dataset.main_path,
                                 config.bb_area,
                                 has_class=True)
    update_config(config, 'dataset.is_cropped', True)




def calculate_dense_sift(data_annos, config):  
  dense_sift_matlab(data_annos, config)
  
  

def create_bow_model(train_annos, config):
#   features = Bow.load_sift(train_annos, config)
#   print "Loaded %d SIFT features from disk" % features.shape[0]
#   print "K-Means CLustering"
#   bow_model = Bow.cluster_to_words(features, config)
  bow_model = Bow.fit_model(train_annos, config)
  Bow.save(bow_model, config.SIFT.BoW.model_file)
  
  
def assign_LLC(dataset, config):
  annos = pd.concat([dataset['train_annos'], dataset['test_annos']], axis=0)
  Bow.create_word_histograms_on_dataset(annos, config)
  
def main():
  config = get_config()
  
#   preprocess_dataset(config)
  
  (dataset, config) = fgu.get_all_metadata(config)
  
  print('DENSE SIFT - train set')
  calculate_dense_sift(dataset['train_annos'], config)
  print('DENSE SIFT - test set')
  calculate_dense_sift(dataset['test_annos'], config)
  
  # Create BoW model
  create_bow_model(dataset['train_annos'], config)
  
  # Assign cluster labels to all images
  print("Assigning to histograms/LLC")
  assign_LLC(dataset, config)
  
    
if __name__ == '__main__':
  main()
  
    
    
  
  
  
  
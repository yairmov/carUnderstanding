'''
Created on Jan 13, 2014

@author: ymovshov
'''

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import fgcomp_dataset_utils as fgu
from configuration import get_config
import small_run
import Bow


def test_fg_utils():
  domain_meta_file = '/usr0/home/ymovshov/Documents/Research/Code/3rd_Party/fgcomp2013/release/domain_meta.txt'
  class_meta_file = '/usr0/home/ymovshov/Documents/Research/Code/3rd_Party/fgcomp2013/release/class_meta.txt'
  train_ann_file = '/usr0/home/ymovshov/Documents/Research/Code/3rd_Party/fgcomp2013/release/dataset.txt'

  print 'before load'
  domain_meta = fgu.read_domain_meta(domain_meta_file)
  class_meta = fgu.read_class_meta(class_meta_file)
  dataset = fgu.read_training_data(train_ann_file)
  print 'after load'

  print domain_meta.dtypes
  print '-------------'
  print class_meta.dtypes
  print '-------------'
  print dataset.dtypes

  print 'Showing the first two lines of dataset'
  print '------------------------------------------'
  print dataset.iloc[0:2,:]

  print ' '
  print 'Trying single function to load all data'
  print '------------------------------------------'
  config = get_config(small=True)
  (dataset, class_meta, domain_meta) = fgu.get_all_metadata(config)
  print dataset.iloc[0:2,:]


def load_hist_from_files(files, config=None):
  features = np.empty(shape=[0, config.SIFT.BoW.num_clusters])
  for curr_file in files:
    hist = Bow.load(file)
    features = np.concatenate([features, hist])

  return features

def dbg_clustering():
  (dataset, config) = small_run.preprocess()
  (features, labels) = small_run.create_feature_matrix(dataset, config)
#   fig, ax = plt.subplots()
#   heatmap = ax.pcolor(features, cmap=plt.cm.Blues)
#   plt.show()

#   imgplot = plt.imshow(features)
#   imgplot.set_cmap('gist_earth')
#   plt.colorbar()
#   plt.show()



def test_work_remote():
    print "working remotely, la la la"
    print "lu lu lu"


if __name__ == '__main__':
#   test_fg_utils()
#   dbg_clustering()
    test_work_remote
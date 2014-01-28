'''
Control the configuration options for the project
Created on Jan 16, 2014

@author: ymovshov
'''

from treedict import TreeDict
import os

def get_config():
  config = TreeDict()

  # main params
#   config.main_path = '/usr0/home/ymovshov/Documents/Research/Code/car_understanding/'
  config.main_path = '.'
  config.bb_width = 200

  # SIFT
  config.SIFT.main_dir = os.path.join(config.main_path, 'SIFT')
  config.SIFT.raw_dir = os.path.join(config.SIFT.main_dir, 'raw')
  config.SIFT.grid_spacing = 4
  config.SIFT.BoW.model_file = os.path.join(config.SIFT.main_dir, 'BoW_model.dat')
  config.SIFT.BoW.hist_dir = os.path.join(config.SIFT.main_dir, 'hist')
  config.SIFT.BoW.num_clusters = 1000
  config.SIFT.BoW.max_desc_per_img = 1000



  # dataset params
  config.dataset.name = 'fgcomp2013'
  config.dataset.main_path = '/usr0/home/ymovshov/Documents/Research/Code/3rd_Party/fgcomp2013/release'
  config.dataset.class_meta_file =  '/usr0/home/ymovshov/Documents/Research/Code/3rd_Party/fgcomp2013/release/class_meta.txt'
  config.dataset.domain_meta_file =  '/usr0/home/ymovshov/Documents/Research/Code/3rd_Party/fgcomp2013/release/domain_meta.txt'
  config.dataset.train_annos_file =  '/usr0/home/ymovshov/Documents/Research/Code/3rd_Party/fgcomp2013/release/train_annos.txt'
  config.dataset.domains = [3] # cars

  return config


if __name__ == '__main__':
    config = get_config()
    print config.makeReport()

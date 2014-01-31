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
  # maybe use os.environ['RESEARCH_DIR'] ?
  config.main_path = '../../../'
  config.cache_dir = os.path.join(config.main_path, 'cache')
  config.bb_width = 200

  # SIFT
  config.SIFT.main_dir = os.path.join(config.main_path, 'SIFT')
  config.SIFT.raw_dir = os.path.join(config.SIFT.main_dir, 'raw')
  config.SIFT.grid_spacing = 4
  config.SIFT.BoW.model_file = os.path.join(config.SIFT.main_dir,
                                            'BoW_model.dat')
  config.SIFT.BoW.hist_dir = os.path.join(config.SIFT.main_dir, 'hist')
  config.SIFT.BoW.num_clusters = 1024
  config.SIFT.BoW.max_desc_per_img = 1000



  # dataset params
  # location of dataset (original one)
#   config.dataset.name = 'fgcomp2013'
#   config.dataset.main_path = '/usr0/home/ymovshov/Documents/Research/Data/fgcomp2013/release'

  # location of dataset (after it was copied and normalized)
  config.dataset.name = 'fgcomp2013_normed'
  config.dataset.main_path = os.path.join(config.main_path,
                                        'fgcomp2013_normed',
                                        'release')



  config.dataset.class_meta_file =  os.path.join(config.dataset.main_path,
                                                 'class_meta.txt')
  config.dataset.domain_meta_file =  os.path.join(config.dataset.main_path,
                                                  'domain_meta.txt')
  config.dataset.train_annos_file =  os.path.join(config.dataset.main_path,
                                                  'train_annos.txt')
  config.dataset.domains = [3] # cars

  return config


if __name__ == '__main__':
    config = get_config()
    print config.makeReport()

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

  # Adding pos/neg class definitions to the config

#   # SUV VS Sedan
#   config.dataset.class_ids.pos = [184,215,216,220,231,233,235,241,245,259]# SUV
#   config.dataset.class_ids.neg = [185,186,188,199,200,203,206,207,209,212]#not SUV
# #   config.dataset.class_ids.neg = [188, 190, 196, 207, 213] # not SUV
# #   config.dataset.class_ids.pos = [184, 220, 231, 235, 303] # Sedan


  # Audi VS BMW
  config.dataset.class_ids.pos = [195, 196, 197, 198, 199, 200,
                                  201, 202, 203, 204, 205, 206, 207, 208] # Audi
  config.dataset.class_ids.neg = [209, 210, 211, 212, 213, 214, 215, 216, 217,
                                  218, 219, 220, 221] # BMW

  return config


if __name__ == '__main__':
    config = get_config()
    print config.makeReport()

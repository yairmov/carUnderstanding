'''
Control the configuration options for the project
Created on Jan 16, 2014

@author: ymovshov
'''

from treedict import TreeDict
import os

def get_config(args):
  config = TreeDict()

  # main params
  # maybe use os.environ['RESEARCH_DIR'] ?
  config.main_path = '../../../'
  config.cache_dir = os.path.join(config.main_path, 'cache')
  config.bb_width = 200
  config.logging.verbose = 3
  
  config.output_dir = os.path.join(config.main_path, 'output')

  # SIFT
  config.SIFT.dir = assign_dir(os.path.join(config.main_path, 'SIFT'))
  config.SIFT.raw_dir = assign_dir(os.path.join(config.SIFT.dir, 'raw'))
  config.SIFT.grid_spacing = 4
  config.SIFT.BoW.model_file = os.path.join(config.SIFT.dir, 'BoW_model.dat')
  config.SIFT.BoW.hist_dir = assign_dir(os.path.join(config.SIFT.dir, 
                                                     'word_hist'))
  config.SIFT.BoW.num_clusters = 1024
  config.SIFT.BoW.max_desc_per_img = 1000
  config.SIFT.BoW.max_desc_total = 4e6



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


  # Attribute Params
#   config.attribute.pos_name = args[1]
#   config.attribute.neg_name = args[2]

  config.attribute.names = args
  config.attribute.dir = assign_dir(os.path.join(config.main_path, 
                                      'attribute_classifiers'))
  
  
  return config



def assign_dir(dir_name):
  if not os.path.isdir(dir_name):
    os.makedirs(dir_name)
  
  return dir_name

if __name__ == '__main__':
    config = get_config()
    print config.makeReport()

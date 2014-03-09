'''
Control the configuration options for the project
Created on Jan 16, 2014

@author: ymovshov
'''

from treedict import TreeDict
import os
import socket

def get_config(args):
  config = TreeDict()

  # main params
  config.hostname = socket.gethostname()
  config.main_path = get_main_path(config.hostname)
  config.cache_dir = assign_dir(os.path.join(config.main_path, 'cache'))
  config.bb_width = 200
  config.logging.verbose = 3
  
  config.output_dir = assign_dir(os.path.join(config.main_path, 'output'))

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
  
  # SIFT-LLC
  config.SIFT.LLC.use = False
  config.SIFT.LLC.knn = 5
  config.SIFT.LLC.beta = 3e-2
  
  # HOG
  config.HOG.orientations = 9
  config.HOG.pixels_per_cell = (8, 8)
  
  # LBP
  config.LPB.P = 8 # Number of circularly symmetric neighbour set points (quantization of the angular space).
  config.LPB.R = 8 # Radius of circle (spatial resolution of the operator).



  # dataset params
  # location of dataset (original one)
#   config.dataset.name = 'fgcomp2013'
#   config.dataset.main_path = '/usr0/home/ymovshov/Documents/Research/Data/fgcomp2013/release'

  # location of dataset (after it was copied and normalized)
  config.dataset.name = 'fgcomp2013_normed'
  config.dataset.index_str = 'FGCOMP_{:>07}'
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
  config.attribute.names = args
  config.attribute.dir = assign_dir(os.path.join(config.main_path, 
                                      'attribute_classifiers'))
  config.attribute.high_thresh = 0.65
  
  
  return config



def get_main_path(hostname):
  if hostname == 'palfrey.vasc.ri.cmu.edu':
    main_path = '../../../'
  elif hostname == 'gs10245.sp.cs.cmu.edu':
    main_path = '/Volumes/palfrey/Documents/Research/Code/carUnderstanding'
  else:
    raise Exception("Unknown hostname! please define config.main_path")
  
  return main_path 

def assign_dir(dir_name):
  if not os.path.isdir(dir_name):
    os.makedirs(dir_name)
  
  return dir_name

if __name__ == '__main__':
    config = get_config()
    print config.makeReport()

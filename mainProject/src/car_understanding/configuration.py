'''
Control the configuration options for the project
Created on Jan 16, 2014

@author: ymovshov
'''

from treedict import TreeDict
import os
import socket
import json
from path import path


def save_to_file(config, fname):
  with open(fname, 'w') as f:
    json.dump(config.convertTo('nested_dict'), f, indent=2)
    
def load_from_file(fname):
  with open(fname, 'r') as f:
    d = json.load(f)
    
  return make_directories(TreeDict.fromdict(d, expand_nested=True))


def make_directories(config):
  '''
  process the ocnfig file to create directories if needed.
  right now you have to manually add new dir names 
  '''
  config.main_path = assign_dir(config.main_path)
  config.cache_dir = assign_dir(config.cache_dir)
  config.output_dir = assign_dir(config.output_dir)
  config.attribute.dir = assign_dir(config.attribute.dir)
  config.dataset.main_path = assign_dir(config.dataset.main_path)
  config.SIFT.dir = assign_dir(config.SIFT.dir)
  config.SIFT.raw_dir = assign_dir(config.SIFT.raw_dir)
  config.SIFT.BoW.hist_dir = assign_dir(config.SIFT.BoW.hist_dir)
  config.SIFT.matlab.raw_dir = assign_dir(config.SIFT.matlab.raw_dir)
  
  return config
  
  
def update_config(config, location, value):
  config[location] = value
  save_to_file(config, config.config_file)
  return config

def get_config(config_file='config.json'):
  config =  load_from_file(config_file)
  config.config_file = config_file
  return config
  
  

# def get_config(args):
#   config = TreeDict()
# 
#   # main params
#   config.hostname = socket.gethostname()
#   config.main_path = get_main_path(config.hostname)
#   config.cache_dir = assign_dir(os.path.join(config.main_path, 'cache'))
#   config.bb_width = 200
#   config.logging.verbose = 3
# 
#   config.output_dir = assign_dir(os.path.join(config.main_path, 'output'))
# 
#   # SIFT
#   config.SIFT.dir = assign_dir(os.path.join(config.main_path, 'SIFT'))
#   config.SIFT.raw_dir = assign_dir(os.path.join(config.SIFT.dir, 'raw'))
#   config.SIFT.grid_spacing = 4
#   config.SIFT.BoW.model_file = os.path.join(config.SIFT.dir, 'BoW_model.dat')
#   config.SIFT.BoW.hist_dir = assign_dir(os.path.join(config.SIFT.dir,
#                                                      'word_hist'))
# 
#   #
#   config.SIFT.BoW.requested_n_clusters = 948
#   # Number of clusters after clustering has been done (<= requested_n_clusters)
#   config.SIFT.BoW.num_clusters = 948 #(TODO: find a nice way to do this)
# #   config.SIFT.BoW.max_desc_per_img = 1000
#   config.SIFT.BoW.max_desc_total = 4e6
# 
#   # Ugly hack that uses matlab code for SIFT
#   config.SIFT.matlab.use = True
#   config.SIFT.matlab.raw_dir = config.SIFT.raw_dir
#   config.SIFT.matlab.sizes = [8, 12, 16, 24, 30]
# 
#   # SIFT-LLC
#   config.SIFT.LLC.use = True
#   config.SIFT.LLC.knn = 5
#   config.SIFT.LLC.beta = 3e-2
# 
#   # HOG
#   config.HOG.orientations = 9
#   config.HOG.pixels_per_cell = (8, 8)
# 
#   # LBP
#   config.LPB.P = 8 # Number of circularly symmetric neighbour set points (quantization of the angular space).
#   config.LPB.R = 8 # Radius of circle (spatial resolution of the operator).
# 
# 
# 
#   # dataset params
#   #---------------
#   # location of dataset (original one)
# #   config.dataset.name = 'fgcomp2013'
# #   config.dataset.main_path = '/usr0/home/ymovshov/Documents/Research/Data/fgcomp2013/release'
# 
#   # location of dataset (after it was copied and normalized)
#   config.dataset.name = 'fgcomp2013_normed'
#   config.dataset.index_str = 'FGCOMP_{:>07}'
#   config.dataset.main_path = os.path.join(config.main_path,
#                                         'fgcomp2013_normed',
#                                         'release')
# 
# 
# 
#   config.dataset.class_meta_file =  os.path.join(config.dataset.main_path,
#                                                  'class_meta.txt')
#   config.dataset.domain_meta_file =  os.path.join(config.dataset.main_path,
#                                                   'domain_meta.txt')
#   config.dataset.train_annos_file =  os.path.join(config.dataset.main_path,
#                                                   'train_annos.txt')
# 
#   # The test annos file doesn't have class indices.
#   # To get test performance you need to submit to a server.
#   config.dataset.test_annos_file =  os.path.join(config.dataset.main_path,
#                                                   'test_annos_track1.txt')
#   config.dataset.domains = [3] # cars
#   # split train set into train/dev sets as we don't have access to the
#   # test data. At some point we need to submit results to their server.
#   config.dataset.dev_set.use = True
#   config.dataset.dev_set.test_size = 10 # per class
#   config.dataset.dev_set.rand_seed = 12
# 
# 
#   # Attribute Params
#   config.attribute.names = args
#   config.attribute.dir = assign_dir(os.path.join(config.main_path,
#                                       'attribute_classifiers'))
#   config.attribute.high_thresh = 0
#   # should classifiers be normalized to output probabilities
#   config.attribute.use_prob = False
# 
# 
# 
#   return config



def get_main_path(hostname):
  if hostname == 'palfrey.vasc.ri.cmu.edu':
    main_path = '../../../'
  elif hostname == 'gs10245.sp.cs.cmu.edu':
    main_path = '/Volumes/palfrey/Documents/Research/Code/carUnderstanding'
  else:
    raise Exception("Unknown hostname! please define config.main_path")

  return main_path

def assign_dir(dir_name):
  dir_path = path(dir_name)
  if not dir_path.isdir():
    dir_path.makedirs()
    
  return dir_path
#   if not os.path.isdir(dir_name):
#     os.makedirs(dir_name)
# 
#   return dir_name

if __name__ == '__main__':
    config = get_config()
    print config.makeReport()

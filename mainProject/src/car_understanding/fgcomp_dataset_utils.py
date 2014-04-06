'''
Some Utils functions to parse and manipulate the dataset files from
the fine-grained competition.
Created on Jan 16, 2014

@author: ymovshov
'''

import pandas as pd
import numpy as np
import os
from configuration import get_config
from path import path
import Image
from util import ProgressBar

# Use this function to create a pandas array for the classes.
# You can't just use pd.read_csv cause the file has names with commas in them...
def read_class_meta(infilename):
  # read file into a string
  with open(infilename, 'r') as infile:
    class_meta_content = infile.readlines()

  # parse string line by line
  class_id_col = []
  class_name_col = []
  domain_id_col = []

  for line in class_meta_content:
    parts = line.strip().split(',')
    # First and last columns go into the proper place
    class_id_col.append(int(parts[0]))
    domain_id_col.append(int(parts.pop()))

    # The rest of the cols are merged into the class name column
    class_name_col.append(str.join(',', parts[1:]))

  all_cols = np.array([class_id_col, class_name_col,domain_id_col]).transpose()

  class_meta =  pd.DataFrame(all_cols,
                             columns=['class_index', 'class_name', 'domain_index'])
  class_meta['class_index'] = class_meta['class_index'].astype(np.int32)
  class_meta['domain_index'] = class_meta['domain_index'].astype(np.int32)
  
  class_meta = class_meta.set_index('class_index', drop=False) 

  return class_meta


# Read the domain metadata from file. row index is the domain_index and the
# column is the domain_name
def read_domain_meta(infilename):
  return pd.read_csv(infilename,
                            header=None, index_col=1, names=['domain_name'])


def read_image_annotations(infilename, has_class_id=True):
  '''
  Reads the training/test annotations from a txt file.
  '''
  if has_class_id:
    names = ['image_index', 'img_path', 'domain_index',
             'class_index', 'xmin', 'xmax', 'ymin', 'ymax']
    types = {'image_index': np.int32,
             'rel_path': 'str',
             'domain_index': np.int32,
             'class_index': np.int32,
             'xmin': np.float64,
             'xmax':np.float64,
             'ymin':np.float64,
             'ymax':np.float64}
  else:
    names = ['image_index', 'img_path', 'domain_index',
             'xmin', 'xmax', 'ymin', 'ymax']
    types = {'image_index': np.int32,
             'rel_path': 'str',
             'domain_index': np.int32,
             'xmin': np.float64,
             'xmax':np.float64,
             'ymin':np.float64,
             'ymax':np.float64}
  dataset =  pd.read_csv(infilename, header=None, names=names, dtype=types,
                     index_col=0)

  dataset['basename'] = dataset.img_path.apply(os.path.basename)

  return dataset


def get_all_metadata(config=None, args=None):
  if config == None and args == None:
    raise Exception('Either config or args need to be not None')
  if config == None:
    config = get_config(args)
    
  class_meta  = read_class_meta(config.dataset.class_meta_file)
  train_annos = read_image_annotations(config.dataset.train_annos_file)
  test_annos = read_image_annotations(config.dataset.test_annos_file,
                                      has_class_id=False)
  domain_meta = read_domain_meta(config.dataset.domain_meta_file)
  train_annos['class_name'] = np.array([class_meta.class_name[class_index] for 
                                         class_index in 
                                         train_annos.class_index])
#   test_annos['class_name'] = np.array([class_meta.class_name[class_index] for 
#                                          class_index in 
#                                          test_annos.class_index])

  # Prepand path to the dataset to each img_path
  train_annos.img_path.map(lambda x: config.dataset.main_path.joinpath(x))

  # Filter the class meta and train/test annotations to just use the 
  # domains defined in config
  class_meta = class_meta[class_meta.domain_index.isin(config.dataset.domains)]
  train_annos = train_annos[train_annos.domain_index.isin(config.dataset.domains)]
  test_annos = test_annos[test_annos.domain_index.isin(config.dataset.domains)]
  
  
  # Create dev set
  dev_annos_train, dev_annos_test = create_dev_set(train_annos, 
                                                   config)

  # Should we use the dev set as the test set
  if config.dataset.dev_set.use:
    train_used, test_used = dev_annos_train, dev_annos_test 
  else:
    train_used, test_used = train_annos, test_annos
    
    
  if config.flip_images:
    train_used = create_flipped_images(train_used, config)

  return ({'real_train_annos': train_annos,
           'real_test_annos': test_annos,
           'train_annos': train_used,
           'test_annos': test_used,
           'dev_annos': dev_annos_test, 
            'class_meta': class_meta,
            'domain_meta': domain_meta},
          config)

def create_dev_set(train_annos, config):
  num_test = config.dataset.dev_set.test_size
  u_ids = train_annos.class_index.unique()
  dev_img_ids_test = []
  dev_img_ids_train = []
  
  # Set random seed for repreducibility of dev set
  if type(config.dataset.dev_set.rand_seed) == int:
    R = np.random.RandomState(config.dataset.dev_set.rand_seed)
    
  for id in u_ids:
    curr = train_annos[train_annos.class_index == id]
    
    r_ids = R.permutation(curr.index)
    dev_img_ids_test.extend(list(r_ids[:num_test]))
    dev_img_ids_train.extend(list(r_ids[num_test:]))
    
  dev_set_test = train_annos.loc[dev_img_ids_test]
  dev_set_train = train_annos.loc[dev_img_ids_train]
  
  return dev_set_train, dev_set_test  


def create_flipped_images(train_annos, config):
  flipped_annos = train_annos.copy()
  
  # Create new ids for the flipped images
  flipped_annos.index = 1e5 + flipped_annos.index
  
  cache_dir = config.cache_dir
  fp_suffix = config.flip_suffix
  rel_to_cache = path(config.dataset.main_path).relpathto(config.cache_dir)
  
  n_imgs = train_annos.shape[0]
  pbar = ProgressBar(n_imgs)
  print('Creating flipped copies of train images')
  for ii in range(n_imgs):
    parts = cache_dir.joinpath(train_annos.basename.iloc[ii]).splitext()
    flipped_file = parts[0] + fp_suffix + parts[1]
#     print "flipped_file: ", flipped_file
    
    img_file = config.dataset.main_path.joinpath(train_annos.rel_path.iloc[ii]) 
    img = Image.open(img_file)
    (width, height) = img.size
    
    # We might need to create the flipped image if it is not in 
    # cache already.
    if not flipped_file.isfile():
      f_img = img.transpose(Image.FLIP_LEFT_RIGHT)
      f_img.save(flipped_file)
      
    # Modify the annotations for it
    flipped_annos.rel_path.iloc[ii] = rel_to_cache.joinpath(flipped_file.basename())
    flipped_annos.basename.iloc[ii] = flipped_file.basename()
    box = (train_annos.iloc[ii].xmin, train_annos.iloc[ii].ymin,
           train_annos.iloc[ii].xmax, train_annos.iloc[ii].ymax)
    (xmin, ymin, xmax, ymax) = flip_box_LR(box, width)
    flipped_annos.xmin.iloc[ii] = xmin
    flipped_annos.xmax.iloc[ii] = xmax
    flipped_annos.ymin.iloc[ii] = ymin
    flipped_annos.ymax.iloc[ii] = ymax
    
    pbar.animate(ii)
    
    
  return pd.concat([train_annos, flipped_annos], axis=0)
    
def flip_box_LR(box, width):
  '''
  box = (xmin, ymin, xmax, ymax)
  '''
  xmin, ymin, xmax, ymax = box    
  n_xmax = width - xmin
  n_xmin = width - xmax
  
  return (n_xmin, ymin, n_xmax, ymax)
  
  

def run_test():
  from configuration import get_config
  config = get_config([])
  
  domain_meta = read_domain_meta(config.dataset.domain_meta_file)
  print("domain_meta:")
  print(domain_meta.head())
  
  class_meta  = read_class_meta(config.dataset.class_meta_file)
  print("class_meta:")
  print(class_meta.head())
  
  train_annos = read_image_annotations(config.dataset.train_annos_file)
  print("train_annos:")
  print(train_annos.head())
  
  test_annos = read_image_annotations(config.dataset.test_annos_file,
                                      has_class_id=False)
  print("test_annos:")
  print(test_annos.head())
  
  print("Using call to get_all_metadata()")
  print("--------------------------------")
  (dataset, config) = get_all_metadata(config)
  
  domain_meta = dataset['domain_meta']
  print("domain_meta:")
  print(domain_meta.head())
  
  class_meta  = dataset['class_meta']
  print("class_meta:")
  print(class_meta.head())
  
  train_annos = dataset['train_annos']
  print("train_annos:")
  print(train_annos.head())
  
  test_annos = dataset['test_annos']
  print("test_annos:")
  print(test_annos.head())
  
  
  

if __name__ == '__main__':
  run_test()

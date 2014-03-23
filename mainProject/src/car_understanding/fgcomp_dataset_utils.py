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
    names = ['image_index', 'rel_path', 'domain_index',
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
    names = ['image_index', 'rel_path', 'domain_index',
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

  dataset['basename'] = dataset.rel_path.apply(os.path.basename)

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
#   train_annos = pd.merge(train_annos, class_meta.iloc[:,0:2], on='class_index') # probably produces WRONG mapping
#   train_annos.index.name = 'image_index'
  train_annos['class_name'] = np.array([class_meta.class_name[class_index] for 
                                         class_index in 
                                         train_annos.class_index])
#   test_annos['class_name'] = np.array([class_meta.class_name[class_index] for 
#                                          class_index in 
#                                          test_annos.class_index])

  # Filter the class meta and train/test annotations to just use the 
  # domains defined in config
  class_meta = class_meta[class_meta.domain_index.isin(config.dataset.domains)]
  train_annos = train_annos[train_annos.domain_index.isin(config.dataset.domains)]
  test_annos = test_annos[test_annos.domain_index.isin(config.dataset.domains)]
  
  
  # Create dev set
  dev_annos = create_dev_set(train_annos, config.dataset.dev_set.test_size)

  # Should we use the dev set as the test set
  if config.dataset.dev_set.use:
    used = dev_annos
  else:
    used = test_annos

  return ({'train_annos': train_annos,
           'real_test_annos': test_annos,
           'test_annos': used,
           'dev_annos': dev_annos, 
            'class_meta': class_meta,
            'domain_meta': domain_meta},
          config)

def create_dev_set(train_annos, num_test=10):
  u_ids = train_annos.class_index.unique()
  dev_img_ids = []
  for id in u_ids:
    curr = train_annos[train_annos.class_index == id]
#     print(curr.head(10))
    c = curr.index[:num_test]
    print(c)
#     dev_img_ids.extend(list(curr.index[:num_test]))
    
  dev_set = train_annos.loc[dev_img_ids]
  return dev_set 


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

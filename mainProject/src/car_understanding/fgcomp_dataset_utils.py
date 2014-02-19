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

def read_training_data(infilename):
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
  train_annos = read_training_data(config.dataset.train_annos_file)
  domain_meta = read_domain_meta(config.dataset.domain_meta_file)
#   train_annos = pd.merge(train_annos, class_meta.iloc[:,0:2], on='class_index') # probably produces WRONG mapping
#   train_annos.index.name = 'image_index'
  train_annos['class_name'] = np.array([class_meta.class_name[class_index] for 
                                         class_index in 
                                         train_annos.class_index])

  # Filter the class meta and train annotations to just use the 
  # domains defined in config
  class_meta = class_meta[class_meta['domain_index'] == config.dataset.domains[0]]
  train_annos = train_annos[train_annos.class_index.isin(class_meta.class_index)]

  return ({'train_annos': train_annos,
             'class_meta': class_meta,
             'domain_meta': domain_meta},
          config)



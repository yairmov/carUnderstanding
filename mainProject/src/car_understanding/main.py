'''
Created on Mar 3, 2014

@author: ymovshov
'''

from configuration import get_config, update_config
import util

if __name__ == '__main__':
  # Load config
  config = get_config()
  
  # Copy data set to cache
  util.copy_dataset('../../../fgcomp2013/release', 
                    config)
  
  # RUN THIS ONCE(it will crop and resize the images)
  if not config.dataset.is_cropped:
    # train
    util.crop_and_resize_dataset(config.dataset.train_annos_file_bk,
                                 config.dataset.train_annos_file,
                                 config.dataset.main_path,
                                 config.bb_area)
    #test
    util.crop_and_resize_dataset(config.dataset.train_annos_file_bk,
                                 config.dataset.train_annos_file,
                                 config.dataset.main_path,
                                 config.bb_area)
    update_config(config, 'dataset.is_cropped', True)
    
    
  
  
  
  
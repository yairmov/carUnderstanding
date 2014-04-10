'''
Created on Mar 3, 2014

@author: ymovshov
'''

from configuration import get_config
import util

if __name__ == '__main__':
  # Load config
  config = get_config()
  
  # Copy data set to cache
  util.copy_dataset('../../../fgcomp2013/release', '../../../cache/dataset')
  
  # RUN THIS ONCE(it will crop and resize the images)
  
  
  
  
  
  
  
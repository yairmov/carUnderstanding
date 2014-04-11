'''
Created on Mar 3, 2014

@author: ymovshov
'''

from configuration import get_config, update_config
import util
from matlab_dense_sift import dense_sift_matlab
import fgcomp_dataset_utils as fgu


def preprocess_dataset(config):
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




def calculate_dense_sift(data_annos, config):  
  dense_sift_matlab(data_annos, config)
  

  
def main():
  config = get_config()
  
  preprocess_dataset(config)
  
  (dataset, config) = fgu.get_all_metadata(config)
  
  print('DENSE SIFT')
  calculate_dense_sift(dataset['train_annos'], config)
  calculate_dense_sift(dataset['test_annos'], config)
  
  # Create BoW model
#   features = load_SIFT_from_files(dataset, config)
#   print "Loaded %d SIFT features from disk" % features.shape[0]
#   print "K-Means CLustering"
#   bow_model = Bow.create_BoW_model(features, config)
  
  
    
if __name__ == '__main__':
  main()
  
    
    
  
  
  
  
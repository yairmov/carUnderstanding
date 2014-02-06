# encoding: utf-8
'''
car_understanding.attribute_classifier -- a single attribute classifier

@author:     Yair Movshovitz-Attias

@copyright:  2014 Yair Movshovitz-Attias. All rights reserved.

@contact:    yair@cs.cmu.edu
'''

from sklearn.externals.joblib import Parallel, delayed, Memory
import sklearn as sk

import Bow

class AttributeClassifier:
  """A module for classifying attributes."""
  
  def __init__(self, config, dataset, pos_inds, name, desc=""):
    """ Ctor.
    
    Args:
          config   - Config object created by configuration.get_config()
          dataset  - Pandas table defining the dataset (see fgcomp_dataset_utils)
          pos_inds - a list or numpy array of image indices to use as positive
          examples for training the classifier).
          name     - Short string name for object.
          desc     - Longer string description of attribute (optional)
          
    """
    self.config   = config
    self.name     = name
    self.pos_inds = pos_inds
    self.dataset  = dataset.copy()
    self.desc     = desc
    self.memory   = Memory(cachedir=config.SIFT.BoW.hist_dir.format(name), 
                           verbose=0)
    self.clf      = None
    
    # Creating memoiztion for functions
#     self.calc_raw_feature = self.memory.cache(self.calc_raw_feature) 
    
#   def calc_raw_feature(self, annotation):
#     rel_path = annotation['rel_path']
#     img_file = os.path.join(self.config.dataset.main_path, rel_path)
#     
#     # Replace extension to .dat and location in config.SIFT.raw_dir
#     (name, ext) = os.path.splitext(os.path.split(img_file)[1])
#     save_name = os.path.join(self.config.SIFT.raw_dir, name + '.dat')
#     
#     if os.path.exists(save_name):
#       return
#     
#     img = cv.imread(img_file)  
#     (kp, desc) = dense_SIFT(img, grid_spacing=self.config.SIFT.grid_spacing)
#     save_to_disk(kp, desc, save_name)
#     
#     
#   def calc_raw_feature_on_dataset(self):
#     '''
#     Just calles calc_dense_SIFT_one_img on all images in dataset using a
#     parallel wrapper.
#     '''
#     Parallel(n_jobs=-1, verbose=self.config.logging.verbose)(
#                    delayed(self.calc_raw_feature)(dataset.iloc[ii])
#                    for ii in range(len(dataset)))
#     
#   def load_raw_from_a_file(self, curr_anno):
#     curr_file = os.path.splitext(curr_anno['basename'])[0] + '.dat'
#     (kp, desc) = load_from_disk(os.path.join(self.config.SIFT.raw_dir, 
#                                              curr_file))
#   
#     # Only keep points that are inside the bounding box
#     box = (curr_anno['xmin'], curr_anno['xmax'],
#            curr_anno['ymin'], curr_anno['ymax'])
#   
#     inds = np.zeros(shape=[len(kp),], dtype=bool)
#     for jj in range(len(kp)):
#       inds[jj] = contains(box, kp[jj].pt)
#   
#     desc = desc[inds, :]
#     kp = np.asarray(kp)[inds].tolist()
#   
#     # Random selection of a subset of the keypojnts/descriptors
#     inds  = np.random.permutation(desc.shape[0])
#     desc = desc[inds, :]
#     desc = desc[:self.config.SIFT.BoW.max_desc_per_img, :]
#   #   kp    = [kp[i] for i in inds]
#   #   kp    = kp[:self.config.SIFT.BoW.max_desc_per_img]
#   
#     return desc
  
  
  def create_feature_matrix(self):
    
    # Preallocate feature matrix
    features = np.empty(shape=[len(self.dataset), 
                               self.config.SIFT.BoW.num_clusters])
  
    # Load histograms from disk into a matrix
    for ii in range(len(self.dataset)):
      img_name = self.dataset.iloc[ii]['basename']
      img_name = os.path.splitext(img_name)[0]
      hist_filename = os.path.join(self.config.SIFT.BoW.hist_dir, 
                                   img_name) + '_hist.dat'
      hist = Bow.load(hist_filename) # hist[0] = values, hist[1] = bin edges
      features[ii, :] = hist[0]
  
  
    # preprocess features
    features = sk.preprocessing.scale(features)
  
    # create pos/neg labels
    labels = self.dataset.class_index.isin(self.pos_inds).values
  
    return (features, labels)
  
  
  def fit(self, features, labels):
    self.clf = sk.svm.SVC(kernel='linear', C=0.0005, class_weight='auto')
    self.clf.fit(features, labels)
    
  def my_print(self, str):
    print self.name + ":" + str
    
  def run_training_pipeline(self):
    """ The full sequence of operations that trains an attribute classifier"""
    
    self.my_print("Loading feature-word histograms from disk, and creating " + 
                  "matrix for attribute classification.")
    (features, labels) = self.create_feature_matrix()
    
    self.my_print("Training classifier")
    self.fit(features, labels)
    
    
    
# "Static" functions
def contains(box, point):
  '''
  box = (xmin, xmax, ymin, ymax)
  point = (x, y)
  '''
  return (box[0] <= point[0] and box[1] >= point[0] and
          box[2] <= point[1] and box[3] >= point[1])
    
    
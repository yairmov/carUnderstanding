'''
Util functions for the carUnderstanding projects
Created on Jan 28, 2014

@author: yair@cs.cmu.edu
'''

from __future__ import print_function
import os
# import scipy.misc
from PIL import Image
import matplotlib.pyplot as plt
# from clint.textui import progress
import sys
import cv2 as cv
from path import path
import distutils.dir_util as dir_util
from numpy import sqrt
import pandas as pd
import numpy as np


# import base64
# import numpy as np
# import pandas as pd
# from sklearn import ensemble, decomposition, manifold
# from PIL import Image
# from path import path

# import Bow


def set_width_to_normalize_bb_width(img, xmin, xmax, to_width):
  w = xmax - xmin
  s = to_width / w
#   img = scipy.misc.imresize(img, s)
  img.resize( [int(s * v) for v in img.size], Image.ANTIALIAS)
  return img, s


def resize_img_to_normalize_bb_area(img, bb, to_area=1e5):
  '''
  bb = (xmin, ymin, xmax, ymax)
  '''
  
  area = float((bb[2] - bb[0]) * (bb[3] - bb[1]))
#   print("area: ", area)
  s = to_area / area
#   print("s: ", s)
  if s != 1:
#     img = scipy.misc.imresize(img, s)
    img.resize( [int(s * siz) for siz in img.size])
  return img, s

def change_bb_loc(scaler, xmin, xmax, ymin, ymax):
  w, h = (xmax - xmin, ymax - ymin)
  nw, nh = (scaler * w, scaler * h)

  center = (xmin + (w/2), ymin + (h/2))
  new_center = (center[0] * scaler, center[1] * scaler)

  xmin, xmax = (new_center[0] - (nw/2), new_center[0] + (nw/2))
  ymin, ymax = (new_center[1] - (nh/2), new_center[1] + (nh/2))


  return xmin, xmax, ymin, ymax

'''
This will change the images in the dataset!!
Call this on a COPY of the dataset
'''
# def normalize_dataset(train_annos_file, main_path, out_file, bb_width):
def normalize_dataset(data_annos_file, main_path, out_file, to_area=1e5):
  # read lines from file
  with open(data_annos_file) as f:
    content = f.readlines()

  out_fid = open(os.path.join(main_path, out_file), 'w')
#   print("Resizing images such that BB is of width = %g" % bb_width)
  print("Resizing images such that BB is of area = %g" % to_area)
  n_imgs = len(content)
  progress = ProgressBar(n_imgs)
  for ii in range(n_imgs):
    progress.animate(ii)
    curr_line = content[ii]
    curr_line = curr_line.strip()
    (img_index, rel_path, domain_index,
     class_index, xmin, xmax, ymin, ymax) = curr_line.split(',')

    xmin, xmax, ymin, ymax = (float(x) for x in (xmin, xmax, ymin, ymax))

    # Read image and resize such that bounding box is of specific size
    img_file = os.path.join(main_path, rel_path)
    img = Image.open(img_file)
#     img, scaler = set_width_to_normalize_bb_width(img, xmin, xmax, bb_width)
    img, scaler = resize_img_to_normalize_bb_area(img, 
                                                  (xmin, ymin, xmax, ymax), 
                                                  to_area=to_area)

    img.save(img_file)

    # change xmin, xmax, ymin, ymax to the new size
    xmin, xmax, ymin, ymax = change_bb_loc(scaler, xmin, xmax, ymin, ymax)
    xmin, xmax, ymin, ymax = (str(int(x)) for x in (xmin, xmax, ymin, ymax))

    # generate new text line
    new_line = ','.join((img_index, rel_path, domain_index,
              class_index, xmin, xmax, ymin, ymax))

    out_fid.write("%s\n" % new_line)

  out_fid.close()


def crop_and_resize_img(img, bb, to_area=1e5):
  '''
  bb = (xmin, ymin, xmax, ymax)
  '''
  
  area = (bb[2] - bb[0]) * (bb[3] - bb[1])
  s = sqrt(to_area / float(area))
  
  img = img.crop(bb)
  img = img.resize( [int(s * siz) for siz in img.size], Image.ANTIALIAS)
  
  return img, s
  
def crop_and_resize_dataset(infile, outfile, main_path, bb_area,
                            has_class=True):
  with open(infile) as f:
    content = f.readlines()
  
  out_fid = open(outfile, 'w')
  
  print("Cropping images and resizing BB such that the area is: %g" % bb_area)
  n_imgs = len(content)
  progress = ProgressBar(n_imgs)
  for ii in range(n_imgs):
    progress.animate(ii)
    curr_line = content[ii]
    curr_line = curr_line.strip()
    if has_class:
      (img_index, rel_path, domain_index,
       class_index, xmin, xmax, ymin, ymax) = curr_line.split(',')
    else:
      (img_index, rel_path, domain_index,
       xmin, xmax, ymin, ymax) = curr_line.split(',')
       
    xmin, xmax, ymin, ymax = (int(x) for x in (xmin, xmax, ymin, ymax))
      
    
    img_file = os.path.join(main_path, rel_path)
    img = Image.open(img_file)
    
    img, scaler = crop_and_resize_img(img,
                                      (xmin, ymin, xmax, ymax), 
                                      to_area=bb_area)
    
    # generate new text line. we output -1 for bb location to mark
    # that the image is cropped
    if has_class:
      new_line = ','.join((img_index, rel_path, domain_index,
                class_index, '-1', '-1', '-1', '-1'))
    else:
      new_line = ','.join((img_index, rel_path, domain_index,
                 '-1', '-1', '-1', '-1'))
    
    out_fid.write("%s\n" % new_line)
    
    img.save(img_file)
    
  out_fid.close()
  print('')
    


def copy_dataset(config):
  if config.dataset.main_path.isdir() and \
     len(config.dataset.main_path.listdir()) > 0:
    print('FOUND COPY OF DATASET, NOT COPYING ANYTHING')
    return
  
  old_path = config.dataset.original_dataset_path
  flist = dir_util.copy_tree(old_path, config.dataset.main_path, 
                             update=1, verbose=3)
  
  # make a backup of the train/test annotation files
  path.copy(path(config.dataset.train_annos_file), path(config.dataset.train_annos_file_bk))
  path.copy(path(config.dataset.test_annos_file), path(config.dataset.test_annos_file_bk))

class ProgressBar:
    def __init__(self, iterations):
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 50
        self.__update_amount(0)

    def animate(self, iter):
        print('\r', self, end='')
        sys.stdout.flush()
        self.update_iteration(iter + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
            (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)



class AccuracyAtN(object):
  def __init__(self, scores, true_labels, class_names=None):
    self.S = pd.DataFrame(data=scores)
    if not class_names is None:
      self.S.columns=class_names
    
    tmp = np.argsort(self.S)
    self.class_order = (self.S.shape[1] - 1) - np.argsort(tmp) 
#     l = self.class_order.lookup(range(self.S.shape[0]), true_labels)
    l = np.zeros(shape=[self.S.shape[0]])
    for ii in range(l.shape[0]):
      print(true_labels[ii])
      l[ii] = self.class_order.iloc[ii][true_labels[ii]]
      
    self.rank_of_true = pd.DataFrame(data=l, index=self.S.index, columns=['Rank']) 
    
  def get_accuracy_at(self, N):
    return np.mean(self.rank_of_true < N)[0]


def makedir_if_needed(name):
  p = path(name)
  if not p.isdir():
    p.makedirs()


if __name__ == '__main__':
  pass





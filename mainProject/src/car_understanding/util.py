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
import distutils

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
#   for ii in [909]:
    progress.animate(ii)
    curr_line = content[ii]
    curr_line = curr_line.strip()
    (img_index, rel_path, domain_index,
     class_index, xmin, xmax, ymin, ymax) = curr_line.split(',')

    xmin, xmax, ymin, ymax = (float(x) for x in (xmin, xmax, ymin, ymax))

    # Read image and resize such that bounding box is of specific size
    img_file = os.path.join(main_path, rel_path)
#     print("img_file: ", img_file)
#     img = scipy.misc.imread(img_file)
    img = Image.open(img_file)
#     img, scaler = set_width_to_normalize_bb_width(img, xmin, xmax, bb_width)
    img, scaler = resize_img_to_normalize_bb_area(img, 
                                                  (xmin, ymin, xmax, ymax), 
                                                  to_area=to_area)
#     print("scaler: ", scaler)

    # Write image back to disk
#     scipy.misc.imsave(img_file, img)
    img.save(img_file)

    # change xmin, xmax, ymin, ymax to the new size
    xmin, xmax, ymin, ymax = change_bb_loc(scaler, xmin, xmax, ymin, ymax)
    xmin, xmax, ymin, ymax = (str(int(x)) for x in (xmin, xmax, ymin, ymax))

    # generate new text line
    new_line = ','.join((img_index, rel_path, domain_index,
              class_index, xmin, xmax, ymin, ymax))

    out_fid.write("%s\n" % new_line)

  out_fid.close()



def copy_dataset(old_path, new_path):
  distutils.dir_util.copy_tree(old_path, new_path, update=1)

def crop_dataset(config):
  
  # Start by copying the dataset
  
  



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



def series_to_iplot(series, name=''):
    '''
    Coverting a Pandas Series to Plotly interface
    '''
    if series.index.__class__.__name__=="DatetimeIndex":
        #Convert the index to MySQL Datetime like strings
        x = series.index.format()
        #Alternatively, directly use x, since DateTime index is np.datetime64
        #see http://nbviewer.ipython.org/gist/cparmer/7721116
        #x=df.index.values.astype('datetime64[s]')
    else:
        x = series.index.values

    line = {}
    line['x'] = x
    line['y'] = series.values
    line['name'] = name

    return [line]


def makedir_if_needed(name):
  p = path(name)
  if not p.isdir():
    p.makedirs()


if __name__ == '__main__':
  pass





'''
Util functions for the carUnderstanding projects
Created on Jan 28, 2014

@author: yair@cs.cmu.edu
'''

import pandas as pd
import os
import cv2 as cv
import scipy.misc
import matplotlib.pyplot as plt

def set_width_to_normalize_bb(img, xmin, xmax, to_width):
  w = xmax - xmin
  s = to_width / w
  img = scipy.misc.imresize(img, s)
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
def normalize_dataset(dataset, config):
  for row_tuple in dataset.iterrows():
    # row_tuple[0]=index row_tuple[1]=data
    row = row_tuple[1]
    rel_path = row['rel_path']
    img_file = os.path.join(config.dataset.main_path, rel_path)

    # Read image and resize such that bounding box is of specific size
    print 'Resize img such that BB is of width = ' + str(config.bb_width)
    img = cv.imread(img_file)
    img, scaler = set_width_to_normalize_bb(img, row['xmin'], 
                                    row['xmax'], config.bb_width)
    cv.imwrite(img, img_file)
    
    # change xmin, xmax, ymin, ymax to the new size
    

def showboxes(img_cv, boxes):
  
  img = cv.cvtColor(img_cv, cv.COLOR_BGR2RGB)
  
  ax = plt.axes()
  ax.imshow(img)
  ax.axis('off')
  
  for box in boxes:
    xmin, xmax = box[0], box[2]
    ymin, ymax = box[1], box[3]
    r = plt.Rectangle((xmin, ymin),
                   xmax-xmin, ymax-ymin,
                     edgecolor='red', facecolor='none')
    ax.add_patch(r)
  
  ax.draw()
  plt.show()
  
    
if __name__ == '__main__':
  img = cv.imread('/usr0/home/ymovshov/Documents/Research/Code/carUnderstanding/fgcomp2013_normed/release/train_images/0010/FGCOMP_0010819.jpg')
  xmin, xmax = 48.0, 441.0
  ymin, ymax = 24.0, 202.0
  img_s, scaler = set_width_to_normalize_bb(img, xmin, xmax, 200)
  xmin_s, xmax_s, ymin_s, ymax_s = change_bb_loc(scaler, xmin, xmax, ymin, ymax)
  
  showboxes(img, ((xmin, ymin, xmax, ymax),))
  plt.figure()
  showboxes(img_s, ((xmin_s, ymin_s, xmax_s, ymax_s),))  
  
  
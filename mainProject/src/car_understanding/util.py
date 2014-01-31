'''
Util functions for the carUnderstanding projects
Created on Jan 28, 2014

@author: yair@cs.cmu.edu
'''

import pandas as pd
import os
import scipy.misc
import matplotlib.pyplot as plt
from clint.textui import progress
import numpy as np
import cv2 as cv

import small_run

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
def normalize_dataset(train_annos_file, main_path, out_file, bb_width):
  # read lines from file
  with open(os.path.join(main_path, train_annos_file)) as f:
    content = f.readlines()

  out_fid = open(os.path.join(main_path, out_file), 'w')
  print "Resizing images such that BB is of width = %g" % bb_width
  for ii in progress.bar(range(len(content))):
    curr_line = content[ii]
    curr_line = curr_line.strip()
    (img_index, rel_path, domain_index,
     class_index, xmin, xmax, ymin, ymax) = curr_line.split(',')

    xmin, xmax, ymin, ymax = (float(x) for x in (xmin, xmax, ymin, ymax))

    # Read image and resize such that bounding box is of specific size
    img_file = os.path.join(main_path, rel_path)
    img = scipy.misc.imread(img_file)
    img, scaler = set_width_to_normalize_bb(img, xmin, xmax, bb_width)

    # Write image back to disk
    scipy.misc.imsave(img_file, img)

    # change xmin, xmax, ymin, ymax to the new size
    xmin, xmax, ymin, ymax = change_bb_loc(scaler, xmin, xmax, ymin, ymax)
    xmin, xmax, ymin, ymax = (str(int(x)) for x in (xmin, xmax, ymin, ymax))

    # generate new text line
    new_line = ','.join((img_index, rel_path, domain_index,
              class_index, xmin, xmax, ymin, ymax))

    out_fid.write("%s\n" % new_line)

  out_fid.close()


# boxes should be a tuple/list of tuples. each inner tuple should be
# of format (xmin, xmax, ymin, ymax)
def showboxes(img_in, boxes, is_opencv=False):

  if is_opencv:
    img = cv.cvtColor(img_in, cv.COLOR_BGR2RGB)
  else:
    img = img_in

  ax = plt.axes()
  ax.imshow(img)
  ax.axis('off')

  for box in boxes:
    xmin, xmax = box[0], box[1]
    ymin, ymax = box[2], box[3]
    r = plt.Rectangle((xmin, ymin),
                   xmax-xmin, ymax-ymin,
                     edgecolor='red', facecolor='none')
    ax.add_patch(r)

  plt.draw()
  plt.show()


def explore_training_data(train_annos, config):
  plt.ion()
  for ii in range(0, len(train_annos), 10):
    curr_anno = train_annos.iloc[ii]
    img = scipy.misc.imread(os.path.join(
                      config.dataset.main_path, curr_anno['rel_path']))
    bbox = (curr_anno['xmin'], curr_anno['xmax'],
            curr_anno['ymin'], curr_anno['ymax'])
    print 'BB width: %g' % (bbox[1] - bbox[0])
    boxes = (bbox,) # create a tuple, as it is expected by showboxes
    showboxes(img, boxes, is_opencv=False)
    c = raw_input('Press any key to continue\n')
    plt.clf()
    if c.startswith('q'):
      plt.close()
      return


  plt.close()



if __name__ == '__main__':
#   img = cv.imread('/usr0/home/ymovshov/Documents/Research/Code/carUnderstanding/fgcomp2013_normed/release/train_images/0010/FGCOMP_0010819.jpg')
#   xmin, xmax = 48.0, 441.0
#   ymin, ymax = 24.0, 202.0
#   img_s, scaler = set_width_to_normalize_bb(img, xmin, xmax, 200)
#   xmin_s, xmax_s, ymin_s, ymax_s = change_bb_loc(scaler, xmin, xmax, ymin, ymax)
#
#   showboxes(img, ((xmin, xmax, ymin, ymax),), is_opencv=True)
#   plt.figure()
#   showboxes(img_s, ((xmin_s, xmax_s, ymin_s, ymax_s),), is_opencv=True)

#   normalize_dataset('train_annos_old.txt', '../../../fgcomp2013_normed/release/',
#                      'train_annos.txt', 200)

  (dataset, config) = small_run.preprocess()
  print config.makeReport()
  train_annos = dataset['train_annos']
  explore_training_data(train_annos, config)


